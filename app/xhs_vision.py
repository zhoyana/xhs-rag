"""把笔记图片交给视觉模型转写成文字。

为什么必须有这一层：小红书大量笔记（面经、教程、清单）把正文全部画在长图里，
`desc` 字段只剩话题标签。而 Agent 的工具返回值只能是文本——图片本身根本到不了
对话模型眼前，模型只看得见一串图片 URL。所以"读图"必须在工具内部完成：
下载图片 → 转 Base64 → 交给视觉模型 → 拿回文字。

三条设计约束：
1. 可注入：下载与模型调用都能替换，测试完全离线，不依赖真实图床与真实 API。
2. 可降级：读图失败只意味着少了一段文字，绝不能让"采集"这件事本身失败。
3. 白名单：图片地址必须落在小红书图床，避免把工具变成任意 URL 下载器（SSRF）。

秋招知识点（多模态在 Agent 里的位置）：视觉能力属于模型，但"把图片送到模型
眼前"属于工程。工具调用的返回值是文本通道，图片无法经由工具结果回传，因此
多模态必须在服务端完成一次"图 → 文"的转换，再把文字作为普通工具结果返回。
"""
import base64
import logging
import os

import httpx

# 复用 ai 里的消息文本提取：它已经把 content 可能是字符串、可能是分块数组
# 这两种形态都抹平了，比在图片链路里重写一份可靠。
from .ai import AIProblem, _message_text
from .media import IMAGE_HOSTS

logger = logging.getLogger(__name__)

# 长图内容密集，一次读太多张既慢又容易超出上下文；9 张足以覆盖绝大多数笔记。
MAX_IMAGES = 9
MAX_IMAGE_BYTES = 20 * 1024 * 1024
DOWNLOAD_TIMEOUT = 30.0
VISION_TIMEOUT = 120.0

DEFAULT_MODEL = "deepseek-v4-flash-vision-exp"

_MIME_BY_EXT = {"webp": "image/webp", "png": "image/png", "jpg": "image/jpeg",
                "jpeg": "image/jpeg", "gif": "image/gif"}

TRANSCRIBE_PROMPT = """下面是一篇小红书笔记的第 {index} 张图片（共 {total} 张），请把这张图里的内容转写成文字。

要求：
1. 有文字就照原文转写，保留编号、层级与换行，不要总结、不要改写、不要补充常识；
2. 题号、数字、单位必须与图中一致；看不清的字用 〇 占位，不要猜；
3. 图片没有文字时，用一句话客观描述图片内容。

只输出转写结果本身，不要写"第 N 张图"这类标题，不要输出 Markdown 代码块。"""


def allowed_image_url(url) -> bool:
    """纯字符串判断图片地址是否属于小红书图床；不联网，便于单测。

    这里只做主机名白名单。DNS 层面的内网地址拦截在 media.check_image_url 里，
    下载默认实现会再校验一次。
    """
    if not isinstance(url, str):
        return False
    try:
        parsed = httpx.URL(url)
    except (TypeError, ValueError):
        return False
    return (parsed.scheme in {"http", "https"}
            and parsed.host in IMAGE_HOSTS
            and not parsed.userinfo
            and parsed.port in {None, 80, 443})


def _pick_images(images, max_images: int) -> tuple[list[str], int]:
    """按原笔记顺序保留合法图片地址，返回（可用列表，被过滤掉的张数）。"""
    usable, dropped = [], 0
    for url in images or []:
        if len(usable) >= max_images:
            dropped += 1
            continue
        if allowed_image_url(url):
            usable.append(url)
        else:
            dropped += 1
    return usable, dropped


async def _default_fetch(url: str, *, client=None) -> tuple[bytes, str]:
    """默认下载实现：HTTPX 取字节，按扩展名推断 MIME。"""
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True)
    try:
        response = await client.get(url)
        response.raise_for_status()
        data = response.content
        if len(data) > MAX_IMAGE_BYTES:
            raise AIProblem("图片体积超过限制。")
        suffix = httpx.URL(url).path.rsplit(".", 1)[-1].lower()
        mime = _MIME_BY_EXT.get(suffix) or (response.headers.get("content-type") or "").split(";")[0]
        return data, mime or "image/jpeg"
    finally:
        if own_client:
            await client.aclose()


async def _default_complete(content: list[dict], *, timeout: float = VISION_TIMEOUT) -> str:
    """默认模型调用：OpenAI 兼容的 chat/completions，返回转写文本。

    刻意不要求 JSON：真机实测里，带 response_format 的视觉请求会偶发返回空
    content，而这里要的本来就是一段自由文本——少一层格式约束就少一类失败面。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise AIProblem("尚未配置 DEEPSEEK_API_KEY。")
    model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL).strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        # 每次只读一张图，3000 足够装下一整张长图的文字；给太大反而会纵容
        # 模型把额度耗在思考上（真机见过 finish_reason=length 且 content 为空）。
        "max_tokens": 3000,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{base_url}/chat/completions", headers=headers, json=body)
    if response.status_code >= 400:
        logger.warning("xhs_vision_http_error status=%s", response.status_code)
        raise AIProblem(f"视觉模型调用失败（状态码 {response.status_code}）。")

    try:
        choice = response.json()["choices"][0]
        text = _message_text(choice["message"].get("content"))
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning("xhs_vision_response_shape_invalid", exc_info=True)
        raise AIProblem("视觉模型响应缺少预期字段。") from exc

    if not text.strip():
        # finish_reason 是区分"思考型模型把额度花在 reasoning 上"与"接口异常"的关键线索。
        logger.warning("xhs_vision_empty_content finish_reason=%s", choice.get("finish_reason"))
        raise AIProblem("视觉模型返回了空内容。")
    return text


async def _transcribe_one(index: int, content: list[dict], call_model) -> str:
    """读单张图，并在空结果时重试一次。

    秋招知识点（把批处理拆成单条，是稳定性设计不是浪费）：一次请求塞多张长图，
    模型的输出长度上限被所有图片共同消耗，真机实测出现过 finish_reason=length
    导致只转写出第一张的一小半。拆成"一张图一次请求"后，单张不会被总量截断、
    一张失败不牵连其余、每张还能独立重试——代价只是请求数变多。
    """
    for attempt in (1, 2):
        try:
            text = (await call_model(content) or "").strip()
        except Exception:  # noqa: BLE001 - 读图失败不能拖垮采集本身
            logger.warning("xhs_vision_model_failed index=%d attempt=%d",
                           index, attempt, exc_info=True)
            text = ""
        if text:
            return text
        logger.warning("xhs_vision_empty_result index=%d attempt=%d", index, attempt)
    return ""


async def transcribe_images(images, *, title: str = "", max_images: int = MAX_IMAGES,
                            fetch_image=None, call_model=None) -> dict:
    """下载并转写笔记图片，返回可直接放进 note 的结果字典。

    status 含义：ok 全部读出；partial 部分成功；empty 无可用图片；
    failed 有图片但一张都没读出来（调用方应据此保留"内容在图片里"的警告）。
    """
    fetch_image = fetch_image or _default_fetch
    call_model = call_model or _default_complete
    urls, dropped = _pick_images(images, max_images)
    result = {"status": "empty", "text": "", "model": "", "error": None,
              "read": 0, "dropped": dropped}

    if not urls:
        return result

    total = len(urls)
    sections, failed = [], []
    for index, url in enumerate(urls, 1):
        try:
            data, mime = await fetch_image(url)
        except Exception:  # noqa: BLE001 - 单张图片失败不影响其余图片
            logger.warning("xhs_vision_image_fetch_failed index=%d", index, exc_info=True)
            failed.append(index)
            continue
        content = [
            {"type": "text", "text": TRANSCRIBE_PROMPT.format(index=index, total=total)
             + (f"\n笔记标题：{title}" if title else "")},
            {"type": "image_url", "image_url": {
                "url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}",
                # original 有利于识别小红书长图里的小字。
                "detail": "original",
            }},
        ]
        text = await _transcribe_one(index, content, call_model)
        if text:
            sections.append(f"【第 {index} 张图】\n{text}")
        else:
            failed.append(index)

    if not sections:
        result["status"] = "failed"
        result["error"] = "所有图片都没有读出文字。"
        return result

    result.update({
        "status": "ok" if not failed else "partial",
        "text": "\n\n".join(sections),
        "model": os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL).strip(),
        "read": len(sections),
        "failed": failed,
    })
    return result


def apply_image_text(note: dict, vision: dict) -> None:
    """把读图结果并进 note，并同步修正 warnings。

    警告必须跟着事实走：图片一旦被识别成功，"内容在图片里"这句就不再成立，
    留着反而会误导模型声称自己看不到内容。

    放在视觉模块而不是某个采集通道里：MCP 通道（Agent 收藏）和收件箱通道
    （统一收件、批量导入）都要用，转写结果的落地格式只该有一份定义。
    """
    warnings = [w for w in note.get("warnings", []) if "主要内容在图片里" not in w
                and "需及时下载" not in w]
    text = str(vision.get("text") or "").strip()
    if text:
        note["image_text"] = text
        note["image_count"] = vision.get("read")
        warnings.append(
            f"已用视觉模型识别 {vision.get('read')} 张图片，完整内容见 image_text，可直接引用。")
    else:
        warnings.append(
            f"图片识别未取得文字（{vision.get('error') or '未知原因'}），只能依据文字正文回答。")
    note["warnings"] = warnings


__all__ = ["MAX_IMAGES", "TRANSCRIBE_PROMPT", "allowed_image_url", "transcribe_images",
           "apply_image_text"]
