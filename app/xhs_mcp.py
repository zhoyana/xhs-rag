"""xiaohongshu-mcp 桥接：把外部 MCP 服务的读类工具挂进本地 Agent。

xiaohongshu-mcp（github.com/xpzouying/xiaohongshu-mcp）以 Streamable HTTP
协议提供 MCP 服务（默认 http://localhost:18060/mcp）。本模块做五件事：
连接、按白名单过滤、把"粘链接采集"补成本地工具、把"收藏进知识库"补成本地工具、
失败降级。MCP 工具与本地工具在 LangChain 里是同一抽象（BaseTool /
StructuredTool），因此 Agent 循环（run_agent_turn）零改动。
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from .media import download_note_assets
from .xhs_link import XhsLinkError, expand_short_link, note_from_feed_detail, parse_xhs_link
from .xhs_vision import transcribe_images

logger = logging.getLogger(__name__)

# 白名单：只暴露读类工具。发布/评论/点赞是真实账号操作，一旦挂给模型，
# 提示词注入或模型幻觉都可能触发真实发布——写权限永远不进 Agent。
# 秋招知识点：外部工具的 description 本身也是不可信输入，可能诱导模型
# 调用危险工具；白名单在"工具进入模型视野之前"收敛，比靠提示词约束可靠。
XHS_MCP_ALLOWED = ("search_feeds", "get_feed_detail", "list_feeds", "get_my_profile")

# 服务地址可用环境变量覆盖；与本仓库 tools 目录下的启动配置保持一致。
XHS_MCP_URL = os.getenv("XHS_MCP_URL", "http://localhost:18060/mcp")

LINK_TOOL_NAME = "collect_note_by_link"
LINK_TOOL_DESCRIPTION = (
    "按小红书链接采集笔记原文，并自动识别图片里的文字。用户直接粘贴小红书链接或分享文案时用它，"
    "比 search_feeds 更准确。小红书很多笔记把正文写在长图里，此时返回值中的 image_text 才是真正的"
    "内容——只要它非空，就必须以它为准来回答，不要再声称无法识别图片。"
    "本工具只读取、不写入知识库；用户要求收藏时改用 save_note_by_link。"
    "链接必须带 xsec_token（从 App 或网页复制的分享链接都带），令牌会过期，"
    "失败就说明链接已失效，应请用户重新复制，不要反复重试。"
)

SAVE_TOOL_NAME = "save_note_by_link"
SAVE_TOOL_DESCRIPTION = (
    "把小红书链接对应的笔记收藏进用户的知识库，并把图片下载到本地保存。"
    "只在用户明确要求「收藏 / 保存 / 存进知识库 / 加进我的资料」时使用；"
    "只是想知道内容用什么都不用写库的 collect_note_by_link，不要两个都调。"
    "返回 note_id、status（imported 新建 / existing 已存在）和图片落盘数量，"
    "必须按返回值如实汇报，不要编造保存结果。"
)

_cache: list | None = None
DEFAULT_TRANSCRIBE = transcribe_images
DEFAULT_DOWNLOAD = download_note_assets


class XhsMcpUnavailable(Exception):
    """站外采集通道本身不可用（服务未启动、未登录、传输失败、超时）。

    与"采集通了但结果不合规"区分开：只有这一类才说明通道有问题，
    采集层据此决定是否熔断，另一类换条通道重试就够了。
    """


class CollectNoteInput(BaseModel):
    """模型只需提供一个链接；是否加载评论由模型按用户意图决定。"""

    link: str = Field(min_length=8, max_length=2000,
                      description="小红书笔记链接，或包含链接的整段分享文案")
    read_images: bool = Field(
        default=True,
        description="是否识别图片里的文字（默认开启）。关闭可更快返回，但图文笔记会拿不到正文。")
    with_comments: bool = Field(default=False, description="是否加载更多评论，会更慢")


class SaveNoteInput(BaseModel):
    """收藏只接收链接：笔记内容由采集得到，模型不能自己编一份写进库。"""

    link: str = Field(min_length=8, max_length=2000,
                      description="小红书笔记链接，或包含链接的整段分享文案")


def reset_cache():
    """测试与配置变更时清空缓存；生产进程内无需调用。"""
    global _cache
    _cache = None


def _failure(code: str, message: str) -> dict:
    return {"ok": False, "data": None, "error": {"code": code, "message": message}}


def _tool_text(raw) -> str | None:
    """取出 MCP 工具返回的文本内容。

    秋招知识点（协议到框架的类型落差）：langchain-mcp-adapters 不会把 MCP 的
    content 数组拍平成字符串，而是原样转成 LangChain 的 content blocks
    —— [{ "type": "text", "text": "..." }]。直接 json.loads 会拿到 list 而报错；
    这里统一压平，顺便兼容旧版本的纯字符串返回。
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict) and raw.get("type") == "text":
        return str(raw.get("text") or "")
    if isinstance(raw, (list, tuple)):
        parts = []
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        # 文本块是同一份内容的切片，必须直接相连；加换行会切断 JSON 字符串。
        return "".join(parts)
    return None


def _detail_args(feed_id: str, xsec_token: str, with_comments: bool) -> dict:
    args = {"feed_id": feed_id, "xsec_token": xsec_token}
    if with_comments:
        args.update({"load_all_comments": True, "limit": 20})
    return args


async def _call_detail(detail_tool, args: dict) -> dict:
    """调用 MCP 的 get_feed_detail，并把三类失败翻译成可诊断的错误。

    三类失败必须分开：传输异常说明通道坏了（可熔断），返回结构不对和返回纯文本
    提示说明这次采集不行（换通道重试即可）。
    """
    try:
        raw = await detail_tool.ainvoke(args)
    except Exception as exc:  # noqa: BLE001 - 站外失败只影响这一个工具
        raise XhsMcpUnavailable("站外采集服务暂时不可用，请稍后再试。") from exc

    text = _tool_text(raw)
    if text is None:
        raise XhsLinkError("采集返回不是预期的结构。", "payload_invalid")
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        # MCP 采集失败时返回的是纯文本提示（如"缺少xsec_token参数"），不是 JSON；
        # 原样透传会把站外实现的措辞泄露给模型，这里换成对用户有意义的话。
        raise XhsLinkError("采集失败，链接可能已过期或笔记不可见。", "collect_failed") from None


def _apply_image_text(note: dict, vision: dict) -> None:
    """把读图结果并进 note，并同步修正 warnings。

    警告必须跟着事实走：图片一旦被识别成功，"内容在图片里"这句就不再成立，
    留着反而会误导模型声称自己看不到内容。
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


async def _collect_note_for_link(link: str, detail_tool, *, read_images: bool,
                                 transcribe, with_comments: bool = False) -> dict:
    """链接 → note（含图片文字）。失败抛 XhsLinkError / XhsMcpUnavailable。

    秋招知识点（多模态的位置）：图片必须在这里就地转成文字。Agent 的工具结果
    走的是文本通道，模型拿不到图片本身；把视觉调用放在工具内部，模型侧零改动。
    """
    try:
        ref = parse_xhs_link(link)
        if ref["short_link"]:
            # 短链不含笔记编号，先跟随 302 再解析。
            ref = parse_xhs_link(await expand_short_link(ref["url"]))
    except XhsLinkError:
        raise
    except Exception as exc:  # noqa: BLE001 - 短链跳转失败不该让整轮对话失败
        logger.warning("xhs_short_link_expand_failed", exc_info=True)
        raise XhsLinkError("短链展开失败，请改用完整的小红书链接。",
                           "short_link_unreachable") from exc

    if not ref["feed_id"]:
        raise XhsLinkError("链接里没有笔记编号，请复制笔记详情页的链接。", "note_id_missing")
    if not ref["xsec_token"]:
        raise XhsLinkError(
            "链接缺少 xsec_token，无法采集；请从 App 或网页重新复制完整分享链接。",
            "xsec_token_missing",
        )

    payload = await _call_detail(
        detail_tool, _detail_args(ref["feed_id"], ref["xsec_token"], with_comments))
    note = note_from_feed_detail(payload)

    # 图片地址带签名且很快过期，必须在采集的同一轮里读掉，不能留着稍后再下载。
    if read_images and note.get("images"):
        try:
            vision = await transcribe(note["images"], title=note.get("title", ""))
        except Exception:  # noqa: BLE001 - 读图失败只降级成"只有文字正文"
            logger.warning("xhs_note_transcribe_failed", exc_info=True)
            vision = {"status": "failed", "text": "", "error": "视觉识别异常。"}
        _apply_image_text(note, vision)
    return note


def build_link_collect_tool(detail_tool, transcribe=None) -> StructuredTool:
    """把 MCP 的 get_feed_detail 包成"粘链接即可采集"的工具。

    秋招知识点：MCP 侧只有 feed_id + xsec_token 的接口，但用户手里只有链接。
    参数形态的差异不该由模型去感知——由编排层补齐，模型只看到"能按链接采集"。
    """
    transcribe = transcribe or DEFAULT_TRANSCRIBE

    async def collect(link: str, read_images: bool = True, with_comments: bool = False) -> dict:
        try:
            note = await _collect_note_for_link(
                link, detail_tool, read_images=read_images,
                transcribe=transcribe, with_comments=with_comments)
        except XhsMcpUnavailable as exc:
            return _failure("collect_unavailable", str(exc))
        except XhsLinkError as exc:
            return _failure(exc.code, str(exc))
        return {"ok": True, "data": note, "error": None}

    return StructuredTool.from_function(
        name=LINK_TOOL_NAME,
        description=LINK_TOOL_DESCRIPTION,
        coroutine=collect,
        args_schema=CollectNoteInput,
    )


def _persist_note(store, media_root, note: dict, downloader=None) -> tuple[dict, str, dict]:
    """入库 + 图片落盘。返回 (保存后的 note, 状态, 素材统计)。

    与收件箱 process_intake 用同一条管线：新建走 save，重复收藏走
    refresh_note_source（刷新带签名的临时图片地址，已下载的文件保持不变）。
    downloader 可注入，单测里换成不联网的替身。
    """
    downloader = downloader or DEFAULT_DOWNLOAD
    now = datetime.now(timezone.utc).isoformat()
    if store.get(note["note_id"]) is None:
        note["created_at"] = now
        saved, _created = store.save(note)
        status = "imported"
    else:
        saved = store.refresh_note_source(note["note_id"], note, now)
        status = "existing"
    store.ensure_assets(saved)
    # 下载是同步的阻塞 IO（httpx + Pillow），由调用方放进线程执行。
    assets = downloader(store, saved, media_root)
    return saved, status, assets


def build_save_note_tool(detail_tool, store, media_root, *, transcribe=None,
                         downloader=None, collector=None) -> StructuredTool:
    """把"采集 + 入库 + 图片落盘"包成收藏工具。

    秋招知识点（写工具单独成一个工具）：采集工具保持纯读，收藏是写操作。
    拆开之后，普通问答时模型视野里没有任何写入口；提示词注入想借"读笔记"
    顺带写库，也必须先让用户明确说了收藏才会走到这里。
    """
    transcribe = transcribe or DEFAULT_TRANSCRIBE
    downloader = downloader or DEFAULT_DOWNLOAD
    collector = collector or _collect_note_for_link

    async def save(link: str) -> dict:
        try:
            note = await collector(link, detail_tool, read_images=True,
                                   transcribe=transcribe, with_comments=False)
        except XhsMcpUnavailable as exc:
            return _failure("collect_unavailable", str(exc))
        except XhsLinkError as exc:
            return _failure(exc.code, str(exc))

        try:
            saved, status, assets = await asyncio.to_thread(
                _persist_note, store, media_root, note, downloader)
        except Exception:  # noqa: BLE001 - 写库失败不该让整轮对话崩掉
            logger.warning("xhs_save_note_failed", exc_info=True)
            return _failure("save_failed", "采集成功但写入知识库失败，请稍后重试。")

        return {"ok": True, "data": {
            "note_id": saved["note_id"],
            "title": saved["title"],
            # 收藏是否新建，直接影响怎么向用户汇报（"已收藏" vs "之前就收藏过"）。
            "status": status,
            "assets": assets,
            "has_image_text": bool(str(saved.get("image_text") or "").strip()),
            "warnings": saved.get("warnings", []),
        }, "error": None}

    return StructuredTool.from_function(
        name=SAVE_TOOL_NAME,
        description=SAVE_TOOL_DESCRIPTION,
        coroutine=save,
        args_schema=SaveNoteInput,
    )


async def _cached_tools() -> list:
    """白名单 MCP 工具 + 本地"粘链接采集"工具，成功后进程内缓存。

    秋招知识点（外部依赖的降级设计）：xiaohongshu-mcp 是增强路径——
    它挂掉只意味着 agent 少几个站外工具，绝不能让一次聊天请求 500。
    成功加载后做缓存：MCP 工具 schema 在服务运行期间是稳定的，
    不必每条聊天消息都做一次 tools/list 往返；失败则不缓存，下次还会重试。
    """
    global _cache
    if _cache is not None:
        return _cache
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({
            "xiaohongshu": {"transport": "streamable_http", "url": XHS_MCP_URL},
        })
        tools = [t for t in await client.get_tools() if t.name in XHS_MCP_ALLOWED]
        # 只有拿得到 get_feed_detail 时才补本地工具，否则它们无从执行。
        detail = get_detail_tool_from(tools)
        if detail is not None:
            tools.append(build_link_collect_tool(detail))
        _cache = tools
        logger.info("xhs_mcp_tools_loaded count=%d url=%s", len(tools), XHS_MCP_URL)
        return tools
    except Exception:  # noqa: BLE001 - 外部服务任何故障都降级为"没有这些工具"
        logger.warning("xhs_mcp_tools_unavailable url=%s", XHS_MCP_URL, exc_info=True)
        return []


def get_detail_tool_from(tools: list):
    return next((t for t in tools if t.name == "get_feed_detail"), None)


async def get_detail_tool():
    """取 MCP 的 get_feed_detail 工具；服务不可用时返回 None。"""
    return get_detail_tool_from(await _cached_tools())


async def fetch_feed_detail(feed_id: str, xsec_token: str, *,
                            with_comments: bool = False) -> dict:
    """给收件箱等非 Agent 调用方用的采集入口。

    与 Agent 工具共用同一条 MCP 通道，因此失败语义一致：抛 XhsMcpUnavailable
    表示通道问题（采集层会熔断），抛 XhsLinkError 表示这次采集结果不可用。
    """
    detail = await get_detail_tool()
    if detail is None:
        raise XhsMcpUnavailable("站外采集服务不可用，请确认 xiaohongshu-mcp 已启动并登录。")
    return await _call_detail(detail, _detail_args(feed_id, xsec_token, with_comments))


async def load_xhs_tools(*, store=None, media_root=None) -> list:
    """加载站外工具。

    不传 store 时返回"白名单 MCP 工具 + 链接采集工具"（无状态调用方，如采集层）；
    传入 store 时再追加收藏工具——写操作需要落地到某个知识库，没有库就没有它。
    """
    tools = await _cached_tools()
    if store is None:
        return tools
    detail = get_detail_tool_from(tools)
    if detail is None:
        return tools
    return tools + [build_save_note_tool(detail, store, media_root)]
