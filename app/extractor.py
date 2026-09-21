"""提取模块：只读取公开网页，不接收 Cookie，不执行网页脚本。"""
import re
from urllib.parse import urlsplit, urljoin

import httpx
import json5


class ImportProblem(Exception):
    # 自定义异常类型：把“预期的导入问题”与意外程序错误区分开。
    # code 只表示失败阶段，不包含分享 URL、Token 或网页内容。
    def __init__(self, message: str, code: str = "import_error"):
        super().__init__(message)
        self.code = code


# 集合用于域名白名单判断。只做精确匹配，不信任“名字里包含小红书”的网站。
# 这只是本地原型的基础限制，不等于完整的公网 SSRF 防护。
HOSTS = {"www.xiaohongshu.com", "xiaohongshu.com", "xhslink.com", "www.xhslink.com"}


def check_url(url: str) -> str:
    # url: str 和 -> str 是类型提示，分别说明输入和返回值应为字符串。
    # 普通 Python 函数不会自动执行类型校验，因此仍需要下面的实际检查。
    try:
        parsed = urlsplit(url)
        # 将 URL 拆成协议、域名、端口等，再拒绝未知域名、用户密码和不允许的端口。
        valid = (parsed.scheme in {"https", "http"} and parsed.hostname in HOSTS
                 and not parsed.username and not parsed.password
                 and parsed.port in {None, 80, 443})
    except ValueError:
        # 例如无法解析的端口号，也要作为不合法 URL 处理。
        valid = False
    if not valid:
        raise ImportProblem("仅接受小红书官方网页或分享域名，不能访问任意地址。")
    return url


def share_url(text: str) -> str:
    # 兼容聊天中复制的 Markdown 转义。
    cleaned = text.replace("\\_", "_").replace("\\&", "&")
    # re.search 用正则表达式寻找第一个匹配项；r'...' 避免 Python 先处理反斜杠。
    # 这个模式从 http(s) 开始，读到空格或常见分享文本分隔符前停止。
    match = re.search(r'https?://[^\s<>"）)\]，]+', cleaned)
    if not match:
        raise ImportProblem("没有找到链接，请粘贴完整分享文本或 URL。")
    # group() 取匹配到的文本；rstrip() 清理尾部标点，然后再检查安全边界。
    return check_url(match.group().rstrip("。；;"))


def note_id_from_url(url: str) -> str | None:
    """只从已知的详情页路径取 ID；不读取查询参数中的访问 token。"""
    match = re.search(r"^/(?:explore|discovery/item)/([a-zA-Z0-9]+)(?:/|$)",
                      urlsplit(url).path)
    return match.group(1) if match else None


async def fetch_page(url: str) -> tuple[str, str]:
    # 返回 (网页 HTML, 可用于解析的详情 URL)。短链接跳转时保留最后见到的详情页地址。
    # 即使最终落到登录页，也不能因此丢掉已知 ID；解析阶段仍会严格核对网页内容。
    detail_url = url if note_id_from_url(url) else None
    # async with 在退出代码块时关闭网络资源；User-Agent 是请求头，不是登录凭据。
    # timeout=15 限制各类网络等待，不是整个导入过程严格的 15 秒总上限。
    # 手动检查每次重定向，防止分享链接被用作任意网址代理。
    async with httpx.AsyncClient(timeout=15, follow_redirects=False,
                                 headers={"User-Agent": "Mozilla/5.0"}) as client:
        for _ in range(5):
            # _ 表示不需要使用循环计数值。最多发起五次请求，防止无限跳转。
            check_url(url)
            async with client.stream("GET", url) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    # 这些 HTTP 状态码表示跳转；Location 头提供新的地址。
                    location = response.headers.get("location")
                    if not location:
                        raise ImportProblem("分享链接跳转地址缺失。")
                    url = urljoin(url, location)
                    if note_id_from_url(url):
                        detail_url = url
                    # continue 跳过本轮后续代码；下一轮会重新验证跳转目标的域名。
                    continue
                if response.status_code >= 400:
                    raise ImportProblem(f"来源站点返回 HTTP {response.status_code}，未导入。")
                chunks = bytearray()
                # 按块读取并累计大小，避免一次性无上限地读入整个网页。
                async for chunk in response.aiter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > 5_000_000:
                        raise ImportProblem("网页超过 5 MB，已停止读取。")
                return chunks.decode("utf-8", errors="replace"), url if note_id_from_url(url) else detail_url or url
    raise ImportProblem("链接跳转次数过多，已停止读取。")


def parse_note(html: str, final_url: str, requested_url: str | None = None) -> dict:
    # 小红书页面可能在脚本中包含初始化数据。这不是官方 API，页面变化会导致失效。
    # re.S 使正则中的点号也能匹配换行；(.*?) 捕获尽可能短的内容。
    match = re.search(r"window\.__INITIAL_STATE__\s*=\s*(.*?)</script>", html, re.S)
    if not match:
        raise ImportProblem("网页没有笔记数据：可能需要登录、内容受限或网页结构已变化。",
                            "initial_state_missing")
    final_id = note_id_from_url(final_url)
    requested_id = note_id_from_url(requested_url) if requested_url else None
    # 原始直链与跳转后的详情 ID 冲突时拒绝，防止导入别人的推荐笔记。
    if final_id and requested_id and final_id != requested_id:
        raise ImportProblem("分享链接与跳转后的笔记编号不一致，未导入。",
                            "redirect_note_id_mismatch")
    note_id = final_id or requested_id
    if not note_id:
        raise ImportProblem("未定位到笔记详情地址，可能跳转到了登录或错误页面。",
                            "note_url_missing")
    try:
        # json5 不执行 JavaScript；只把字符串外的 undefined 转成 null。
        literal = match.group(1).strip().rstrip(";")
        # 下方较复杂的正则优先匹配完整引号字符串，保留字符串里的 "undefined"。
        # lambda 是匿名小函数：有字符串捕获组就原样保留，否则替换 undefined。
        literal = re.sub(r'("(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27)|\bundefined\b',
                         lambda m: m.group(1) if m.group(1) else "null", literal)
        try:
            state = json5.loads(literal)
        except (ValueError, TypeError):
            raise ImportProblem("网页中的笔记状态无法解析，未导入。", "state_json_invalid") from None
        # JSON5 只解析文本、不执行 JavaScript；每个结构节点单独诊断。
        note_state = state.get("note") if isinstance(state, dict) else None
        details = note_state.get("noteDetailMap") if isinstance(note_state, dict) else None
        if not isinstance(details, dict):
            raise ImportProblem("网页缺少笔记详情映射，可能是页面结构变化或内容受限。",
                                "note_detail_map_missing")
        # 不直接取第一条：页面可能有推荐笔记，必须与链接中的 ID 对上。
        entry = details.get(note_id)
        if entry is None:
            # 部分页面的 map 键不同，但内部 noteId 仍是目标笔记 ID。
            matches = [value for value in details.values()
                       if isinstance(value, dict)
                       and isinstance(value.get("note"), dict)
                       and value["note"].get("noteId") == note_id]
            if len(matches) == 1:
                entry = matches[0]
        if not isinstance(entry, dict):
            raise ImportProblem("网页没有与分享链接匹配的笔记详情，未导入。",
                                "requested_note_id_missing")
        note = entry.get("note")
        if not isinstance(note, dict):
            raise ImportProblem("网页中的笔记字段结构已变化，未导入。", "note_fields_invalid")
        title = note.get("title", "").strip()
        # 字典.get(key, 默认值) 在键不存在时返回默认值；strip() 去掉首尾空白。
        content = note.get("desc", "").strip()
        if not title or not content:
            raise ImportProblem("缺少标题或正文，未按完整笔记入库；请核对原笔记。",
                                "note_content_missing")
        images = []
        for item in note.get("imageList", []):
            image_url = item.get("urlDefault") or item.get("url", "")
            # or 在第一个值为空时使用备用值；isinstance 检查是否确实是字符串。
            if isinstance(image_url, str) and image_url.startswith(("https://", "http://")):
                images.append(image_url)
        kind = note.get("type", "unknown")
        # 这里不再写"图片只存了远程地址、未下载"那句：它在采集这一刻为真，但同一轮
        # 落盘之后就成了假话，而 payload 里的警告没人会回头改（详情页因此自相矛盾）。
        # 存储状态与识别状态都随后的动作而变，改由读取时按 assets / image_text 现算，
        # 见 media.image_state_warnings。
        warnings = []
        if kind == "video":
            warnings.append("只导入视频笔记的文字描述，未提取或转写视频。")
        return {"note_id": note_id, "title": title, "content": content,
                # f"...{变量}..." 是格式化字符串。这里只保存不带访问 token 的来源地址。
                "source_url": f"https://www.xiaohongshu.com/explore/{note_id}",
                "images": images, "kind": kind, "warnings": warnings}
    except ImportProblem:
        # 单独写 raise 会把当前异常原样向外传递，保留已经写好的具体错误说明。
        raise
    except (ValueError, TypeError, KeyError, AttributeError):
        # 捕获解析与字段结构错误；不要用一个宽泛的 except 隐藏所有编程错误。
        raise ImportProblem("笔记字段不符合预期，未导入；需要更新解析器。",
                            "note_fields_invalid") from None
