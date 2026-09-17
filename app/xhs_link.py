"""小红书链接解析：把用户复制的链接（或整段分享文案）转成采集所需参数。

链路：
    链接 --parse_xhs_link--> (feed_id, xsec_token)
         --MCP get_feed_detail--> 原始 payload
         --note_from_feed_detail--> 知识库标准 note

本模块刻意做成"纯函数 + 零数据库依赖"，只有短链展开需要网络，单独放在
expand_short_link 里，便于单测与替换。

秋招知识点（外部资源的可信边界）：链接来自用户输入，属于不可信数据。
这里只做白名单式提取——认不出笔记编号就拒绝，绝不把原始链接当请求目标
去访问任意站点（避免 SSRF）。
"""
import re
from urllib.parse import parse_qs, urljoin, urlsplit

# 分享文案里的链接常常夹在中文之间，先按"非空白、非中文标点"切出候选链接。
_URL_PATTERN = re.compile(r"https?://[^\s\u4e00-\u9fff，。！？；：、（）()【】《》“”]+")
# 笔记编号在路径里：/explore/<id> 或 /discovery/item/<id>。只接受十六进制风格的 ID。
_NOTE_ID_PATTERN = re.compile(r"/(?:explore|discovery/item)/([0-9a-zA-Z]{8,32})")
_SHORT_HOSTS = {"xhslink.com", "www.xhslink.com"}
# 小红书话题标签形如 "#话题名[话题]#"。这里必须用贪婪匹配：[^#\s]{0,40}? 这样的
# 懒惰写法会先尝试匹配 0 个字符，而结尾的 #? 又能匹配空，于是一个标签只被删掉
# 开头那个 #，标签正文照样留下——实测中十来个标签拼起来就超过了正文阈值，
# "内容其实在图片里"的笔记被误判成有正文。
_TOPIC_PATTERN = re.compile(r"#[^#\s]{1,40}#?")

# 去掉话题标签后正文短于这个长度，就认为内容主要不在文字里。
MIN_TEXT_CONTENT = 40

# 评论：MCP 的 get_feed_detail **不传任何参数**就会带回前 10 条一级评论（实测 2026-09），
# 所以评论是"白拿"的。这里的上限只是防止开了 load_all_comments 之后把上百条评论灌进 note。
MAX_COMMENTS = 20
# 每条一级评论最多留几条二级回复，避免"长尾回复"把工具返回值撑大。
MAX_REPLIES_PER_COMMENT = 3
# 二级回复多数很短；超过这个长度的基本是复制粘贴的长评，截断即可。
MAX_REPLY_CHARS = 200


def text_body_length(content: str) -> int:
    """去掉话题标签后的正文长度。

    小红书的话题标签（#xxx[话题]#）不算内容，判断"内容是不是都在长图里"
    要看剥掉它们之后还剩多少字。
    """
    return len(_TOPIC_PATTERN.sub("", content or "").strip())


def content_lives_in_images(note: dict) -> bool:
    """正文是不是空壳（真正的内容在长图里）。

    采集层据此决定要不要花一次视觉调用来读图：正文够长的笔记不必付这个成本，
    只有正文只剩话题标签的才需要。
    """
    return text_body_length(note.get("content", "")) < MIN_TEXT_CONTENT

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36")


class XhsLinkError(ValueError):
    """链接或采集结果本身的问题；code 用于让 Agent 和日志可诊断。"""

    def __init__(self, message: str, code: str = "link_invalid"):
        super().__init__(message)
        self.code = code


def extract_url(text: str) -> str:
    """从整段分享文案里取出第一个链接。"""
    if not isinstance(text, str) or not text.strip():
        raise XhsLinkError("没有找到链接，请粘贴小红书的分享链接。", "url_missing")
    match = _URL_PATTERN.search(text)
    if match is None:
        raise XhsLinkError("没有找到链接，请粘贴小红书的分享链接。", "url_missing")
    # 分享文案的链接结尾常跟着中文标点或多余字符，交给 urlsplit 再规范一次。
    return match.group(0)


def parse_xhs_link(text: str) -> dict:
    """从链接或分享文案提取采集参数。

    返回 url / feed_id / xsec_token / xsec_source / short_link。
    feed_id 可能为 None（短链、或链接被截断）；xsec_token 同理。
    """
    url = extract_url(text)
    parts = urlsplit(url)
    query = parse_qs(parts.query)
    token = query.get("xsec_token", [None])[0]
    match = _NOTE_ID_PATTERN.search(parts.path)
    return {
        "url": url,
        "feed_id": match.group(1) if match else None,
        "xsec_token": token or None,
        "xsec_source": query.get("xsec_source", [None])[0],
        "short_link": parts.netloc.lower() in _SHORT_HOSTS,
    }


async def expand_short_link(url: str, *, client=None) -> str:
    """跟随 xhslink 短链的 302，拿到带笔记编号的真实地址。

    短链本身不含 note_id，必须先跳转；跳转目标同样要接受后续的白名单校验。
    """
    import httpx

    own_client = client is None
    client = client or httpx.AsyncClient(
        timeout=10, follow_redirects=False,
        headers={"User-Agent": USER_AGENT},
    )
    try:
        target = url
        for _ in range(5):
            response = await client.get(target)
            if response.status_code not in {301, 302, 303, 307, 308}:
                return target
            location = response.headers.get("location")
            if not location:
                raise XhsLinkError("短链跳转缺少目标地址。", "short_link_target_missing")
            target = urljoin(target, location)
        raise XhsLinkError("短链跳转次数过多。", "short_link_loop")
    finally:
        if own_client:
            await client.aclose()


def _comment_entry(item) -> dict | None:
    """把 MCP 的一条评论清洗成紧凑字段；没有正文的评论直接丢弃。"""
    if not isinstance(item, dict):
        return None
    content = str(item.get("content") or "").strip()
    if not content:
        return None
    user = item.get("userInfo") if isinstance(item.get("userInfo"), dict) else {}
    return {
        # 昵称在 userInfo.nickname；同一结构里的 nickName 实测是空串，别取错。
        "author": str(user.get("nickname") or "").strip(),
        "content": content,
        # 小红书给的是热度表示（"1千+"）而不是数字，原样保留比硬转成数字更诚实。
        "like_count": str(item.get("likeCount") or "").strip(),
        "ip_location": str(item.get("ipLocation") or "").strip(),
    }


def extract_comments(data) -> list[dict]:
    """从 get_feed_detail 的 payload 里抽出评论（一级 + 精简的二级回复）。

    实测（2026-09）确认的两件事，直接决定了这里怎么写：

    1. 容器与 note **平级**：``data.comments = {"cursor", "hasMore", "list"}``，
       而且**没有 count/total 字段**，别按总数写代码。
    2. **不传任何参数，MCP 也会返回前 10 条一级评论**，二级回复也已经带了一部分。
       所以评论不需要额外开关、不额外耗时——它此前完全是"白拿没要"，
       payload 里一直都有，只是这里没读 `data.comments` 就丢掉了。

    秋招知识点（外部数据的清洗边界）：评论是不可信输入，只做白名单式取字段，
    不认识的结构直接忽略，别把整个 dict 原样存进知识库。
    """
    if not isinstance(data, dict):
        return []
    box = data.get("comments")
    if isinstance(box, dict):
        raw = box.get("list")
    elif isinstance(box, list):
        raw = box
    else:
        return []
    if not isinstance(raw, list):
        return []

    comments = []
    for item in raw:
        entry = _comment_entry(item)
        if entry is None:
            continue
        replies = []
        subs = item.get("subComments") if isinstance(item, dict) else None
        if isinstance(subs, list):
            for sub in subs[:MAX_REPLIES_PER_COMMENT]:
                reply = _comment_entry(sub)
                if reply is None:
                    continue
                if len(reply["content"]) > MAX_REPLY_CHARS:
                    reply["content"] = reply["content"][:MAX_REPLY_CHARS] + "…"
                replies.append(reply)
        if replies:
            entry["replies"] = replies
        comments.append(entry)
        if len(comments) >= MAX_COMMENTS:
            break
    return comments


def note_from_feed_detail(payload) -> dict:
    """把 MCP get_feed_detail 的返回转成知识库标准 note。

    字段与 extractor.parse_note 对齐，因此产出的 note 可以直接交给
    NoteStore.save() 走原有的入库、素材下载与切分管线。

    比网页解析通道多一个 `comments` 字段（评论区内容），由 extract_comments 抽取。
    """
    if not isinstance(payload, dict):
        raise XhsLinkError("采集返回不是预期的结构。", "payload_invalid")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    note = data.get("note") if isinstance(data.get("note"), dict) else None
    if not note:
        raise XhsLinkError("采集结果里没有笔记内容。", "note_missing")

    note_id = str(note.get("noteId") or payload.get("feed_id") or "").strip()
    if not note_id:
        raise XhsLinkError("采集结果里没有笔记编号。", "note_id_missing")

    title = str(note.get("title") or "").strip()
    content = str(note.get("desc") or "").strip()
    if not content:
        raise XhsLinkError("采集到的笔记没有正文，可能内容受限。", "note_content_missing")

    images = []
    for item in note.get("imageList") or []:
        if not isinstance(item, dict):
            continue
        url = item.get("urlDefault") or item.get("urlPre") or item.get("url") or ""
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            images.append(url)

    kind = str(note.get("type") or "unknown")
    warnings = []
    # 小红书大量笔记把内容写在长图里，正文只剩话题标签；这里必须如实上报，
    # 否则模型会把空壳当正文，编出不存在的内容。
    text_only = _TOPIC_PATTERN.sub("", content).strip()
    if len(text_only) < MIN_TEXT_CONTENT:
        warnings.append(
            f"去掉话题标签后文字正文仅 {len(text_only)} 字，主要内容在图片里，"
            "需要对图片做识别才能得到完整内容。"
        )
    if kind == "video":
        warnings.append("视频笔记：只取到文字描述，未转写视频内容。")
    if not title:
        title = text_only[:30] or "未命名笔记"
        warnings.append("原笔记没有标题，已用正文开头代替。")
    if images:
        warnings.append(f"笔记含 {len(images)} 张图片，链接为带签名的临时地址，需及时下载。")

    return {
        "note_id": note_id,
        "title": title,
        "content": content,
        # 与 parse_note 一致：只保存不带访问令牌的来源地址，避免 token 落库。
        "source_url": f"https://www.xiaohongshu.com/explore/{note_id}",
        "images": images,
        "kind": kind,
        "warnings": warnings,
        # 评论随笔记一起落库（存进 notes.payload，无需改表结构）。
        # 这里**始终**带上这个键：采集侧用 `.get("comments")` 判断"这条通道能不能给评论"，
        # 而网页解析通道不产生该键，于是刷新正文时它会保留上一次采到的评论而不是清空。
        "comments": extract_comments(data),
    }
