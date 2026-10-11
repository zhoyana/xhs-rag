"""小红书「我的收藏」：读取后归一成可勾选的条目。

链路：
    get_my_profile(tab="fav") ──xhs_mcp.fetch_my_favorites──▶ 原始 payload
                              ──parse_profile──────────────▶ {profile, items}
    item.feed_id + xsec_token ──note_link──────────────────▶ 采集用笔记链接
                              ──import_from_url────────────▶ 入库（复用现有链路）

本模块是纯函数层：不联网、不碰数据库，单测喂一份 payload 就能跑；网络那一段在
`xhs_mcp.fetch_my_favorites`。

秋招知识点（"工具在白名单里"不等于"能力可达"）：`get_my_profile` 早就挂在白名单
里，但调用时从没带过 `tab` 参数，所以"读自己的收藏"这道能力其实一直不可达——
白名单按**工具名**收敛，真正可达的能力却由**参数**决定，两者的差集就是漏网的口子。
"""
from urllib.parse import quote, urlsplit
import re

from .media import IMAGE_HOSTS

# 笔记编号：实测形如 `6ab2412a000000003703c455`。它会被拼进采集链接，所以校验
# 必须严格——接口层的同名校验直接引用这一份，两处各写一遍正则迟早改漏一处。
FEED_ID_PATTERN = r"^[0-9a-zA-Z]{8,32}$"
_FEED_ID_RE = re.compile(FEED_ID_PATTERN)

# 服务只返回主页首屏已渲染出来的收藏，没有 cursor/offset。这不是我们的截断，
# 是能力边界：要翻页得改采集服务（Go）的源码。
# 实测同一账号两次读取分别拿到 20 条和 10 条——页面懒加载渲染到哪儿就抓到哪儿，
# 所以这里只说"最多"多少，不承诺固定条数。
FAVORITES_PER_PAGE = 20
LIMIT_NOTE = ("采集服务只返回主页首屏的收藏（最多 20 条，实测会随页面渲染情况"
              "在 10–20 条之间浮动），没有翻页参数；"
              "要同步更早的收藏需要改 xiaohongshu-mcp 的源码。")

NOTE_URL_TEMPLATE = ("https://www.xiaohongshu.com/explore/{feed_id}"
                     "?xsec_token={token}&xsec_source=pc_user")


class FavoritesError(Exception):
    """列表拿到了但内容不能用（结构变了、未登录返回了纯文本等）。

    与 `XhsMcpUnavailable` 分开：那类是通道问题（可熔断、要人去看服务），
    这类是这次返回的东西不对，同一条通道换次时机再来可能就对了。
    """

    def __init__(self, message, code="favorites_invalid"):
        super().__init__(message)
        self.code = code


def note_link(feed_id: str, xsec_token: str) -> str:
    """拼出采集用的笔记链接。

    feed_id 与 xsec_token 都来自服务返回，token 里含 `=`、`+`、`/`，必须百分号
    编码——`+` 在 query 里会被解析成空格，令牌当场作废。
    """
    return NOTE_URL_TEMPLATE.format(feed_id=quote(str(feed_id or ""), safe=""),
                                    token=quote(str(xsec_token or ""), safe=""))


def _text(value) -> str:
    return str(value or "").strip()


def _cover_url(cover) -> str:
    """封面地址：过与图片下载同一份图床白名单。

    服务返回的 cover 里有 urlPre / urlDefault / infoList 好几个地址，前端只拿它显示
    缩略图。这里复刻 media 的白名单校验，避免把外部返回的任意 URL 直接塞进页面
    （一条收藏被改写成内网地址或非 http 协议，就等于给页面开了个口子）。
    """
    if not isinstance(cover, dict):
        return ""
    for key in ("urlDefault", "urlPre", "url"):
        url = _text(cover.get(key))
        if not url.startswith(("http://", "https://")):
            continue
        if (urlsplit(url).hostname or "") in IMAGE_HOSTS:
            return url
    return ""


def _profile_summary(payload: dict) -> dict:
    """只留主页顶部的展示字段，别把整个 userBasicInfo 透给前端。"""
    info = payload.get("userBasicInfo")
    info = info if isinstance(info, dict) else {}
    interactions = {}
    for entry in payload.get("interactions") or []:
        if isinstance(entry, dict) and entry.get("name"):
            interactions[_text(entry["name"])] = _text(entry.get("count"))
    return {
        "nickname": _text(info.get("nickname")),
        "ip_location": _text(info.get("ipLocation")),
        "interactions": interactions,
    }


def parse_profile(payload) -> dict:
    """把 get_my_profile 的返回归一成 {profile, items, skipped}。

    skipped 是被丢掉的条目数：缺编号或缺令牌的条目采不了，与其在列表里点不动，
    不如直接不显示——但数量要如实回报，否则"收藏 20 条只列出 18 条"就成了静默丢数据。
    """
    if not isinstance(payload, dict):
        raise FavoritesError("收藏列表返回不是预期的结构。", "payload_invalid")
    feeds = payload.get("feeds")
    if not isinstance(feeds, list):
        raise FavoritesError(
            "返回里没有收藏列表；通常是采集服务未登录，或页面没渲染出「收藏」子 tab。",
            "feeds_missing",
        )

    items, skipped = [], 0
    for position, feed in enumerate(feeds):
        if not isinstance(feed, dict):
            skipped += 1
            continue
        card = feed.get("noteCard")
        card = card if isinstance(card, dict) else {}
        user = card.get("user")
        user = user if isinstance(user, dict) else {}
        interact = card.get("interactInfo")
        interact = interact if isinstance(interact, dict) else {}

        feed_id = _text(feed.get("id"))
        token = _text(feed.get("xsecToken"))
        if not _FEED_ID_RE.match(feed_id) or not token:
            # 缺令牌就调不动 get_feed_detail，编号不合规则会在导入时被接口拒掉（422）。
            # 两种情况都列不出来，直接不显示、只计数。
            skipped += 1
            continue

        items.append({
            "feed_id": feed_id,
            "xsec_token": token,
            "title": _text(card.get("displayTitle")) or "(无标题)",
            "author": _text(user.get("nickname")),
            "kind": _text(card.get("type")) or "normal",
            "liked_count": _text(interact.get("likedCount")),
            "cover": _cover_url(card.get("cover")),
            "position": position,
        })

    return {"profile": _profile_summary(payload), "items": items, "skipped": skipped}
