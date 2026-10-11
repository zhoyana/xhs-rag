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
import time

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


# 会话令牌（xsec_token）的有效期远长于一次对话，但也确实会过期；半小时是个折中：
# 足够用户"先看看有哪些、过十分钟再说要哪几篇"，又不至于拿着隔夜的令牌去采集。
DEFAULT_TOKEN_TTL = 30 * 60


class TokenCache:
    """记下「读收藏」那一步拿到的 xsec_token，供「导入收藏」按编号取用。

    为什么要服务端记：令牌是 44 个字符的 base64，里面有 `+` `/` `=`。让模型把它
    原样抄进下一次工具调用，等于给幻觉留了个必中的口子——抄错一个字符，表现不是
    "报错"而是"这条采集失败"，用户看到的是"导入失败"却不知道为什么。
    工具之间传数据的那条通道不该经过模型：模型只需要说清"要哪几篇"，令牌由服务端
    自己接上。

    单进程假设：这里就是一个带过期的 dict，进程重启即失效（下次调用会重新读一遍
    收藏，代价是几十秒，不会有错误结果）。多实例部署时必须换成 Redis 或落库，
    否则"A 实例读到列表、请求落到 B 实例"就会查不到令牌。
    """

    def __init__(self, ttl: float = DEFAULT_TOKEN_TTL, clock=None):
        self._ttl = float(ttl)
        # 注入时钟，单测里把半小时拨过去不必真的等。
        self._clock = clock or time.monotonic
        self._entries: dict[str, dict] = {}

    def remember(self, items) -> int:
        """记下这批条目里能用来采集的那些（缺令牌的条目本来就采不了）。"""
        now = self._clock()
        for item in items or []:
            if not isinstance(item, dict):
                continue
            feed_id = _text(item.get("feed_id"))
            token = _text(item.get("xsec_token"))
            if feed_id and token:
                self._entries[feed_id] = {
                    "xsec_token": token,
                    "title": _text(item.get("title")),
                    "at": now,
                }
        self._expire(now)
        return len(self._entries)

    def feed_ids(self) -> list[str]:
        """当前还能用的编号，按记下的顺序（= 收藏列表的展示顺序）。"""
        self._expire(self._clock())
        return list(self._entries)

    def entry(self, feed_id) -> dict | None:
        self._expire(self._clock())
        found = self._entries.get(_text(feed_id))
        return dict(found) if found else None

    def forget_all(self) -> None:
        self._entries.clear()

    def _expire(self, now: float) -> None:
        for feed_id in [k for k, v in self._entries.items()
                        if now - v["at"] > self._ttl]:
            del self._entries[feed_id]
