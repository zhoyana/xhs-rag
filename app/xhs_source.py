"""笔记采集源：MCP 优先、网页解析回退。

同一条链接有两条采集通道：

    通道      实现                                    优点 / 代价
    MCP      xiaohongshu-mcp 的真实浏览器            和用户网页上看到的一致，
                                                     但依赖本地服务与登录态，
                                                     单次 10~30 秒
    网页解析  extractor.fetch_page + parse_note      无额外依赖，但只要页面
                                                     结构变化或跳登录页就失败

两条通道产出同一个 note 结构（字段与 parse_note 对齐），所以收件箱、批量导入、
Agent 工具都不需要知道自己走的是哪条；来源会连同 note 一起返回，写进收件箱
任务里，排查"这次为什么只采到空壳"时一眼就能看出来。

秋招知识点（多通道降级的两个细节）：
1. 只有"通道故障"才熔断。链接缺 xsec_token、笔记内容受限这类问题换网页解析
   仍可能成功，把它们也算作熔断条件会让一条坏链接拖累后面的请求。
2. 降级要有超时预算。MCP 是浏览器自动化，服务僵死时调用会长时间挂住；
   批量导入 20 条就是 20 倍的等待，所以这里给单次调用加超时，失败后短时间
   内直接跳过 MCP 通道（熔断），让整批导入快速退回网页解析。
"""
import asyncio
import inspect
import logging
import os
import time

from .extractor import fetch_page, parse_note
from .xhs_link import XhsLinkError, expand_short_link, note_from_feed_detail, parse_xhs_link
from .xhs_mcp import XhsMcpUnavailable, fetch_feed_detail

logger = logging.getLogger("xhs.source")

# 采集来源：写进收件箱任务，便于区分"是 MCP 采的"还是"网页解析采的"。
SOURCE_MCP = "mcp"
SOURCE_HTML = "html"

# 单次 MCP 采集的时间上限。浏览器自动化正常 10~30 秒，超过就当作通道故障。
MCP_TIMEOUT = float(os.getenv("XHS_MCP_TIMEOUT", "60"))
# 通道故障后的熔断时长：这段时间内直接走网页解析，不再逐条等超时。
BREAKER_SECONDS = float(os.getenv("XHS_MCP_BREAKER_SECONDS", "60"))

_blocked_until = 0.0


def reset_breaker():
    """清空熔断状态；测试与手动恢复时使用。"""
    global _blocked_until
    _blocked_until = 0.0


def circuit_open() -> bool:
    return time.monotonic() < _blocked_until


def _trip_breaker():
    global _blocked_until
    _blocked_until = time.monotonic() + BREAKER_SECONDS


async def collect_via_mcp(url: str) -> dict | None:
    """尝试用 MCP 采集一篇笔记。

    返回 note 表示采集成功；返回 None 表示这条通道这次用不了，调用方应回退到
    网页解析。这里刻意不做日志噪音控制：失败原因只记一次、不带链接正文。
    """
    if circuit_open():
        return None

    try:
        ref = parse_xhs_link(url)
        if ref["short_link"]:
            # 短链要先跟随 302 才能拿到带令牌的真实地址。
            ref = parse_xhs_link(await expand_short_link(ref["url"]))
    except XhsLinkError as exc:
        logger.info("xhs_source_link_unparsable code=%s", exc.code)
        return None
    except Exception:  # noqa: BLE001 - 短链跳转失败不是通道故障，网页解析照样能走
        logger.warning("xhs_source_short_link_failed", exc_info=True)
        return None

    if not ref["feed_id"] or not ref["xsec_token"]:
        # 没有签名令牌就没法调 MCP。这不是通道故障，不能熔断，
        # 否则一条裸链接会顺带把后续请求也踢出 MCP 通道。
        return None

    try:
        payload = await asyncio.wait_for(
            fetch_feed_detail(ref["feed_id"], ref["xsec_token"]), MCP_TIMEOUT
        )
    except asyncio.TimeoutError:
        _trip_breaker()
        logger.warning("xhs_source_mcp_timeout timeout=%s", MCP_TIMEOUT)
        return None
    except XhsMcpUnavailable:
        _trip_breaker()
        logger.warning("xhs_source_mcp_unavailable", exc_info=True)
        return None

    try:
        return note_from_feed_detail(payload)
    except XhsLinkError as exc:
        # 通道是通的，只是这篇的结果不合规（内容受限、字段变化等），
        # 换网页解析再试一次即可，不需要熔断。
        logger.info("xhs_source_mcp_payload_rejected code=%s", exc.code)
        return None


def parse_html_note(parser, html, final_url, requested_url):
    """调用 parse_note 解析网页。

    另一条并行改动给 parse_note 增加了 requested_url（用于发现"分享链接与跳转后
    笔记编号不一致"）。这里按签名适配，保证两条线在合并前各自都能跑，
    合并后自动用上更严格的校验，而不是把对方的工作覆盖掉。
    """
    try:
        accepts = "requested_url" in inspect.signature(parser).parameters
    except (TypeError, ValueError):
        accepts = False
    if accepts:
        return parser(html, final_url, requested_url=requested_url)
    return parser(html, final_url)


async def default_html_note(url: str) -> dict:
    """默认的网页解析通道。"""
    html, final_url = await fetch_page(url)
    return parse_html_note(parse_note, html, final_url, url)


async def collect_note(url: str, *, html_note=None) -> tuple[dict, str]:
    """统一采集入口，返回 (note, 来源)。

    html_note 用于注入网页解析实现（HTTP 层需要让 fetch_page / parse_note
    仍是可被测试替换的模块级引用），不传时用本模块的默认实现。
    """
    note = await collect_via_mcp(url)
    if note is not None:
        return note, SOURCE_MCP
    fallback = html_note or default_html_note
    return await fallback(url), SOURCE_HTML
