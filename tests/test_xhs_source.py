"""采集通道测试：MCP 优先、网页回退、超时与熔断，全部离线。"""
import asyncio
import json

import pytest

from app import xhs_mcp, xhs_source
from app.extractor import ImportProblem
from app.xhs_link import XhsLinkError

PC_LINK = ("https://www.xiaohongshu.com/explore/6aa3c6ff0000000025036dd4"
           "?xsec_token=ABXOT8pzCMmo0h7NC6K6n-HpMsyKDQF-ltrOlOCVMrQG8=&xsec_source=pc_user")
BARE_LINK = "https://www.xiaohongshu.com/explore/6aa3c6ff0000000025036dd4"

FEED_DETAIL = {"feed_id": "6aa3c6ff0000000025036dd4", "data": {"note": {
    "noteId": "6aa3c6ff0000000025036dd4",
    "title": "淘宝AI Agent开发面经",
    "desc": "一面问了 ReAct 和 Function Calling 的执行链路，二面考多 Agent 协作。",
    "type": "normal",
    "imageList": [],
}}}


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean_breaker():
    """熔断状态是进程级的，用例之间必须隔离。"""
    xhs_source.reset_breaker()
    yield
    xhs_source.reset_breaker()


def html_note(title="网页解析标题"):
    async def fake(url):
        fake.calls.append(url)
        return {"note_id": "6aa3c6ff0000000025036dd4", "title": title,
                "content": "网页解析正文", "source_url": url, "images": [],
                "kind": "normal", "warnings": []}

    fake.calls = []
    return fake


def patch_detail(monkeypatch, *, payload=None, error=None, delay=0.0):
    """替换采集通道最内层的网络调用，记录被调用次数。"""
    calls = []

    async def fake(feed_id, xsec_token, **_kwargs):
        calls.append((feed_id, xsec_token))
        if delay:
            await asyncio.sleep(delay)
        if error is not None:
            raise error
        return payload

    monkeypatch.setattr(xhs_source, "fetch_feed_detail", fake)
    return calls


# ---------------- 通道选择 ----------------

def test_mcp_is_preferred_when_available(monkeypatch):
    """MCP 能采到时就不该再去抓网页。"""
    calls = patch_detail(monkeypatch, payload=FEED_DETAIL)
    fallback = html_note()
    note, source = run(xhs_source.collect_note(PC_LINK, html_note=fallback))

    assert source == xhs_source.SOURCE_MCP
    assert note["title"] == "淘宝AI Agent开发面经"
    assert calls == [("6aa3c6ff0000000025036dd4",
                      "ABXOT8pzCMmo0h7NC6K6n-HpMsyKDQF-ltrOlOCVMrQG8=")]
    assert fallback.calls == []  # 有 MCP 结果就不该走网页解析


def test_missing_token_skips_mcp_without_calling_it(monkeypatch):
    """裸链接没有 xsec_token，压根不该去连 MCP，也不该熔断。"""
    async def exploding(*args, **kwargs):
        raise AssertionError("没有令牌时不应调用 MCP")

    monkeypatch.setattr(xhs_source, "fetch_feed_detail", exploding)
    fallback = html_note()
    note, source = run(xhs_source.collect_note(BARE_LINK, html_note=fallback))

    assert source == xhs_source.SOURCE_HTML
    assert note["title"] == "网页解析标题"
    assert xhs_source.circuit_open() is False


def test_mcp_unavailable_falls_back_and_trips_breaker(monkeypatch):
    """服务挂了要退回网页解析，并短时间内不再逐条重试 MCP。"""
    calls = patch_detail(monkeypatch, error=xhs_mcp.XhsMcpUnavailable("服务没启动"))

    first_fallback = html_note()
    _note, source = run(xhs_source.collect_note(PC_LINK, html_note=first_fallback))
    assert source == xhs_source.SOURCE_HTML
    assert xhs_source.circuit_open() is True

    # 熔断期间的第二次采集不应再触碰 MCP，否则批量导入会逐条等超时。
    second_fallback = html_note()
    _note, source = run(xhs_source.collect_note(PC_LINK, html_note=second_fallback))
    assert source == xhs_source.SOURCE_HTML
    assert len(calls) == 1


def test_mcp_timeout_falls_back(monkeypatch):
    """浏览器自动化卡住时必须靠超时兜住，不能让收件箱一直挂着。"""
    monkeypatch.setattr(xhs_source, "MCP_TIMEOUT", 0.01)
    patch_detail(monkeypatch, payload=FEED_DETAIL, delay=0.5)
    fallback = html_note()

    _note, source = run(xhs_source.collect_note(PC_LINK, html_note=fallback))
    assert source == xhs_source.SOURCE_HTML
    assert xhs_source.circuit_open() is True


def test_rejected_payload_does_not_trip_breaker(monkeypatch):
    """通道是通的、只是这篇采不到，不该熔断整条 MCP 通道。"""
    patch_detail(monkeypatch, payload={"data": {"note": {"noteId": "x", "desc": ""}}})
    fallback = html_note()

    _note, source = run(xhs_source.collect_note(PC_LINK, html_note=fallback))
    assert source == xhs_source.SOURCE_HTML
    assert xhs_source.circuit_open() is False


def test_short_link_is_expanded_before_mcp(monkeypatch):
    """短链不带编号，必须先跟随跳转拿到带令牌的地址。"""
    async def fake_expand(url, **_kwargs):
        assert url == "http://xhslink.com/a/AbCdEf"
        return PC_LINK

    monkeypatch.setattr(xhs_source, "expand_short_link", fake_expand)
    calls = patch_detail(monkeypatch, payload=FEED_DETAIL)

    note, source = run(xhs_source.collect_note("http://xhslink.com/a/AbCdEf",
                                               html_note=html_note()))
    assert source == xhs_source.SOURCE_MCP
    assert note["note_id"] == "6aa3c6ff0000000025036dd4"
    assert len(calls) == 1


def test_short_link_expansion_failure_falls_back(monkeypatch):
    """短链跳转失败不是通道故障，网页解析同样能跟随跳转。"""
    async def fake_expand(url, **_kwargs):
        raise XhsLinkError("短链跳转次数过多。", "short_link_loop")

    monkeypatch.setattr(xhs_source, "expand_short_link", fake_expand)
    fallback = html_note()
    _note, source = run(xhs_source.collect_note("http://xhslink.com/a/AbCdEf",
                                                html_note=fallback))
    assert source == xhs_source.SOURCE_HTML
    assert xhs_source.circuit_open() is False


def test_both_channels_failing_propagates_import_problem(monkeypatch):
    """两条通道都失败时，错误要原样带上去，让收件箱记录失败阶段。"""
    patch_detail(monkeypatch, error=xhs_mcp.XhsMcpUnavailable("服务没启动"))

    async def failing_html(_url):
        raise ImportProblem("网页没有笔记数据", "initial_state_missing")

    with pytest.raises(ImportProblem) as excinfo:
        run(xhs_source.collect_note(PC_LINK, html_note=failing_html))
    assert excinfo.value.code == "initial_state_missing"


# ---------------- parse_note 签名适配 ----------------

def test_parse_html_note_passes_requested_url_when_supported():
    """支持 requested_url 的解析器要收到原始链接，用于检测跳转后编号不一致。"""
    seen = {}

    def parser(html, final_url, requested_url=None):
        seen.update(html=html, final_url=final_url, requested_url=requested_url)
        return {"note_id": "x"}

    xhs_source.parse_html_note(parser, "<html>", "https://final", PC_LINK)
    assert seen["requested_url"] == PC_LINK
    assert seen["final_url"] == "https://final"


def test_parse_html_note_works_with_two_argument_parser():
    """旧版解析器只有两个参数，同样要能调用，不能因此报 TypeError。"""
    def parser(html, final_url):
        return {"note_id": "x", "final_url": final_url}

    note = xhs_source.parse_html_note(parser, "<html>", "https://final", PC_LINK)
    assert note["final_url"] == "https://final"


def test_default_html_note_uses_extractor(monkeypatch):
    """默认网页通道走 extractor，产出与 parse_note 同构的 note。"""
    html = ("<script>window.__INITIAL_STATE__=" + json.dumps({"note": {"noteDetailMap": {
        "abc123": {"note": {"title": "标题", "desc": "正文", "type": "normal",
                            "imageList": []}}}}}) + ";</script>")

    async def fake_fetch(url):
        return html, url

    monkeypatch.setattr(xhs_source, "fetch_page", fake_fetch)
    note = run(xhs_source.default_html_note("https://www.xiaohongshu.com/explore/abc123"))
    assert note["note_id"] == "abc123"
    assert note["title"] == "标题"
