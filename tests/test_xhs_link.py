"""链接采集测试：全部离线，stub 掉 MCP 调用与短链跳转。"""
import asyncio
import json

import pytest

from app import xhs_mcp
from app.xhs_link import (
    XhsLinkError,
    expand_short_link,
    note_from_feed_detail,
    parse_xhs_link,
)


def run(coro):
    return asyncio.run(coro)


PC_LINK = ("https://www.xiaohongshu.com/explore/6aa3c6ff0000000025036dd4"
           "?xsec_token=ABXOT8pzCMmo0h7NC6K6n-HpMsyKDQF-ltrOlOCVMrQG8=&xsec_source=pc_user")


def feed_detail_payload(**note_overrides):
    note = {
        "noteId": "6aa3c6ff0000000025036dd4",
        "xsecToken": "ABXOT8pz",
        "title": "淘宝AI Agent开发面经",
        "desc": "一面问了 ReAct 和 Function Calling 的执行链路，二面考多 Agent 协作。",
        "type": "normal",
        "user": {"nickname": "抹茶小熊"},
        "imageList": [
            {"width": 1242, "height": 1660,
             "urlDefault": "http://sns-webpic-qc.xhscdn.com/a/1!nd_dft_wlteh_webp_3"},
            {"width": 1242, "height": 1660,
             "urlDefault": "http://sns-webpic-qc.xhscdn.com/a/2!nd_dft_wlteh_webp_3"},
        ],
    }
    note.update(note_overrides)
    return {"feed_id": note["noteId"], "data": {"note": note,
                                                "comments": {"list": [], "hasMore": False}}}


class FakeDetailTool:
    """get_feed_detail 替身：记录参数，返回预设文本或抛错。"""

    def __init__(self, result="", error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def ainvoke(self, args):
        self.calls.append(args)
        if self.error is not None:
            raise self.error
        return self.result


# ---------------- 链接解析 ----------------

def test_parse_pc_link_keeps_token_and_source():
    """PC 详情页链接：编号在路径、令牌在 query，都要取到。"""
    ref = parse_xhs_link(PC_LINK)
    assert ref["feed_id"] == "6aa3c6ff0000000025036dd4"
    assert ref["xsec_token"] == "ABXOT8pzCMmo0h7NC6K6n-HpMsyKDQF-ltrOlOCVMrQG8="
    assert ref["xsec_source"] == "pc_user"
    assert ref["short_link"] is False


def test_parse_share_text_around_link():
    """App 分享文案里链接夹在中文之间，也要能切出来。"""
    text = ("78 淘宝AI Agent面经，值得一看 😆 复制本条信息，打开【小红书】App查看精彩内容！ "
            "https://www.xiaohongshu.com/discovery/item/6aa3c6ff0000000025036dd4"
            "?xsec_token=ABabcdefg&xsec_source=pc_share 完")
    ref = parse_xhs_link(text)
    assert ref["feed_id"] == "6aa3c6ff0000000025036dd4"
    assert ref["xsec_token"] == "ABabcdefg"


def test_parse_short_link_has_no_note_id():
    """短链本身不含编号，必须由调用方先展开。"""
    ref = parse_xhs_link("http://xhslink.com/a/AbCdEf")
    assert ref["short_link"] is True
    assert ref["feed_id"] is None


def test_parse_rejects_text_without_link():
    with pytest.raises(XhsLinkError) as excinfo:
        parse_xhs_link("这篇笔记真不错")
    assert excinfo.value.code == "url_missing"


def test_link_without_token_is_reported_not_silently_accepted():
    """裸链接（只有编号）必须能被识别出来，否则采集会以更难懂的方式失败。"""
    ref = parse_xhs_link("https://www.xiaohongshu.com/explore/6aa3c6ff0000000025036dd4")
    assert ref["feed_id"] == "6aa3c6ff0000000025036dd4"
    assert ref["xsec_token"] is None


# ---------------- 返回结构转换 ----------------

def test_note_from_feed_detail_matches_knowledge_base_shape():
    """产出的 note 必须与 extractor.parse_note 同字段，才能直接入库。"""
    note = note_from_feed_detail(feed_detail_payload())
    assert set(note) == {"note_id", "title", "content", "source_url",
                         "images", "kind", "warnings"}
    assert note["note_id"] == "6aa3c6ff0000000025036dd4"
    assert note["title"] == "淘宝AI Agent开发面经"
    assert note["kind"] == "normal"
    assert len(note["images"]) == 2
    # 来源地址不带访问令牌，避免 token 随笔记落库。
    assert note["source_url"] == "https://www.xiaohongshu.com/explore/6aa3c6ff0000000025036dd4"
    assert "xsec_token" not in note["source_url"]


def test_image_only_note_warns_that_content_is_in_pictures():
    """正文只剩话题标签时，必须如实警告，否则模型会把空壳当正文。"""
    note = note_from_feed_detail(feed_detail_payload(desc="#面经[话题]# #ai[话题]#"))
    assert any("主要内容在图片里" in w for w in note["warnings"])
    assert any("临时地址" in w for w in note["warnings"])


def test_many_topic_tags_still_count_as_empty_body():
    """标签数量一多，去掉标签后拼起来可能超过阈值——这正是真机踩到的坑。"""
    desc = ("#24h新鲜面经[话题]# #面试经验howto[话题]# #面经[话题]# #开发[话题]# #ai[话题]#"
            "#agent[话题]# #大模型[话题]# #程序员[话题]# #淘宝[话题]#")
    note = note_from_feed_detail(feed_detail_payload(desc=desc))
    assert any("主要内容在图片里" in w for w in note["warnings"])


def test_real_text_body_does_not_warn():
    """确实有正文的笔记不应该乱报"内容在图片里"。"""
    note = note_from_feed_detail(feed_detail_payload(
        desc="一面问了 ReAct、CoT 和 Self-reflection 的区别，二面重点考多 Agent 协作模式。"))
    assert not any("主要内容在图片里" in w for w in note["warnings"])


def test_missing_desc_is_rejected():
    with pytest.raises(XhsLinkError) as excinfo:
        note_from_feed_detail(feed_detail_payload(desc=""))
    assert excinfo.value.code == "note_content_missing"


def test_payload_without_note_is_rejected():
    with pytest.raises(XhsLinkError):
        note_from_feed_detail({"data": {"comments": {"list": []}}})


# ---------------- 采集工具 ----------------

def collect_tool(result="", error=None):
    return xhs_mcp.build_link_collect_tool(FakeDetailTool(result=result, error=error))


def test_collect_uses_link_parameters():
    """工具要把链接翻译成 feed_id + xsec_token 再调 MCP。"""
    detail = FakeDetailTool(result=json.dumps(feed_detail_payload()))
    tool = xhs_mcp.build_link_collect_tool(detail)
    result = run(tool.ainvoke({"link": PC_LINK}))

    assert result["ok"] is True
    assert result["data"]["note_id"] == "6aa3c6ff0000000025036dd4"
    assert detail.calls[0]["feed_id"] == "6aa3c6ff0000000025036dd4"
    assert detail.calls[0]["xsec_token"].startswith("ABXOT8pz")
    # 默认不加载全部评论，避免拖慢一次聊天。
    assert "load_all_comments" not in detail.calls[0]


def test_collect_accepts_content_block_list():
    """langchain-mcp-adapters 返回的是 content blocks 列表，不是纯字符串。"""
    blocks = [{"type": "text", "text": json.dumps(feed_detail_payload())}]
    detail = FakeDetailTool(result=blocks)
    tool = xhs_mcp.build_link_collect_tool(detail)
    result = run(tool.ainvoke({"link": PC_LINK}))
    assert result["ok"] is True
    assert result["data"]["title"] == "淘宝AI Agent开发面经"
    assert result["data"]["note_id"] == "6aa3c6ff0000000025036dd4"


def test_collect_joins_multiple_text_blocks():
    """多个文本块要拼成一份 JSON 再解析，不能只取第一块。"""
    payload = json.dumps(feed_detail_payload())
    blocks = [{"type": "text", "text": payload[:20]},
              {"type": "text", "text": payload[20:]}]
    tool = xhs_mcp.build_link_collect_tool(FakeDetailTool(result=blocks))
    result = run(tool.ainvoke({"link": PC_LINK}))
    assert result["ok"] is True


def test_collect_rejects_unknown_payload_shape():
    tool = collect_tool(result=12345)
    result = run(tool.ainvoke({"link": PC_LINK}))
    assert result["ok"] is False
    assert result["error"]["code"] == "payload_invalid"


def test_collect_without_token_explains_how_to_fix():
    tool = collect_tool(result="{}")
    result = run(tool.ainvoke(
        {"link": "https://www.xiaohongshu.com/explore/6aa3c6ff0000000025036dd4"}))
    assert result["ok"] is False
    assert result["error"]["code"] == "xsec_token_missing"
    assert "重新复制" in result["error"]["message"]


def test_collect_maps_plain_text_failure_to_error():
    """MCP 采集失败时返回的是纯文本提示，不能当正常结果交给模型。"""
    tool = collect_tool(result="获取Feed详情失败: feed 6aa3 not found in noteDetailMap")
    result = run(tool.ainvoke({"link": PC_LINK}))
    assert result["ok"] is False
    assert result["error"]["code"] == "collect_failed"


def test_collect_survives_mcp_transport_error():
    tool = collect_tool(error=RuntimeError("connection reset"))
    result = run(tool.ainvoke({"link": PC_LINK}))
    assert result["ok"] is False
    assert result["error"]["code"] == "collect_unavailable"


def test_collect_expands_short_link(monkeypatch):
    """短链先跟随跳转，再用真实地址里的编号与令牌采集。"""
    async def fake_expand(url, **_kwargs):
        assert url == "http://xhslink.com/a/AbCdEf"
        return PC_LINK

    monkeypatch.setattr("app.xhs_mcp.expand_short_link", fake_expand)
    detail = FakeDetailTool(result=json.dumps(feed_detail_payload()))
    tool = xhs_mcp.build_link_collect_tool(detail)
    result = run(tool.ainvoke({"link": "http://xhslink.com/a/AbCdEf"}))

    assert result["ok"] is True
    assert detail.calls[0]["feed_id"] == "6aa3c6ff0000000025036dd4"


def test_expand_short_link_reports_missing_location():
    """跳转响应没有 Location 时要给出可诊断错误，而不是空指针。"""
    class FakeResponse:
        status_code = 302
        headers: dict = {}

    class FakeClient:
        async def get(self, url):
            return FakeResponse()

        async def aclose(self):
            return None

    with pytest.raises(XhsLinkError) as excinfo:
        run(expand_short_link("http://xhslink.com/a/x", client=FakeClient()))
    assert excinfo.value.code == "short_link_target_missing"


# ---------------- 与主链路的接线 ----------------

def test_link_tool_is_mounted_alongside_whitelisted_tools(monkeypatch):
    """MCP 可用时，链接采集工具要跟着白名单工具一起挂进 agent。"""
    from langchain_core.tools import StructuredTool

    xhs_mcp.reset_cache()

    class FakeMCPClient:
        def __init__(self, config):
            self.config = config

        async def get_tools(self):
            return [StructuredTool.from_function(func=lambda: "", name=name,
                                                 description="测试")
                    for name in ("search_feeds", "get_feed_detail", "publish_content")]

    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient", FakeMCPClient)
    try:
        tools = run(xhs_mcp.load_xhs_tools())
    finally:
        xhs_mcp.reset_cache()

    names = [tool.name for tool in tools]
    assert names == ["search_feeds", "get_feed_detail", xhs_mcp.LINK_TOOL_NAME]
    assert "publish_content" not in names
