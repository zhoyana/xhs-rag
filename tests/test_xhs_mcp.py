"""xiaohongshu-mcp 桥接测试：stub MCP 客户端，不联网。"""
import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.tools import StructuredTool

from app import xhs_mcp
from app.database import NoteStore
from app.main import create_app
from app.xhs_link import XhsLinkError


def run(coro):
    """与项目其他测试保持一致：asyncio.run 驱动协程，不依赖 pytest 异步插件。"""
    return asyncio.run(coro)


def fake_tool(name):
    return StructuredTool.from_function(func=lambda: "", name=name, description="测试工具")


SAVED_NOTE_ID = "6aa3c6ff0000000025036dd4"
PC_LINK = (f"https://www.xiaohongshu.com/explore/{SAVED_NOTE_ID}"
           "?xsec_token=ABXOT8pzCMmo0h7NC6K6n-HpMsyKDQF-ltrOlOCVMrQG8=&xsec_source=pc_user")


def feed_detail_text(images=1, desc=None):
    note = {
        "noteId": SAVED_NOTE_ID,
        "title": "淘宝AI Agent开发面经",
        "desc": desc or "一面问了 ReAct 和 Function Calling 的执行链路，二面考多 Agent 协作。",
        "type": "normal",
        "imageList": [{"urlDefault": f"http://sns-webpic-qc.xhscdn.com/a/{i}!nd_dft_wlteh_webp_3"}
                      for i in range(1, images + 1)],
    }
    return json.dumps({"feed_id": SAVED_NOTE_ID, "data": {"note": note,
                                                          "comments": {"list": []}}})


class FakeDetailTool:
    """get_feed_detail 替身：记录调用参数，返回预设内容。"""

    def __init__(self, result=""):
        self.result = result
        self.calls = []

    async def ainvoke(self, args):
        self.calls.append(args)
        return self.result


@pytest.fixture(autouse=True)
def _offline_local_tools(monkeypatch):
    """收藏工具默认会读图并下载图片；单测把这两个默认实现替换成替身。

    默认实现以模块级常量暴露，工具构造时才读取，替换常量即替换默认行为。
    """
    async def fake_transcribe(images, title="", **_kwargs):
        return {"status": "ok", "text": "（测试替身转写）", "images": [], "model": "fake",
                "read": len(images), "dropped": 0, "error": None}

    def fake_download(store, note, root):
        total = len(note.get("images", []))
        return {"status": "complete", "total": total, "downloaded": total,
                "skipped": 0, "failed": 0}

    monkeypatch.setattr(xhs_mcp, "DEFAULT_TRANSCRIBE", fake_transcribe)
    monkeypatch.setattr(xhs_mcp, "DEFAULT_DOWNLOAD", fake_download)


class FakeMCPClient:
    """MultiServerMCPClient 替身：记录配置，返回预设工具或抛错。"""

    tools: list = []
    error: Exception | None = None
    builds = 0
    last_config: dict | None = None

    def __init__(self, config):
        FakeMCPClient.builds += 1
        FakeMCPClient.last_config = config
        if FakeMCPClient.error is not None:
            raise FakeMCPClient.error

    async def get_tools(self):
        return list(FakeMCPClient.tools)


@pytest.fixture
def fake_mcp(monkeypatch):
    """替换 MCP 客户端并清理缓存，保证每条测试互不污染。"""
    xhs_mcp.reset_cache()
    FakeMCPClient.tools = []
    FakeMCPClient.error = None
    FakeMCPClient.builds = 0
    FakeMCPClient.last_config = None
    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient", FakeMCPClient)
    yield FakeMCPClient
    xhs_mcp.reset_cache()


def test_only_whitelisted_tools_are_loaded(fake_mcp):
    """发布/点赞等写工具即使服务提供了，也绝不进入 agent 工具列表。"""
    fake_mcp.tools = [
        fake_tool("search_feeds"), fake_tool("publish_content"),
        fake_tool("get_feed_detail"), fake_tool("like_feed"),
    ]
    tools = run(xhs_mcp.load_xhs_tools())
    # 有 get_feed_detail 时会额外补一个"粘链接采集"的本地工具。
    assert [t.name for t in tools] == ["search_feeds", "get_feed_detail",
                                       xhs_mcp.LINK_TOOL_NAME]

    # 连接配置：Streamable HTTP 指向本地 MCP 服务。
    assert fake_mcp.last_config["xiaohongshu"]["transport"] == "streamable_http"
    # 而且必须带"不走代理"的客户端工厂：本机服务走系统代理会被代理成 502。
    assert fake_mcp.last_config["xiaohongshu"]["httpx_client_factory"] is xhs_mcp.local_only_client


def test_service_down_degrades_to_empty_and_retries(fake_mcp):
    """服务不可用时返回空列表且不缓存失败结果，下次调用会重试。"""
    fake_mcp.error = RuntimeError("connection refused")
    assert run(xhs_mcp.load_xhs_tools()) == []

    fake_mcp.error = None
    fake_mcp.tools = [fake_tool("search_feeds")]
    tools = run(xhs_mcp.load_xhs_tools())
    assert [t.name for t in tools] == ["search_feeds"]
    assert fake_mcp.builds == 2  # 失败那次没被缓存


def test_success_result_is_cached(fake_mcp):
    """成功加载后进程内缓存：同一进程内不重复做 tools/list 往返。"""
    fake_mcp.tools = [fake_tool("search_feeds")]
    first = run(xhs_mcp.load_xhs_tools())
    fake_mcp.tools = [fake_tool("search_feeds"), fake_tool("get_feed_detail")]
    second = run(xhs_mcp.load_xhs_tools())
    assert first is second
    assert fake_mcp.builds == 1


def test_chat_message_receives_mcp_tools(tmp_path, monkeypatch, fake_mcp):
    """端到端接线：MCP 工具与本地工具一起出现在 agent 的工具列表里。"""
    fake_mcp.tools = [fake_tool("search_feeds"), fake_tool("publish_content")]

    seen = {}

    async def fake_agent(history, tools, **_kwargs):
        seen["names"] = {tool.name for tool in tools}
        return {"content": "好的。", "tool_calls": [], "retrieved_note_ids": [],
                "cited_note_ids": [], "source_note_ids": [], "plan_id": None}

    monkeypatch.setattr("app.main.run_agent_turn", fake_agent)

    with TestClient(create_app(tmp_path / "mcp.sqlite3")) as client:
        thread_id = client.post("/threads", json={"title": "测试"}).json()["id"]
        response = client.post(f"/threads/{thread_id}/messages",
                               json={"content": "帮我看看站外有什么"})
        assert response.status_code == 200

    # 本地工具与白名单 MCP 工具混编；写工具被白名单挡在模型视野之外。
    assert {"search_notes", "get_note", "recall_memories",
            "search_feeds"} <= seen["names"]
    assert "publish_content" not in seen["names"]


def test_mcp_down_keeps_chat_working(tmp_path, monkeypatch, fake_mcp):
    """MCP 服务挂掉时聊天照常工作，只是工具列表里没有站外工具。"""
    fake_mcp.error = RuntimeError("connection refused")

    seen = {}

    async def fake_agent(history, tools, **_kwargs):
        seen["names"] = {tool.name for tool in tools}
        return {"content": "好的。", "tool_calls": [], "retrieved_note_ids": [],
                "cited_note_ids": [], "source_note_ids": [], "plan_id": None}

    monkeypatch.setattr("app.main.run_agent_turn", fake_agent)

    with TestClient(create_app(tmp_path / "mcp2.sqlite3")) as client:
        thread_id = client.post("/threads", json={"title": "测试"}).json()["id"]
        response = client.post(f"/threads/{thread_id}/messages",
                               json={"content": "你好"})
        assert response.status_code == 200

    assert "search_notes" in seen["names"]
    assert "search_feeds" not in seen["names"]


# ---------------- 收件箱用的采集通道 ----------------

def test_fetch_feed_detail_reports_channel_missing(fake_mcp):
    """服务不可用时是"通道故障"，采集层据此决定是否熔断。"""
    fake_mcp.tools = [fake_tool("search_feeds")]
    with pytest.raises(xhs_mcp.XhsMcpUnavailable):
        run(xhs_mcp.fetch_feed_detail("abc", "token"))


def test_call_detail_parses_json_text():
    detail = FakeDetailTool(result=feed_detail_text())
    payload = run(xhs_mcp._call_detail(detail, {"feed_id": "abc", "xsec_token": "token"}))
    assert payload["data"]["note"]["noteId"] == SAVED_NOTE_ID


def test_call_detail_accepts_content_blocks():
    """langchain-mcp-adapters 返回的是 content blocks 列表，不是纯字符串。"""
    detail = FakeDetailTool(result=[{"type": "text", "text": feed_detail_text()}])
    payload = run(xhs_mcp._call_detail(detail, {"feed_id": "abc", "xsec_token": "token"}))
    assert payload["feed_id"] == SAVED_NOTE_ID


def test_call_detail_rejects_unexpected_shape():
    """既不是字符串也不是 content blocks 时，属于"结果不合规"而非通道故障。"""
    detail = FakeDetailTool(result=12345)
    with pytest.raises(XhsLinkError) as excinfo:
        run(xhs_mcp._call_detail(detail, {"feed_id": "abc", "xsec_token": "token"}))
    assert excinfo.value.code == "payload_invalid"


def test_call_detail_maps_plain_text_failure():
    """纯文本失败提示属于"这次采集不行"，不该被当成通道故障。"""
    detail = FakeDetailTool(result="获取Feed详情失败: feed abc not found")
    with pytest.raises(XhsLinkError) as excinfo:
        run(xhs_mcp._call_detail(detail, {"feed_id": "abc", "xsec_token": "token"}))
    assert excinfo.value.code == "collect_failed"


# ---------------- 收藏工具 ----------------

def save_tool(store, media_root, *, images=1, downloader=None, transcribe=None, indexer=None):
    detail = FakeDetailTool(result=feed_detail_text(images=images))
    return xhs_mcp.build_save_note_tool(
        detail, store, media_root, transcribe=transcribe, downloader=downloader,
        indexer=indexer), detail


def test_save_tool_writes_note_assets_and_image_text(tmp_path):
    """收藏要真的落库：笔记入库、图片登记，转写出的正文也一起保存。"""
    store = NoteStore(tmp_path / "save.sqlite3")
    tool, detail = save_tool(store, tmp_path / "media")
    result = run(tool.ainvoke({"link": PC_LINK}))

    assert result["ok"] is True
    assert result["data"]["status"] == "imported"
    assert result["data"]["assets"]["downloaded"] == 1
    assert result["data"]["has_image_text"] is True

    saved = store.get(SAVED_NOTE_ID)
    assert saved["title"] == "淘宝AI Agent开发面经"
    # 图片承载型笔记的价值就在这里：转写结果随笔记落库，之后不用重采也能检索。
    assert saved["image_text"] == "（测试替身转写）"
    assert [asset["position"] for asset in store.assets(SAVED_NOTE_ID)] == [1]
    assert detail.calls[0]["feed_id"] == SAVED_NOTE_ID


def test_save_tool_reports_existing_instead_of_claiming_new(tmp_path):
    """重复收藏要如实回报 existing，否则模型会对用户谎报"已收藏成功"。"""
    store = NoteStore(tmp_path / "again.sqlite3")
    tool, _detail = save_tool(store, tmp_path / "media")

    first = run(tool.ainvoke({"link": PC_LINK}))
    second = run(tool.ainvoke({"link": PC_LINK}))

    assert first["data"]["status"] == "imported"
    assert second["data"]["status"] == "existing"
    # 重复收藏不该把图片登记成两条。
    assert len(store.assets(SAVED_NOTE_ID)) == 1


def test_save_tool_rejects_link_without_token(tmp_path):
    store = NoteStore(tmp_path / "notoken.sqlite3")
    tool, _detail = save_tool(store, tmp_path / "media")
    result = run(tool.ainvoke(
        {"link": f"https://www.xiaohongshu.com/explore/{SAVED_NOTE_ID}"}))

    assert result["ok"] is False
    assert result["error"]["code"] == "xsec_token_missing"
    assert store.get(SAVED_NOTE_ID) is None


def test_save_tool_survives_transport_error(tmp_path):
    store = NoteStore(tmp_path / "down.sqlite3")

    class BrokenDetail:
        async def ainvoke(self, args):
            raise RuntimeError("connection reset")

    tool = xhs_mcp.build_save_note_tool(BrokenDetail(), store, tmp_path / "media")
    result = run(tool.ainvoke({"link": PC_LINK}))

    assert result["ok"] is False
    assert result["error"]["code"] == "collect_unavailable"
    assert store.get(SAVED_NOTE_ID) is None


def test_save_tool_reports_write_failure_without_crashing(tmp_path):
    """写库失败只影响这次收藏，不能让整轮对话崩掉。"""
    store = NoteStore(tmp_path / "broken.sqlite3")

    def failing_download(_store, _note, _root):
        raise OSError("disk full")

    tool, _detail = save_tool(store, tmp_path / "media", downloader=failing_download)
    result = run(tool.ainvoke({"link": PC_LINK}))

    assert result["ok"] is False
    assert result["error"]["code"] == "save_failed"


# ---------------- 收藏后自动建索引 ----------------

def test_save_tool_builds_index_so_the_note_is_searchable(tmp_path):
    """收藏完要能马上语义检索：索引器必须拿到刚入库的 note_id。"""
    store = NoteStore(tmp_path / "indexed.sqlite3")
    seen = []

    def recording_indexer(_store, note_id, **_kwargs):
        seen.append(note_id)
        return {"status": "indexed", "note_id": note_id, "chunk_count": 3,
                "vector_total": 3, "searchable": True}

    tool, _detail = save_tool(store, tmp_path / "media", indexer=recording_indexer)
    result = run(tool.ainvoke({"link": PC_LINK}))

    assert result["ok"] is True
    assert seen == [SAVED_NOTE_ID]
    assert result["data"]["index"] == {"status": "indexed", "searchable": True,
                                       "chunk_count": 3, "vector_total": 3}


def test_save_tool_keeps_the_note_when_indexing_fails(tmp_path):
    """索引是可重建的派生数据：它失败不该把已存好的笔记和图片一起丢掉。"""
    store = NoteStore(tmp_path / "indexfail.sqlite3")

    def broken_indexer(_store, _note_id, **_kwargs):
        raise RuntimeError("向量库不可用")

    tool, _detail = save_tool(store, tmp_path / "media", indexer=broken_indexer)
    result = run(tool.ainvoke({"link": PC_LINK}))

    assert result["ok"] is True
    assert result["data"]["index"]["status"] == "failed"
    assert result["data"]["index"]["searchable"] is False
    assert store.get(SAVED_NOTE_ID) is not None
    assert len(store.assets(SAVED_NOTE_ID)) == 1


def test_save_tool_reports_skipped_index_for_textless_note(tmp_path):
    """纯图片且没转写出文字：索引跳过，但收藏本身仍然成功且如实回报。"""
    store = NoteStore(tmp_path / "skipped.sqlite3")

    def skipping_indexer(_store, note_id, **_kwargs):
        return {"status": "skipped", "note_id": note_id, "reason": "标题和正文不能为空"}

    tool, _detail = save_tool(store, tmp_path / "media", indexer=skipping_indexer)
    result = run(tool.ainvoke({"link": PC_LINK}))

    assert result["ok"] is True
    assert result["data"]["index"]["status"] == "skipped"
    assert result["data"]["index"]["searchable"] is False
    assert store.get(SAVED_NOTE_ID) is not None


# ---------------- 读工具与写工具的边界 ----------------

def test_collect_tool_stays_read_only(tmp_path):
    """采集工具不能有任何写入口：普通问答不该改动知识库。"""
    store = NoteStore(tmp_path / "pure-read.sqlite3")
    tool = xhs_mcp.build_link_collect_tool(FakeDetailTool(result=feed_detail_text()))
    result = run(tool.ainvoke({"link": PC_LINK}))

    assert result["ok"] is True
    assert set(tool.args) == {"link", "read_images", "with_comments"}
    assert store.list(10, 0) == []


def test_save_tool_is_mounted_only_when_a_store_is_available(fake_mcp):
    """写工具必须挂在某个知识库上；没有库就不该出现在工具列表里。

    读收藏（list_my_favorites）是只读的，但它和导入工具成对使用，所以一起随
    store 挂载：留着单独一个"只能看不能导"的入口，模型只会多一次无效调用。
    """
    fake_mcp.tools = [fake_tool("search_feeds"), fake_tool("get_feed_detail")]

    without_store = [t.name for t in run(xhs_mcp.load_xhs_tools())]
    assert xhs_mcp.SAVE_TOOL_NAME not in without_store
    assert xhs_mcp.IMPORT_FAVORITES_TOOL_NAME not in without_store

    with_store = [t.name for t in run(xhs_mcp.load_xhs_tools(
        store=object(), media_root="C:/tmp"))]
    assert with_store[:2] == ["search_feeds", "get_feed_detail"]
    # 白名单工具在前，本地补的四个工具跟在后面（顺序无关紧要，成员关系才是契约）：
    # 按链接采集 + 按链接收藏 + 读收藏夹 + 导入收藏夹。
    assert set(with_store[2:]) == {"collect_note_by_link", xhs_mcp.SAVE_TOOL_NAME,
                                   xhs_mcp.LIST_FAVORITES_TOOL_NAME,
                                   xhs_mcp.IMPORT_FAVORITES_TOOL_NAME}

    # 不传 store 时"按链接采集"仍然在（它不落库），但三写/读收藏的入口一个都不该有。
    assert "collect_note_by_link" in without_store


def test_chat_agent_can_save_notes(tmp_path, monkeypatch, fake_mcp):
    """接线检查：聊天时收藏工具与白名单 MCP 工具一起可见。"""
    fake_mcp.tools = [fake_tool("search_feeds"), fake_tool("get_feed_detail")]

    seen = {}

    async def fake_agent(history, tools, **_kwargs):
        seen["names"] = {tool.name for tool in tools}
        return {"content": "好的。", "tool_calls": [], "retrieved_note_ids": [],
                "cited_note_ids": [], "source_note_ids": [], "plan_id": None}

    monkeypatch.setattr("app.main.run_agent_turn", fake_agent)

    with TestClient(create_app(tmp_path / "save-mount.sqlite3")) as client:
        thread_id = client.post("/threads", json={"title": "测试"}).json()["id"]
        assert client.post(f"/threads/{thread_id}/messages",
                           json={"content": "帮我收藏这篇"}).status_code == 200

    assert xhs_mcp.SAVE_TOOL_NAME in seen["names"]
    assert "publish_content" not in seen["names"]


def test_chat_agent_can_import_my_favorites(tmp_path, monkeypatch, fake_mcp):
    """接线检查："让 Agent 帮我导收藏"这条路上，两个工具都得真的在模型视野里。

    `load_xhs_tools` 的单元用例证明工具造得出来，这条证明它们被**装进了一次真实
    对话**。两者缺一都可能出现"功能写好了但模型看不见"——那种故障只有用户点开
    聊天才会发现。
    """
    fake_mcp.tools = [fake_tool("search_feeds"), fake_tool("get_feed_detail")]

    seen = {}

    async def fake_agent(history, tools, **_kwargs):
        seen["names"] = {tool.name for tool in tools}
        return {"content": "好的。", "tool_calls": [], "retrieved_note_ids": [],
                "cited_note_ids": [], "source_note_ids": [], "plan_id": None}

    monkeypatch.setattr("app.main.run_agent_turn", fake_agent)

    with TestClient(create_app(tmp_path / "fav-mount.sqlite3")) as client:
        thread_id = client.post("/threads", json={"title": "测试"}).json()["id"]
        assert client.post(f"/threads/{thread_id}/messages",
                           json={"content": "把我收藏的面经都存进来"}).status_code == 200

    assert xhs_mcp.LIST_FAVORITES_TOOL_NAME in seen["names"]
    assert xhs_mcp.IMPORT_FAVORITES_TOOL_NAME in seen["names"]


# ---------------------- 本机服务不能被系统代理劫持 ----------------------
# 真机现场：采集服务在 18060 正常监听、登录也有效，但所有 MCP 工具都加载不出来，
# 日志只有一句 xhs_mcp_tools_unavailable，异常里是 `502 Bad Gateway for
# http://localhost:18060/mcp`。根因不是采集服务挂了，而是 Windows 注册表里配了
# 系统代理（实测 getproxies() 返回 http://127.0.0.1:7892），httpx 默认
# trust_env=True 把 localhost 也塞进了代理，代理对 localhost 回 502。
# 下面这条用例把"本机服务不走代理"钉死：只要哪天有人把工厂去掉，它就会红。

def test_local_mcp_client_bypasses_the_system_proxy(monkeypatch):
    import http.server
    import threading

    import httpx

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server 要求的接口名
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # 访问日志不必出现在测试输出里
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/mcp"

    # 指向一个必然连不上的代理：默认客户端会走它而失败，我们的客户端直连本机。
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    async def probe():
        # 工厂返回的是 AsyncClient（MCP 适配器要的就是异步客户端）。
        async with xhs_mcp.local_only_client() as client:
            response = await client.get(url)
            assert response.status_code == 200

    try:
        run(probe())
        # 对照组：默认设置确实会被代理带走，否则上面那条断言就没有意义了。
        with httpx.Client() as proxied:
            with pytest.raises(httpx.HTTPError):
                proxied.get(url)
    finally:
        server.shutdown()
        server.server_close()
