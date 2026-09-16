"""xiaohongshu-mcp 桥接测试：stub MCP 客户端，不联网。"""
import asyncio

import pytest
from fastapi.testclient import TestClient
from langchain_core.tools import StructuredTool

from app import xhs_mcp
from app.main import create_app


def run(coro):
    """与项目其他测试保持一致：asyncio.run 驱动协程，不依赖 pytest 异步插件。"""
    return asyncio.run(coro)


def fake_tool(name):
    return StructuredTool.from_function(func=lambda: "", name=name, description="测试工具")


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
