"""对话 Agent 测试：全部使用假模型，不联网、不消耗 Token。"""
import asyncio

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.chat_agent import plan_requested, run_agent_turn
from app.main import create_app
from app.tools import build_langchain_tools


def test_thread_messages_are_persisted(tmp_path, monkeypatch):
    async def fake_agent(history, tools):
        assert history[-1]["content"] == "帮我找 RAG 笔记"
        assert "draft_plan" not in {tool.name for tool in tools}
        return {"content": "找到一篇笔记。", "tool_calls": [{"name": "search_notes", "ok": True}],
                "retrieved_note_ids": ["n1"], "cited_note_ids": ["n1"],
                "source_note_ids": ["n1"], "plan_id": None}

    monkeypatch.setattr("app.main.run_agent_turn", fake_agent)
    with TestClient(create_app(tmp_path / "chat.sqlite3")) as client:
        created = client.post("/threads", json={"title": "秋招准备"})
        assert created.status_code == 201
        thread_id = created.json()["id"]
        response = client.post(f"/threads/{thread_id}/messages", json={"content": "帮我找 RAG 笔记"})
        assert response.status_code == 200
        thread = client.get(f"/threads/{thread_id}").json()
        assert [item["role"] for item in thread["messages"]] == ["user", "assistant"]
        assert thread["messages"][-1]["metadata"]["source_note_ids"] == ["n1"]


def test_agent_executes_tool_then_generates_answer():
    class FakeModel:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[{
                    "name": "search_notes", "args": {"query": "RAG", "limit": 2},
                    "id": "call-1", "type": "tool_call",
                }])
            return AIMessage(content="根据你的收藏，建议先学习混合检索。")

    async def fake_draft(goal, ids, date):
        return {"status": "draft"}

    tools = build_langchain_tools(
        searcher=lambda query, limit: [{"note_id": "n1", "title": "RAG 检索"}],
        getter=lambda note_id: None,
        recaller=lambda query, limit: [],
        drafter=fake_draft,
    )
    result = asyncio.run(run_agent_turn(
        [{"role": "user", "content": "我收藏过哪些 RAG 内容？"}], tools, model=FakeModel()
    ))
    assert result["retrieved_note_ids"] == ["n1"]
    assert result["cited_note_ids"] == []
    assert result["tool_calls"][0]["name"] == "search_notes"
    assert result["tool_calls"][0]["result_count"] == 1
    assert "混合检索" in result["content"]


def test_plan_tool_requires_explicit_user_intent():
    assert plan_requested("帮我检索 RAG 笔记并告诉我怎么准备") is False
    assert plan_requested("请根据这些笔记制定一周学习计划") is True
