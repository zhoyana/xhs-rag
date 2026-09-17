"""对话 Agent 测试：全部使用假模型，不联网、不消耗 Token。"""
import asyncio

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from app.chat_agent import (
    asked_question_last,
    asks_clarification,
    build_hits,
    plan_requested,
    plan_thread_pending,
    run_agent_turn,
    run_agent_turn_stream,
)
from app.main import create_app
from app.tools import build_langchain_tools


def test_build_hits_keeps_display_fields_and_trims_noise():
    """命中原片是给界面看的，只留必要字段，正文压成单行并截断。"""
    hits = build_hits([
        {"note_id": "n1", "title": "面经", "score": 0.83,
         "content": "第一行\n第二行\t带制表符" + "x" * 400},
        {"title": "缺少 note_id，无法跳转"},   # 没有 ID 就没法点开，直接丢
        "不是字典",
    ])
    assert len(hits) == 1
    assert hits[0]["note_id"] == "n1" and hits[0]["score"] == 0.83
    assert "\n" not in hits[0]["snippet"] and "\t" not in hits[0]["snippet"]
    assert len(hits[0]["snippet"]) == 240
    assert build_hits([], limit=3) == []
    assert build_hits(None) == []


def test_build_hits_respects_limit():
    items = [{"note_id": f"n{i}", "content": "内容"} for i in range(20)]
    assert len(build_hits(items, limit=5)) == 5


def test_one_shot_endpoint_also_persists_retrieved_hits(tmp_path, monkeypatch):
    """一次性接口（脚本 / 安卓端）也要带上命中片段，两条链路的元数据不能分叉。"""

    async def fake_agent(history, tools, **_kwargs):
        return {"content": "找到 1 篇。", "tool_calls": [],
                "retrieved_note_ids": ["n1"], "retrieved_hits": [
                    {"note_id": "n1", "title": "面经", "score": 0.7, "snippet": "笔试考了什么"}],
                "cited_note_ids": ["n1"], "source_note_ids": ["n1"], "plan_id": None}

    monkeypatch.setattr("app.main.run_agent_turn", fake_agent)
    with TestClient(create_app(tmp_path / "hits.sqlite3")) as client:
        thread_id = client.post("/threads", json={"title": "命中"}).json()["id"]
        client.post(f"/threads/{thread_id}/messages", json={"content": "笔试考什么"})
        meta = client.get(f"/threads/{thread_id}").json()["messages"][-1]["metadata"]
        assert meta["retrieved_hits"][0]["note_id"] == "n1"


def test_thread_messages_are_persisted(tmp_path, monkeypatch):
    async def fake_agent(history, tools, **_kwargs):
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
    assert plan_requested("请根据刚才找到的笔记，制定一个从 2026-09-16 开始的一周面试准备计划") is True
    assert plan_requested("先不用制定计划，只告诉我有哪些笔记") is False


def test_explicit_plan_request_forces_draft_tool_when_model_only_writes_text():
    class ReluctantModel:
        def __init__(self):
            self.calls = 0
            self.was_forced = False

        def bind_tools(self, tools, **kwargs):
            if kwargs.get("tool_choice") == "draft_plan":
                self.was_forced = True
            return self

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="我直接写一份七天计划。")
            if self.calls == 2:
                return AIMessage(content="", tool_calls=[{
                    "name": "draft_plan",
                    "args": {"goal": "准备面试", "source_note_ids": ["n1"],
                             "start_date": "2099-09-16"},
                    "id": "draft-1", "type": "tool_call",
                }])
            return AIMessage(content="草案已经保存，请确认后再创建任务。")

    async def fake_draft(goal, ids, date):
        return {"status": "draft", "plan": {"id": 12}}

    model = ReluctantModel()
    tools = build_langchain_tools(
        searcher=lambda query, limit: [], getter=lambda note_id: None,
        recaller=lambda query, limit: [], drafter=fake_draft,
    )
    result = asyncio.run(run_agent_turn(
        [{"role": "user", "content": "制定计划", "metadata": {}}],
        tools, model=model, required_tool="draft_plan",
    ))
    assert model.was_forced is True
    assert result["plan_id"] == 12
    assert result["tool_calls"][0]["name"] == "draft_plan"


def test_plan_conversation_helpers_keep_multi_turn_context():
    """"先聊清楚再起草"就靠这三个判断，回归时最容易被当成样板删掉。"""
    assert asks_clarification("你打算从哪天开始？") is True
    assert asks_clarification("这份草案已经准备好了。") is False

    history = [
        {"role": "user", "content": "帮我制定一份一周面试准备计划"},
        {"role": "assistant", "content": "你每天能投入多少时间？"},
        {"role": "user", "content": "大概两小时，从周五开始"},
    ]
    # 用户回答里没有"计划"二字，但会话仍在商量计划，工具不能被收回，
    # 否则 agent 只能在文字里空谈，草案永远落不了库。
    assert plan_requested(history[-1]["content"]) is False
    assert asked_question_last(history) is True
    assert plan_thread_pending(history) is True

    assert plan_thread_pending([{"role": "user", "content": "帮我检索 RAG 笔记"}]) is False
    assert asked_question_last([{"role": "user", "content": "你好"}]) is False


def test_clarifying_question_is_not_treated_as_disobedience():
    """模型追问目标时不该被强行纠正：目标都没问清，逼出来的草案只会凭空定目标。"""

    class AskingModel:
        def __init__(self):
            self.forced = False

        def bind_tools(self, tools, **kwargs):
            if kwargs.get("tool_choice"):
                self.forced = True
            return self

        async def ainvoke(self, messages):
            return AIMessage(content="这份计划你想覆盖到哪天面试？每天能投入多久？")

    model = AskingModel()
    result = asyncio.run(run_agent_turn(
        [{"role": "user", "content": "帮我制定一份计划", "metadata": {}}],
        [], model=model, required_tool="draft_plan",
    ))
    assert model.forced is False
    assert result["plan_id"] is None
    assert "每天能投入多久" in result["content"]


def test_stream_clarifying_question_is_kept_not_reset():
    """流式链路同理：追问要原样保留，不能发 reset 把用户看到的字抹掉。"""

    class AskingModel:
        def bind_tools(self, tools, **kwargs):
            return self

        async def astream(self, messages):
            yield AIMessageChunk(content="你想从哪天开始？")

    async def collect():
        return [event async for event in run_agent_turn_stream(
            [{"role": "user", "content": "帮我制定一份计划", "metadata": {}}],
            [], model=AskingModel(), required_tool="draft_plan",
        )]

    events = asyncio.run(collect())
    assert [event["type"] for event in events] == ["delta", "done"]
    assert events[-1]["plan_id"] is None
    assert events[-1]["content"] == "你想从哪天开始？"


def test_normalize_tool_result_flattens_mcp_content_blocks():
    """MCP 工具返回 content blocks 列表，本地工具返回自研 dict，编排层要抹平。"""
    from app.chat_agent import normalize_tool_result

    blocks = [{"type": "text", "text": '{"title":'}, {"type": "text", "text": '"面经"}'}]
    result = normalize_tool_result(blocks)
    assert result["ok"] is True
    # 分块是同一份内容的切片，必须直接相连，不能插入换行。
    assert result["data"] == '{"title":"面经"}'

    already = {"ok": True, "data": 1, "error": None}
    assert normalize_tool_result(already) is already
    assert normalize_tool_result("纯文本")["data"] == "纯文本"


def test_agent_survives_tool_returning_content_blocks():
    """真机 bug 回归：站外 MCP 工具返回列表时，旧代码在下游 .get() 直接崩。"""
    from langchain_core.tools import StructuredTool

    class ToolCallingModel:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[{
                    "name": "collect_note_by_link", "args": {"link": "https://x/1"},
                    "id": "call-1", "type": "tool_call",
                }])
            return AIMessage(content="这篇笔记讲的是多 Agent 协作。")

    async def remote_tool(link: str) -> list:
        # langchain-mcp-adapters 包装后的真实返回形态。
        return [{"type": "text", "text": '{"title": "面经"}'}]

    tool = StructuredTool.from_function(coroutine=remote_tool,
                                        name="collect_note_by_link", description="测试")
    result = asyncio.run(run_agent_turn(
        [{"role": "user", "content": "帮我把这个链接收藏进来"}],
        [tool], model=ToolCallingModel(),
    ))
    assert result["tool_calls"][0]["name"] == "collect_note_by_link"
    assert result["tool_calls"][0]["ok"] is True
    assert "多 Agent 协作" in result["content"]
