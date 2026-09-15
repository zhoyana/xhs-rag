import asyncio

from app.tools import get_note_tool, recall_memories_tool, search_notes_tool, draft_plan_tool


def test_knowledge_tools_have_stable_result_shape():
    assert search_notes_tool(lambda q, n: [{"title": q}], "  rag ", 1)["data"][0]["title"] == "rag"
    assert get_note_tool(lambda _: None, "missing")["error"] == "笔记不存在"


def test_tools_validate_inputs():
    assert search_notes_tool(lambda *_: [], "", 5)["ok"] is False
    assert recall_memories_tool(lambda *_: [], limit=0)["ok"] is False


def test_draft_tool_is_explicitly_a_draft():
    async def fake(goal, ids, date):
        return {"status": "draft", "goal": goal, "source_note_ids": ids, "start_date": date}

    result = asyncio.run(draft_plan_tool(fake, "准备 RAG", ["n1"], "2026-09-16"))
    assert result["ok"] is True
    assert result["data"]["status"] == "draft"
