import asyncio

from app.tools import build_langchain_tools


def test_langchain_tools_expose_stable_names_and_schema():
    tools = build_langchain_tools(searcher=lambda q, n: [], getter=lambda _: None,
                                  recaller=lambda q, n: [], drafter=lambda *args: None)
    assert [tool.name for tool in tools] == ["search_notes", "get_note", "recall_memories", "draft_plan"]
    # 秋招知识点：Agent 根据 JSON Schema 构造工具参数，参数校验在业务执行前完成。
    schema = tools[0].args_schema.model_json_schema()
    assert schema["properties"]["limit"]["maximum"] == 20
    assert tools[0].invoke({"query": "RAG", "limit": 3})["ok"] is True


def test_async_draft_tool_can_be_invoked():
    async def fake(goal, ids, date):
        return {"status": "draft", "goal": goal}

    tools = build_langchain_tools(searcher=lambda q, n: [], getter=lambda _: None,
                                  recaller=lambda q, n: [], drafter=fake)
    result = asyncio.run(tools[3].ainvoke({"goal": "面试", "source_note_ids": ["n1"], "start_date": "2026-09-16"}))
    assert result["data"]["status"] == "draft"
