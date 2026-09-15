"""把框架无关的业务函数适配成 LangChain StructuredTool。

这里不绑定具体模型：Tool 只描述“能做什么”，模型由上层 Agent 决定。
秋招知识点：这是适配器模式；业务层不用知道 LangChain 的存在。
"""

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, StructuredTool

from .knowledge_tools import get_note_tool, search_notes_tool
from .memory_tools import recall_memories_tool
from .plan_tools import draft_plan_tool


class SearchNotesInput(BaseModel):
    """Pydantic 会把本模型转成 JSON Schema，供 LLM 生成合法参数。"""

    query: str = Field(min_length=1, max_length=500, description="用户想查找的主题或问题")
    limit: int = Field(default=5, ge=1, le=20, description="最多返回多少篇不同笔记")


class GetNoteInput(BaseModel):
    note_id: str = Field(min_length=1, description="search_notes 返回的笔记 ID")


class RecallMemoriesInput(BaseModel):
    query: str | None = Field(default=None, max_length=500, description="当前问题；留空时读取最近记忆")
    limit: int = Field(default=10, ge=1, le=20, description="最多返回多少条有效记忆")


class DraftPlanInput(BaseModel):
    goal: str = Field(min_length=1, max_length=1000, description="用户明确表达的目标")
    source_note_ids: list[str] = Field(
        min_length=1, max_length=10,
        description="只能填写检索得到、并决定采用的笔记 ID",
    )
    start_date: str = Field(description="计划开始日期，格式 YYYY-MM-DD")


def build_langchain_tools(
    *,
    searcher: Callable[[str, int], list[dict]],
    getter: Callable[[str], dict | None],
    recaller: Callable[[str | None, int], list[dict]],
    drafter: Callable[[str, list[str], str], Awaitable[dict]],
) -> list[BaseTool]:
    """注入底层依赖，构造可直接交给 Agent 的工具列表。

    秋招知识点：依赖注入让 Tool 与数据库、模型解耦，测试时可传入 fake 函数。
    """
    async def draft(goal: str, source_note_ids: list[str], start_date: str) -> dict:
        return await draft_plan_tool(drafter, goal, source_note_ids, start_date)

    # description 不是普通接口文档，而是模型选择工具的依据；要说明何时用、何时不用。
    return [
        StructuredTool.from_function(
            name="search_notes",
            description="从用户知识库检索相关笔记。回答收藏内容或制定计划前优先调用。",
            func=lambda query, limit=5: search_notes_tool(searcher, query, limit),
            args_schema=SearchNotesInput,
        ),
        StructuredTool.from_function(
            name="get_note",
            description="根据笔记 ID 查看完整内容；仅在检索片段不足时调用。",
            func=lambda note_id: get_note_tool(getter, note_id),
            args_schema=GetNoteInput,
        ),
        StructuredTool.from_function(
            name="recall_memories",
            description="读取用户已确认的偏好、习惯和纠正，用于个性化回答与计划。",
            func=lambda query=None, limit=10: recall_memories_tool(recaller, query, limit),
            args_schema=RecallMemoriesInput,
        ),
        StructuredTool.from_function(
            name="draft_plan",
            description="根据用户目标和已检索笔记生成计划草案；不确认计划、不创建任务。",
            coroutine=draft,
            args_schema=DraftPlanInput,
        ),
    ]
