"""把业务函数适配成 LangChain Tool。

这里不绑定具体模型：Tool 只描述“能做什么”，模型由上层 Agent 决定。
"""

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from .knowledge_tools import get_note_tool, search_notes_tool
from .memory_tools import recall_memories_tool
from .plan_tools import draft_plan_tool


class SearchNotesInput(BaseModel):
    query: str = Field(description="用户想查找的主题或问题")
    limit: int = Field(default=5, ge=1, le=20, description="最多返回多少条结果")


class GetNoteInput(BaseModel):
    note_id: str = Field(description="笔记的唯一 ID")


class RecallMemoriesInput(BaseModel):
    query: str | None = Field(default=None, description="与用户问题相关的关键词")
    limit: int = Field(default=10, ge=1, le=20)


class DraftPlanInput(BaseModel):
    goal: str = Field(description="用户希望达成的目标")
    source_note_ids: list[str] = Field(description="允许参考的笔记 ID")
    start_date: str = Field(description="计划开始日期，格式 YYYY-MM-DD")


def build_langchain_tools(*, searcher, getter, recaller, drafter) -> list[StructuredTool]:
    """注入底层依赖，构造可直接交给 Agent 的工具列表。

    秋招知识点：依赖注入让 Tool 与数据库、模型解耦，测试时可传入 fake 函数。
    """
    async def draft(goal: str, source_note_ids: list[str], start_date: str) -> dict:
        return await draft_plan_tool(drafter, goal, source_note_ids, start_date)

    return [
        StructuredTool.from_function(
            name="search_notes", description="检索知识库中的相关笔记", func=lambda query, limit=5: search_notes_tool(searcher, query, limit), args_schema=SearchNotesInput,
        ),
        StructuredTool.from_function(
            name="get_note", description="查看一篇笔记的完整内容", func=lambda note_id: get_note_tool(getter, note_id), args_schema=GetNoteInput,
        ),
        StructuredTool.from_function(
            name="recall_memories", description="召回用户的学习偏好和长期记忆", func=lambda query=None, limit=10: recall_memories_tool(recaller, query, limit), args_schema=RecallMemoriesInput,
        ),
        StructuredTool.from_function(
            name="draft_plan", description="根据目标和来源笔记生成学习计划草案，不会自动确认", coroutine=draft, args_schema=DraftPlanInput,
        ),
    ]
