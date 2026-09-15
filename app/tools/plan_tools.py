"""计划工具：只生成草案，不自动确认或创建任务。"""


async def draft_plan_tool(drafter, goal: str, source_note_ids: list[str], start_date: str) -> dict:
    """让 AI 根据目标和来源笔记生成草案；确认动作仍由用户触发。"""
    if not goal.strip() or not source_note_ids:
        return {"ok": False, "data": None, "error": "goal 和 source_note_ids 不能为空"}
    result = await drafter(goal.strip(), source_note_ids, start_date)
    return {"ok": True, "data": result, "error": None}
