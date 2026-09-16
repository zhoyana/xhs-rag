"""计划工具：只生成草案，不自动确认或创建任务。"""
from datetime import datetime
from zoneinfo import ZoneInfo


async def draft_plan_tool(drafter, goal: str, source_note_ids: list[str], start_date: str) -> dict:
    """让 AI 根据目标和来源笔记生成草案；确认动作仍由用户触发。"""
    if not goal.strip() or not source_note_ids:
        return {"ok": False, "data": None,
                "error": {"code": "invalid_arguments", "message": "目标和来源笔记不能为空"}}
    try:
        parsed_start = datetime.strptime(start_date, "%Y-%m-%d").date()
    except ValueError:
        return {"ok": False, "data": None,
                "error": {"code": "invalid_start_date", "message": "日期格式必须为 YYYY-MM-DD"}}
    if parsed_start < datetime.now(ZoneInfo("Asia/Shanghai")).date():
        return {"ok": False, "data": None,
                "error": {"code": "past_start_date", "message": "计划开始日期不能早于今天"}}
    result = await drafter(goal.strip(), source_note_ids, start_date)
    return {"ok": True, "data": result, "error": None}
