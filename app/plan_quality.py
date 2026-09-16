"""计划确认前的确定性质量门禁。

秋招知识点：把“能否执行”的硬校验与“是否值得执行”的软建议分开。
这里不假装能用规则判断内容是否真的相关；语义质量留给用户和后续评测。
"""
from collections import Counter
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def check_plan_quality(plan: dict, now: datetime | None = None) -> dict:
    """只读检查，不改变计划、任务或笔记。时间统一换成 UTC 后比较。"""
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    blockers, warnings = [], []
    items = plan["items"]
    if not items:
        blockers.append({"code": "empty_plan", "message": "计划没有步骤，不能确认。"})

    day_counts = Counter()
    used_note_ids = set()
    for item in items:
        label = item["title"]
        try:
            scheduled = datetime.fromisoformat(item["scheduled_at"])
            if scheduled.tzinfo is None:
                raise ValueError("naive datetime")
            scheduled = scheduled.astimezone(timezone.utc)
        except (ValueError, TypeError):
            blockers.append({"code": "invalid_schedule", "item_id": item["id"],
                             "message": f"「{label}」的执行时间无效。"})
            continue
        if scheduled <= current:
            blockers.append({"code": "past_schedule", "item_id": item["id"],
                             "message": f"「{label}」的执行时间已经过去，请先调整。"})

        reminder_text = item.get("reminder_at")
        if reminder_text:
            try:
                reminder = datetime.fromisoformat(reminder_text)
                if reminder.tzinfo is None:
                    raise ValueError("naive datetime")
                reminder = reminder.astimezone(timezone.utc)
            except (ValueError, TypeError):
                blockers.append({"code": "invalid_reminder", "item_id": item["id"],
                                 "message": f"「{label}」的提醒时间无效。"})
            else:
                if reminder <= current:
                    blockers.append({"code": "past_reminder", "item_id": item["id"],
                                     "message": f"「{label}」的提醒时间已经过去，请先调整或清空。"})

        day_counts[scheduled.astimezone(ZoneInfo(item["timezone"])).date().isoformat()] += 1
        sources = item.get("source_note_ids", [])
        used_note_ids.update(sources)
        if not sources:
            warnings.append({"code": "missing_item_source", "item_id": item["id"],
                             "message": f"「{label}」没有标明参考笔记。"})

    if len(items) > 10:
        warnings.append({"code": "many_tasks", "message":
                         f"计划有 {len(items)} 个步骤，确认后通常会生成同样数量的任务，请核对工作量。"})
    for day, count in sorted(day_counts.items()):
        if count > 2:
            warnings.append({"code": "busy_day", "date": day,
                             "message": f"{day} 安排了 {count} 项任务，请核对当天可用时间。"})
    unused = sorted(set(plan.get("source_note_ids", [])) - used_note_ids)
    if unused:
        warnings.append({"code": "unused_plan_sources", "note_ids": unused,
                         "message": "部分计划级参考笔记没有被任何步骤引用。"})

    return {"plan_id": plan["id"], "can_confirm": not blockers,
            "task_count_if_created": len(items), "blockers": blockers, "warnings": warnings}
