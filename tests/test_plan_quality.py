"""质量门禁的纯函数测试：固定时间，不依赖真实时钟或模型。"""
from datetime import datetime, timezone

from app.plan_quality import check_plan_quality


def test_quality_separates_blockers_from_warnings():
    plan = {
        "id": 4, "source_note_ids": ["used", "unused"],
        "items": [
            {"id": 1, "title": "已错过", "scheduled_at": "2026-09-16T01:00:00+00:00",
             "reminder_at": None, "timezone": "Asia/Shanghai", "source_note_ids": ["used"]},
            {"id": 2, "title": "待完成", "scheduled_at": "2026-09-17T01:00:00+00:00",
             "reminder_at": None, "timezone": "Asia/Shanghai", "source_note_ids": []},
        ],
    }
    result = check_plan_quality(plan, datetime(2026, 9, 16, 8, tzinfo=timezone.utc))
    assert result["can_confirm"] is False
    assert {item["code"] for item in result["blockers"]} == {"past_schedule"}
    assert {item["code"] for item in result["warnings"]} == {
        "missing_item_source", "unused_plan_sources"
    }
    assert result["task_count_if_created"] == 2
