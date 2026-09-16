"""计划草案与任务转换测试：验证用户确认门槛和重复调用幂等性。"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from app.database import NoteStore
from app.main import create_app


def future_at(days: int) -> str:
    """测试数据相对当前日期生成，避免固定日期过期导致测试随时间失效。"""
    day = datetime.now(ZoneInfo("Asia/Shanghai")) + timedelta(days=days)
    return day.replace(hour=20, minute=0, second=0, microsecond=0).isoformat()


def test_plan_draft_confirm_and_create_tasks(tmp_path):
    db_path = tmp_path / "plans.sqlite3"
    NoteStore(db_path).save({
        "note_id": "plan-note", "title": "Agent 记忆", "content": "学习内容", "images": []
    })
    with TestClient(create_app(db_path)) as client:
        payload = {
            "title": "一周掌握 Agent 记忆系统",
            "goal": "准备秋招项目",
            "source_note_ids": ["plan-note"],
            "items": [{
                "title": "理解短期记忆和长期记忆",
                "scheduled_at": future_at(1)
            }, {
                "title": "实现一个简单的记忆模块",
                "scheduled_at": future_at(2)
            }]
        }
        created = client.post("/plans", json=payload)
        assert created.status_code == 201
        plan_id = created.json()["id"]
        assert created.json()["status"] == "draft"
        assert client.post(f"/plans/{plan_id}/create-tasks").status_code == 409
        confirmed = client.post(f"/plans/{plan_id}/confirm").json()
        assert confirmed["status"] == "confirmed"
        result = client.post(f"/plans/{plan_id}/create-tasks").json()
        assert result["status"] == "created" and len(result["task_ids"]) == 2
        repeated = client.post(f"/plans/{plan_id}/create-tasks").json()
        assert repeated["status"] == "existing" and repeated["task_ids"] == []
        assert len(client.get("/tasks").json()["items"]) == 2


def test_plan_rejects_invalid_times_and_missing_note(tmp_path):
    db_path = tmp_path / "plans.sqlite3"
    with TestClient(create_app(db_path)) as client:
        base = {"title": "计划", "goal": "目标", "items": [{
            "title": "步骤", "scheduled_at": future_at(1)
        }]}
        assert client.post("/plans", json={**base, "source_note_ids": ["missing"]}).status_code == 404
        invalid = {**base, "items": [{
            "title": "步骤", "scheduled_at": future_at(1),
            "reminder_at": future_at(2)
        }]}
        assert client.post("/plans", json=invalid).status_code == 400


def test_ai_draft_plan_is_saved_as_draft(tmp_path, monkeypatch):
    db_path = tmp_path / "ai-plans.sqlite3"
    NoteStore(db_path).save({"note_id": "source", "title": "RAG", "content": "检索", "images": []})

    async def fake_draft(goal, notes, intents, start_date):
        return {"title": "AI 计划", "goal": goal, "items": [{
            "title": "理解 RAG", "details": "整理概念",
            "scheduled_at": "2026-09-16T20:00:00+08:00", "reminder_at": None
        }]}, "test-model"

    monkeypatch.setattr("app.main.draft_plan", fake_draft)
    with TestClient(create_app(db_path)) as client:
        response = client.post("/plans/draft-from-ai", json={
            "goal": "准备秋招", "source_note_ids": ["source"], "start_date": "2026-09-16"
        })
        assert response.status_code == 201
        assert response.json()["status"] == "draft"
        assert response.json()["plan"]["items"][0]["title"] == "理解 RAG"


def test_plan_items_can_be_edited_deleted_and_reordered_only_in_draft(tmp_path):
    db_path = tmp_path / "plan-edit.sqlite3"
    NoteStore(db_path).save({"note_id": "source", "title": "资料", "content": "内容", "images": []})
    with TestClient(create_app(db_path)) as client:
        plan = client.post("/plans", json={
            "title": "计划", "goal": "目标", "source_note_ids": ["source"],
            "items": [
                {"title": "第一步", "scheduled_at": future_at(1)},
                {"title": "第二步", "scheduled_at": future_at(2)},
                {"title": "第三步", "scheduled_at": future_at(3)}
            ]
        }).json()
        ids = [item["id"] for item in plan["items"]]
        edited = client.put(f"/plans/{plan['id']}/items/{ids[0]}", json={"title": "修改后的第一步"})
        assert edited.status_code == 200 and edited.json()["title"] == "修改后的第一步"
        reordered = client.post(f"/plans/{plan['id']}/items/reorder", json={"item_ids": ids[::-1]})
        assert [item["id"] for item in reordered.json()["items"]] == ids[::-1]
        deleted = client.delete(f"/plans/{plan['id']}/items/{ids[1]}")
        assert len(deleted.json()["items"]) == 2
        assert [item["item_index"] for item in deleted.json()["items"]] == [0, 1]
        client.post(f"/plans/{plan['id']}/confirm")
        assert client.put(f"/plans/{plan['id']}/items/{ids[0]}", json={"title": "禁止修改"}).status_code == 409


def test_quality_check_blocks_past_steps_and_warns_about_many_tasks(tmp_path):
    db_path = tmp_path / "quality.sqlite3"
    NoteStore(db_path).save({"note_id": "source", "title": "资料", "content": "内容", "images": []})
    with TestClient(create_app(db_path)) as client:
        plan = client.post("/plans", json={
            "title": "密集计划", "goal": "准备面试", "source_note_ids": ["source"],
            "items": [{"title": f"步骤 {index}", "scheduled_at": future_at(index + 1)}
                      for index in range(11)],
        }).json()
        plan_id = plan["id"]
        first_id = plan["items"][0]["id"]
        check = client.get(f"/plans/{plan_id}/quality-check").json()
        assert check["can_confirm"] is True
        assert check["task_count_if_created"] == 11
        assert "many_tasks" in {issue["code"] for issue in check["warnings"]}

        client.put(f"/plans/{plan_id}/items/{first_id}", json={
            "scheduled_at": "2020-01-01T20:00:00+08:00"
        })
        blocked = client.post(f"/plans/{plan_id}/confirm")
        assert blocked.status_code == 409
        assert "past_schedule" in {issue["code"] for issue in blocked.json()["detail"]["blockers"]}
        assert client.get(f"/plans/{plan_id}").json()["status"] == "draft"
        assert client.get("/tasks").json()["items"] == []

        client.put(f"/plans/{plan_id}/items/{first_id}", json={"scheduled_at": future_at(1)})
        assert client.post(f"/plans/{plan_id}/confirm").status_code == 200


def test_expired_reminder_can_be_cleared_before_confirmation(tmp_path):
    db_path = tmp_path / "reminder-quality.sqlite3"
    with TestClient(create_app(db_path)) as client:
        plan = client.post("/plans", json={
            "title": "学习计划", "goal": "学习", "items": [{
                "title": "明天学习", "scheduled_at": future_at(1),
                "reminder_at": "2020-01-01T10:00:00+08:00",
            }],
        }).json()
        plan_id = plan["id"]
        item_id = plan["items"][0]["id"]
        check = client.get(f"/plans/{plan_id}/quality-check").json()
        assert "past_reminder" in {issue["code"] for issue in check["blockers"]}
        edited = client.put(f"/plans/{plan_id}/items/{item_id}", json={"reminder_at": None})
        assert edited.status_code == 200 and edited.json()["reminder_at"] is None
        assert client.get(f"/plans/{plan_id}/quality-check").json()["can_confirm"] is True
