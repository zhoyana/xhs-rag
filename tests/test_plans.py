"""计划草案与任务转换测试：验证用户确认门槛和重复调用幂等性。"""
from fastapi.testclient import TestClient

from app.database import NoteStore
from app.main import create_app


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
                "scheduled_at": "2026-09-16T20:00:00+08:00"
            }, {
                "title": "实现一个简单的记忆模块",
                "scheduled_at": "2026-09-17T20:00:00+08:00"
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
            "title": "步骤", "scheduled_at": "2026-09-16T20:00:00+08:00"
        }]}
        assert client.post("/plans", json={**base, "source_note_ids": ["missing"]}).status_code == 404
        invalid = {**base, "items": [{
            "title": "步骤", "scheduled_at": "2026-09-16T20:00:00+08:00",
            "reminder_at": "2026-09-17T20:00:00+08:00"
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
