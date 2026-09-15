"""任务接口测试：验证确认门槛、时间处理、修改和软删除。"""
from fastapi.testclient import TestClient

from app.database import NoteStore
from app.main import create_app


def prepared_store(path, confirmed=True):
    store = NoteStore(path)
    store.save({"note_id": "note1", "title": "学习", "images": []})
    if confirmed:
        store.save_intent_confirmation(
            "note1", "confirmed", "学习", "练习 Python",
            ["完成小项目"], "2026-09-04T00:00:00+00:00"
        )
    return store


def test_task_lifecycle_and_persistence(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    prepared_store(db_path)
    body = {
        "title": "完成时序预测小项目",
        "details": "先完成 baseline",
        "scheduled_at": "2026-09-06T20:00:00+08:00",
        "reminder_at": "2026-09-06T19:30:00+08:00",
        "timezone": "Asia/Shanghai",
        "source_action": "做一个时序预测小项目",
    }
    with TestClient(create_app(db_path)) as client:
        created = client.post("/notes/note1/tasks", json=body)
        assert created.status_code == 200
        task = created.json()
        assert task["scheduled_at"] == "2026-09-06T12:00:00+00:00"
        assert task["status"] == "pending"
        task_id = task["id"]

        assert client.get(f"/tasks/{task_id}").json()["title"] == body["title"]
        assert len(client.get("/tasks?status=pending").json()["items"]) == 1

        changed = client.put(f"/tasks/{task_id}", json={
            "title": "完成并复盘时序预测项目",
            "scheduled_at": "2026-09-07T20:00:00+08:00",
            "reminder_at": None,
        }).json()
        assert changed["title"] == "完成并复盘时序预测项目"
        assert changed["reminder_at"] is None

        completed = client.post(f"/tasks/{task_id}/complete").json()
        assert completed["status"] == "completed"
        cancelled = client.delete(f"/tasks/{task_id}").json()
        assert cancelled["status"] == "cancelled"

    # 应用重启后任务仍存在，说明不是只放在内存中。
    with TestClient(create_app(db_path)) as client:
        assert client.get(f"/tasks/{task_id}").json()["status"] == "cancelled"


def test_task_requires_confirmation_and_valid_time(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    prepared_store(db_path, confirmed=False)
    base = {
        "title": "测试任务",
        "scheduled_at": "2026-09-06T20:00:00+08:00",
        "timezone": "Asia/Shanghai",
    }
    with TestClient(create_app(db_path)) as client:
        assert client.post("/notes/note1/tasks", json=base).status_code == 409

    # 添加确认后，再分别验证不带时区、无效时区和提醒晚于执行时间。
    prepared_store(db_path).save_intent_confirmation(
        "note1", "confirmed", "学习", "测试", [], "2026-09-04T00:00:00+00:00"
    )
    with TestClient(create_app(db_path)) as client:
        invalid = {**base, "scheduled_at": "2026-09-06T20:00:00"}
        assert client.post("/notes/note1/tasks", json=invalid).status_code == 400
        invalid = {**base, "timezone": "Mars/Beijing"}
        assert client.post("/notes/note1/tasks", json=invalid).status_code == 400
        invalid = {**base, "reminder_at": "2026-09-06T21:00:00+08:00"}
        assert client.post("/notes/note1/tasks", json=invalid).status_code == 400
