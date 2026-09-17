"""任务接口测试：验证确认门槛、时间处理、修改，以及取消/删除两种删除语义。"""
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


def create_task(client, title):
    return client.post("/notes/note1/tasks", json={
        "title": title,
        "scheduled_at": "2026-09-06T20:00:00+08:00",
        "timezone": "Asia/Shanghai",
    }).json()


def test_cancel_keeps_the_row_and_hard_delete_removes_it(tmp_path):
    """取消与删除是两种语义，必须能被分别验证。

    取消：任务是真的，只是不做了，记录要留着，回看时能知道当时的决定。
    删除：这行记录本身就是错的（误建、调试刷出来的），只能真的从库里抹掉。
    只有取消的话，这类脏数据会永远挂在「已取消」里清不掉。
    """
    db_path = tmp_path / "test.sqlite3"
    prepared_store(db_path)
    with TestClient(create_app(db_path)) as client:
        cancelled_task = create_task(client, "不打算做了")
        deleted_task = create_task(client, "误建的脏数据")

        assert client.delete(f"/tasks/{cancelled_task['id']}").json()["status"] == "cancelled"
        assert client.get(f"/tasks/{cancelled_task['id']}").status_code == 200

        removed = client.delete(f"/tasks/{deleted_task['id']}?hard=true").json()
        assert removed == {"id": deleted_task["id"], "status": "deleted", "hard": True}
        assert client.get(f"/tasks/{deleted_task['id']}").status_code == 404

        # 删除只影响目标行，别的任务不受牵连。
        assert client.get(f"/tasks/{cancelled_task['id']}").status_code == 200

    # 重启后仍然是删掉的：证明确实落到了数据库，而不是只在内存里过滤。
    with TestClient(create_app(db_path)) as client:
        assert client.get(f"/tasks/{deleted_task['id']}").status_code == 404
        assert len(client.get("/tasks").json()["items"]) == 1
        # 不存在的任务返回 404，而不是静默成功。
        assert client.delete("/tasks/99999?hard=true").status_code == 404
