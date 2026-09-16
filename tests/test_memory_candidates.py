"""候选确认与用户主动记忆：验证状态机、幂等和来源区分。"""
from fastapi.testclient import TestClient

from app.database import NoteStore
from app.main import create_app


def test_candidate_review_and_direct_user_memory(tmp_path):
    db_path = tmp_path / "candidates.sqlite3"
    store = NoteStore(db_path)
    store.save_memory_candidates("test-ai", [
        {"type": "preference", "content": "用户偏好用中文解释 RAG"},
    ], "note-a", "2026-09-16T00:00:00+00:00")
    candidate_id = store.memory_candidates()[0]["id"]
    with TestClient(create_app(db_path)) as client:
        assert client.get("/memories?status=active").json()["items"] == []
        approved = client.post(f"/memory-candidates/{candidate_id}/review", json={
            "decision": "confirmed"
        })
        assert approved.status_code == 200
        assert approved.json()["memory"]["model"] == "test-ai"
        assert client.post(f"/memory-candidates/{candidate_id}/review", json={
            "decision": "confirmed"
        }).status_code == 409
        direct = client.post("/memories", json={
            "type": "habit", "content": "用户通常晚上学习 Agent 工程"
        })
        assert direct.status_code == 201
        assert direct.json()["model"] == "user"
        assert direct.json()["source_note_id"] is None
        assert client.post("/memories", json={
            "type": "habit", "content": "用户通常晚上学习 Agent 工程"
        }).status_code == 409
        assert len(client.get("/memories?status=active").json()["items"]) == 2


def test_dismissed_candidate_never_becomes_active(tmp_path):
    db_path = tmp_path / "dismiss.sqlite3"
    store = NoteStore(db_path)
    store.save_memory_candidates("test-ai", [
        {"type": "correction", "content": "用户纠正了收藏夹分类"},
    ], "note-a", "2026-09-16T00:00:00+00:00")
    candidate_id = store.memory_candidates()[0]["id"]
    with TestClient(create_app(db_path)) as client:
        result = client.post(f"/memory-candidates/{candidate_id}/review", json={
            "decision": "dismissed"
        })
        assert result.status_code == 200
        assert result.json()["memory"] is None
        assert client.get("/memories?status=active").json()["items"] == []
