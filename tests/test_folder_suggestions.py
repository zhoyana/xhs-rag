"""AI 收藏夹建议测试：使用假模型，不联网、不消耗真实 API 额度。"""
from fastapi.testclient import TestClient

from app.database import NoteStore
from app.main import create_app


def prepared_store(db_path):
    store = NoteStore(db_path)
    store.save({
        "note_id": "agent-note", "title": "LangGraph 项目",
        "content": "用 RAG 和 Agent 工作流准备秋招项目。", "images": [],
    })
    store.save_analysis("agent-note", "test-model", {
        "summary": "Agent 项目实践", "category": "学习",
        "inferred_intent": "准备秋招项目", "intent_confidence": 0.9,
        "tags": ["RAG", "Agent"], "suggested_actions": [],
    }, "2026-01-01T00:00:00+00:00")
    return store


def test_ai_suggestion_is_cached_then_confirmed(tmp_path, monkeypatch):
    db_path = tmp_path / "suggestions.sqlite3"
    prepared_store(db_path)
    calls = []

    async def fake_suggest(note, analysis, confirmation, folders):
        calls.append([folder["id"] for folder in folders])
        return {"suggestions": [
            {"folder_id": folders[0]["id"], "confidence": 0.95, "reason": "属于 Agent 技术"},
            {"folder_id": folders[1]["id"], "confidence": 0.88, "reason": "用于秋招项目"},
        ], "uncertainties": []}, "test-model"

    monkeypatch.setattr("app.main.suggest_folders", fake_suggest)
    with TestClient(create_app(db_path)) as client:
        ai_folder = client.post("/folders", json={"name": "AI 应用工程"}).json()
        job_folder = client.post("/folders", json={"name": "秋招与面试"}).json()
        first = client.post("/notes/agent-note/folder-suggestions")
        assert first.status_code == 200 and first.json()["operation"] == "suggested"
        assert first.json()["status"] == "pending"
        assert first.json()["selected_folder_ids"] is None

        cached = client.post("/notes/agent-note/folder-suggestions")
        assert cached.json()["operation"] == "cached" and len(calls) == 1
        confirmed = client.post("/notes/agent-note/folder-suggestions/confirm", json={
            "folder_ids": [ai_folder["id"], job_folder["id"]]
        })
        assert confirmed.json()["selected_folder_ids"] == [ai_folder["id"], job_folder["id"]]
        assigned = client.get("/notes/agent-note/folders").json()["items"]
        assert {item["name"] for item in assigned} == {"AI 应用工程", "秋招与面试"}

    # 模拟服务重启后仍可读取，证明建议与确认均已持久化。
    with TestClient(create_app(db_path)) as client:
        assert client.get("/notes/agent-note/folder-suggestions").json()["status"] == "confirmed"


def test_suggestion_requires_analysis_and_active_folders(tmp_path, monkeypatch):
    db_path = tmp_path / "suggestions.sqlite3"
    store = NoteStore(db_path)
    store.save({"note_id": "plain", "title": "未分析", "content": "内容", "images": []})
    with TestClient(create_app(db_path)) as client:
        assert client.post("/notes/plain/folder-suggestions").status_code == 409

        client.post("/folders", json={"name": "待整理"})
        # 有收藏夹但无分析，仍应拒绝，避免模型在缺少多模态信息时草率分类。
        assert client.post("/notes/plain/folder-suggestions").status_code == 409


def test_confirmation_validates_folder_ids(tmp_path, monkeypatch):
    db_path = tmp_path / "suggestions.sqlite3"
    prepared_store(db_path)

    async def fake_suggest(note, analysis, confirmation, folders):
        return {"suggestions": [{
            "folder_id": folders[0]["id"], "confidence": 0.8, "reason": "相关"
        }], "uncertainties": []}, "test-model"

    monkeypatch.setattr("app.main.suggest_folders", fake_suggest)
    with TestClient(create_app(db_path)) as client:
        folder = client.post("/folders", json={"name": "AI"}).json()
        client.post("/notes/agent-note/folder-suggestions")
        duplicate = client.post("/notes/agent-note/folder-suggestions/confirm", json={
            "folder_ids": [folder["id"], folder["id"]]
        })
        assert duplicate.status_code == 400
        missing = client.post("/notes/agent-note/folder-suggestions/confirm", json={
            "folder_ids": [99999]
        })
        assert missing.status_code == 404
