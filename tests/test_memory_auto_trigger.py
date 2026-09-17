"""消息接口的自动记忆提炼：用假 Agent/假模型验证，不访问网络。"""
from fastapi.testclient import TestClient

from app.database import NoteStore
from app.main import create_app, memory_triggered


def _fake_agent():
    async def answer(*_args, **_kwargs):
        return {"content": "收到。", "tool_calls": [], "retrieved_note_ids": [],
                "cited_note_ids": [], "source_note_ids": [], "plan_id": None}
    return answer


def test_trigger_only_for_stable_preferences_or_memory_commands():
    assert memory_triggered("请记住：我每天只能学习两小时")
    assert memory_triggered("我平时不看视频课，只刷题")
    assert memory_triggered("我是在职备考")
    assert not memory_triggered("帮我找一篇 RAG 笔记")
    assert not memory_triggered("不要记住刚才的话")


def test_message_triggers_review_candidate_but_not_active_memory(tmp_path, monkeypatch):
    db_path = tmp_path / "chat.sqlite3"
    monkeypatch.setattr("app.main.run_agent_turn", _fake_agent())
    calls = []

    async def extract(messages, existing, thread_id):
        # 后台开始时，本轮用户消息和助手回复都已写入数据库。
        calls.append([item["role"] for item in messages])
        return [{"type": "preference", "content": "用户每天只能学习两小时"}], "fake-model"

    monkeypatch.setattr("app.main.extract_chat_memories_safe", extract)
    with TestClient(create_app(db_path)) as client:
        thread_id = client.post("/threads", json={"title": "学习"}).json()["id"]
        response = client.post(f"/threads/{thread_id}/messages",
                               json={"content": "请记住：我每天只能学习两小时"})
        assert response.status_code == 200
        assert response.json()["message"]["content"] == "收到。"
        candidates = client.get("/memory-candidates").json()["items"]
        assert len(candidates) == 1
        assert candidates[0]["status"] == "pending"
        assert NoteStore(db_path).recall_memories() == []

        reviewed = client.post(f"/memory-candidates/{candidates[0]['id']}/review",
                               json={"decision": "confirmed"})
        assert reviewed.status_code == 200
        assert len(NoteStore(db_path).recall_memories()) == 1
    assert calls == [["user", "assistant"]]


def test_auto_trigger_deduplicates_and_ignores_ordinary_messages(tmp_path, monkeypatch):
    db_path = tmp_path / "chat.sqlite3"
    monkeypatch.setattr("app.main.run_agent_turn", _fake_agent())
    calls = []

    async def extract(*_args):
        calls.append(1)
        return [{"type": "habit", "content": "用户平时只刷题"}], "fake-model"

    monkeypatch.setattr("app.main.extract_chat_memories_safe", extract)
    with TestClient(create_app(db_path)) as client:
        thread_id = client.post("/threads", json={"title": "学习"}).json()["id"]
        for content in ["我平时只刷题", "帮我找 RAG 笔记", "我平时只刷题"]:
            assert client.post(f"/threads/{thread_id}/messages",
                               json={"content": content}).status_code == 200
    assert len(calls) == 2
    assert len(NoteStore(db_path).memory_candidates()) == 1


def test_auto_trigger_retries_once_after_failure(tmp_path, monkeypatch):
    db_path = tmp_path / "chat.sqlite3"
    monkeypatch.setattr("app.main.run_agent_turn", _fake_agent())
    attempts = []

    async def extract(*_args):
        attempts.append(1)
        if len(attempts) == 1:
            return [], None  # safe 包装以 model=None 表示本次调用失败。
        return [{"type": "preference", "content": "用户喜欢中文注释"}], "fake-model"

    async def no_delay(_seconds):
        pass

    monkeypatch.setattr("app.main.extract_chat_memories_safe", extract)
    monkeypatch.setattr("app.main.asyncio.sleep", no_delay)
    with TestClient(create_app(db_path)) as client:
        thread_id = client.post("/threads", json={"title": "学习"}).json()["id"]
        response = client.post(f"/threads/{thread_id}/messages",
                               json={"content": "请记住：我喜欢中文注释"})
    assert response.status_code == 200
    assert len(attempts) == 2
    assert len(NoteStore(db_path).memory_candidates(status="pending")) == 1


def test_auto_trigger_failure_keeps_chat_response(tmp_path, monkeypatch):
    db_path = tmp_path / "chat.sqlite3"
    monkeypatch.setattr("app.main.run_agent_turn", _fake_agent())
    attempts = []

    async def extract(*_args):
        attempts.append(1)
        return [], None

    async def no_delay(_seconds):
        pass

    monkeypatch.setattr("app.main.extract_chat_memories_safe", extract)
    monkeypatch.setattr("app.main.asyncio.sleep", no_delay)
    with TestClient(create_app(db_path)) as client:
        thread_id = client.post("/threads", json={"title": "学习"}).json()["id"]
        response = client.post(f"/threads/{thread_id}/messages",
                               json={"content": "请记住：我喜欢中文注释"})
    assert response.status_code == 200
    assert response.json()["message"]["content"] == "收到。"
    assert len(attempts) == 2
    assert NoteStore(db_path).memory_candidates() == []
