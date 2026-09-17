"""会话 → 长期记忆沉淀通道测试：替换提炼函数，不联网也不产生费用。"""
import asyncio

from fastapi.testclient import TestClient

from app.ai import _chat_memory_prompt, extract_chat_memories
from app.database import NoteStore
from app.main import create_app


def chat_store(path, thread_id="t1"):
    """准备一个有四条消息（两轮对话）的会话，内容里埋了两条稳定偏好。"""
    store = NoteStore(path)
    store.create_thread(thread_id, "考研规划", "2026-09-16T00:00:00+00:00")
    messages = [
        ("user", "我是在职备考，只有晚上有空"),
        ("assistant", "好的，之后都会按晚上安排"),
        ("user", "我平时不看视频课，只刷题"),
        ("assistant", "明白，计划里会以刷题为主"),
    ]
    for role, content in messages:
        store.add_chat_message(thread_id, role, content, {},
                               "2026-09-16T00:01:00+00:00")
    return store


def short_store(path):
    """只有一轮消息的会话：信息量不足以提炼稳定偏好。"""
    store = NoteStore(path)
    store.create_thread("t2", "闲聊", "2026-09-16T00:00:00+00:00")
    store.add_chat_message("t2", "user", "帮我查个天气", {},
                           "2026-09-16T00:01:00+00:00")
    return store


def fake_chat_extractor(memories, model="test-memory-model"):
    """提炼函数的替身：记录调用参数，返回预设结果。"""
    calls = []

    async def _fake(messages, existing_memories, thread_id=""):
        calls.append({
            "thread_id": thread_id,
            "message_count": len(messages),
            "existing": [item["content"] for item in existing_memories],
        })
        return memories, model

    _fake.calls = calls
    return _fake


def test_missing_thread_returns_404_without_model_call(tmp_path, monkeypatch):
    """会话不存在时 404 快速失败，一次模型调用都不发生。"""
    db_path = tmp_path / "test.sqlite3"
    NoteStore(db_path)
    extractor = fake_chat_extractor([])
    monkeypatch.setattr("app.main.extract_chat_memories_safe", extractor)

    with TestClient(create_app(db_path)) as client:
        response = client.post("/threads/nope/consolidate")

    assert response.status_code == 404
    assert extractor.calls == []


def test_short_conversation_is_skipped(tmp_path, monkeypatch):
    """少于两轮对话不值得花模型调用：响应仍是 queued，但提炼没发生。"""
    db_path = tmp_path / "test.sqlite3"
    short_store(db_path)
    extractor = fake_chat_extractor([
        {"type": "preference", "content": "用户喜欢在雨天查天气"}])
    monkeypatch.setattr("app.main.extract_chat_memories_safe", extractor)

    with TestClient(create_app(db_path)) as client:
        response = client.post("/threads/t2/consolidate")

    assert response.status_code == 200
    assert response.json()["consolidation"] == "queued"
    assert extractor.calls == []
    assert NoteStore(db_path).memory_candidates() == []


def test_consolidation_creates_review_candidates(tmp_path, monkeypatch):
    """从会话提炼的记忆进候选区等审核，不自动进入 active。"""
    db_path = tmp_path / "test.sqlite3"
    chat_store(db_path)
    extractor = fake_chat_extractor([
        {"type": "habit", "content": "用户只在晚上有空学习"},
        {"type": "preference", "content": "用户偏好刷题不看视频课"},
    ])
    monkeypatch.setattr("app.main.extract_chat_memories_safe", extractor)

    with TestClient(create_app(db_path)) as client:
        response = client.post("/threads/t1/consolidate")

    assert response.status_code == 200
    assert response.json()["consolidation"] == "queued"

    # 提炼时带上了消息和已有记忆，模型才能自查重复。
    assert extractor.calls[0]["thread_id"] == "t1"
    assert extractor.calls[0]["message_count"] == 4

    store = NoteStore(db_path)
    candidates = store.memory_candidates(status="pending")
    assert len(candidates) == 2
    # 溯源：候选记录来源是这个会话，审核确认后记忆也继承这个来源。
    assert candidates[0]["source_note_id"] == "t1"
    assert candidates[0]["model"] == "test-memory-model"
    assert store.active_memories() == []

    # 用户审核确认后，才真正进入长期记忆层。
    with TestClient(create_app(db_path)) as client:
        reviewed = client.post(f"/memory-candidates/{candidates[0]['id']}/review",
                               json={"decision": "confirmed"})
    assert reviewed.status_code == 200
    assert reviewed.json()["memory"]["status"] == "active"
    assert len(NoteStore(db_path).active_memories()) == 1


def test_repeated_consolidation_does_not_duplicate_candidates(tmp_path, monkeypatch):
    """同一会话重复触发：UNIQUE(source_note_id, content_hash) 挡住重复候选。"""
    db_path = tmp_path / "test.sqlite3"
    chat_store(db_path)
    extractor = fake_chat_extractor([
        {"type": "habit", "content": "用户只在晚上有空学习"}])
    monkeypatch.setattr("app.main.extract_chat_memories_safe", extractor)

    with TestClient(create_app(db_path)) as client:
        client.post("/threads/t1/consolidate")
        client.post("/threads/t1/consolidate")

    assert len(extractor.calls) == 2
    # 第二次提炼能看到尚未审核的第一条候选，模型可避免换词重提。
    assert extractor.calls[1]["existing"] == ["用户只在晚上有空学习"]
    assert len(NoteStore(db_path).memory_candidates()) == 1


def test_pending_candidate_is_labeled_and_exact_duplicate_is_filtered(monkeypatch):
    """待审核内容提供给模型避重，也参与提炼后的精确去重。"""
    known = [
        {"status": "active", "content": "用户在职备考"},
        {"status": "pending", "content": "用户只在晚上有空学习"},
    ]
    prompt = _chat_memory_prompt([{"role": "user", "content": "我只有晚上有空"}], known)
    assert "已确认记忆（与其中任何一条重复都不要输出）：\n- 用户在职备考" in prompt
    assert "待审核候选（尚未成为有效记忆；与其中任何一条重复也不要输出）：\n- 用户只在晚上有空学习" in prompt

    async def fake_completion(_prompt):
        return {"memories": [{"type": "habit", "content": "用户只在晚上有空学习"}]}, "fake-model"

    monkeypatch.setattr("app.ai._memory_completion", fake_completion)
    memories, model = asyncio.run(extract_chat_memories(
        [{"role": "user", "content": "我只有晚上有空"}], known
    ))
    assert model == "fake-model"
    assert memories == []


def test_extraction_failure_keeps_endpoint_successful(tmp_path, monkeypatch):
    """提炼失败不影响接口：照常 200，只是没有新候选。"""
    db_path = tmp_path / "test.sqlite3"
    chat_store(db_path)
    monkeypatch.setattr("app.main.extract_chat_memories_safe",
                        fake_chat_extractor([], model=None))

    with TestClient(create_app(db_path)) as client:
        response = client.post("/threads/t1/consolidate")

    assert response.status_code == 200
    assert response.json()["consolidation"] == "queued"
    assert NoteStore(db_path).memory_candidates() == []
