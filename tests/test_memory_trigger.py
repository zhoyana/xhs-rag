"""意图确认触发记忆提炼的接口测试：替换提炼函数，不联网也不产生费用。"""
from fastapi.testclient import TestClient

from app.database import NoteStore
from app.main import create_app


def prepared_store(path, existing_memories=None):
    """准备一篇已完成分析、等待确认意图的笔记。"""
    store = NoteStore(path)
    store.save({"note_id": "note1", "title": "一周减脂餐安排",
                "content": "少油少盐，按周备餐。", "images": []})
    store.save_analysis("note1", "test-model", {
        "category": "做饭", "inferred_intent": "想开始控制饮食",
        "intent_confidence": 0.7, "suggested_actions": ["排下周菜单"],
    }, "2026-09-15T00:00:00+00:00")
    if existing_memories:
        store.save_memories("test-model", existing_memories,
                            "note0", "2026-09-14T00:00:00+00:00")
    return store


def fake_extractor(memories, model="test-memory-model"):
    """提炼函数的替身：记录调用参数，返回预设结果。"""
    calls = []

    async def _fake(note, confirmation, existing_memories):
        calls.append({
            "note_id": note["note_id"],
            "decision": confirmation["decision"],
            "existing": [item["content"] for item in existing_memories],
        })
        return memories, model

    _fake.calls = calls
    return _fake


def test_confirm_triggers_extraction_and_creates_review_candidate(tmp_path, monkeypatch):
    """模型只能提议；用户确认候选后才成为 active 长期记忆。"""
    db_path = tmp_path / "test.sqlite3"
    prepared_store(db_path, existing_memories=[
        {"type": "preference", "content": "用户偏好居家徒手训练"}])
    extractor = fake_extractor([
        {"type": "habit", "content": "用户习惯把减脂餐归为健身"},
    ])
    monkeypatch.setattr("app.main.extract_memories_safe", extractor)

    with TestClient(create_app(db_path)) as client:
        response = client.put("/notes/note1/intent", json={"decision": "confirmed"})

    assert response.status_code == 200
    body = response.json()
    # 原有字段保持不变，只是新增 memory_job，老客户端不受影响。
    assert body["status"] == "confirmed"
    assert body["confirmation"]["intent"] == "想开始控制饮食"
    assert body["memory_job"] == "queued"

    # 提炼时带上了已有记忆，模型才能在提示词内自查重复。
    assert extractor.calls[0]["existing"] == ["用户偏好居家徒手训练"]

    # 候选没有自动进入 active，只有用户确认后才会影响后续回答。
    store = NoteStore(db_path)
    assert len(store.active_memories()) == 1
    candidates = store.memory_candidates(status="pending")
    assert len(candidates) == 1
    assert candidates[0]["model"] == "test-memory-model"
    assert candidates[0]["source_note_id"] == "note1"
    with TestClient(create_app(db_path)) as client:
        reviewed = client.post(f"/memory-candidates/{candidates[0]['id']}/review", json={
            "decision": "confirmed", "content": "用户习惯把减脂餐用于健身备餐"
        })
    assert reviewed.status_code == 200
    assert reviewed.json()["memory"]["status"] == "active"
    assert len(NoteStore(db_path).active_memories()) == 2


def test_dismissed_does_not_extract(tmp_path, monkeypatch):
    """用户未采用建议时不提炼：响应标记 skipped，且一次模型调用都没发生。"""
    db_path = tmp_path / "test.sqlite3"
    prepared_store(db_path)
    extractor = fake_extractor([])
    monkeypatch.setattr("app.main.extract_memories_safe", extractor)

    with TestClient(create_app(db_path)) as client:
        body = client.put("/notes/note1/intent", json={"decision": "dismissed"}).json()

    assert body["memory_job"] == "skipped"
    assert extractor.calls == []
    assert NoteStore(db_path).memories() == []


def test_extraction_failure_keeps_confirmation_successful(tmp_path, monkeypatch):
    """提炼失败不影响确认结果：接口照常成功，只是没有新记忆。"""
    db_path = tmp_path / "test.sqlite3"
    prepared_store(db_path)
    # 降级路径的返回值：没有记忆、也没有模型名。
    monkeypatch.setattr("app.main.extract_memories_safe",
                        fake_extractor([], model=None))

    with TestClient(create_app(db_path)) as client:
        response = client.put("/notes/note1/intent", json={
            "decision": "edited", "category": "健身", "intent": "开始控制饮食",
        })

    assert response.status_code == 200
    assert response.json()["memory_job"] == "queued"
    # 用户的确认已经落库；记忆为空，但流程没有报错。
    assert NoteStore(db_path).intent_confirmation("note1")["category"] == "健身"
    assert NoteStore(db_path).memories() == []


def test_repeated_confirmation_does_not_duplicate_memory(tmp_path, monkeypatch):
    """重复确认同一结果时，内容指纹在数据库层挡住重复记忆。"""
    db_path = tmp_path / "test.sqlite3"
    prepared_store(db_path)
    extractor = fake_extractor([
        {"type": "habit", "content": "用户习惯把减脂餐归为健身"}])
    monkeypatch.setattr("app.main.extract_memories_safe", extractor)

    with TestClient(create_app(db_path)) as client:
        client.put("/notes/note1/intent", json={"decision": "confirmed"})
        client.put("/notes/note1/intent", json={"decision": "confirmed"})

    assert len(extractor.calls) == 2
    assert NoteStore(db_path).memories() == []
    assert len(NoteStore(db_path).memory_candidates()) == 1
