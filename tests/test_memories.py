"""记忆层存储测试：验证建表约束、幂等去重、软删除与指纹同步。"""
import sqlite3

import pytest

from app.database import NoteStore


def make_store(path):
    return NoteStore(path)


def sample_records():
    return [
        {"type": "preference", "content": "用户偏好居家徒手训练，不去健身房"},
        {"type": "correction", "content": "用户习惯把减脂餐归为健身而非做饭"},
    ]


def test_memories_table_persists_across_reopen(tmp_path):
    """表结构随库文件持久化：重新打开 NoteStore 记忆仍在。"""
    db_path = tmp_path / "test.sqlite3"
    store = make_store(db_path)
    saved = store.save_memories("deepseek-chat", sample_records(),
                                "note1", "2026-09-15T00:00:00+00:00")
    assert len(saved) == 2

    reopened = make_store(db_path)
    assert len(reopened.memories()) == 2
    assert reopened.active_memories()[0]["type"] == "preference"


def test_save_memories_returns_new_records_only(tmp_path):
    """重复提交同一批提取结果不会产生重复记忆（数据库层幂等）。"""
    store = make_store(tmp_path / "test.sqlite3")
    now = "2026-09-15T00:00:00+00:00"
    first = store.save_memories("deepseek-chat", sample_records(), "note1", now)
    assert len(first) == 2

    # 同内容再次沉淀：UNIQUE 约束让 INSERT OR IGNORE 全部跳过。
    second = store.save_memories("deepseek-chat", sample_records(), "note2", now)
    assert second == []
    assert len(store.memories()) == 2
    # source_note_id 保持首次来源，不被第二次尝试覆盖。
    assert all(memory["source_note_id"] == "note1" for memory in store.memories())


def test_check_constraint_rejects_invalid_type(tmp_path):
    """CHECK 约束兜底：非法 type 即使绕过应用层也写不进数据库。"""
    store = make_store(tmp_path / "test.sqlite3")
    with pytest.raises(sqlite3.IntegrityError):
        with sqlite3.connect(store.path) as db:
            db.execute("""INSERT INTO memories
                (type, content, content_hash, model, source_note_id,
                 status, created_at, updated_at)
                VALUES ('gossip', '不该存在的记录', 'x', 'm', NULL,
                        'active', '2026', '2026')""")


def test_active_memories_orders_by_recency_and_excludes_dismissed(tmp_path):
    """注入用查询：只取 active，按 updated_at 倒序（最近的最先注入）。"""
    store = make_store(tmp_path / "test.sqlite3")
    old = store.save_memories("deepseek-chat", sample_records()[:1],
                              "note1", "2026-09-14T00:00:00+00:00")
    new = store.save_memories("deepseek-chat", sample_records()[1:],
                              "note2", "2026-09-15T00:00:00+00:00")

    active = store.active_memories()
    assert [memory["id"] for memory in active] == [new[0]["id"], old[0]["id"]]

    # 软删除后不再进入注入列表，但记录仍可查询（可审计）。
    store.update_memory(old[0]["id"], {"status": "dismissed"},
                        "2026-09-15T01:00:00+00:00")
    assert [memory["id"] for memory in store.active_memories()] == [new[0]["id"]]
    assert store.memory(old[0]["id"])["status"] == "dismissed"


def test_update_memory_recomputes_content_hash(tmp_path):
    """编辑内容时指纹必须同步重算，否则唯一约束会挡住后续合法写入。"""
    store = make_store(tmp_path / "test.sqlite3")
    saved = store.save_memories("deepseek-chat", sample_records()[:1],
                                "note1", "2026-09-15T00:00:00+00:00")
    memory_id = saved[0]["id"]

    updated = store.update_memory(memory_id, {"content": "用户偏好居家训练"},
                                  "2026-09-15T02:00:00+00:00")
    assert updated["content"] == "用户偏好居家训练"
    assert updated["content_hash"] == NoteStore.memory_content_hash("用户偏好居家训练")
    assert updated["updated_at"] == "2026-09-15T02:00:00+00:00"

    # 更新后的指纹正确：相同新内容不会被误判为重复。
    again = store.save_memories("deepseek-chat",
                                [{"type": "preference", "content": "用户偏好居家训练"}],
                                "note3", "2026-09-15T03:00:00+00:00")
    assert again == []


def test_update_memory_ignores_fields_outside_whitelist(tmp_path):
    """字段白名单：model、source_note_id 等不由用户直接修改。"""
    store = make_store(tmp_path / "test.sqlite3")
    saved = store.save_memories("deepseek-chat", sample_records()[:1],
                                "note1", "2026-09-15T00:00:00+00:00")
    updated = store.update_memory(
        saved[0]["id"],
        {"content": "用户偏好居家训练", "model": "伪造模型", "evil": "x"},
        "2026-09-15T02:00:00+00:00")
    assert updated["model"] == "deepseek-chat"
    assert "evil" not in updated
