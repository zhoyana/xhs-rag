"""记忆召回（recall）测试：统一入口、向量排序、惰性回填与降级。

全部用假向量函数替身，不加载真实 BGE 模型、不联网。重点覆盖：
1. query 为空时与 active_memories 行为一致（存量调用方零感知）；
2. query 非空时按余弦相似度排序并带 score 字段，缺失向量先惰性回填；
3. 向量链路任何故障都降级成时间排序，不抛错（增强路径不拖垮主链路）；
4. superseded 三态：不进召回、接口可过滤、禁止手动设置；
5. 旧版两态表打开时自动迁移到三态 + 向量列，数据不丢。
"""
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.database import NoteStore
from app.main import create_app

NOW = "2026-09-14T00:00:00+00:00"
DIM = 512  # 与 app.embeddings.EXPECTED_DIMENSION 一致


def make_store(path):
    return NoteStore(path)


def seed(store):
    return store.save_memories("test-model", [
        {"type": "preference", "content": "用户偏好居家徒手训练"},
        {"type": "correction", "content": "用户习惯把减脂餐归为健身"},
    ], "note0", NOW)


def fake_vector(has_fitness):
    """确定性 512 维向量：命中“健身”语义的第 0 维为 1，否则全零。"""
    vector = [0.0] * DIM
    if has_fitness:
        vector[0] = 1.0
    return vector


def fake_embed_query(text):
    return fake_vector("健身" in text)


def fake_embed_texts(texts):
    return [fake_vector("健身" in text) for text in texts]


def install_fake_embeddings(monkeypatch, embed_texts=fake_embed_texts,
                            embed_query=fake_embed_query):
    """只替换模型相关的两个函数；blob/cosine 等纯数学函数用真实现。"""
    monkeypatch.setattr("app.embeddings.embed_texts", embed_texts)
    monkeypatch.setattr("app.embeddings.embed_query", embed_query)


def test_recall_without_query_behaves_like_active_memories(tmp_path):
    """query 为空（含空白串）时必须与现有 active_memories 完全等价。"""
    store = make_store(tmp_path / "test.sqlite3")
    seed(store)
    assert store.recall_memories() == store.active_memories()
    assert store.recall_memories("   ") == store.active_memories()
    assert store.recall_memories(None, limit=1) == store.active_memories(limit=1)


def test_recall_with_query_on_empty_store_returns_empty_list(tmp_path, monkeypatch):
    """空记忆时带 query 召回返回空数组，不报错（清单第 6 条的硬要求）。"""
    store = make_store(tmp_path / "test.sqlite3")
    install_fake_embeddings(monkeypatch)
    assert store.recall_memories("健身") == []


def test_recall_by_similarity_ranks_with_score(tmp_path, monkeypatch):
    """query 非空时按余弦相似度排序，带 score 字段，且向量列不外泄。"""
    store = make_store(tmp_path / "test.sqlite3")
    seed(store)
    install_fake_embeddings(monkeypatch)

    result = store.recall_memories("健身")
    assert len(result) == 2
    # 命中“健身”语义的记忆排在最前，score 接近 1。
    assert result[0]["content"] == "用户习惯把减脂餐归为健身"
    assert result[0]["score"] == pytest.approx(1.0)
    assert result[0]["score"] >= result[1]["score"]
    # 内部向量缓存列是对外不可见的实现细节。
    assert "embedding" not in result[0]
    assert "embedding_model" not in result[0]


def test_recall_backfills_missing_vectors_once(tmp_path, monkeypatch):
    """第一次召回触发惰性回填；向量就绪后第二次调用不再重复计算。"""
    store = make_store(tmp_path / "test.sqlite3")
    seed(store)
    calls = []

    def counting_embed_texts(texts):
        calls.append(list(texts))
        return fake_embed_texts(texts)

    install_fake_embeddings(monkeypatch, embed_texts=counting_embed_texts)

    store.recall_memories("健身")
    assert len(calls) == 1 and len(calls[0]) == 2
    store.recall_memories("健身")
    assert len(calls) == 1  # 向量缓存生效


def test_recall_degrades_to_recency_when_embedding_fails(tmp_path, monkeypatch):
    """模型缺失/推理异常时降级成时间排序，绝不向上抛错。"""
    store = make_store(tmp_path / "test.sqlite3")
    seed(store)

    def broken(*args, **kwargs):
        raise RuntimeError("embedding 模型不可用")

    install_fake_embeddings(monkeypatch, embed_texts=broken, embed_query=broken)
    result = store.recall_memories("健身")
    assert [m["id"] for m in result] == [m["id"] for m in store.active_memories()]


def test_superseded_memory_excluded_from_recall(tmp_path, monkeypatch):
    """superseded 与 dismissed 一样不进 active/召回，但可按状态查到。"""
    store = make_store(tmp_path / "test.sqlite3")
    saved = seed(store)
    install_fake_embeddings(monkeypatch)

    store.update_memory(saved[0]["id"], {"status": "superseded"}, NOW)
    assert all(m["id"] != saved[0]["id"] for m in store.recall_memories("健身"))
    assert [m["id"] for m in store.active_memories()] == [saved[1]["id"]]
    assert [m["id"] for m in store.memories(status="superseded")] == [saved[0]["id"]]


def test_api_filters_superseded_and_rejects_manual_set(tmp_path):
    """接口层：?status=superseded 可过滤；手动 PUT 设置被 400 拒绝。"""
    db_path = tmp_path / "test.sqlite3"
    store = make_store(db_path)
    saved = seed(store)
    store.update_memory(saved[0]["id"], {"status": "superseded"}, NOW)

    with TestClient(create_app(db_path)) as client:
        items = client.get("/memories?status=superseded").json()["items"]
        blocked = client.put(f"/memories/{saved[1]['id']}",
                             json={"status": "superseded"})
        # 删除/恢复等合法状态迁移不受影响。
        restored = client.put(f"/memories/{saved[0]['id']}",
                              json={"status": "active"})

    assert len(items) == 1 and items[0]["id"] == saved[0]["id"]
    assert blocked.status_code == 400
    assert restored.status_code == 200
    assert restored.json()["status"] == "active"


def test_old_two_state_table_auto_migrates(tmp_path):
    """旧版两态表第一次打开时自动重建为三态 + 向量列，数据完整保留。"""
    db_path = tmp_path / "old.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute("""CREATE TABLE memories (
            id INTEGER PRIMARY KEY,
            type TEXT NOT NULL CHECK(type IN ('preference', 'habit', 'correction')),
            content TEXT NOT NULL, content_hash TEXT NOT NULL UNIQUE,
            model TEXT NOT NULL, source_note_id TEXT,
            status TEXT NOT NULL DEFAULT 'active'
                CHECK(status IN ('active', 'dismissed')),
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )""")
        db.execute("""INSERT INTO memories VALUES
            (1, 'preference', '旧记忆内容', 'hash-old', 'm', 'note0',
             'active', '2026-09-01', '2026-09-01')""")

    store = make_store(db_path)
    rows = store.memories()
    assert len(rows) == 1
    assert rows[0]["content"] == "旧记忆内容"
    assert "embedding" not in rows[0]  # 旧数据没有向量，也不伪造空列

    # 向量列可写：模拟回填后能被内部查询读到。
    store.save_memory_embeddings([
        {"id": 1, "vector": b"\x00" * (DIM * 4), "model": "bge",
         "content_hash": "hash-old"}])
    assert store.memories_missing_embeddings("bge") == []
    assert len(store._active_rows_with_embeddings("bge")) == 1

    # 三态 CHECK 生效：superseded 现在是合法状态。
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE memories SET status = 'superseded' WHERE id = 1")
    assert store.memories(status="superseded")[0]["id"] == 1
