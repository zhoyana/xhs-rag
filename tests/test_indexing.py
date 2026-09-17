"""索引 + 向量一步到位：验证收藏后能立刻语义检索，且失败可降级。"""
import pytest

from app.database import NoteStore
from app.embeddings import EXPECTED_DIMENSION
from app.indexing import index_and_embed, index_note, embed_document
from app.indexing import EmptyContent


def sample_note(note_id="auto-index", repeat=60, **extra):
    note = {
        "note_id": note_id,
        "title": "Agent 工程学习路线",
        "content": "。".join(["先理解检索，再完成评测，然后学习工具调用"] * repeat),
        "source_url": f"https://www.xiaohongshu.com/explore/{note_id}",
        "images": [],
    }
    note.update(extra)
    return note


def fake_embedder(calls):
    """返回固定单位向量的假模型：不联网、不加载权重。"""
    def embed(texts):
        calls.append(list(texts))
        return [[1.0] + [0.0] * (EXPECTED_DIMENSION - 1) for _ in texts]
    return embed


def test_index_and_embed_builds_chunks_and_vectors(tmp_path):
    store = NoteStore(tmp_path / "index.sqlite3")
    store.save(sample_note())
    calls = []

    result = index_and_embed(store, "auto-index", embedder=fake_embedder(calls))

    assert result["status"] == "indexed"
    assert result["chunk_count"] > 1
    # 每个 Chunk 都要有向量，否则"收藏完能搜到"只对一部分内容成立。
    assert result["vector_total"] == result["chunk_count"]
    assert result["searchable"] is True
    assert len(calls) == 1


def test_index_and_embed_is_idempotent(tmp_path):
    """重复收藏同一篇笔记不该重复花钱算向量。"""
    store = NoteStore(tmp_path / "again.sqlite3")
    store.save(sample_note())
    calls = []

    first = index_and_embed(store, "auto-index", embedder=fake_embedder(calls))
    second = index_and_embed(store, "auto-index", embedder=fake_embedder(calls))

    assert first["status"] == "indexed"
    assert second["status"] == "existing"
    assert second["embedded"] == 0
    assert second["vector_total"] == first["vector_total"]
    assert len(calls) == 1


def test_index_and_embed_skips_note_without_indexable_text(tmp_path):
    """正文为空且没有转写结果时应当跳过，而不是抛错让收藏失败。"""
    store = NoteStore(tmp_path / "empty.sqlite3")
    store.save(sample_note(content="   "))

    result = index_and_embed(store, "auto-index", embedder=fake_embedder([]))

    assert result["status"] == "skipped"
    assert result["searchable"] is False
    assert result["reason"]
    assert store.document_for_note("auto-index") is None


def test_image_carried_note_becomes_searchable(tmp_path):
    """正文只剩话题标签的图片笔记，靠 image_text 也要能被索引到。"""
    store = NoteStore(tmp_path / "image.sqlite3")
    store.save(sample_note(
        content="#面试[话题]#",
        image_text="一面：讲讲 Redis 的持久化方案。" * 20,
    ))
    calls = []

    result = index_and_embed(store, "auto-index", embedder=fake_embedder(calls))

    assert result["status"] == "indexed"
    assert result["searchable"] is True
    chunks = store.document_chunks(result["document_id"])
    assert any("Redis 的持久化方案" in chunk["content"] for chunk in chunks)


def test_reindex_after_content_change_rebuilds_vectors(tmp_path):
    """笔记内容变了（例如重新采集后转写更全）必须重建向量，不能吃旧缓存。"""
    store = NoteStore(tmp_path / "changed.sqlite3")
    store.save(sample_note(note_id="changing", content="旧的正文。"))
    calls = []
    embedder = fake_embedder(calls)

    index_and_embed(store, "changing", embedder=embedder)
    store.refresh_note_source("changing", sample_note(note_id="changing", content="全新的正文内容。"),
                              "2026-09-17T00:00:00+00:00")
    again = index_and_embed(store, "changing", embedder=embedder)

    assert again["status"] == "indexed"
    assert again["embedded"] == again["chunk_count"]
    assert len(calls) == 2


def test_embed_document_reports_missing_document(tmp_path):
    store = NoteStore(tmp_path / "missing.sqlite3")
    result = embed_document(store, 999, embedder=fake_embedder([]))
    assert result["status"] == "missing"


def test_index_note_raises_for_unknown_note(tmp_path):
    store = NoteStore(tmp_path / "unknown.sqlite3")
    with pytest.raises(KeyError):
        index_note(store, "nope")


def test_empty_content_is_still_a_value_error(tmp_path):
    """EmptyContent 继承 ValueError，旧调用方的 except ValueError 不会被破坏。"""
    store = NoteStore(tmp_path / "legacy.sqlite3")
    store.save(sample_note(content=""))
    with pytest.raises(ValueError):
        index_note(store, "auto-index")
    assert issubclass(EmptyContent, ValueError)
