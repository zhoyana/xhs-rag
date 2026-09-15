"""Embedding 持久化测试：普通测试使用假向量，不加载或下载真实模型。"""
from fastapi.testclient import TestClient

from app.database import NoteStore
from app.embeddings import EXPECTED_DIMENSION, blob_to_vector, cosine_similarity, vector_to_blob
from app.main import create_app


def test_blob_round_trip():
    # 秋招知识点：序列化 round-trip 要保证保存前后维度和值一致。
    original = [index / EXPECTED_DIMENSION for index in range(EXPECTED_DIMENSION)]
    blob = vector_to_blob(original)
    restored = blob_to_vector(blob, EXPECTED_DIMENSION)
    assert len(blob) == EXPECTED_DIMENSION * 4
    assert len(restored) == EXPECTED_DIMENSION
    assert abs(float(restored[100]) - original[100]) < 1e-6


def test_cosine_similarity():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_document_embedding_is_idempotent(tmp_path, monkeypatch):
    db_path = tmp_path / "embedding.sqlite3"
    NoteStore(db_path).save({
        "note_id": "embed-note",
        "title": "RAG 学习",
        "content": "先切分文档，再生成向量，最后进行语义检索。" * 30,
        "source_url": "https://www.xiaohongshu.com/explore/embed-note",
        "images": [],
    })
    calls = []

    def fake_embed(texts):
        calls.append(list(texts))
        return [[1.0] + [0.0] * (EXPECTED_DIMENSION - 1) for _ in texts]

    monkeypatch.setattr("app.main.embed_texts", fake_embed)
    with TestClient(create_app(db_path)) as client:
        document = client.post("/notes/embed-note/index").json()["document"]
        first = client.post(f"/documents/{document['id']}/embed").json()
        second = client.post(f"/documents/{document['id']}/embed").json()

    assert first["status"] == "embedded"
    assert first["embedded"] == document["chunk_count"]
    assert second["status"] == "existing"
    assert second["embedded"] == 0
    assert second["skipped"] == document["chunk_count"]
    assert len(calls) == 1
    assert all(text.startswith("标题：RAG 学习\n正文片段：") for text in calls[0])
    assert all(text != "标题：RAG 学习" for text in calls[0])


def test_semantic_search_returns_most_related_chunk(tmp_path, monkeypatch):
    db_path = tmp_path / "search.sqlite3"
    store = NoteStore(db_path)
    notes = [
        {"note_id": "forecast", "title": "时间序列", "content": "先建立 baseline 再做预测。",
         "source_url": "https://www.xiaohongshu.com/explore/forecast", "images": []},
        {"note_id": "cooking", "title": "家常菜", "content": "西红柿炒鸡蛋的做法。",
         "source_url": "https://www.xiaohongshu.com/explore/cooking", "images": []},
    ]
    for note in notes:
        store.save(note)

    def fake_embed(texts):
        return [
            ([1.0, 0.0] if "baseline" in text else [0.0, 1.0])
            + [0.0] * (EXPECTED_DIMENSION - 2)
            for text in texts
        ]

    query_vector = [1.0, 0.0] + [0.0] * (EXPECTED_DIMENSION - 2)
    monkeypatch.setattr("app.main.embed_texts", fake_embed)
    monkeypatch.setattr("app.main.embed_search_query", lambda _query: query_vector)

    with TestClient(create_app(db_path)) as client:
        for note in notes:
            document = client.post(f"/notes/{note['note_id']}/index").json()["document"]
            client.post(f"/documents/{document['id']}/embed")
        result = client.post("/search", json={"query": "怎么做预测基线？", "top_k": 1})

    assert result.status_code == 200
    body = result.json()
    assert body["total_candidates"] == 2
    assert body["items"][0]["note_id"] == "forecast"
    assert body["items"][0]["score"] == 1.0
    assert "vector" not in body["items"][0]
