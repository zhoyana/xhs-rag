"""混合检索测试：中文单字分词、BM25 召回、RRF 融合，以及老库降级。

全部离线且不加载 Embedding 模型——向量由测试直接写进库，所以每条用例都是
秒级完成。检索质量本身由 scripts/retrieval_eval.py 在真实评测集上衡量，
这里钉的是"机制对不对"：分词口径、融合数学、索引同步、降级路径。
"""
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import indexing
from app.database import NoteStore
from app.embeddings import EXPECTED_DIMENSION, MODEL_NAME, vector_to_blob
from app.fts import build_match_query, segment, tokens
from app.main import create_app
from app.retrieval import reciprocal_rank_fusion, search_knowledge

NOW = "2026-09-22T00:00:00+00:00"


def make_note(note_id, title, content):
    return {
        "note_id": note_id, "title": title, "content": content,
        "source_url": f"https://www.xiaohongshu.com/explore/{note_id}", "images": [],
    }


def unit_vector(*values):
    """构造 512 维向量列表：只用传入的几维表达方向，其余为 0。

    返回 list 而不是 BLOB，是为了与真实 embed_query 的输出形态一致——
    查询向量是内存里的浮点数组，库里的向量是 float32 字节，混用会直接在
    cosine_similarity 里抛错。
    """
    return [*values, *([0.0] * (EXPECTED_DIMENSION - len(values)))]


def build_store(tmp_path, name, notes):
    """建库、存笔记、建索引；向量由调用方按需再写。"""
    store = NoteStore(tmp_path / name)
    for note in notes:
        store.save(note)
        indexing.index_note(store, note["note_id"])
    return store


def embed_note(store, note_id, vector):
    """给一篇笔记的所有 Chunk 写入同一个向量（测试正文短，通常只有一个 Chunk）。"""
    blob = vector_to_blob(vector)
    document = store.document_for_note(note_id)
    chunks = store.document_chunks(document["id"])
    store.save_embeddings([{
        "chunk_id": chunk["id"], "model": MODEL_NAME, "dimension": EXPECTED_DIMENSION,
        "vector": blob, "content_hash": chunk["content_hash"],
    } for chunk in chunks], NOW)


# ----------------------------- 分词与查询构造 -----------------------------

def test_segment_splits_cjk_per_character():
    assert segment("小红书") == "小 红 书"
    # 汉字与英文之间必须也断开：unicode61 视字母为同类，
    # 不断开就会切出"书abc"这种混合 token，两边都搜不到。
    assert segment("小红书abc") == "小 红 书 abc"
    assert segment("abc小红书") == "abc 小 红 书"
    assert segment("") == ""
    assert segment("   ") == ""


def test_segment_keeps_non_cjk_intact():
    assert segment("bge-small-zh 512") == "bge-small-zh 512"


def test_tokens_keep_underscored_word_together():
    # unicode61 把下划线当 token 字符。查询侧若在 `_` 处切分，
    # a_b 会变成 a 与 b 两个 token，跟索引里的单词 a_b 对不上。
    assert tokens("a_b") == ["a_b"]


def test_tokens_drop_punctuation_only_text():
    assert tokens("，。！？（）") == []


def test_build_match_query_wraps_fragment_in_phrase():
    # 引号把 token 序列变成短语：这是单字索引下唯一能表达"这几个字挨在一起"的方式。
    assert build_match_query("排版") == '"排 版"'


def test_build_match_query_ors_fragments():
    # 片段之间用 OR：其中一个词没命中时，不该让整条查询空手而归。
    match = build_match_query("小红书 排版")
    assert match.startswith('"小 红 书"')
    assert '"排 版"' in match
    assert " OR " in match


def test_build_match_query_falls_back_to_bigrams_for_long_query():
    """整句粘进来时不能只靠整串短语——那样几乎必然空手，bigram 才是兜底。"""
    match = build_match_query("我想知道秋招简历应该怎么排版")
    phrases = [part.strip('"') for part in match.split(" OR ")]
    assert all(len(phrase.split()) == 2 for phrase in phrases)
    # 结尾的关键词必须被覆盖到，否则位于句尾的意图永远搜不到。
    assert "排 版" in phrases
    assert "秋 招" in phrases


def test_build_match_query_returns_none_without_usable_words():
    assert build_match_query("") is None
    assert build_match_query("！！！？？…—") is None


# ----------------------------- RRF 融合 -----------------------------

def test_rrf_rewards_chunks_hit_by_both_channels():
    vector = [{"chunk_id": 1, "note_id": "a"}, {"chunk_id": 2, "note_id": "b"}]
    keyword = [{"chunk_id": 2, "note_id": "b"}, {"chunk_id": 3, "note_id": "c"}]
    fused = reciprocal_rank_fusion([("vector", vector), ("keyword", keyword)], k=60)
    by_id = {item["chunk_id"]: item for item in fused}

    # 手工核对公式：score = Σ 1/(k + rank)
    assert by_id[2]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert by_id[1]["rrf_score"] == pytest.approx(1 / 61)
    assert by_id[3]["rrf_score"] == pytest.approx(1 / 62)
    # 两路都命中的累加两次，必然排在只被单路命中之前
    assert [item["chunk_id"] for item in fused] == [2, 1, 3]
    assert by_id[2]["channels"] == ["vector", "keyword"]
    assert by_id[1]["channels"] == ["vector"]
    # score 必须等于排序依据，展示的数字和实际顺序不能是两回事
    assert by_id[2]["score"] == pytest.approx(by_id[2]["rrf_score"])


def test_rrf_uses_rank_only_not_raw_scores():
    """RRF 的立论基础：原始分数换成另一套量纲，融合结果不变。"""
    first = [{"chunk_id": 1, "vector_score": 0.99}, {"chunk_id": 2, "vector_score": 0.98}]
    second = [{"chunk_id": 1, "bm25_score": 14.7}, {"chunk_id": 2, "bm25_score": 0.12}]
    left = reciprocal_rank_fusion([("x", list(first))], k=60)
    right = reciprocal_rank_fusion([("y", list(second))], k=60)
    assert [item["chunk_id"] for item in left] == [item["chunk_id"] for item in right]
    assert [item["rrf_score"] for item in left] == [item["rrf_score"] for item in right]


def test_rrf_keeps_fields_from_both_channels():
    """两路各自带来的分数字段要都留下，否则无从解释某条为什么排上来。"""
    vector = [{"chunk_id": 1, "vector_score": 0.7}]
    keyword = [{"chunk_id": 1, "bm25_score": 2.5}]
    fused = reciprocal_rank_fusion([("vector", vector), ("keyword", keyword)], k=60)
    assert fused[0]["vector_score"] == 0.7
    assert fused[0]["bm25_score"] == 2.5


# ----------------------------- 索引同步 -----------------------------

def test_fts_index_follows_chunks(tmp_path):
    store = build_store(tmp_path, "sync.sqlite3", [
        make_note("a", "秋招时间线", "秋招从七月开始，提前批比正式批容易。")])
    assert store.fts_stats()["in_sync"]
    assert [hit["note_id"] for hit in store.search_chunks_fts("提前批")] == ["a"]


def test_fts_drops_rows_when_chunks_replaced(tmp_path):
    """正文改掉之后，旧 Chunk 不该留在索引里——否则删掉的字还能搜到。"""
    db_path = tmp_path / "replace.sqlite3"
    store = NoteStore(db_path)
    store.save(make_note("a", "护肤", "防晒是护肤里最有效的一步。"))
    indexing.index_note(store, "a")
    assert store.search_chunks_fts("防晒")

    store.update_note_fields("a", {"content": "换成了完全无关的内容，讲简历排版。"})
    indexing.index_note(store, "a")

    assert store.search_chunks_fts("防晒") == []
    assert [hit["note_id"] for hit in store.search_chunks_fts("排版")] == ["a"]
    assert store.fts_stats()["in_sync"]


def test_title_weight_outranks_body_mention(tmp_path):
    """标题命中比正文顺带提及更相关——这就是 bm25() 列权重存在的理由。"""
    store = build_store(tmp_path, "weight.sqlite3", [
        make_note("title-hit", "简历排版", "一页纸讲清楚重点就够了。"),
        make_note("body-hit", "日常记录", "排版这件事别写太长。"),
    ])
    scores = {hit["note_id"]: hit["bm25_score"] for hit in store.search_chunks_fts("排版", limit=5)}
    assert scores["title-hit"] > scores["body-hit"]


def test_search_chunks_fts_returns_empty_without_usable_query(tmp_path):
    store = build_store(tmp_path, "empty.sqlite3", [make_note("a", "标题", "正文内容。")])
    # 纯标点无从匹配，返回空而不是拿一个空表达式去扫全表
    assert store.search_chunks_fts("！！！") == []


def test_long_spoken_query_still_reaches_keyword_channel(tmp_path):
    """整句粘进来时 BM25 不能空手——空手就等于混合检索静默退化成纯向量。

    这是真实踩到的坑：查询"怎么做预测基线"整串当短语命不中任何东西，
    而语料里明明有"再做预测"。靠 bigram 里的"预 测"才能捞回来。
    """
    store = build_store(tmp_path, "bigram.sqlite3", [
        make_note("a", "预测", "先建立 baseline 再做预测，最后看误差。")])
    assert [hit["note_id"] for hit in store.search_chunks_fts("怎么做预测基线")] == ["a"]


def test_rebuild_fts_index_backfills_legacy_rows(tmp_path):
    """老库（引入混合检索前建的）只有 chunks 没有 FTS 行，重建一次就补齐。"""
    db_path = tmp_path / "legacy.sqlite3"
    store = NoteStore(db_path)
    store.save(make_note("a", "排版", "简历排版不要超过一页。"))
    indexing.index_note(store, "a")
    with sqlite3.connect(db_path) as db:
        db.execute("DELETE FROM chunks_fts")     # 模拟老库：索引缺失

    assert store.search_chunks_fts("排版") == []
    assert store.fts_stats()["in_sync"] is False

    result = store.rebuild_fts_index()
    assert result == {"status": "rebuilt", "indexed": store.fts_stats()["chunks"]}
    assert [hit["note_id"] for hit in store.search_chunks_fts("排版")] == ["a"]
    # 纯派生数据重建，重复执行没有副作用
    assert store.rebuild_fts_index() == result


# ----------------------------- 检索模式 -----------------------------

def test_hybrid_recovers_phrase_hit_that_vector_misses(tmp_path):
    """关键词路要能把精确短语命中顶回来。

    构造：查询向量只跟"换个说法"的那篇同向，真正含关键词的那篇向量相似度是 0。
    纯向量会被语义相近的那篇骗走，混合检索必须把词面命中的那篇排到第一。
    """
    store = build_store(tmp_path, "hybrid.sqlite3", [
        make_note("semantic", "换个说法", "这段文字和问题是一个意思，但用词完全不同。"),
        make_note("keyword", "精确短语", "简历排版不要超过一页。"),
    ])
    embed_note(store, "semantic", unit_vector(1.0))
    embed_note(store, "keyword", unit_vector(0.0, 1.0))

    def fake_query_embedder(_query):
        return unit_vector(1.0)

    vector_only = search_knowledge(
        store, "简历排版", query_embedder=fake_query_embedder, mode="vector")
    hybrid = search_knowledge(
        store, "简历排版", query_embedder=fake_query_embedder, mode="hybrid")

    assert vector_only[0]["note_id"] == "semantic"     # 纯向量被同向的那篇赢走
    assert hybrid[0]["note_id"] == "keyword"           # 双路命中把词面那篇顶到第一
    assert hybrid[0]["channels"] == ["vector", "keyword"]
    assert vector_only[0]["score"] == pytest.approx(1.0)          # vector 模式保留余弦分
    assert hybrid[0]["score"] == pytest.approx(1 / 62 + 1 / 61)   # hybrid 模式是融合分


def test_hybrid_degrades_to_vector_without_keyword_index(tmp_path):
    """老库还没补建 FTS 时，hybrid 必须等于纯向量，而不是返回空。"""
    db_path = tmp_path / "degrade.sqlite3"
    store = build_store(tmp_path, "degrade.sqlite3", [
        make_note("a", "秋招", "秋招从七月开始。")])
    embed_note(store, "a", unit_vector(1.0))
    with sqlite3.connect(db_path) as db:
        db.execute("DELETE FROM chunks_fts")

    def fake_query_embedder(_query):
        return unit_vector(1.0)

    hybrid = search_knowledge(store, "秋招", query_embedder=fake_query_embedder, mode="hybrid")
    vector_only = search_knowledge(
        store, "秋招", query_embedder=fake_query_embedder, mode="vector")
    assert [item["note_id"] for item in hybrid] == [item["note_id"] for item in vector_only]
    assert hybrid[0]["score"] == pytest.approx(hybrid[0]["vector_score"])


def test_search_knowledge_returns_empty_without_vectors(tmp_path):
    store = build_store(tmp_path, "novec.sqlite3", [make_note("a", "标题", "正文内容。")])
    assert search_knowledge(store, "正文") == []


# ----------------------------- 接口 -----------------------------

def test_search_endpoint_reports_retrieval_mode(tmp_path, monkeypatch):
    db_path = tmp_path / "api.sqlite3"
    store = NoteStore(db_path)
    store.save(make_note("kw", "简历排版", "简历排版不要超过一页。"))
    store.save(make_note("sem", "换个说法", "用词完全不同但意思一样的一段话。"))
    for note_id in ("kw", "sem"):
        indexing.index_note(store, note_id)
    embed_note(store, "kw", unit_vector(0.0, 1.0))
    embed_note(store, "sem", unit_vector(1.0))

    monkeypatch.setattr("app.main.embed_search_query", lambda _query: unit_vector(1.0))
    with TestClient(create_app(db_path)) as client:
        body = client.post("/search", json={"query": "简历排版"}).json()
        legacy = client.post("/search", json={"query": "简历排版", "mode": "vector"}).json()
        rebuilt = client.post("/search/rebuild-index").json()

    assert body["retrieval_mode"] == "hybrid"
    assert body["items"][0]["note_id"] == "kw"
    assert body["fts_rows"] == 2
    assert legacy["retrieval_mode"] == "vector"
    assert legacy["items"][0]["note_id"] == "sem"
    assert rebuilt["status"] == "rebuilt"


def test_search_endpoint_rejects_unknown_mode(tmp_path):
    with TestClient(create_app(tmp_path / "bad.sqlite3")) as client:
        response = client.post("/search", json={"query": "排版", "mode": "magic"})
    assert response.status_code == 422
