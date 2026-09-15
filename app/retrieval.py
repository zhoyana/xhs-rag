"""检索结果后处理：把 Chunk 级候选整理成用户真正需要的笔记级结果。"""

from .embeddings import MODEL_NAME, blob_to_vector, cosine_similarity, embed_query


def select_top_k(ranked: list[dict], top_k: int, deduplicate_notes: bool = True) -> list[dict]:
    """按分数选 Top-K；默认每篇笔记只保留得分最高的一个 Chunk。"""
    ordered = sorted(ranked, key=lambda item: item["score"], reverse=True)
    if not deduplicate_notes:
        return ordered[:top_k]

    # 秋招知识点：这是检索的 diversity（多样性）约束，减少同一文档霸占上下文窗口。
    selected = []
    seen_note_ids = set()
    for item in ordered:
        if item["note_id"] in seen_note_ids:
            continue
        seen_note_ids.add(item["note_id"])
        selected.append(item)
        if len(selected) == top_k:
            break
    return selected


def search_knowledge(
    store, query: str, top_k: int = 5, deduplicate_notes: bool = True,
    query_embedder=embed_query,
) -> list[dict]:
    """执行语义检索，供 FastAPI 路由和 Agent Tool 共同复用。

    秋招知识点：把检索从 Controller 中抽成 Service，避免 HTTP 接口与 Agent
    各写一套算法。以后加入 BM25、RRF 或 Reranker，只改这里即可。
    """
    candidates = store.searchable_embeddings(MODEL_NAME)
    if not candidates:
        return []
    query_vector = query_embedder(query.strip())
    ranked = []
    for original in candidates:
        # 不在原字典上 pop，避免调用者缓存候选时被意外修改。
        candidate = dict(original)
        vector = candidate.pop("vector")
        chunk_vector = blob_to_vector(vector, candidate["dimension"])
        ranked.append({**candidate, "score": cosine_similarity(query_vector, chunk_vector)})
    return select_top_k(ranked, top_k, deduplicate_notes)
