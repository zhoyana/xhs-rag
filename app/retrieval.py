"""检索编排：向量语义路、关键词路，以及把两路名次合并的 RRF。

本模块只做"候选 → 排名"，不碰 HTTP、不碰提示词，所以可以在单测里注入假向量
直接跑。三种模式的取舍集中在 search_knowledge 一处，路由与 Agent 工具都从这里
取结果，避免两个入口各写一套排序规则而慢慢漂移。
"""

from .embeddings import MODEL_NAME, blob_to_vector, cosine_similarity, embed_query

MODE_VECTOR = "vector"
MODE_HYBRID = "hybrid"

# RRF 的平滑常数与原论文一致取 60。
RRF_K = 60
# 每一路参与融合的名次深度。两路取同一个值，是为了让融合前两路的"视野"
# 相当——一路看 5 条、另一路看 100 条时，名次之间的可比性就没了。
FUSION_LIMIT = 30
# 关键词路召回条数。
BM25_LIMIT = 30


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


def vector_ranking(candidates: list[dict], query: str, query_embedder=embed_query) -> list[dict]:
    """给候选 Chunk 打余弦分，按分数降序返回。

    分数写在 vector_score 而不是 score：融合后的排序依据是 RRF 名次分，
    两套量纲不同的分数共用一个字段名，迟早会出现"拿余弦分当融合分排序"
    这类看不出错在哪的 bug。
    """
    if not candidates:
        return []
    query_vector = query_embedder(query.strip())
    ranked = []
    for original in candidates:
        # 不在原字典上 pop，避免调用者缓存候选时被意外修改。
        candidate = dict(original)
        vector = candidate.pop("vector")
        chunk_vector = blob_to_vector(vector, candidate["dimension"])
        candidate["vector_score"] = cosine_similarity(query_vector, chunk_vector)
        ranked.append(candidate)
    return sorted(ranked, key=lambda item: item["vector_score"], reverse=True)


def reciprocal_rank_fusion(channels, k: int = RRF_K, limit: int = FUSION_LIMIT) -> list[dict]:
    """把多路排名合并：score = Σ 1/(k + rank)，只用名次、不用原始分数。

    这是选 RRF 而不是加权求和的根本原因——BM25 的分数无上界且随语料变化，
    余弦相似度固定在 [0, 1]，两者直接加权必须先归一化，而归一化方式本身又
    成了一个要调的参数；名次天然无量纲，省掉这层调参。

    k=60 压平头部差距，让某一路的 Top1 不至于单方面决定最终排序。

    channels 是 [(名称, 已排序候选), ...]；同一个 Chunk 被两路同时命中会累加
    贡献，这正是"两路都认为相关"应当得到的奖励，也是混合检索优于单路的地方。
    """
    merged: dict = {}
    for name, ranking in channels:
        for rank, item in enumerate(ranking[:limit], start=1):
            key = item["chunk_id"]
            entry = merged.get(key)
            if entry is None:
                entry = dict(item)
                entry["rrf_score"] = 0.0
                entry["channels"] = []
                merged[key] = entry
            else:
                # 补齐另一路才带的分数字段（正文、标题两边都有，不必覆盖）。
                for field in ("vector_score", "bm25_score"):
                    if field in item and field not in entry:
                        entry[field] = item[field]
            entry["rrf_score"] += 1.0 / (k + rank)
            if name not in entry["channels"]:
                entry["channels"].append(name)

    ordered = sorted(merged.values(), key=lambda entry: entry["rrf_score"], reverse=True)
    for entry in ordered:
        # score 直接暴露融合分：展示出来的数字与排序依据是同一个，
        # 不会出现"看着 0.9 却排在 0.8 后面"这种解释不清的现象。
        entry["score"] = entry["rrf_score"]
    return ordered


def search_knowledge(
    store, query: str, top_k: int = 5, deduplicate_notes: bool = True,
    query_embedder=embed_query, mode: str = MODE_HYBRID, rrf_k: int = RRF_K,
) -> list[dict]:
    """执行检索，供 FastAPI 路由和 Agent Tool 共同复用。

    hybrid（默认）：关键词（FTS5/BM25）与向量两路召回，按 RRF 融合名次。
      两路强项互补——向量认语义近义与换一种说法的提问，关键词认专有名词、
      型号、术语与精确短语，后者正是中文短查询下纯向量最容易失手的地方。
    vector：只跑向量余弦。保留它不是为了兼容旧行为，而是为了消融对比——
      "混合到底带来多少提升"必须有同一套评测集下的单路基线才说得清。

    任一路空手时自动退化成另一路：老库还没补建 FTS 索引、或查询里只有标点时，
    结果不会比单路更差，也不会抛错。
    """
    candidates = store.searchable_embeddings(MODEL_NAME)
    if not candidates:
        return []

    ranked_vector = vector_ranking(candidates, query, query_embedder)
    keyword_ranking = (
        store.search_chunks_fts(query, limit=BM25_LIMIT)
        if mode == MODE_HYBRID else []
    )
    if not keyword_ranking:
        for item in ranked_vector:
            item["score"] = item["vector_score"]
        return select_top_k(ranked_vector, top_k, deduplicate_notes)

    fused = reciprocal_rank_fusion(
        [("vector", ranked_vector), ("keyword", keyword_ranking)], k=rrf_k)
    return select_top_k(fused, top_k, deduplicate_notes)
