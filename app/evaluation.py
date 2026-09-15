"""检索评测指标。

本模块只处理“人工标签 + 排名结果”，不读取数据库、不加载模型，因此可以快速单元测试。
"""
from collections.abc import Iterable


def reciprocal_rank(ranked_note_ids: list[str], relevant_note_ids: set[str]) -> float:
    """计算单个问题的倒数排名；第一个相关结果在第 r 位时得分为 1/r。"""
    for rank, note_id in enumerate(ranked_note_ids, 1):
        if note_id in relevant_note_ids:
            return 1.0 / rank
    return 0.0


def hit_at_k(ranked_note_ids: list[str], relevant_note_ids: set[str], k: int) -> float:
    """前 K 个 Chunk 中只要出现任意相关笔记，就记为命中。"""
    return float(any(note_id in relevant_note_ids for note_id in ranked_note_ids[:k]))


def summarize_metrics(answerable_results: Iterable[dict], ks=(1, 3, 5)) -> dict:
    """汇总有答案问题的 Hit@K 和 MRR。"""
    results = list(answerable_results)
    if not results:
        return {**{f"hit_at_{k}": 0.0 for k in ks}, "mrr": 0.0}

    summary = {}
    for k in ks:
        summary[f"hit_at_{k}"] = sum(item[f"hit_at_{k}"] for item in results) / len(results)
    summary["mrr"] = sum(item["reciprocal_rank"] for item in results) / len(results)
    return summary
