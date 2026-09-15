"""评测指标单元测试：不加载真实模型，不访问正式数据库。"""
from app.evaluation import hit_at_k, reciprocal_rank, summarize_metrics


def test_hit_at_k_and_reciprocal_rank():
    ranked = ["wrong", "target", "other"]
    relevant = {"target"}
    assert hit_at_k(ranked, relevant, 1) == 0.0
    assert hit_at_k(ranked, relevant, 3) == 1.0
    assert reciprocal_rank(ranked, relevant) == 0.5


def test_duplicate_chunks_keep_their_real_ranks():
    # 秋招知识点：同一错误笔记占据两个 Chunk 时，相关笔记真实排名是第 3，不能偷偷去重美化指标。
    ranked = ["wrong", "wrong", "target"]
    assert reciprocal_rank(ranked, {"target"}) == 1 / 3
    assert hit_at_k(ranked, {"target"}, 1) == 0.0
    assert hit_at_k(ranked, {"target"}, 3) == 1.0


def test_summary_averages_answerable_cases():
    results = [
        {"hit_at_1": 1.0, "hit_at_3": 1.0, "hit_at_5": 1.0, "reciprocal_rank": 1.0},
        {"hit_at_1": 0.0, "hit_at_3": 1.0, "hit_at_5": 1.0, "reciprocal_rank": 0.5},
    ]
    summary = summarize_metrics(results)
    assert summary == {"hit_at_1": 0.5, "hit_at_3": 1.0, "hit_at_5": 1.0, "mrr": 0.75}
