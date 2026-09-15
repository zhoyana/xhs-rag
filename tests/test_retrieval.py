"""检索后处理测试：不加载真实模型，快速验证笔记级去重。"""
from app.retrieval import select_top_k


def test_select_top_k_deduplicates_notes_by_default():
    ranked = [
        {"note_id": "a", "chunk_id": 1, "score": 0.90},
        {"note_id": "a", "chunk_id": 2, "score": 0.80},
        {"note_id": "b", "chunk_id": 3, "score": 0.70},
    ]
    selected = select_top_k(ranked, top_k=2)
    assert [(item["note_id"], item["chunk_id"]) for item in selected] == [("a", 1), ("b", 3)]


def test_select_top_k_can_return_raw_chunks_for_experiments():
    ranked = [
        {"note_id": "a", "chunk_id": 1, "score": 0.90},
        {"note_id": "a", "chunk_id": 2, "score": 0.80},
        {"note_id": "b", "chunk_id": 3, "score": 0.70},
    ]
    selected = select_top_k(ranked, top_k=2, deduplicate_notes=False)
    assert [item["chunk_id"] for item in selected] == [1, 2]
