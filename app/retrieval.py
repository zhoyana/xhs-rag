"""检索结果后处理：把 Chunk 级候选整理成用户真正需要的笔记级结果。"""


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
