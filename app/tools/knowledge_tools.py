"""知识库工具：为 LangChain/MCP 保留稳定的输入输出协议。"""
from ..note_text import note_markdown


def search_notes_tool(searcher, query: str, limit: int = 5) -> dict:
    """检索笔记；searcher 是现有检索函数，未来可替换为混合检索。"""
    query = query.strip()
    if not query:
        return {"ok": False, "data": [], "error": "query 不能为空"}
    if not 1 <= limit <= 20:
        return {"ok": False, "data": [], "error": "limit 必须在 1 到 20 之间"}
    return {"ok": True, "data": searcher(query, limit), "error": None}


def get_note_tool(getter, note_id: str) -> dict:
    """按 ID 查看完整笔记；不存在时不让异常泄漏到 Agent。

    返回渲染后的 markdown，而不是原始 payload。模型要的是"这篇讲了什么"，
    而 payload 里 `image_text`、评论常常是**不存在的键**——它只能靠"少了一个键"
    去察觉，很容易答成"我拿不到图片里的字"（其实库里就没有）。markdown 会把
    缺失的段落明写成"（无：原因）"，把"没有"变成明确事实。

    `note_id` 必须保留：`chat_agent` 靠它记录本轮引用过的笔记（`viewed_note_ids`），
    少了它引用追踪会静默失效。
    """
    note = getter(note_id.strip()) if note_id else None
    if not isinstance(note, dict):
        return {"ok": False, "data": None, "error": "笔记不存在"}
    return {"ok": True, "data": {"note_id": note.get("note_id"),
                                "markdown": note_markdown(note)}, "error": None}
