"""知识库工具：为 LangChain/MCP 保留稳定的输入输出协议。"""


def search_notes_tool(searcher, query: str, limit: int = 5) -> dict:
    """检索笔记；searcher 是现有检索函数，未来可替换为混合检索。"""
    query = query.strip()
    if not query:
        return {"ok": False, "data": [], "error": "query 不能为空"}
    if not 1 <= limit <= 20:
        return {"ok": False, "data": [], "error": "limit 必须在 1 到 20 之间"}
    return {"ok": True, "data": searcher(query, limit), "error": None}


def get_note_tool(getter, note_id: str) -> dict:
    """按 ID 查看完整笔记；不存在时不让异常泄漏到 Agent。"""
    note = getter(note_id.strip()) if note_id else None
    return {"ok": note is not None, "data": note, "error": None if note else "笔记不存在"}
