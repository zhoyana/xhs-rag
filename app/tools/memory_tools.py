"""用户记忆工具。"""


def recall_memories_tool(recaller, query: str | None = None, limit: int = 10) -> dict:
    """召回用户偏好；底层可从时间排序升级为向量召回而不改 Agent 协议。"""
    if not 1 <= limit <= 20:
        return {"ok": False, "data": [], "error": "limit 必须在 1 到 20 之间"}
    return {"ok": True, "data": recaller(query.strip() if query else None, limit), "error": None}
