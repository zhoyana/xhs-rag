"""LangChain 对话 Agent：让模型选择工具，并把工具结果送回模型。"""
import json
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI


SYSTEM_PROMPT = """你是用户的小红书知识库学习助手。
回答收藏内容前先使用 search_notes；需要完整原文时再使用 get_note。
制定计划前应检索相关笔记并读取用户记忆，之后只能生成 draft 草案。
绝不能替用户确认计划或创建任务。回答使用简洁中文，并说明参考了哪些笔记。
"""


def deepseek_chat_model():
    """通过 OpenAI 兼容协议创建 LangChain ChatModel；不会读取或输出密钥内容。"""
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ValueError("尚未配置 DEEPSEEK_API_KEY。")
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash-vision-exp").strip(),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
        temperature=0,
        timeout=90,
        max_retries=1,
    )


async def run_agent_turn(history: list[dict], tools: list, model=None, max_rounds: int = 6) -> dict:
    """运行一次 ReAct 风格的“模型 → 工具 → 模型”循环。

    max_rounds 是保险丝，防止模型反复调用工具形成无限循环。
    """
    llm = model or deepseek_chat_model()
    model_with_tools = llm.bind_tools(tools)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for item in history:
        message_type = HumanMessage if item["role"] == "user" else AIMessage
        messages.append(message_type(content=item["content"]))

    tool_map = {tool.name: tool for tool in tools}
    traces, source_note_ids, plan_id = [], [], None
    for _ in range(max_rounds):
        response = await model_with_tools.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return {
                "content": response.content if isinstance(response.content, str) else str(response.content),
                "tool_calls": traces,
                "source_note_ids": list(dict.fromkeys(source_note_ids)),
                "plan_id": plan_id,
            }

        for call in response.tool_calls:
            tool = tool_map.get(call["name"])
            if tool is None:
                result = {"ok": False, "data": None, "error": "模型请求了不存在的工具"}
            else:
                try:
                    result = await tool.ainvoke(call["args"])
                except Exception as exc:  # 工具失败应回传给模型，让它解释或换策略。
                    result = {"ok": False, "data": None, "error": type(exc).__name__}
            traces.append({"name": call["name"], "args": call["args"], "ok": result.get("ok", False)})
            if call["name"] == "search_notes" and result.get("ok"):
                source_note_ids.extend(item["note_id"] for item in result["data"] if item.get("note_id"))
            if call["name"] == "draft_plan" and result.get("ok"):
                data = result.get("data") or {}
                plan = data.get("plan", data)
                plan_id = plan.get("id") if isinstance(plan, dict) else None
            messages.append(ToolMessage(
                content=json.dumps(result, ensure_ascii=False, default=str),
                tool_call_id=call["id"],
            ))
    raise RuntimeError("Agent 工具调用次数过多，请缩小问题范围后重试。")
