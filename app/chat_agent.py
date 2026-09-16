"""LangChain 对话 Agent：让模型选择工具，并把工具结果送回模型。"""
import json
import os
import re
from datetime import datetime
from time import perf_counter
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI


SYSTEM_PROMPT = """你是用户的小红书知识库学习助手。当前日期是 {today}。
回答收藏内容前先使用 search_notes；需要完整原文时再使用 get_note。
需要站外小红书内容时使用 search_feeds，查看站外笔记原文用 get_feed_detail；
这些站外工具可能不可用，调用失败就说明仅基于知识库回答，不要反复重试。
制定计划前应检索相关笔记并读取用户记忆，之后只能生成 draft 草案。
绝不能替用户确认计划或创建任务。回答使用简洁中文，并说明参考了哪些笔记。
"""


PLAN_ACTION_PATTERN = re.compile(r"制定|生成|创建|安排|规划|出一版|做一份|做一个")
PLAN_OBJECT_PATTERN = re.compile(r"计划|日程|学习安排|复习安排|草案")
PLAN_NEGATION_PATTERN = re.compile(
    r"(不要|不用|无需|不需要|暂时不|先不).{0,12}(计划|日程|草案|制定|生成|创建)"
)


def plan_requested(text: str) -> bool:
    """只有用户明确要求制定计划时，才授权 Agent 看见 draft_plan 工具。"""
    # 动作词和对象词可以相距较远，例如“制定一个从周三开始的一周面试准备计划”。
    # 不能再依赖固定字符窗口，否则自然语言稍长就会漏判。
    if PLAN_NEGATION_PATTERN.search(text):
        return False
    return bool(PLAN_ACTION_PATTERN.search(text) and PLAN_OBJECT_PATTERN.search(text))


def safe_tool_error(exc: Exception) -> dict:
    """保留可诊断错误码，但不把异常堆栈、URL 或密钥交给模型。"""
    status_code = getattr(exc, "status_code", None)
    detail = getattr(exc, "detail", None)
    if isinstance(status_code, int):
        return {"code": f"http_{status_code}", "message": str(detail or "工具请求失败")}
    return {"code": "tool_exception", "message": "工具执行失败，请稍后重试。"}


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


async def run_agent_turn(
    history: list[dict], tools: list, model=None, max_rounds: int = 6,
    required_tool: str | None = None,
) -> dict:
    """运行一次 ReAct 风格的“模型 → 工具 → 模型”循环。

    max_rounds 是保险丝，防止模型反复调用工具形成无限循环。
    """
    llm = model or deepseek_chat_model()
    model_with_tools = llm.bind_tools(tools)
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    messages = [SystemMessage(content=SYSTEM_PROMPT.format(today=today))]
    for item in history:
        message_type = HumanMessage if item["role"] == "user" else AIMessage
        messages.append(message_type(content=item["content"]))
        # 数据库中的来源元数据比从自然语言中重新猜 ID 更可靠，供下一轮直接复用。
        cited = item.get("metadata", {}).get("cited_note_ids", [])
        if item["role"] == "assistant" and cited:
            messages.append(SystemMessage(
                content=f"上一轮已引用且可继续使用的笔记 ID：{', '.join(cited)}"
            ))

    tool_map = {tool.name: tool for tool in tools}
    traces, retrieved_note_ids, viewed_note_ids, plan_id = [], [], [], None
    attempted_tools: set[str] = set()
    force_required_next = False
    for _ in range(max_rounds):
        # 明确要求计划时，如果模型企图直接输出文字，下一轮强制它走 draft_plan。
        # 秋招知识点：业务不变量必须由编排层保证，不能只依赖 Prompt 的软约束。
        round_model = (llm.bind_tools(tools, tool_choice=required_tool)
                       if force_required_next and required_tool else model_with_tools)
        response = await round_model.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            if required_tool and required_tool not in attempted_tools:
                messages.append(HumanMessage(content=(
                    f"用户明确要求生成可保存的计划草案。不要直接输出文字计划；"
                    f"现在必须调用 {required_tool}，并使用已检索或上一轮引用的笔记 ID。"
                )))
                force_required_next = True
                continue
            content = response.content if isinstance(response.content, str) else str(response.content)
            # cited 只记录模型最终回答中出现过，或通过 get_note 明确展开过的来源。
            cited = list(viewed_note_ids)
            cited.extend(note_id for note_id in retrieved_note_ids
                         if note_id in content or note_id[:8] in content)
            cited = list(dict.fromkeys(cited))
            return {
                "content": content,
                "tool_calls": traces,
                "retrieved_note_ids": list(dict.fromkeys(retrieved_note_ids)),
                "cited_note_ids": cited,
                # 兼容旧客户端；新代码应优先读取 cited_note_ids。
                "source_note_ids": cited,
                "plan_id": plan_id,
            }

        for call in response.tool_calls:
            attempted_tools.add(call["name"])
            force_required_next = False
            tool = tool_map.get(call["name"])
            started = perf_counter()
            if tool is None:
                result = {"ok": False, "data": None,
                          "error": {"code": "unknown_tool", "message": "工具不存在"}}
            else:
                try:
                    result = await tool.ainvoke(call["args"])
                except Exception as exc:  # 工具失败应回传给模型，让它解释或换策略。
                    result = {"ok": False, "data": None, "error": safe_tool_error(exc)}
            data = result.get("data")
            error = result.get("error")
            scores = ([item.get("score") for item in data
                       if isinstance(item.get("score"), (int, float))]
                      if isinstance(data, list) else [])
            trace = {
                "name": call["name"], "args": call["args"], "ok": result.get("ok", False),
                "duration_ms": round((perf_counter() - started) * 1000, 1),
                "result_count": len(data) if isinstance(data, list) else None,
                "top_score": max(scores) if scores else None,
                "error_code": (error.get("code") if isinstance(error, dict)
                               else "validation_error" if error else None),
            }
            traces.append(trace)
            if call["name"] == "search_notes" and result.get("ok"):
                retrieved_note_ids.extend(item["note_id"] for item in result["data"] if item.get("note_id"))
            if call["name"] == "get_note" and result.get("ok") and isinstance(data, dict):
                if data.get("note_id"):
                    viewed_note_ids.append(data["note_id"])
            if call["name"] == "draft_plan" and result.get("ok"):
                data = result.get("data") or {}
                plan = data.get("plan", data)
                plan_id = plan.get("id") if isinstance(plan, dict) else None
            messages.append(ToolMessage(
                content=json.dumps(result, ensure_ascii=False, default=str),
                tool_call_id=call["id"],
            ))
    raise RuntimeError("Agent 工具调用次数过多，请缩小问题范围后重试。")
