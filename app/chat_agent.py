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
需要站外小红书内容时使用 search_feeds；用户直接粘贴小红书链接或分享文案时，
用 collect_note_by_link 按链接采集原文。这些站外工具可能不可用，调用失败就说明
仅基于知识库回答，不要反复重试。
小红书很多笔记把正文写在长图里：collect_note_by_link 已内置视觉识别，返回值里的
image_text 就是图片中的文字。只要 image_text 非空，就必须以它为准来回答，
不要再声称自己无法识别图片；只有 image_text 为空时，才可以说内容看不清。
采集结果若带 warnings，必须如实转达，绝不能凭空编造图片里的内容。
用户明确要求收藏、保存这条链接时改用 save_note_by_link（它会写入知识库、把图片存到
本地，并自动建立语义索引），并如实汇报返回值里的 status、图片落盘数量与 index：
status 是 existing 就说明之前已经收藏过，不要说成刚收藏成功。index.searchable 为 true
表示这篇笔记现在就能被 search_notes 检索到；index.status 是 skipped 时说明它没有可检索
的正文，应告诉用户这次没建索引，不要谎称可以搜到。同一篇笔记不要两个工具都调。
制定计划前应检索相关笔记并读取用户记忆，之后只能生成 draft 草案。
目标还不清楚时（不知道想解决什么、从哪天开始、每天能投入多久），先用一两句话
问清楚再起草，不要凭空替用户定目标；追问的那条回答要以问号结尾。
起草之后必须说明"这是草案，需要你确认才会写入日历"，绝不能替用户确认或创建任务。
回答使用简洁中文，并说明参考了哪些笔记。
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


QUESTION_TAIL = re.compile(r"[？?]\s*$")


def asks_clarification(content: str) -> bool:
    """模型这句话是不是在向用户追问（以问号结尾）。

    用户说"帮我排个计划"却没给目标，此时强行逼它调 draft_plan，只会得到一份
    凭空定目标的草案。允许它以追问收尾，把"想清楚再落库"这一步留给用户。
    """
    return bool(QUESTION_TAIL.search((content or "").strip()))


def asked_question_last(history: list[dict]) -> bool:
    """上一轮助手是不是在追问。

    追问之后用户回的那句往往没有"计划"二字（例如"每天两小时，从周五开始"），
    只看最后一句会把 draft_plan 工具收回，计划就只能在文字里空谈、落不了库。
    """
    for item in reversed(history):
        if item.get("role") != "assistant":
            continue
        return asks_clarification(item.get("content") or "")
    return False


def plan_thread_pending(history: list[dict], window: int = 6) -> bool:
    """最近几轮里出现过明确的计划请求——会话还在"商量计划"的状态中。"""
    user_turns = [item.get("content") or "" for item in history
                  if item.get("role") == "user"][-window:]
    return any(plan_requested(text) for text in user_turns)


def safe_tool_error(exc: Exception) -> dict:
    """保留可诊断错误码，但不把异常堆栈、URL 或密钥交给模型。"""
    status_code = getattr(exc, "status_code", None)
    detail = getattr(exc, "detail", None)
    if isinstance(status_code, int):
        return {"code": f"http_{status_code}", "message": str(detail or "工具请求失败")}
    return {"code": "tool_exception", "message": "工具执行失败，请稍后重试。"}


def normalize_tool_result(result) -> dict:
    """把任意工具的返回值统一成 {ok, data, error} 契约。

    秋招知识点（混合工具来源的契约收敛）：本地工具返回的是自研 dict，
    而 MCP 工具经 langchain-mcp-adapters 包装后返回的是 MCP 的 content
    blocks 列表 —— [{ "type": "text", "text": "..." }]。编排层若不抹平这个
    差异，下游一句 result.get("data") 就会直接崩，而且是在真实对话里才暴露。
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, (list, tuple)):
        parts = [str(block.get("text") or "") for block in result
                 if isinstance(block, dict) and block.get("type") == "text"]
        text = "".join(parts)
        # 文本块能拼出内容就交文本；拼不出则原样透传，至少不丢数据。
        return {"ok": True, "data": text or list(result), "error": None}
    return {"ok": True, "data": result, "error": None}


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


def seed_messages(history: list[dict]) -> list:
    """把库里的历史消息还原成模型消息，并补上“上一轮引用过的笔记 ID”。

    数据库中的来源元数据比从自然语言里重新猜 ID 更可靠，所以引用的笔记直接作为
    系统消息注入，供下一轮复用。流式与非流式两条链路共用这段铺垫，避免逻辑漂移。
    """
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    messages = [SystemMessage(content=SYSTEM_PROMPT.format(today=today))]
    for item in history:
        message_type = HumanMessage if item["role"] == "user" else AIMessage
        messages.append(message_type(content=item["content"]))
        cited = item.get("metadata", {}).get("cited_note_ids", [])
        if item["role"] == "assistant" and cited:
            messages.append(SystemMessage(
                content=f"上一轮已引用且可继续使用的笔记 ID：{', '.join(cited)}"
            ))
    return messages


def cited_ids_for(content: str, viewed_note_ids: list, retrieved_note_ids: list) -> list:
    """引用来源只认“模型真的用过”的：展开过全文的，或回答里点名 ID 的。"""
    cited = list(viewed_note_ids)
    cited.extend(note_id for note_id in retrieved_note_ids
                 if note_id in content or note_id[:8] in content)
    return list(dict.fromkeys(cited))


HIT_LIMIT = 8            # 单轮最多留存多少条命中原片
HIT_SNIPPET_CHARS = 240  # 单条片段字符上限


def build_hits(items, limit: int = HIT_LIMIT) -> list[dict]:
    """把一次 search_notes 的命中压成可展示的原始片段。

    「检索」视图并进对话后，前端要在回答下方按需展开"这一轮到底命中了什么"。
    真到了展开那一刻再去重新检索，既慢又白付一次 embedding 费用，所以在编排层
    顺手留存影本。这里只留展示必需的字段，正文截断，避免整轮结果灌进消息表。
    """
    hits = []
    for item in (items if isinstance(items, list) else []):
        if not isinstance(item, dict) or not item.get("note_id"):
            continue
        hits.append({
            "note_id": item["note_id"],
            "title": str(item.get("title") or ""),
            "score": item.get("score"),
            "snippet": " ".join(str(item.get("content") or "").split())[:HIT_SNIPPET_CHARS],
        })
        if len(hits) == limit:
            break
    return hits


def build_trace(call: dict, result: dict, duration_ms: float) -> dict:
    """把一次工具调用压成可展示、可回归的轨迹记录（不含工具返回的正文）。"""
    data = result.get("data")
    error = result.get("error")
    scores = ([item.get("score") for item in data
               if isinstance(item.get("score"), (int, float))]
              if isinstance(data, list) else [])
    return {
        "name": call["name"], "args": call["args"], "ok": result.get("ok", False),
        "duration_ms": duration_ms,
        "result_count": len(data) if isinstance(data, list) else None,
        "top_score": max(scores) if scores else None,
        "error_code": (error.get("code") if isinstance(error, dict)
                       else "validation_error" if error else None),
    }


async def run_agent_turn(
    history: list[dict], tools: list, model=None, max_rounds: int = 6,
    required_tool: str | None = None,
) -> dict:
    """运行一次 ReAct 风格的“模型 → 工具 → 模型”循环。

    max_rounds 是保险丝，防止模型反复调用工具形成无限循环。
    """
    llm = model or deepseek_chat_model()
    model_with_tools = llm.bind_tools(tools)
    messages = seed_messages(history)

    tool_map = {tool.name: tool for tool in tools}
    traces, retrieved_note_ids, viewed_note_ids, plan_id = [], [], [], None
    retrieved_hits: list[dict] = []
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
            content = response.content if isinstance(response.content, str) else str(response.content)
            # 以追问收尾不算"不听话"：目标都没问清就把草稿逼出来，只会得到一份
            # 凭空定目标的计划，用户还得推翻重来。
            if (required_tool and required_tool not in attempted_tools
                    and not asks_clarification(content)):
                messages.append(HumanMessage(content=(
                    f"用户明确要求生成可保存的计划草案。不要直接输出文字计划；"
                    f"现在必须调用 {required_tool}，并使用已检索或上一轮引用的笔记 ID。"
                    f"还缺必要信息就先用一句问句问清楚，不要凭空假设。"
                )))
                force_required_next = True
                continue
            # cited 只记录模型最终回答中出现过，或通过 get_note 明确展开过的来源。
            cited = cited_ids_for(content, viewed_note_ids, retrieved_note_ids)
            return {
                "content": content,
                "tool_calls": traces,
                "retrieved_note_ids": list(dict.fromkeys(retrieved_note_ids)),
                "retrieved_hits": retrieved_hits,
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
                    result = normalize_tool_result(await tool.ainvoke(call["args"]))
                except Exception as exc:  # 工具失败应回传给模型，让它解释或换策略。
                    result = {"ok": False, "data": None, "error": safe_tool_error(exc)}
            data = result.get("data")
            trace = build_trace(call, result, round((perf_counter() - started) * 1000, 1))
            traces.append(trace)
            if call["name"] == "search_notes" and result.get("ok"):
                retrieved_note_ids.extend(item["note_id"] for item in result["data"] if item.get("note_id"))
                # 只留第一批命中的片段：多轮检索时越靠前的越接近用户原问。
                if not retrieved_hits:
                    retrieved_hits = build_hits(result["data"])
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


async def run_agent_turn_stream(
    history: list[dict], tools: list, model=None, max_rounds: int = 6,
    required_tool: str | None = None,
):
    """与 run_agent_turn 同一套策略，但把过程作为事件流吐给前端。

    一次性版本要等模型全部说完才返回，界面上只能干等；流式版把“检索 → 读原文 →
    逐字作答”的过程暴露出来，才有看得见过程的对话体验。事件类型：

    - delta      文本增量（逐字追加到当前气泡）
    - reset      丢弃本段草稿：模型该调工具却想直接作答，编排层要求它重来
    - tool_start 工具开始执行（名字 + 参数，可展示成一行步骤）
    - tool_end   工具执行结束（耗时 / 命中条数 / 错误码）
    - done       本轮收尾（附引用笔记、计划 ID 等元数据，前端据此处落库结果）
    - error      失败（前端据此提示，不必再等回复）

    工具失败照样回传给模型、仍由 max_rounds 兜住死循环——安全策略与一次性版本一致。
    """
    llm = model or deepseek_chat_model()
    model_with_tools = llm.bind_tools(tools)
    messages = seed_messages(history)

    tool_map = {tool.name: tool for tool in tools}
    traces, retrieved_note_ids, viewed_note_ids, plan_id = [], [], [], None
    retrieved_hits: list[dict] = []
    attempted_tools: set[str] = set()
    force_required_next = False
    for _ in range(max_rounds):
        round_model = (llm.bind_tools(tools, tool_choice=required_tool)
                       if force_required_next and required_tool else model_with_tools)
        gathered = None
        # AIMessageChunk 支持累加：只有拼回完整消息才能进下一轮上下文，
        # 否则模型会看不见自己刚才发起的 tool_calls。
        async for chunk in round_model.astream(messages):
            gathered = chunk if gathered is None else gathered + chunk
            piece = chunk.content if isinstance(chunk.content, str) else ""
            if piece:
                yield {"type": "delta", "text": piece}
        messages.append(gathered)

        if not gathered.tool_calls:
            content = gathered.content if isinstance(gathered.content, str) else str(gathered.content)
            # 以追问收尾不算"不听话"：保留这段文字让用户回答，不逼它凭空起草。
            if (required_tool and required_tool not in attempted_tools
                    and not asks_clarification(content)):
                messages.append(HumanMessage(content=(
                    f"用户明确要求生成可保存的计划草案。不要直接输出文字计划；"
                    f"现在必须调用 {required_tool}，并使用已检索或上一轮引用的笔记 ID。"
                    f"还缺必要信息就先用一句问句问清楚，不要凭空假设。"
                )))
                force_required_next = True
                # 刚吐出去的只是模型的“口头答应”，不是答案，通知前端丢弃这段草稿。
                yield {"type": "reset"}
                continue
            yield {
                "type": "done",
                "content": content,
                "tool_calls": traces,
                "retrieved_note_ids": list(dict.fromkeys(retrieved_note_ids)),
                "retrieved_hits": retrieved_hits,
                "cited_note_ids": cited_ids_for(content, viewed_note_ids, retrieved_note_ids),
                "plan_id": plan_id,
            }
            return

        for call in gathered.tool_calls:
            attempted_tools.add(call["name"])
            force_required_next = False
            tool = tool_map.get(call["name"])
            yield {"type": "tool_start", "name": call["name"], "args": call["args"]}
            started = perf_counter()
            if tool is None:
                result = {"ok": False, "data": None,
                          "error": {"code": "unknown_tool", "message": "工具不存在"}}
            else:
                try:
                    result = normalize_tool_result(await tool.ainvoke(call["args"]))
                except Exception as exc:
                    result = {"ok": False, "data": None, "error": safe_tool_error(exc)}
            data = result.get("data")
            trace = build_trace(call, result, round((perf_counter() - started) * 1000, 1))
            traces.append(trace)
            if call["name"] == "search_notes" and result.get("ok"):
                retrieved_note_ids.extend(item["note_id"] for item in result["data"] if item.get("note_id"))
                # 只留第一批命中的片段：多轮检索时越靠前的越接近用户原问。
                if not retrieved_hits:
                    retrieved_hits = build_hits(result["data"])
            if call["name"] == "get_note" and result.get("ok") and isinstance(data, dict):
                if data.get("note_id"):
                    viewed_note_ids.append(data["note_id"])
            if call["name"] == "draft_plan" and result.get("ok"):
                data = result.get("data") or {}
                plan = data.get("plan", data)
                plan_id = plan.get("id") if isinstance(plan, dict) else None
            yield {"type": "tool_end", **trace}
            messages.append(ToolMessage(
                content=json.dumps(result, ensure_ascii=False, default=str),
                tool_call_id=call["id"],
            ))
    yield {"type": "error",
           "message": "Agent 工具调用次数过多，请缩小问题范围后重试。"}
