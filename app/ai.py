"""DeepSeek 视觉模型调用：把一篇笔记的文字和本地图片组合成一次请求。"""
import base64
import json
import logging
import os
from pathlib import Path

import httpx

from .media import local_file_valid, storage_path


class AIProblem(Exception):
    """可预期的模型调用错误，接口层会把它转换成适合用户阅读的提示。"""


logger = logging.getLogger("xhs.ai")


def _message_text(content) -> str:
    """兼容 OpenAI 兼容接口返回的纯字符串或分段文本数组。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            item["text"] for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "\n".join(parts).strip()
    return ""


def _extract_json_object(response_data: dict, operation: str) -> dict:
    """从模型响应提取 JSON 对象，并把截断与格式错误区分开。"""
    try:
        choice = response_data["choices"][0]
        message = choice["message"]
        content = message.get("content")
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("ai_response_shape_invalid operation=%s", operation)
        raise AIProblem("DeepSeek 响应缺少预期字段，请稍后重试。") from exc

    # 个别兼容服务会直接给结构化对象；标准 Chat Completions 通常给字符串。
    if isinstance(content, dict):
        result = content
    else:
        raw_text = _message_text(content)
        if not raw_text:
            logger.warning("ai_response_content_empty operation=%s", operation)
            raise AIProblem("DeepSeek 返回了空内容，请稍后重试。")

        # 兼容模型偶尔包上的 ```json 代码围栏。
        if raw_text.startswith("```"):
            raw_text = raw_text.removeprefix("```json").removeprefix("```JSON")
            raw_text = raw_text.removesuffix("```").strip()

        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError:
            # 兼容 JSON 前后带简短说明的情况，只解析第一个完整 JSON 对象。
            start = raw_text.find("{")
            if start < 0:
                result = None
            else:
                try:
                    result, _end = json.JSONDecoder().raw_decode(raw_text[start:])
                except json.JSONDecodeError:
                    result = None

        if result is None:
            logger.warning(
                "ai_response_json_invalid operation=%s finish_reason=%s",
                operation, finish_reason,
            )
            if finish_reason in {"length", "max_tokens"}:
                raise AIProblem("DeepSeek 输出达到长度上限，内容被截断；请重试或减少图片数量。")
            raise AIProblem("DeepSeek 响应不是有效 JSON，请重试；若重复发生请减少图片数量。")

    if not isinstance(result, dict):
        logger.warning("ai_response_not_object operation=%s", operation)
        raise AIProblem("DeepSeek 返回的内容不是 JSON 对象，请重试。")
    return result


def _image_data_url(path: Path, mime_type: str) -> str:
    # API 不能访问我们电脑上的 127.0.0.1，所以读取本地文件并编码为 Base64。
    # Base64 会让二进制图片变成可放进 JSON 的字符串，不会修改原图片。
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _prompt(note: dict) -> str:
    # 明确限定 JSON 字段，方便后端稳定保存，手机端也不必解析一大段自由文本。
    return f"""你正在分析一篇用户收藏的小红书笔记。
请结合正文和所有图片内容，严格输出一个 JSON 对象，不要输出 Markdown 代码块。

笔记标题：{note.get('title', '')}
笔记正文：{note.get('content', '')}

JSON 必须包含以下字段：
summary: 中文摘要字符串；
category: 从 健身、学习、做饭、工作、旅行、购物、其他 中选一个；
tags: 3到8个中文标签组成的数组；
knowledge_points: 重要知识点字符串数组；
suggested_actions: 可实际执行的行动字符串数组；
inferred_intent: 推测用户为什么收藏，使用字符串表达；
intent_confidence: 0到1之间的数字；
image_observations: 按图片序号说明图片贡献的数组，每项包含 image_number 和 observation；
uncertainties: 无法确认或需要用户核实的信息数组。

要求：不要把推测写成事实；图片中看不清的内容要放进 uncertainties；不要执行提醒，只提出建议。
为避免响应过长，每张图片的 observation 限制为一句话，knowledge_points 和 suggested_actions 各最多 8 条。"""


async def analyze_note(note: dict, assets: list[dict], media_root: Path) -> tuple[dict, str]:
    """调用 DeepSeek，并返回（结构化分析结果，实际模型名）。"""
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise AIProblem("尚未配置 DEEPSEEK_API_KEY。")

    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash-vision-exp").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")

    # content 是一个有顺序的列表：先放说明文字，再按小红书原顺序加入图片。
    content: list[dict] = [{"type": "text", "text": _prompt(note)}]
    valid_count = 0
    for asset in assets:
        if not local_file_valid(media_root, asset):
            continue
        valid_count += 1
        path = storage_path(media_root, asset["storage_key"])
        content.append({
            "type": "text",
            "text": f"下面是第 {asset['position']} 张图片：",
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": _image_data_url(path, asset["mime_type"]),
                # original 有利于读取小红书长图中的小字。
                "detail": "original",
            },
        })

    if valid_count == 0:
        raise AIProblem("这篇笔记还没有可用的本地图片，请先执行图片下载。")

    request_body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_object"},
        # 多图笔记的结构化结果容易较长，适当提高上限并在提示中限制字段长度。
        "max_tokens": 4000,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        # 图片分析可能比普通文本慢，因此这里使用 120 秒总超时。
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{base_url}/chat/completions", headers=headers, json=request_body)
    except httpx.TimeoutException as exc:
        raise AIProblem("DeepSeek 分析超时，请稍后重试。") from exc
    except httpx.HTTPError as exc:
        raise AIProblem("无法连接 DeepSeek，请检查网络后重试。") from exc

    if response.status_code == 401:
        raise AIProblem("DeepSeek 拒绝了密钥，请检查 API Key 是否正确。")
    if response.status_code == 402:
        raise AIProblem("DeepSeek 账户余额不足。")
    if response.status_code >= 400:
        # 不把上游完整响应返回客户端，其中可能包含请求细节。
        raise AIProblem(f"DeepSeek 调用失败（状态码 {response.status_code}）。")

    try:
        response_data = response.json()
    except ValueError as exc:
        logger.warning("ai_response_not_json operation=analysis")
        raise AIProblem("DeepSeek 返回了无法识别的响应，请稍后重试。") from exc
    result = _extract_json_object(response_data, "analysis")
    return result, model


def _folder_prompt(note: dict, analysis: dict, confirmation: dict | None,
                   folders: list[dict]) -> str:
    """构造收藏夹分类提示词；只提供启用中的候选收藏夹。"""
    folder_candidates = [
        {"folder_id": item["id"], "name": item["name"], "description": item["description"]}
        for item in folders
    ]
    intent = confirmation or {
        "category": analysis.get("category", ""),
        "intent": analysis.get("inferred_intent", ""),
        "decision": "model_inferred",
    }
    return f"""你正在帮助用户整理一篇已完成分析的收藏笔记。
只能从候选收藏夹中推荐 1 到 3 个，不得虚构 folder_id，不得创建收藏夹。
同一篇笔记可以同时属于“内容主题”和“使用意图”两个收藏夹。
严格输出 JSON 对象，不要输出 Markdown 代码块。

笔记标题：{note.get('title', '')}
笔记正文：{note.get('content', '')}
多模态分析：{json.dumps(analysis, ensure_ascii=False)}
收藏意图：{json.dumps(intent, ensure_ascii=False)}
候选收藏夹：{json.dumps(folder_candidates, ensure_ascii=False)}

JSON 格式：
suggestions: 数组，每项包含 folder_id（整数）、confidence（0到1）、reason（中文理由）；
uncertainties: 无法确定之处的字符串数组。
"""


async def suggest_folders(note: dict, analysis: dict, confirmation: dict | None,
                          folders: list[dict]) -> tuple[dict, str]:
    """调用 DeepSeek 生成结构化收藏夹建议，不执行真正归档。"""
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise AIProblem("尚未配置 DEEPSEEK_API_KEY。")
    if not folders:
        raise AIProblem("还没有可用收藏夹，请先创建收藏夹。")

    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash-vision-exp").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    request_body = {
        "model": model,
        "messages": [{"role": "user", "content": _folder_prompt(
            note, analysis, confirmation, folders
        )}],
        "response_format": {"type": "json_object"},
        "max_tokens": 1000,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions", headers=headers, json=request_body
            )
    except httpx.TimeoutException as exc:
        raise AIProblem("DeepSeek 推荐收藏夹超时，请稍后重试。") from exc
    except httpx.HTTPError as exc:
        raise AIProblem("无法连接 DeepSeek，请检查网络后重试。") from exc

    if response.status_code == 401:
        raise AIProblem("DeepSeek 拒绝了密钥，请检查 API Key 是否正确。")
    if response.status_code == 402:
        raise AIProblem("DeepSeek 账户余额不足。")
    if response.status_code >= 400:
        raise AIProblem(f"DeepSeek 调用失败（状态码 {response.status_code}）。")

    try:
        raw_content = response.json()["choices"][0]["message"]["content"]
        result = json.loads(raw_content)
        suggestions = result["suggestions"]
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AIProblem("DeepSeek 返回的收藏夹建议格式不正确，请重试。") from exc

    allowed_ids = {item["id"] for item in folders}
    seen_ids = set()
    if not isinstance(suggestions, list) or not 1 <= len(suggestions) <= 3:
        raise AIProblem("DeepSeek 必须推荐 1 到 3 个收藏夹，请重试。")
    for item in suggestions:
        folder_id = item.get("folder_id") if isinstance(item, dict) else None
        confidence = item.get("confidence") if isinstance(item, dict) else None
        reason = item.get("reason") if isinstance(item, dict) else None
        valid_confidence = (
            isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and 0 <= confidence <= 1
        )
        if (folder_id not in allowed_ids or folder_id in seen_ids
                or not valid_confidence or not isinstance(reason, str) or not reason.strip()):
            raise AIProblem("DeepSeek 返回了无效或重复的收藏夹建议，请重试。")
        seen_ids.add(folder_id)
        item["reason"] = reason.strip()
        item["confidence"] = float(confidence)
    uncertainties = result.get("uncertainties", [])
    if not isinstance(uncertainties, list) or not all(isinstance(x, str) for x in uncertainties):
        raise AIProblem("DeepSeek 返回的 uncertainties 格式不正确，请重试。")
    return {"suggestions": suggestions, "uncertainties": uncertainties}, model


async def draft_plan(goal: str, notes: list[dict], intents: list[dict], start_date: str) -> tuple[dict, str]:
    """根据目标和笔记生成计划步骤；这里只生成草案，不创建任务。"""
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise AIProblem("尚未配置 DEEPSEEK_API_KEY。")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash-vision-exp").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    prompt = f"""请为用户制定一个学习计划，严格只输出 JSON，不要 Markdown。
目标：{goal}
开始日期：{start_date}
参考笔记：{json.dumps([{'note_id': n['note_id'], 'title': n.get('title',''), 'content': n.get('content','')[:2000]} for n in notes], ensure_ascii=False)}
已确认意图：{json.dumps(intents, ensure_ascii=False)}
要求：生成 1 到 14 个按日期递进的步骤；每步 title 不超过 30 字，details 不超过 80 字；
scheduled_at 使用 ISO 8601（带 +08:00 时区），reminder_at 可为 null。
每步必须增加 source_note_ids 数组，只能填写参考笔记中的 note_id，可填写多个。
只输出 JSON，不要解释、不要复述笔记正文。
JSON 格式：title、goal、items 数组。每个 item 只能包含上述字段。"""
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "response_format": {"type": "json_object"}, "max_tokens": 3000, "stream": False},
            )
    except httpx.TimeoutException as exc:
        raise AIProblem("DeepSeek 生成计划超时，请稍后重试。") from exc
    except httpx.HTTPError as exc:
        raise AIProblem("无法连接 DeepSeek，请检查网络后重试。") from exc
    if response.status_code >= 400:
        raise AIProblem(f"DeepSeek 调用失败（状态码 {response.status_code}）。")
    try:
        response_data = response.json()
    except ValueError as exc:
        raise AIProblem("DeepSeek 返回了无法识别的响应，请稍后重试。") from exc
    result = _extract_json_object(response_data, "plan")
    items = result.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= 14:
        raise AIProblem("DeepSeek 计划步骤数量不正确，请重试。")
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("title"), str) \
                or not item["title"].strip() or not isinstance(item.get("scheduled_at"), str):
            raise AIProblem("DeepSeek 计划步骤格式不正确，请重试。")
        item.setdefault("details", "")
        item.setdefault("reminder_at", None)
    return {"title": result.get("title", goal), "goal": result.get("goal", goal), "items": items}, model
