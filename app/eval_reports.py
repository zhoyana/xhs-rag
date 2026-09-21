"""评估看板的数据层：只读扫描评测报告 + 聚合工具调用痕迹。

本模块不写库、不加载 Embedding、不调用模型，只做两件事：把
`data/eval_reports/*.json` 归一成同一种形状，把 chat_messages 里的
tool_calls 聚合成次数与分位数。两件事都能离线单测，不需要先跑一遍真实评测。

为什么报告目录跟数据库同目录（`Path(db_path).parent / "eval_reports"`）：
与 `media_root` 同一约定。这样"换个库就看另一批报告"是自然行为，不需要额外
配置项；本地库与线上库各带各的报告，也不会出现"看板显示的报告其实来自别的库"。
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

# 报告文件名以 eval_kind 字段为准做分派，不靠文件名猜：两类报告的指标完全不同
# （单跳是 hit_at_k/mrr，多跳是 coverage/all_hit/hop_recall），按名字硬编码
# 迟早会在新增一类评测时静默算错。
KIND_SINGLE = "single_hop"
KIND_MULTIHOP = "multihop"
KIND_UNREADABLE = "unreadable"

REPORT_SUFFIX = ".json"
REPORT_DIRNAME = "eval_reports"

# 只暴露给界面的指标键：摘要里其余字段（模型名、候选数…）单独列。
# 白名单而不是黑名单：报告文件是外部产物，不能让它决定响应里出现什么键。
SINGLE_METRICS = ("hit_at_1", "hit_at_3", "hit_at_5", "mrr")
MULTIHOP_METRICS = (
    "coverage_at_1", "all_hit_at_1", "hop_recall_at_1", "all_hops_recall_at_1",
    "coverage_at_3", "all_hit_at_3", "hop_recall_at_3", "all_hops_recall_at_3",
    "coverage_at_5", "all_hit_at_5", "hop_recall_at_5", "all_hops_recall_at_5",
)


def report_dir(db_path) -> Path:
    """报告目录 = 数据库所在目录下的 eval_reports。"""
    return Path(db_path).parent / REPORT_DIRNAME


def classify(summary: dict) -> str:
    """按 summary 里的特征字段判定报告类型；判不出来就说判不出来。"""
    if summary.get("eval_kind") == KIND_MULTIHOP:
        return KIND_MULTIHOP
    if any(key in summary for key in SINGLE_METRICS):
        return KIND_SINGLE
    return KIND_UNREADABLE


def _number(value):
    """只让有限数值过关：NaN / Infinity 进了 JSON 会让前端 JSON.parse 直接炸。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def summarize_report(name: str, raw: dict, stat: dict) -> dict:
    """把一份报告压成列表页需要的摘要；缺的字段一律 None，不编造默认值。"""
    summary = raw.get("summary") or {}
    kind = classify(summary)
    metrics_pool = MULTIHOP_METRICS if kind == KIND_MULTIHOP else SINGLE_METRICS
    metrics = {key: _number(summary.get(key)) for key in metrics_pool
               if _number(summary.get(key)) is not None}
    cases = raw.get("cases")
    return {
        "file": name,
        "size_bytes": stat.get("size_bytes"),
        "modified_at": stat.get("modified_at"),
        "kind": kind,
        "model": summary.get("model"),
        "retrieval_mode": summary.get("retrieval_mode"),
        "deduplicate_notes": summary.get("deduplicate_notes"),
        "case_count": summary.get("case_count") if isinstance(summary.get("case_count"), int)
        else (len(cases) if isinstance(cases, list) else None),
        "answerable_count": summary.get("answerable_count"),
        "no_answer_count": summary.get("no_answer_count"),
        "chunk_candidate_count": summary.get("chunk_candidate_count"),
        "average_latency_ms": _number(summary.get("average_latency_ms")),
        "average_hop_latency_ms": _number(summary.get("average_hop_latency_ms")),
        "average_duplicate_slots_top5": _number(summary.get("average_duplicate_slots_top5")),
        "metrics": metrics,
        # 类型分布、无答案题最高分这类结构化成因，原样透传给界面做注解。
        "type_counts": summary.get("type_counts"),
        "no_answer_max_scores": [_number(v) for v in (summary.get("no_answer_max_scores") or [])],
        "missing_hop_vector_scores": summary.get("missing_hop_vector_scores") or [],
    }


def list_reports(directory) -> list[dict]:
    """扫描报告目录，按修改时间倒序返回摘要。

    单个文件读不出来（写了一半、手工编辑坏了、不是 JSON）**不影响其它报告**：
    它以 kind=unreadable + error 出现在列表里。一份坏文件让整个看板打不开，
    是最容易在真实使用中踩到的坑——报告本来就是手工产出的文件。
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    items = []
    for path in sorted(directory.glob(f"*{REPORT_SUFFIX}")):
        if not path.is_file():
            continue
        info = path.stat()
        stat = {
            "size_bytes": info.st_size,
            "modified_at": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
        }
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            items.append({
                "file": path.name, **stat,
                "kind": KIND_UNREADABLE, "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        if not isinstance(raw, dict):
            items.append({
                "file": path.name, **stat,
                "kind": KIND_UNREADABLE, "error": "报告顶层不是 JSON 对象",
            })
            continue
        items.append(summarize_report(path.name, raw, stat))
    items.sort(key=lambda item: item.get("modified_at") or "", reverse=True)
    return items


def resolve_report(directory, name: str) -> Path:
    """把用户给的文件名解析成报告目录内的真实路径，越界一律拒绝。

    三层校验缺一不可：
    1. `PurePath(name).name == name` —— 挡住 `../x.json`、`C:/x.json`、`a/b.json`；
    2. 必须 `.json` 结尾 —— 顺带挡住目录名与无后缀文件；
    3. resolve 之后的父目录必须仍是报告目录 —— 挡住符号链接指到目录外面。
    只看文件名白名单（第 1 层）在"目录里恰好有个软链接"时会漏，所以第 3 层不是多余的。
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("报告文件名不能为空。")
    candidate = name.strip()
    if Path(candidate).name != candidate or not candidate.endswith(REPORT_SUFFIX):
        raise ValueError("报告文件名不合法。")
    directory = Path(directory).resolve()
    path = (directory / candidate).resolve()
    if path.parent != directory:
        raise ValueError("报告文件名不合法。")
    if not path.is_file():
        raise FileNotFoundError(candidate)
    return path


def load_report(directory, name: str) -> dict:
    """读取单份报告的完整内容（摘要 + 逐题），损坏时抛 ValueError 由接口层翻译。"""
    path = resolve_report(directory, name)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"报告不是合法 JSON：{exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("报告顶层不是 JSON 对象。")
    info = path.stat()
    cases = raw.get("cases") if isinstance(raw.get("cases"), list) else []
    return {
        "summary": summarize_report(path.name, raw, {
            "size_bytes": info.st_size,
            "modified_at": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
        }),
        "cases": [simplify_case(case) for case in cases if isinstance(case, dict)],
    }


def simplify_case(case: dict) -> dict:
    """逐题明细只留界面要用的字段。

    `top_results` 原样带过来会塞进 512 维向量（第一版报告里真的有），
    一页 10 题就是几 MB 的响应。这里只取排在最前的几条片段所需字段。
    """
    tops = []
    for hit in (case.get("top_results") or [])[:5]:
        if not isinstance(hit, dict):
            continue
        tops.append({
            "note_id": hit.get("note_id"),
            "title": hit.get("title"),
            "chunk_index": hit.get("chunk_index"),
            "score": _number(hit.get("score")),
            "content": (hit.get("content") or "")[:240],
        })
    item = {
        "id": case.get("id"),
        "query": case.get("query"),
        "difficulty": case.get("difficulty"),
        "type": case.get("type"),
        "relevant_note_ids": case.get("relevant_note_ids") or [],
        "latency_ms": _number(case.get("latency_ms")),
        "top_results": tops,
        "duplicate_slots_top5": case.get("duplicate_slots_top5"),
    }
    for key in ("hit_at_1", "hit_at_3", "hit_at_5", "reciprocal_rank",
                "coverage_at_1", "all_hit_at_1", "hop_hit_at_1",
                "coverage_at_3", "all_hit_at_3", "hop_hit_at_3",
                "coverage_at_5", "all_hit_at_5", "hop_hit_at_5",
                "no_answer_max_score"):
        if case.get(key) is not None:
            item[key] = case[key] if isinstance(case[key], bool) else _number(case[key])
    # 多跳题的每一跳：查了什么、命中哪些笔记、各自的 Top1。
    if case.get("hop_queries"):
        hops = []
        hop_tops = case.get("hop_top_results") or []
        hop_latency = case.get("hop_latency_ms") or []
        for index, query in enumerate(case["hop_queries"]):
            hop_top = hop_tops[index] if index < len(hop_tops) else []
            first = hop_top[0] if isinstance(hop_top, list) and hop_top else {}
            hops.append({
                "query": query,
                "latency_ms": _number(hop_latency[index]) if index < len(hop_latency) else None,
                "top_note_id": first.get("note_id") if isinstance(first, dict) else None,
                "top_title": first.get("title") if isinstance(first, dict) else None,
                "top_score": _number(first.get("score")) if isinstance(first, dict) else None,
            })
        item["hops"] = hops
    return item


def attach_reasons(report: dict, notes: dict) -> dict:
    """把人工标注的 reason 贴到逐题上（评测集里有就贴，没有就算了）。

    评测集是仓库内文件、报告是 `data/` 下的产物，两者的题目版本可能不同步，
    所以匹配不到就当没有标注 —— 界面少一行说明，好过显示一条错位的解释。
    """
    for case in report.get("cases", []):
        note = notes.get(case.get("id"))
        if note:
            case["reason"] = note.get("reason")
            if not case.get("relevant_note_ids"):
                case["relevant_note_ids"] = note.get("relevant_note_ids") or []
    return report


def load_case_notes(case_files) -> dict:
    """把若干评测集 jsonl 合起来读成 {题目 id: 标注}；读不了一律当空。"""
    notes = {}
    for path in case_files:
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and isinstance(row.get("id"), str):
                notes.setdefault(row["id"], row)
    return notes


def percentile(values, quantile: float):
    """线性插值分位数；样本为空返回 None（不返回 0，0 会被读成"快得没耗时"）。"""
    numbers = sorted(value for value in values if isinstance(value, (int, float)))
    if not numbers:
        return None
    if len(numbers) == 1:
        return float(numbers[0])
    position = (len(numbers) - 1) * quantile
    low = math.floor(position)
    high = min(low + 1, len(numbers) - 1)
    return float(numbers[low] + (numbers[high] - numbers[low]) * (position - low))


def _mean(values):
    numbers = [value for value in values if isinstance(value, (int, float))]
    return sum(numbers) / len(numbers) if numbers else None


def summarize_traces(turns) -> dict:
    """把带工具调用的助手消息聚合成看板需要的统计。

    输入是「一轮回答」的列表，每项形如
    `{thread_id, created_at, label, tool_calls, retrieved_note_ids, cited_note_ids}`。
    这里刻意不认识数据库：它是纯函数，测试里直接喂假数据就能覆盖空库、全失败、
    没有检索工具这些分支。
    """
    tools = {}
    durations = []
    failures = []
    slowest = []
    step_counts = []
    turn_total_ms = []
    retrieved = cited = 0
    retrieving_turns = citing_turns = 0

    for turn in turns:
        calls = turn.get("tool_calls")
        # 空轮次直接跳过，而不是"算进轮数但步数为 0"：后者会把平均步数拉低，
        # 让人误以为 agent 变懒了。函数自己就该自洽，不依赖上游先过滤干净。
        if not isinstance(calls, list) or not calls:
            continue
        step_counts.append(len(calls))
        turn_ms = 0.0
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name") or "unknown"
            bucket = tools.setdefault(name, {"calls": 0, "failed": 0, "durations": [],
                                             "result_counts": []})
            bucket["calls"] += 1
            duration = _number(call.get("duration_ms"))
            if duration is not None:
                bucket["durations"].append(duration)
                durations.append(duration)
                turn_ms += duration
                slowest.append({"tool": name, "duration_ms": duration,
                                "thread_id": turn.get("thread_id"),
                                "label": turn.get("label"),
                                "created_at": turn.get("created_at")})
            if call.get("result_count") is not None:
                bucket["result_counts"].append(call["result_count"])
            if call.get("ok") is False:
                bucket["failed"] += 1
                failures.append({"tool": name, "error_code": call.get("error_code"),
                                 "thread_id": turn.get("thread_id"),
                                 "label": turn.get("label"),
                                 "created_at": turn.get("created_at")})
        turn_total_ms.append(turn_ms)
        # 引用率刻意不去重：同一篇笔记在两轮里各召回一次，就是两次"可被引用的机会"，
        # 按笔记去重会把它压成一次，让比率虚高。另给"有几轮真的引用了"作为补充口径。
        retrieved += len(turn.get("retrieved_note_ids") or [])
        cited += len(turn.get("cited_note_ids") or [])
        if turn.get("retrieved_note_ids"):
            retrieving_turns += 1
        if turn.get("cited_note_ids"):
            citing_turns += 1

    tool_rows = []
    for name, bucket in tools.items():
        tool_rows.append({
            "name": name,
            "calls": bucket["calls"],
            "failed": bucket["failed"],
            "avg_ms": _mean(bucket["durations"]),
            "p95_ms": percentile(bucket["durations"], 0.95),
            "avg_results": _mean(bucket["result_counts"]),
        })
    tool_rows.sort(key=lambda row: (-row["calls"], row["name"]))
    slowest.sort(key=lambda row: -(row["duration_ms"] or 0))
    failures.sort(key=lambda row: row.get("created_at") or "", reverse=True)

    return {
        "turn_count": len(step_counts),
        "call_count": sum(step_counts),
        "failed_count": len(failures),
        "steps_per_turn": {
            "avg": _mean(step_counts),
            "max": max(step_counts) if step_counts else None,
        },
        "duration": {
            "avg_ms": _mean(durations),
            "p95_ms": percentile(durations, 0.95),
            "avg_turn_ms": _mean(turn_total_ms),
            "max_turn_ms": max(turn_total_ms) if turn_total_ms else None,
        },
        # 引用率＝回答标注的出处 / 检索到的笔记。检索了 4 篇只引用 1 篇是正常现象，
        # 但比率突然掉到 0 就是"检索到了却用不上"，值得在界面上被看见。
        "citation": {
            "retrieved": retrieved,
            "cited": cited,
            "rate": (cited / retrieved) if retrieved else None,
            "retrieving_turns": retrieving_turns,
            "citing_turns": citing_turns,
        },
        "tools": tool_rows,
        "failures": failures[:20],
        "slowest": slowest[:10],
    }
