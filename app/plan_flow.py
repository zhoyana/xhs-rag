"""计划链的 LangGraph 版本：草案 → 人工闸门 → 写进日历，可以暂停、也可以恢复。

**为什么值得换成图**（而不是继续用两个接口串起来）：

原来的流程是前端连打 `POST /plans/{id}/confirm` 和 `POST /plans/{id}/create-tasks`。
两个请求之间没有任何东西代表"这次确认正在进行中"，所以中间断网 / 服务重启 /
用户关掉页面，就会留下一个"计划已 confirmed、日历里却没有任务"的半成品。
而且这段暂停不是几毫秒——用户可能第二天才回来点确认。

LangGraph 在这一步真正的价值不是"把流程画成图"，而是 **checkpoint**：
闸门处的完整状态会被序列化进 sqlite（`plan_flow_checkpoints.sqlite3`），
于是"停在闸门上等人"变成一件可以被持久化、被恢复、被检视的事。
`thread_id = plan:{plan_id}` 就是这份状态的钥匙；进程重启后拿同一把钥匙
继续跑，代码一行都不用改。

**一个必须守住的边界**：`interrupt()` 恢复时，节点函数会从第一行重跑，
只有 `interrupt()` 的返回值变成 resume 传进来的东西。所以 `human_review`
节点在 interrupt 之前不做任何写操作——真正的写操作都在它后面的 `commit` 节点里。
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .plan_quality import check_plan_quality

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "plan_flow_checkpoints.sqlite3"

# 检查点库放在业务库旁边，测试里就是 tmp_path，不会污染真实数据。
_savers: dict[str, SqliteSaver] = {}
# FastAPI 的同步接口跑在线程池里，SqliteSaver 共用一条连接，需要自己串行化。
_lock = threading.Lock()


class PlanFlowState(TypedDict, total=False):
    plan_id: int
    draft: dict        # 闸门要展示给人看的东西（含每步来源笔记）
    quality: dict      # 确定性质量门禁的结果
    review: dict       # 用户在闸门做的决定（resume 传进来）
    applied_item_ids: list[int]
    outcome: str       # awaiting_review / committed / already / blocked / missing / missing_source
    task_ids: list[int]


def thread_id_for(plan_id: int) -> str:
    return f"plan:{plan_id}"


def _saver_for(store) -> SqliteSaver:
    raw = getattr(store, "path", None)
    base = Path(raw).resolve() if raw else Path.cwd() / "notes.sqlite3"
    target = base.parent / CHECKPOINT_FILENAME
    key = str(target)
    if key not in _savers:
        conn = sqlite3.connect(target, check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        _savers[key] = saver
    return _savers[key]


def _draft_payload(plan: dict) -> dict:
    """闸门展示用：只给需要看的字段，且必须是纯 dict（要能被序列化进 checkpoint）。"""
    return {
        "plan_id": plan["id"],
        "title": plan["title"],
        "goal": plan["goal"],
        "source_note_ids": list(plan.get("source_note_ids") or []),
        "items": [{
            "id": item["id"],
            "title": item["title"],
            "scheduled_at": item["scheduled_at"],
            "source_note_ids": list(item.get("source_note_ids") or []),
        } for item in plan["items"]],
    }


def _build(store):
    """把 store 通过闭包绑进节点。编译很便宜，检查点库才是跨请求共享的那部分。"""

    def load_draft(state: PlanFlowState) -> dict:
        plan = store.plan(state["plan_id"])
        if plan is None:
            return {"outcome": "missing"}
        if plan["status"] != "draft":
            # 已经落地过：不重复建任务，直接告诉调用方现状。
            return {"outcome": "already"}
        return {
            "outcome": "awaiting_review",
            "draft": _draft_payload(plan),
            "quality": check_plan_quality(plan),
        }

    def human_review(state: PlanFlowState) -> dict:
        decision = interrupt({
            "ask": "确认这份计划吗？",
            "draft": state["draft"],
            "quality": state["quality"],
        })
        # 站在这里的是"重跑后的自己"：只有 interrupt 的返回值是新的。
        return {"review": decision}

    def apply_edits(state: PlanFlowState) -> dict:
        """把用户在闸门里改过的步骤落库，再进 commit。

        先改后建，任务拿到的就是用户最后确认的那个版本 —— 这也是"人工审核"
        和"事后改任务"的区别。
        """
        edits = (state.get("review") or {}).get("items") or []
        now = datetime.now(timezone.utc).isoformat()
        applied = []
        for edit in edits:
            # 只认白名单字段。注意不能顺手丢掉 None：显式传 reminder_at=null
            # 表示"清空提醒"，和"没传这个字段"是两件事。
            fields = {key: value for key, value in edit.items()
                      if key in {"title", "scheduled_at", "reminder_at"}}
            if not fields or not edit.get("id"):
                continue
            store.update_plan_item(edit["id"], fields, now)
            applied.append(edit["id"])
        return {"applied_item_ids": applied}

    def commit(state: PlanFlowState) -> dict:
        plan = store.plan(state["plan_id"])
        if plan is None:
            return {"outcome": "missing"}
        if not plan["source_note_ids"]:
            return {"outcome": "missing_source"}
        quality = check_plan_quality(plan)
        if not quality["can_confirm"]:
            # 门禁没过：什么都不写，把结构化结果交回接口层转成 409。
            return {"outcome": "blocked", "quality": quality}
        now = datetime.now(timezone.utc).isoformat()
        if plan["status"] == "draft":
            store.update_plan_status(plan["id"], "confirmed", now)
        _, task_ids = store.create_plan_tasks(plan["id"], plan["source_note_ids"][0], now)
        return {"outcome": "committed", "task_ids": task_ids, "quality": quality}

    def after_load(state: PlanFlowState) -> str:
        return "human_review" if state.get("outcome") == "awaiting_review" else END

    builder = StateGraph(PlanFlowState)
    builder.add_node("load_draft", load_draft)
    builder.add_node("human_review", human_review)
    builder.add_node("apply_edits", apply_edits)
    builder.add_node("commit", commit)
    builder.add_edge(START, "load_draft")
    builder.add_conditional_edges("load_draft", after_load,
                                  {"human_review": "human_review", END: END})
    builder.add_edge("human_review", "apply_edits")
    builder.add_edge("apply_edits", "commit")
    builder.add_edge("commit", END)
    return builder.compile(checkpointer=_saver_for(store))


def _config(plan_id: int) -> dict:
    return {"configurable": {"thread_id": thread_id_for(plan_id)}}


def start_review(store, plan_id: int) -> dict:
    """草案生成后立刻调用：让图跑到人工闸门停下，状态落进 checkpoint。

    已经在闸门上等着的不重复启动（幂等）——重复启动会多出一份无意义的历史。
    """
    with _lock:
        graph = _build(store)
        config = _config(plan_id)
        snapshot = graph.get_state(config)
        if not snapshot.next:
            graph.invoke({"plan_id": plan_id}, config)
            snapshot = graph.get_state(config)
        return {
            "thread_id": thread_id_for(plan_id),
            "awaiting_review": bool(snapshot.next),
            "outcome": snapshot.values.get("outcome"),
            "draft": snapshot.values.get("draft"),
            "quality": snapshot.values.get("quality"),
        }


def commit_plan(store, plan_id: int, items: list[dict] | None = None) -> dict:
    """闸门上的决定：把用户的修改和"确认"一起送回去，图接着跑完。

    两种入口都支持：
    - 已经挂起在闸门上（正常路径，草案生成时启动的）→ 直接 resume；
    - 没有挂起（老数据、或草案由 POST /plans 手工建的）→ 先启动到闸门再 resume。
    对调用方来说没有区别，也不用关心，这正是把这段状态收进 checkpoint 的收益。
    """
    with _lock:
        graph = _build(store)
        config = _config(plan_id)
        if not graph.get_state(config).next:
            graph.invoke({"plan_id": plan_id}, config)
            snapshot = graph.get_state(config)
            if not snapshot.next:
                # 没有闸门可恢复：计划不存在、已经落地过、或来源笔记缺失。
                # 返回结构和正常路径保持一致，调用方不用记两套字段。
                return {
                    "outcome": snapshot.values.get("outcome", "missing"),
                    "task_ids": [],
                    "quality": snapshot.values.get("quality"),
                    "applied_item_ids": [],
                    "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
                }
        result = graph.invoke(Command(resume={"action": "approve", "items": items or []}), config)
        snapshot = graph.get_state(config)
        return {
            "outcome": result.get("outcome"),
            "task_ids": result.get("task_ids") or [],
            "quality": result.get("quality"),
            "applied_item_ids": result.get("applied_item_ids") or [],
            "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
        }


def flow_snapshot(store, plan_id: int, history_limit: int = 20) -> dict:
    """把这条计划流的执行路径读出来：停在哪、走过哪些节点、每步的 checkpoint id。

    这条接口本身也是"Agent 执行路径可观测"的最小实现：状态不在日志里，在库里。
    """
    graph = _build(store)
    config = _config(plan_id)
    snapshot = graph.get_state(config)
    steps = []
    for index, item in enumerate(graph.get_state_history(config)):
        if index >= history_limit:
            break
        steps.append({
            "next": list(item.next),
            "outcome": item.values.get("outcome"),
            "applied_item_ids": item.values.get("applied_item_ids") or [],
            "checkpoint_id": item.config["configurable"].get("checkpoint_id"),
        })
    return {
        "thread_id": thread_id_for(plan_id),
        "interrupted": bool(snapshot.next),
        "next": list(snapshot.next),
        "outcome": snapshot.values.get("outcome"),
        "steps": steps,
    }
