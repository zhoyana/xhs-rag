"""计划链的 LangGraph 版本：人工闸门、编辑后落地、幂等、以及跨进程恢复。

这里最重要的一条是 test_checkpoint_survives_a_fresh_process —— 它把内存里的
checkpointer 全部丢掉再重开 app，模拟"服务重启过"。如果状态只是活在进程内存里，
这条用例必挂；能过就说明暂停窗口真的落在磁盘上。
"""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from app import plan_flow
from app.database import NoteStore
from app.main import create_app


def future_at(days: int) -> str:
    """相对当前日期生成，避免固定日期过期导致测试随时间失效。"""
    day = datetime.now(ZoneInfo("Asia/Shanghai")) + timedelta(days=days)
    return day.replace(hour=20, minute=0, second=0, microsecond=0).isoformat()


def as_utc_text(local_text: str) -> str:
    return datetime.fromisoformat(local_text).astimezone(timezone.utc).isoformat()


def make_plan(client, items: int = 2) -> dict:
    return client.post("/plans", json={
        "title": "一周掌握 Agent 记忆系统",
        "goal": "准备秋招",
        "source_note_ids": ["plan-note"],
        "items": [{"title": f"第 {index + 1} 步", "scheduled_at": future_at(index + 1)}
                  for index in range(items)],
    }).json()


def seed_note(db_path):
    NoteStore(db_path).save({
        "note_id": "plan-note", "title": "Agent 记忆", "content": "学习内容", "images": []
    })


def test_commit_confirms_and_creates_tasks_in_one_call(tmp_path):
    db_path = tmp_path / "commit.sqlite3"
    seed_note(db_path)
    with TestClient(create_app(db_path)) as client:
        plan = make_plan(client)
        assert client.post(f"/plans/{plan['id']}/create-tasks").status_code == 409  # 没确认不能建

        result = client.post(f"/plans/{plan['id']}/commit").json()
        assert result["status"] == "created"
        assert len(result["task_ids"]) == 2
        assert result["plan"]["status"] == "active"
        assert result["checkpoint_id"]
        assert len(client.get("/tasks").json()["items"]) == 2


def test_commit_is_idempotent(tmp_path):
    db_path = tmp_path / "commit-twice.sqlite3"
    seed_note(db_path)
    with TestClient(create_app(db_path)) as client:
        plan_id = make_plan(client)["id"]
        first = client.post(f"/plans/{plan_id}/commit").json()
        assert first["status"] == "created"
        second = client.post(f"/plans/{plan_id}/commit").json()
        assert second["status"] == "existing" and second["task_ids"] == []
        assert len(client.get("/tasks").json()["items"]) == 2


def test_commit_applies_the_edits_made_at_the_gate(tmp_path):
    db_path = tmp_path / "commit-edits.sqlite3"
    seed_note(db_path)
    with TestClient(create_app(db_path)) as client:
        plan = make_plan(client)
        first_id = plan["items"][0]["id"]
        moved = future_at(9)

        result = client.post(f"/plans/{plan['id']}/commit", json={
            "items": [{"id": first_id, "scheduled_at": moved, "title": "改过名字的第一步"}]
        }).json()
        assert result["applied_item_ids"] == [first_id]

        saved = client.get(f"/plans/{plan['id']}").json()["items"][0]
        assert saved["title"] == "改过名字的第一步"
        assert saved["scheduled_at"] == as_utc_text(moved)
        # 任务拿到的必须是用户最后确认的那一版，而不是草案那一版。
        task = next(t for t in client.get("/tasks").json()["items"] if t["title"] == "改过名字的第一步")
        assert task["scheduled_at"] == as_utc_text(moved)


def test_commit_can_be_retried_after_the_quality_gate_blocks_it(tmp_path):
    db_path = tmp_path / "commit-blocked.sqlite3"
    seed_note(db_path)
    with TestClient(create_app(db_path)) as client:
        plan = make_plan(client)
        first_id = plan["items"][0]["id"]
        client.put(f"/plans/{plan['id']}/items/{first_id}",
                   json={"scheduled_at": "2020-01-01T20:00:00+08:00"})

        blocked = client.post(f"/plans/{plan['id']}/commit")
        assert blocked.status_code == 409
        assert "past_schedule" in {issue["code"] for issue in blocked.json()["detail"]["blockers"]}
        assert client.get("/tasks").json()["items"] == []

        # 门禁拦住之后没有留下任何半成品：直接把改好的时间随下一次 resume 一起交回去。
        fixed = client.post(f"/plans/{plan['id']}/commit", json={
            "items": [{"id": first_id, "scheduled_at": future_at(2)}]
        }).json()
        assert fixed["status"] == "created" and len(fixed["task_ids"]) == 2


def test_flow_snapshot_shows_the_pause_and_the_history(tmp_path, monkeypatch):
    db_path = tmp_path / "flow-view.sqlite3"
    NoteStore(db_path).save({"note_id": "source", "title": "RAG", "content": "检索", "images": []})

    async def fake_draft(goal, notes, intents, start_date):
        return {"title": "AI 计划", "goal": goal, "items": [{
            "title": "理解 RAG", "details": "整理概念",
            "scheduled_at": future_at(1), "reminder_at": None}]}, "test-model"

    monkeypatch.setattr("app.main.draft_plan", fake_draft)
    with TestClient(create_app(db_path)) as client:
        draft = client.post("/plans/draft-from-ai", json={
            "goal": "准备秋招", "source_note_ids": ["source"],
            "start_date": future_at(1)[:10],
        }).json()
        plan_id = draft["plan"]["id"]
        # 草案一生成就已经停在闸门上了，而不是等到用户点确认才启动。
        assert draft["flow"]["awaiting_review"] is True

        paused = client.get(f"/plans/{plan_id}/flow").json()
        assert paused["interrupted"] is True and paused["next"] == ["human_review"]
        assert paused["outcome"] == "awaiting_review"

        client.post(f"/plans/{plan_id}/commit")

        done = client.get(f"/plans/{plan_id}/flow").json()
        assert done["interrupted"] is False and done["next"] == []
        assert done["outcome"] == "committed"
        visited = [step["next"] for step in done["steps"]]
        assert ["human_review"] in visited and ["commit"] in visited


def test_checkpoint_survives_a_fresh_process(tmp_path, monkeypatch):
    """模拟进程重启：内存里的连接全部丢掉，只能从磁盘读回 checkpoint。"""
    db_path = tmp_path / "flow-restart.sqlite3"
    NoteStore(db_path).save({"note_id": "source", "title": "RAG", "content": "检索", "images": []})

    async def fake_draft(goal, notes, intents, start_date):
        return {"title": "AI 计划", "goal": goal, "items": [{
            "title": "理解 RAG", "details": "整理概念",
            "scheduled_at": future_at(1), "reminder_at": None}]}, "test-model"

    monkeypatch.setattr("app.main.draft_plan", fake_draft)
    with TestClient(create_app(db_path)) as first_process:
        plan_id = first_process.post("/plans/draft-from-ai", json={
            "goal": "准备秋招", "source_note_ids": ["source"],
            "start_date": future_at(1)[:10],
        }).json()["plan"]["id"]

    plan_flow._savers.clear()  # 新进程里这个字典本来就是空的

    with TestClient(create_app(db_path)) as second_process:
        flow = second_process.get(f"/plans/{plan_id}/flow").json()
        assert flow["interrupted"] is True and flow["next"] == ["human_review"]
        result = second_process.post(f"/plans/{plan_id}/commit").json()
        assert result["status"] == "created" and len(result["task_ids"]) == 1
        assert len(second_process.get("/tasks").json()["items"]) == 1


def test_commit_reports_missing_source_plan(tmp_path):
    db_path = tmp_path / "flow-no-source.sqlite3"
    with TestClient(create_app(db_path)) as client:
        plan = client.post("/plans", json={
            "title": "没有来源笔记的计划", "goal": "目标",
            "items": [{"title": "步骤", "scheduled_at": future_at(1)}],
        }).json()
        blocked = client.post(f"/plans/{plan['id']}/commit")
        assert blocked.status_code == 409
        assert "来源笔记" in blocked.json()["detail"]
        assert client.get("/tasks").json()["items"] == []


def test_commit_on_missing_plan_is_404(tmp_path):
    with TestClient(create_app(tmp_path / "flow-404.sqlite3")) as client:
        assert client.post("/plans/9999/commit").status_code == 404
        assert client.get("/plans/9999/flow").status_code == 404
