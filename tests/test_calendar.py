"""日历相关测试：区间取数、草案确认后才落成任务、丢弃草案。

日历这条链有两个容易想当然的地方，都在这里钉住：
1. "取一批任务"不等于"取这个月的任务"——按时间升序 + limit 会先返回最早的历史
   任务，任务攒多了当月反而是空的。所以必须验证区间是真的在过滤。
2. agent 起草的计划在确认之前不该出现在日历上：起草是建议，确认才是用户的决定。
"""
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.database import NoteStore
from app.main import create_app


@pytest.fixture(autouse=True)
def _no_memory_extraction(monkeypatch):
    """确认整理结果后不触发真实模型调用。"""

    async def fake_extract(note, confirmation, existing):
        return [], None

    monkeypatch.setattr("app.main.extract_memories_safe", fake_extract)


def make_store(db_path):
    store = NoteStore(db_path)
    store.save({
        "note_id": "n1", "title": "网易 Agent 一面面经",
        "content": "一面问了 RAG 的召回策略与 Agent 的工具编排。", "images": [],
    })
    # 任务只能挂在"已经确认过收藏意图"的笔记上（接口层有门禁），
    # 先补一条确认记录，让下面的建任务走的是真实路径。
    store.save_intent_confirmation(
        "n1", "confirmed", "秋招与面试", "想系统过一遍面经", [], "2026-01-01T00:00:00+00:00"
    )
    return store


def add_task(client, title, scheduled_at, note_id="n1"):
    response = client.post(f"/notes/{note_id}/tasks", json={
        "title": title, "scheduled_at": scheduled_at, "timezone": "Asia/Shanghai",
    })
    # 建失败要当场炸：否则"日历空了"会被误读成筛选取数有问题。
    assert response.status_code == 200, response.text
    return response


def plan_payload(title="一周面试冲刺",
                 first="2099-09-18T20:00:00+08:00",
                 second="2099-09-19T20:00:00+08:00"):
    """两个步骤的草案；时间放到 2099 年，确认前的质量门禁会拒绝过去的时间。"""
    return {
        "title": title,
        "goal": "把面经吃透",
        "source_note_ids": ["n1"],
        "items": [
            {"title": "复盘一面", "scheduled_at": first,
             "timezone": "Asia/Shanghai", "source_note_ids": ["n1"]},
            {"title": "补 RAG 原理", "scheduled_at": second,
             "timezone": "Asia/Shanghai", "source_note_ids": ["n1"]},
        ],
    }


def test_task_list_filters_by_date_window(tmp_path):
    db_path = tmp_path / "cal.sqlite3"
    make_store(db_path)
    with TestClient(create_app(db_path)) as client:
        # 先堆 30 条历史任务：这正是"日历为什么总是空的"的复现条件。
        for i in range(30):
            month = 1 + i // 28
            add_task(client, f"旧任务{i}", f"2026-{month:02d}-{i % 28 + 1:02d}T20:00:00+08:00")
        add_task(client, "本月的事", "2026-09-18T20:00:00+08:00")
        add_task(client, "下个月的事", "2026-10-02T20:00:00+08:00")

        window = client.get("/tasks", params={
            "limit": 500,
            # 9 月整月：+08:00 的月初与次月月初，换算成 UTC 的绝对时刻。
            "date_from": "2026-08-31T16:00:00Z",
            "date_to": "2026-09-30T16:00:00Z",
        }).json()
        assert [t["title"] for t in window["items"]] == ["本月的事"]

        # 不带区间时确实能拿到全部——证明上面是"过滤生效"，不是碰巧只剩一条。
        assert len(client.get("/tasks").json()["items"]) == 32


def test_date_window_requires_timezone(tmp_path):
    db_path = tmp_path / "cal.sqlite3"
    make_store(db_path)
    with TestClient(create_app(db_path)) as client:
        # 不带时区的时间无法定位到绝对时刻，宁可 400 也不要猜用户的意思。
        naive = client.get("/tasks", params={"date_from": "2026-09-01T00:00:00"})
        assert naive.status_code == 400
        bad = client.get("/tasks", params={"date_from": "九月一号"})
        assert bad.status_code == 400


def test_plan_draft_lands_on_calendar_only_after_confirmation(tmp_path):
    db_path = tmp_path / "cal.sqlite3"
    make_store(db_path)
    with TestClient(create_app(db_path)) as client:
        draft = client.post("/plans", json=plan_payload()).json()
        assert draft["status"] == "draft"

        # 起草阶段：日历上一个任务都没有，草案另走一条查询。
        assert client.get("/tasks").json()["items"] == []
        assert len(client.get("/plans?status=draft").json()["items"]) == 1

        assert client.post(f"/plans/{draft['id']}/confirm").status_code == 200
        created = client.post(f"/plans/{draft['id']}/create-tasks").json()
        assert created["status"] == "created"
        assert len(created["task_ids"]) == 2

        window = client.get("/tasks", params={
            "date_from": "2099-09-01T00:00:00+08:00",
            "date_to": "2099-10-01T00:00:00+08:00",
        }).json()["items"]
        assert [t["title"] for t in window] == ["复盘一面", "补 RAG 原理"]
        # 成了正式计划就不该再挂在"未确认草案"里。
        assert client.get("/plans?status=draft").json()["items"] == []


def test_discard_draft_plan_but_confirmed_plan_is_kept(tmp_path):
    db_path = tmp_path / "cal.sqlite3"
    make_store(db_path)
    with TestClient(create_app(db_path)) as client:
        dropped = client.post("/plans", json=plan_payload(title="随手起草的")).json()
        kept = client.post("/plans", json=plan_payload(title="要认真执行的")).json()

        assert client.delete(f"/plans/{dropped['id']}").status_code == 200
        assert client.get(f"/plans/{dropped['id']}").status_code == 404
        assert [p["title"] for p in client.get("/plans?status=draft").json()["items"]] \
            == ["要认真执行的"]

        # 已确认的计划不能丢：它已经变成任务，抹掉计划会让那些任务失去来源。
        client.post(f"/plans/{kept['id']}/confirm")
        client.post(f"/plans/{kept['id']}/create-tasks")
        rejected = client.delete(f"/plans/{kept['id']}")
        assert rejected.status_code == 409
        assert client.get(f"/plans/{kept['id']}").status_code == 200

    # 步骤与来源关系要跟着草案一起删，不能留下指向不存在计划的孤儿行。
    with sqlite3.connect(db_path) as db:
        assert db.execute(
            "SELECT COUNT(*) FROM plan_items WHERE plan_id = ?", (dropped["id"],)
        ).fetchone()[0] == 0
        assert db.execute(
            "SELECT COUNT(*) FROM plan_item_sources WHERE plan_item_id NOT IN"
            " (SELECT id FROM plan_items)"
        ).fetchone()[0] == 0
