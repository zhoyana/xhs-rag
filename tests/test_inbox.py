"""收件箱测试：导入之后的"分析 → 意图 → 收藏夹 → 确认"这条链。

全程用假模型，不联网、不消耗额度。这里要验证的不是模型答得好不好，而是
编排层有没有把这几步按顺序做对、失败时会不会留下半成品。
"""
import pytest
from fastapi.testclient import TestClient

from app.ai import AIProblem
from app.database import NoteStore
from app.main import create_app


@pytest.fixture(autouse=True)
def _no_memory_extraction(monkeypatch):
    """确认意图后的记忆提炼保持静默：单测不该因此产生真实模型调用。"""

    async def fake_extract(note, confirmation, existing):
        return [], None

    monkeypatch.setattr("app.main.extract_memories_safe", fake_extract)


def make_store(db_path, titles=("待整理的一篇",)):
    store = NoteStore(db_path)
    for index, title in enumerate(titles):
        store.save({
            "note_id": f"n{index + 1}", "title": title,
            "content": f"正文内容{index + 1}：这是一篇讲 RAG 与 Agent 的笔记。",
            "images": [],
        })
    return store


def fake_analyzer(calls, category="秋招与面试"):
    async def analyze(note, assets, media_root, memories=None, category_hints=None):
        calls.append(("analyze", category_hints))
        return {
            "summary": "面经摘要", "category": category,
            "inferred_intent": "准备秋招面试", "intent_confidence": 0.82,
            "tags": ["Agent", "面试"],
            "suggested_actions": ["复盘一面", "补 RAG 原理"],
        }, "test-model"

    return analyze


def fake_suggester(calls):
    async def suggest(note, analysis, confirmation, folders, memories=None):
        calls.append(("suggest", None))
        return {"suggestions": [
            {"folder_id": folders[0]["id"], "confidence": 0.9, "reason": "讲的是面试内容"},
        ], "uncertainties": []}, "test-model"

    return suggest


def test_inbox_lists_only_unconfirmed_notes(tmp_path):
    db_path = tmp_path / "inbox.sqlite3"
    store = make_store(db_path, ("待整理的", "已经整理过的"))
    store.save_intent_confirmation(
        "n2", "confirmed", "学习", "想系统学一遍", [], "2026-01-01T00:00:00+00:00"
    )
    with TestClient(create_app(db_path)) as client:
        body = client.get("/inbox").json()
        assert body["total"] == 1
        assert [item["note_id"] for item in body["items"]] == ["n1"]
        assert body["items"][0]["excerpt"].startswith("正文内容1")
        assert body["items"][0]["has_analysis"] is False
        assert body["items"][0]["has_suggestion"] is False


def test_prepare_analyzes_then_suggests_and_caches(tmp_path, monkeypatch):
    db_path = tmp_path / "inbox.sqlite3"
    make_store(db_path)
    calls = []
    monkeypatch.setattr("app.main.analyze_note", fake_analyzer(calls))
    monkeypatch.setattr("app.main.suggest_folders", fake_suggester(calls))

    with TestClient(create_app(db_path)) as client:
        folder = client.post("/folders", json={"name": "秋招与面试"}).json()
        body = client.post("/notes/n1/prepare").json()

        assert body["steps"] == ["analyzed", "suggested"]
        assert body["blockers"] == []
        assert body["analysis"]["category"] == "秋招与面试"
        assert body["suggestion"]["suggestions"][0]["folder_id"] == folder["id"]
        assert [f["name"] for f in body["folders"]] == ["秋招与面试"]
        # 分类的取值范围是用户自己的收藏夹名，而不是写死的那套通用标签。
        assert calls[0] == ("analyze", ["秋招与面试"])

        # 第二次全部命中缓存：不再花一次模型调用，也不重复计费。
        again = client.post("/notes/n1/prepare").json()
        assert again["steps"] == []
        assert again["analysis"]["category"] == "秋招与面试"
        assert len(calls) == 2


def test_prepare_without_folders_keeps_analysis_and_reports_blocker(tmp_path, monkeypatch):
    """没有收藏夹时，分析照做，只是推荐不了——要把原因说清楚，而不是整个失败。"""
    db_path = tmp_path / "inbox.sqlite3"
    make_store(db_path)
    calls = []
    monkeypatch.setattr("app.main.analyze_note", fake_analyzer(calls))
    monkeypatch.setattr("app.main.suggest_folders", fake_suggester(calls))

    with TestClient(create_app(db_path)) as client:
        body = client.post("/notes/n1/prepare").json()
        assert body["steps"] == ["analyzed"]
        assert body["suggestion"] is None
        assert body["blockers"] and "收藏夹" in body["blockers"][0]
        assert [name for name, _ in calls] == ["analyze"]


def test_prepare_reports_model_failure_without_breaking_the_queue(tmp_path, monkeypatch):
    """队列里的批量操作：一条失败不该让整个整理流程断掉，错误如实交给界面。"""
    db_path = tmp_path / "inbox.sqlite3"
    make_store(db_path)

    async def broken(note, assets, media_root, memories=None, category_hints=None):
        raise AIProblem("DeepSeek 账户余额不足。")

    monkeypatch.setattr("app.main.analyze_note", broken)
    with TestClient(create_app(db_path)) as client:
        client.post("/folders", json={"name": "秋招与面试"})
        response = client.post("/notes/n1/prepare")
        assert response.status_code == 200
        body = response.json()
        assert body["analysis"] is None
        assert body["steps"] == []
        assert "余额不足" in body["blockers"][0]
        # 分析没成功，笔记仍留在收件箱里等用户重试。
        assert client.get("/inbox").json()["total"] == 1


def test_triage_confirms_and_archives_in_one_step(tmp_path, monkeypatch):
    db_path = tmp_path / "inbox.sqlite3"
    make_store(db_path)
    calls = []
    monkeypatch.setattr("app.main.analyze_note", fake_analyzer(calls))
    monkeypatch.setattr("app.main.suggest_folders", fake_suggester(calls))

    with TestClient(create_app(db_path)) as client:
        folder = client.post("/folders", json={"name": "秋招与面试"}).json()
        client.post("/notes/n1/prepare")

        result = client.post("/notes/n1/triage", json={
            "decision": "confirmed", "folder_ids": [folder["id"]],
        })
        assert result.status_code == 200
        body = result.json()
        assert body["status"] == "confirmed"
        assert [f["name"] for f in body["folders"]] == ["秋招与面试"]
        assert body["memory_job"] == "queued"

        # 一次确认同时落定意图和归档关系，两边都不能少。
        intent = client.get("/notes/n1/intent").json()
        assert intent["status"] == "confirmed"
        assert intent["confirmation"]["intent"] == "准备秋招面试"
        assert intent["confirmation"]["selected_actions"] == ["复盘一面", "补 RAG 原理"]
        assert [f["name"] for f in client.get("/notes/n1/folders").json()["items"]] == ["秋招与面试"]
        # 整理完就该移出收件箱
        assert client.get("/inbox").json()["total"] == 0


def test_triage_edited_uses_user_text_and_drops_dismissed_action(tmp_path, monkeypatch):
    db_path = tmp_path / "inbox.sqlite3"
    make_store(db_path, ("第一篇", "第二篇"))
    calls = []
    monkeypatch.setattr("app.main.analyze_note", fake_analyzer(calls))
    monkeypatch.setattr("app.main.suggest_folders", fake_suggester(calls))

    with TestClient(create_app(db_path)) as client:
        folder = client.post("/folders", json={"name": "秋招与面试"}).json()
        client.post("/notes/n1/prepare")

        edited = client.post("/notes/n1/triage", json={
            "decision": "edited", "category": "面试复盘", "intent": "下次面试前重看",
            "folder_ids": [folder["id"]],
            # 只保留第 2 条行动
            "action_indexes": [1],
        }).json()
        assert edited["confirmation"]["category"] == "面试复盘"
        assert edited["confirmation"]["intent"] == "下次面试前重看"
        assert edited["confirmation"]["selected_actions"] == ["补 RAG 原理"]

        # edited 只填一半（有分类没意图）就是 400：不能存进去半条意图。
        bad = client.post("/notes/n2/triage", json={
            "decision": "edited", "category": "只有分类", "folder_ids": [folder["id"]],
        })
        assert bad.status_code == 404   # n2 还没分析，连提案都没有，先被挡住
        client.post("/notes/n2/prepare")
        half = client.post("/notes/n2/triage", json={
            "decision": "edited", "category": "只有分类", "folder_ids": [folder["id"]],
        })
        assert half.status_code == 400
        assert client.get("/inbox").json()["total"] == 1   # n2 仍留在收件箱


def test_triage_dismissed_leaves_inbox_without_archiving(tmp_path, monkeypatch):
    db_path = tmp_path / "inbox.sqlite3"
    make_store(db_path)
    calls = []
    monkeypatch.setattr("app.main.analyze_note", fake_analyzer(calls))
    monkeypatch.setattr("app.main.suggest_folders", fake_suggester(calls))

    with TestClient(create_app(db_path)) as client:
        folder = client.post("/folders", json={"name": "秋招与面试"}).json()
        client.post("/notes/n1/prepare")
        body = client.post("/notes/n1/triage", json={
            # 带着 folder_ids 提交也不该归档：忽略就是什么都不做。
            "decision": "dismissed", "folder_ids": [folder["id"]],
        }).json()
        assert body["status"] == "dismissed"
        assert body["folders"] == []
        assert body["tasks"] == []
        assert body["confirmation"]["selected_actions"] == []
        # 模型猜错了就没有可复用的偏好，不该再花一次调用去提炼记忆。
        assert body["memory_job"] == "skipped"
        assert client.get("/notes/n1/folders").json()["items"] == []
        assert client.get("/inbox").json()["total"] == 0


def test_triage_validates_input_before_writing(tmp_path, monkeypatch):
    db_path = tmp_path / "inbox.sqlite3"
    make_store(db_path, ("第一篇", "第二篇"))
    calls = []
    monkeypatch.setattr("app.main.analyze_note", fake_analyzer(calls))
    monkeypatch.setattr("app.main.suggest_folders", fake_suggester(calls))

    with TestClient(create_app(db_path)) as client:
        folder = client.post("/folders", json={"name": "秋招与面试"}).json()
        client.post("/notes/n1/prepare")

        # 未分析就确认：先去做分析（与 /intent 保持一致，同样是 404）
        assert client.post("/notes/n2/triage", json={
            "decision": "confirmed", "folder_ids": [folder["id"]],
        }).status_code == 404
        # 既不给收藏夹也不忽略：400
        assert client.post("/notes/n1/triage", json={
            "decision": "confirmed", "folder_ids": [],
        }).status_code == 400
        # 收藏夹不存在：404，且不允许先写意图再报错
        assert client.post("/notes/n1/triage", json={
            "decision": "confirmed", "folder_ids": [99999],
        }).status_code == 404
        # 超过 3 个收藏夹：由 Pydantic 直接挡掉
        extra = [client.post("/folders", json={"name": f"夹{i}"}).json()["id"] for i in range(3)]
        assert client.post("/notes/n1/triage", json={
            "decision": "confirmed", "folder_ids": [folder["id"], *extra],
        }).status_code == 422

        # 上面每一次失败都不该留下痕迹：这篇还是"待整理"。
        assert client.get("/inbox").json()["total"] == 2
        assert client.get("/notes/n1/intent").json()["status"] == "pending"


def test_triage_can_create_tasks_for_selected_actions(tmp_path, monkeypatch):
    db_path = tmp_path / "inbox.sqlite3"
    make_store(db_path)
    calls = []
    monkeypatch.setattr("app.main.analyze_note", fake_analyzer(calls))
    monkeypatch.setattr("app.main.suggest_folders", fake_suggester(calls))

    with TestClient(create_app(db_path)) as client:
        folder = client.post("/folders", json={"name": "秋招与面试"}).json()
        client.post("/notes/n1/prepare")

        # 勾了行动却没给时间：400，不做"猜一个时间"这种不可逆的默认值。
        assert client.post("/notes/n1/triage", json={
            "decision": "confirmed", "folder_ids": [folder["id"]], "create_tasks": True,
        }).status_code == 400

        body = client.post("/notes/n1/triage", json={
            "decision": "confirmed", "folder_ids": [folder["id"]],
            "action_indexes": [0, 1], "create_tasks": True,
            "scheduled_at": "2026-09-18T20:00:00+08:00", "timezone": "Asia/Shanghai",
        }).json()
        assert [t["title"] for t in body["tasks"]] == ["复盘一面", "补 RAG 原理"]
        assert all(t["status"] == "pending" for t in body["tasks"])
        assert client.get("/tasks?status=pending").json()["items"][0]["scheduled_at"] == \
            "2026-09-18T12:00:00+00:00"
