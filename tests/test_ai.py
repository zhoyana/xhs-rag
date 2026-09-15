"""AI 接口测试：替换真实模型调用，因此测试不会联网或产生费用。"""
from fastapi.testclient import TestClient

from app.database import NoteStore
from app.main import create_app
from app.ai import AIProblem, _extract_json_object


def test_ai_response_parser_accepts_fenced_json_and_surrounding_text():
    payload = {"choices": [{"finish_reason": "stop", "message": {
        "content": "结果如下：```json\n{\"summary\": \"摘要\"}\n```"
    }}]}
    assert _extract_json_object(payload, "test") == {"summary": "摘要"}


def test_ai_response_parser_accepts_segmented_text():
    payload = {"choices": [{"finish_reason": "stop", "message": {
        "content": [{"type": "text", "text": "前置说明 {\"summary\":"},
                    {"type": "text", "text": "\"分段响应\"} 后置说明"}]
    }}]}
    assert _extract_json_object(payload, "test") == {"summary": "分段响应"}


def test_ai_response_parser_explains_truncated_output_without_logging_content(caplog):
    payload = {"choices": [{"finish_reason": "length", "message": {
        "content": "{\"summary\": \"private note text"
    }}]}
    try:
        _extract_json_object(payload, "analysis")
    except AIProblem as exc:
        assert "被截断" in str(exc)
    else:
        raise AssertionError("预期模型输出截断时提示可读错误")
    assert "private note text" not in caplog.text


def test_analyze_save_cache_and_force(tmp_path, monkeypatch):
    db_path = tmp_path / "test.sqlite3"
    store = NoteStore(db_path)
    store.save({
        "note_id": "note1", "title": "学习笔记", "content": "Python 学习方法", "images": []
    })

    calls = []
    async def fake_analyze(note, assets, media_root, memories=None):
        calls.append(note["note_id"])
        return {
            "summary": "学习 Python 的方法",
            "category": "学习",
            "tags": ["Python"],
            "knowledge_points": ["先练习再复盘"],
            "suggested_actions": ["完成一个小项目"],
            "inferred_intent": "提升编程能力",
            "intent_confidence": 0.8,
            "image_observations": [],
            "uncertainties": [],
        }, "test-vision-model"

    # monkeypatch 只在这项测试期间替换函数，不会修改正式代码。
    monkeypatch.setattr("app.main.analyze_note", fake_analyze)
    with TestClient(create_app(db_path)) as client:
        first = client.post("/notes/note1/analyze").json()
        assert first["status"] == "analyzed" and first["analysis"]["category"] == "学习"
        assert client.get("/notes/note1/analysis").status_code == 200

        second = client.post("/notes/note1/analyze").json()
        assert second["status"] == "cached" and len(calls) == 1

        forced = client.post("/notes/note1/analyze?force=true").json()
        assert forced["status"] == "analyzed" and len(calls) == 2


def test_analysis_missing(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    NoteStore(db_path).save({"note_id": "note1", "title": "test", "images": []})
    with TestClient(create_app(db_path)) as client:
        assert client.get("/notes/note1/analysis").status_code == 404
        assert client.get("/notes/missing/analysis").status_code == 404


def test_user_confirms_and_edits_intent(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = NoteStore(db_path)
    store.save({"note_id": "note1", "title": "跑步", "images": []})
    store.save_analysis("note1", "test-model", {
        "category": "健身",
        "inferred_intent": "开始跑步训练",
        "intent_confidence": 0.7,
        "suggested_actions": ["周一跑步20分钟", "准备跑鞋"],
    }, "2026-01-01T00:00:00+00:00")

    with TestClient(create_app(db_path)) as client:
        pending = client.get("/notes/note1/intent").json()
        assert pending["status"] == "pending" and pending["confirmation"] is None

        # 直接确认时不必重复提交文字，后端沿用模型建议。
        confirmed = client.put("/notes/note1/intent", json={
            "decision": "confirmed",
            # 模拟 Swagger 的占位值；直接确认时后端必须忽略它们。
            "category": "string", "intent": "string", "selected_actions": ["string"],
        }).json()
        assert confirmed["status"] == "confirmed"
        assert confirmed["confirmation"]["intent"] == "开始跑步训练"
        assert len(confirmed["confirmation"]["selected_actions"]) == 2

        # edited 明确表示用户修改了建议，因此分类和意图必须同时填写。
        invalid = client.put("/notes/note1/intent", json={
            "decision": "edited", "intent": "准备参加五公里比赛"
        })
        assert invalid.status_code == 400

        edited = client.put("/notes/note1/intent", json={
            "decision": "edited",
            "category": "健身",
            "intent": "准备参加五公里比赛",
            "selected_actions": ["本周跑两次"],
        }).json()
        assert edited["confirmation"]["intent"] == "准备参加五公里比赛"
        assert edited["confirmation"]["selected_actions"] == ["本周跑两次"]

        # 重新创建应用后仍能读取，证明确认结果真正写入了 SQLite。
    with TestClient(create_app(db_path)) as client:
        saved = client.get("/notes/note1/intent").json()
        assert saved["status"] == "edited"
        assert saved["confirmation"]["category"] == "健身"


def test_intent_requires_analysis(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    NoteStore(db_path).save({"note_id": "note1", "title": "test", "images": []})
    with TestClient(create_app(db_path)) as client:
        assert client.get("/notes/note1/intent").status_code == 404
        assert client.put("/notes/note1/intent", json={"decision": "confirmed"}).status_code == 404
