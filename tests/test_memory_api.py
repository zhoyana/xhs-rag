"""长期记忆管理接口测试：查看、修正、删除、恢复。

全部走 TestClient，不联网。重点覆盖两件事：
1. 记忆错了用户能改——包括内容指纹同步、空白内容拦截、内容重复的 409；
2. 服务端字段不可伪造——请求体多塞 model / source_note_id 一律被忽略。
"""
from fastapi.testclient import TestClient

from app.database import NoteStore
from app.main import create_app

NOW = "2026-09-14T00:00:00+00:00"


def prepared_store(path):
    """预置两条记忆，模拟已经从用户确认的意图里沉淀过的状态。"""
    store = NoteStore(path)
    store.save_memories("test-model", [
        {"type": "preference", "content": "用户偏好居家徒手训练"},
        {"type": "habit", "content": "用户习惯把减脂餐归为健身"},
    ], "note0", NOW)
    return store


def test_list_memories_and_status_filter(tmp_path):
    """默认列表能看到被删掉的记忆，status 过滤才只看生效中的。"""
    db_path = tmp_path / "test.sqlite3"
    store = prepared_store(db_path)
    dismissed_id = store.memories()[0]["id"]
    store.update_memory(dismissed_id, {"status": "dismissed"}, NOW)

    with TestClient(create_app(db_path)) as client:
        everything = client.get("/memories").json()["items"]
        active_only = client.get("/memories?status=active").json()["items"]

    assert len(everything) == 2          # 软删除的记录依然查得到，可追溯
    assert len(active_only) == 1
    assert active_only[0]["status"] == "active"


def test_get_memory_and_missing_returns_404(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = prepared_store(db_path)
    memory_id = store.memories()[0]["id"]

    with TestClient(create_app(db_path)) as client:
        found = client.get(f"/memories/{memory_id}")
        missing = client.get("/memories/9999")

    assert found.status_code == 200
    assert found.json()["source_note_id"] == "note0"
    assert missing.status_code == 404


def test_update_memory_content_and_type(tmp_path):
    """修正内容与类型；content 变了指纹必须跟着变，否则去重会失效。"""
    db_path = tmp_path / "test.sqlite3"
    store = prepared_store(db_path)
    memory_id = store.memories()[0]["id"]
    before = store.memory(memory_id)

    with TestClient(create_app(db_path)) as client:
        body = client.put(f"/memories/{memory_id}", json={
            "content": "用户偏好在家做力量训练",
            "type": "habit",
        }).json()

    assert body["content"] == "用户偏好在家做力量训练"
    assert body["type"] == "habit"
    assert body["content_hash"] != before["content_hash"]
    assert body["updated_at"] != before["updated_at"]


def test_update_memory_ignores_server_owned_fields(tmp_path):
    """请求体多塞 model / source_note_id / 未知键，一律被忽略。

    这就是字段级越权（批量赋值）的防护：请求模型只声明三个可改字段，
    Pydantic 丢弃未声明的键，存储层的 allowed 集合再兜一次底。
    """
    db_path = tmp_path / "test.sqlite3"
    store = prepared_store(db_path)
    memory_id = store.memories()[0]["id"]

    with TestClient(create_app(db_path)) as client:
        body = client.put(f"/memories/{memory_id}", json={
            "content": "用户偏好居家徒手训练（手动修正）",
            "model": "伪造模型",
            "source_note_id": "别人的笔记",
            "evil": "x",
        }).json()

    assert body["model"] == "test-model"        # 产出模型不可伪造
    assert body["source_note_id"] == "note0"    # 来源不可伪造
    assert body["content"] == "用户偏好居家徒手训练（手动修正）"


def test_update_memory_rejects_duplicate_content(tmp_path):
    """改成另一条记忆已有的内容时，由 content_hash 的 UNIQUE 约束兜底 → 409。"""
    db_path = tmp_path / "test.sqlite3"
    store = prepared_store(db_path)
    first, second = store.memories()

    with TestClient(create_app(db_path)) as client:
        response = client.put(f"/memories/{first['id']}",
                              json={"content": second["content"]})

    assert response.status_code == 409
    assert "相同内容" in response.json()["detail"]


def test_update_memory_rejects_blank_content(tmp_path):
    """纯空格能过 Pydantic 的 min_length，但过不了业务层的 strip 校验。"""
    db_path = tmp_path / "test.sqlite3"
    store = prepared_store(db_path)
    memory_id = store.memories()[0]["id"]

    with TestClient(create_app(db_path)) as client:
        response = client.put(f"/memories/{memory_id}", json={"content": "   "})

    assert response.status_code == 400


def test_delete_is_soft_and_recoverable(tmp_path):
    """DELETE 只改状态：记录还在、仍能查到、改回 active 即可恢复。"""
    db_path = tmp_path / "test.sqlite3"
    store = prepared_store(db_path)
    memory_id = store.memories()[0]["id"]

    with TestClient(create_app(db_path)) as client:
        deleted = client.delete(f"/memories/{memory_id}").json()
        assert deleted["status"] == "dismissed"
        # 物理删除的话这里会 404——软删除保留了可审计性
        assert client.get(f"/memories/{memory_id}").status_code == 200

        # 删除后不再参与提示词注入
        injected = [item["id"] for item in NoteStore(db_path).active_memories()]
        assert memory_id not in injected

        restored = client.put(f"/memories/{memory_id}",
                              json={"status": "active"}).json()
        assert restored["status"] == "active"

    injected_again = [item["id"] for item in NoteStore(db_path).active_memories()]
    assert memory_id in injected_again
