"""记忆页要用到的接口：来源解析、总览计数、向量新鲜度与重建。

页面是前端的事，但这些字段是"看得见依据、改得动"的前提：

1. 来源必须解析成人话（笔记标题 / 会话名）。只回一串 source_note_id 的话，
   用户面对"要不要确认这条记忆"没有任何判断依据。
2. 注入预算与向量过期数必须由服务端给。前端拉列表自己数会因 limit 截断而算错，
   而这两个数字恰好决定"哪些记忆真进了提示词"和"要不要点重建"。
3. 向量重建要么真的修好，要么如实报错——静默失败会让用户以为修好了，
   而召回还在按旧口径匹配。
"""
import pytest
from fastapi.testclient import TestClient

from app.ai import MEMORY_INJECT_LIMIT
from app.database import NoteStore
from app.embeddings import EXPECTED_DIMENSION
from app.main import create_app

NOW = "2026-09-15T00:00:00+00:00"


def seed_note(store, note_id, title):
    """存一篇笔记：来源解析要从 payload 的标题里取，不是从 id 拼。"""
    store.save({"note_id": note_id, "title": title, "content": "正文",
                "source_url": f"https://www.xiaohongshu.com/explore/{note_id}"})


def test_candidate_source_resolves_to_note_title(tmp_path):
    """从笔记确认意图时提炼的候选，来源要显示成笔记标题。"""
    db_path = tmp_path / "page.sqlite3"
    store = NoteStore(db_path)
    seed_note(store, "n1", "网易 Agent 开发一面")
    store.save_memory_candidates(
        "test-model", [{"type": "preference", "content": "偏好居家徒手训练"}], "n1", NOW)

    item = store.memory_candidates(status="pending")[0]

    assert item["source_kind"] == "note"
    assert item["source_title"] == "网易 Agent 开发一面"


def test_candidate_source_resolves_to_thread_title(tmp_path):
    """对话里固化的候选挂在 thread_id 上，同样要能显示成人话。

    两种来源都是字符串 id，形态不可靠（笔记 id 是十六进制、会话 id 是 uuid，
    但那是巧合不是契约），所以判类型只能靠"哪个表查得到"。
    """
    db_path = tmp_path / "page.sqlite3"
    store = NoteStore(db_path)
    store.create_thread("t1", "秋招准备", NOW)
    store.save_memory_candidates(
        "test-model", [{"type": "habit", "content": "用户习惯用中文追问"}], "t1", NOW)

    item = store.memory_candidates(status="pending")[0]

    assert item["source_kind"] == "thread"
    assert item["source_title"] == "秋招准备"


def test_memory_stats_counts_every_state_and_reads_inject_limit(tmp_path):
    """总览计数覆盖三态，注入上限从服务端下发（前端不写死）。"""
    db_path = tmp_path / "page.sqlite3"
    store = NoteStore(db_path)
    saved = store.save_memories("test-model", [
        {"type": "preference", "content": "偏好居家徒手训练"},
        {"type": "habit", "content": "团课记到健身夹"},
    ], "n1", NOW)
    store.update_memory(saved[1]["id"], {"status": "dismissed"}, NOW)
    store.save_memory_candidates(
        "test-model", [{"type": "correction", "content": "别把减脂餐归到做饭"}], "n1", NOW)

    with TestClient(create_app(db_path)) as client:
        body = client.get("/memory-stats").json()

    assert body["memories"]["active"] == 1
    assert body["memories"]["dismissed"] == 1
    assert body["candidates"]["pending"] == 1
    assert body["inject_limit"] == MEMORY_INJECT_LIMIT


def test_user_added_memory_is_marked_as_user_source(tmp_path):
    """用户自己加的没有来源 id，要能与"AI 从某处提炼"区分开。"""
    db_path = tmp_path / "page.sqlite3"
    NoteStore(db_path)

    with TestClient(create_app(db_path)) as client:
        created = client.post("/memories", json={
            "type": "preference", "content": "我习惯用中文写复盘"}).json()

    assert created["source_kind"] == "user"
    assert created["source_title"] == ""
    assert created["embedding_fresh"] is False   # 还没算向量，如实说


def test_embedding_freshness_follows_content_change(tmp_path):
    """内容改了、向量没重算时必须报"不新鲜"。

    这是向量过期唯一的可见信号：不暴露它，用户改了记忆却仍被旧口径匹配，
    界面上看不出任何异常。
    """
    db_path = tmp_path / "page.sqlite3"
    store = NoteStore(db_path)
    saved = store.save_memories(
        "test-model", [{"type": "habit", "content": "团课记到健身夹"}], "n1", NOW)
    memory_id = saved[0]["id"]
    assert saved[0]["embedding_fresh"] is False

    store.save_memory_embeddings([{"id": memory_id, "vector": b"\x00\x01",
                                   "model": "test", "content_hash": saved[0]["content_hash"]}])
    assert store.memory(memory_id)["embedding_fresh"] is True

    updated = store.update_memory(memory_id, {"content": "团课记到训练夹"}, NOW)
    assert updated["embedding_fresh"] is False    # 指纹变了，向量还是旧的那份


def _fake_embedder(dimension=EXPECTED_DIMENSION):
    def embed(texts):
        return [[0.1] * dimension for _ in texts]
    return embed


def test_rebuild_endpoint_refreshes_stale_vectors(tmp_path, monkeypatch):
    """重建要真的把过期向量换掉，而不是只回一个 200。"""
    db_path = tmp_path / "page.sqlite3"
    store = NoteStore(db_path)
    saved = store.save_memories(
        "test-model", [{"type": "habit", "content": "团课记到健身夹"}], "n1", NOW)
    monkeypatch.setattr("app.main.embed_texts", _fake_embedder())

    with TestClient(create_app(db_path)) as client:
        assert client.get("/memory-stats").json()["stale_embeddings"] == 1
        body = client.post("/memories/embeddings/rebuild", json={}).json()
        after = client.get("/memory-stats").json()

    assert body == {"rebuilt": 1, "still_stale": 0}
    assert after["stale_embeddings"] == 0
    assert NoteStore(db_path).memory(saved[0]["id"])["embedding_fresh"] is True


def test_rebuild_reports_unavailable_model_instead_of_pretending(tmp_path, monkeypatch):
    """模型不可用时如实报 503：假装成功会让用户以为向量已经修好了。"""
    db_path = tmp_path / "page.sqlite3"
    store = NoteStore(db_path)
    saved = store.save_memories(
        "test-model", [{"type": "habit", "content": "团课记到健身夹"}], "n1", NOW)

    def boom(_texts):
        raise RuntimeError("模型权重不在本地")

    monkeypatch.setattr("app.main.embed_texts", boom)

    with TestClient(create_app(db_path)) as client:
        response = client.post("/memories/embeddings/rebuild", json={})

    assert response.status_code == 503
    assert NoteStore(db_path).memory(saved[0]["id"])["embedding_fresh"] is False


def test_rebuild_with_nothing_stale_is_a_noop(tmp_path, monkeypatch):
    """没有过期向量时不该白算一遍（重建是一次真实推理，不是空转）。

    顺带钉住：第一次重建会调模型，第二次必须一次都不调——否则用户每点一下
    「重建」都在为同一批向量重复买单。
    """
    db_path = tmp_path / "page.sqlite3"
    NoteStore(db_path).save_memories(
        "test-model", [{"type": "habit", "content": "团课记到健身夹"}], "n1", NOW)
    calls = []

    def counting_embed(texts):
        calls.append(list(texts))
        return [[0.1] * EXPECTED_DIMENSION for _ in texts]

    monkeypatch.setattr("app.main.embed_texts", counting_embed)

    with TestClient(create_app(db_path)) as client:
        first = client.post("/memories/embeddings/rebuild", json={}).json()
        second = client.post("/memories/embeddings/rebuild", json={}).json()

    assert first["rebuilt"] == 1
    assert second == {"rebuilt": 0, "still_stale": 0}
    assert len(calls) == 1


def test_active_list_order_matches_injection_order(tmp_path):
    """前端靠"列表前 inject_limit 条就是会注入的那些"来标注可信度。

    这个前提必须成立，否则界面会把一条带不进去的记忆标成"会注入"——
    那种错没人会去核对，但会实实在在影响用户对系统的信任。
    """
    db_path = tmp_path / "page.sqlite3"
    store = NoteStore(db_path)
    for index, content in enumerate(["偏好甲", "偏好乙", "偏好丙"]):
        store.save_memories("test-model", [{"type": "preference", "content": content}],
                            f"n{index}", f"2026-09-1{index + 1}T00:00:00+00:00")

    listed = [m["id"] for m in store.memories(status="active")]
    injected = [m["id"] for m in store.active_memories(limit=MEMORY_INJECT_LIMIT)]

    assert listed == injected
    assert len(listed) == 3
