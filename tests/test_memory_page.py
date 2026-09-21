"""记忆页要用到的接口：来源解析、总览计数、向量新鲜度与重建。

页面是前端的事，但这些字段是"看得见依据、改得动"的前提：

1. 来源必须解析成人话（笔记标题 / 会话名）。只回一串 source_note_id 的话，
   用户面对"要不要确认这条记忆"没有任何判断依据。
2. 注入预算与向量过期数必须由服务端给。前端拉列表自己数会因 limit 截断而算错，
   而这两个数字恰好决定"哪些记忆真进了提示词"和"要不要点重建"。
3. 向量重建要么真的修好，要么如实报错——静默失败会让用户以为修好了，
   而召回还在按旧口径匹配。
"""
import sqlite3

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


# ---------------------- 来源标题快照 ----------------------
# 现场：两张候选卡的来源笔记已被删除，界面上只剩一串十六进制 id
# （来源：6aa7e7a20000000027014556）。可"这条从哪儿来"正是用户判断该不该信它的
# 唯一依据——来源查不到，审核就退化成凭感觉点确认。快照在提炼那一刻写下，
# 之后不再依赖源表还活着。


def test_deleted_source_note_still_shows_the_title_it_came_from(tmp_path):
    """来源笔记被删掉之后，候选仍要说出它当初是从哪篇提炼的。"""
    db_path = tmp_path / "page.sqlite3"
    store = NoteStore(db_path)
    seed_note(store, "n1", "搜狐 AI 面经")
    store.save_memory_candidates(
        "test-model", [{"type": "preference", "content": "偏好居家徒手训练"}], "n1", NOW)

    store.delete_note("n1")   # 记忆活得比笔记久：候选不受影响，来源标签也不该丢

    item = store.memory_candidates(status="pending")[0]
    assert item["source_kind"] == "deleted"
    assert item["source_title"] == "搜狐 AI 面经"
    # 快照列是实现细节，接口只需要人话版的 kind/title。
    assert "source_label" not in item


def test_confirmed_candidate_carries_the_title_into_the_memory(tmp_path):
    """候选确认成记忆时快照要跟着走，否则删掉笔记后记忆那边又开始显示 id。"""
    db_path = tmp_path / "page.sqlite3"
    store = NoteStore(db_path)
    seed_note(store, "n1", "搜狐 AI 面经")
    candidate = store.save_memory_candidates(
        "test-model", [{"type": "habit", "content": "团课记到健身夹"}], "n1", NOW)[0]

    created = store.review_memory_candidate(candidate["id"], "confirmed", NOW)["memory"]

    store.delete_note("n1")
    memory = store.memory(created["id"])
    assert memory["source_kind"] == "deleted"
    assert memory["source_title"] == "搜狐 AI 面经"


def test_source_title_follows_a_rename_while_the_note_exists(tmp_path):
    """源还在库里时以现查为准：笔记改了标题，界面上的来源要跟得上（点进去对得上）。"""
    db_path = tmp_path / "page.sqlite3"
    store = NoteStore(db_path)
    seed_note(store, "n1", "旧标题")
    store.save_memory_candidates(
        "test-model", [{"type": "preference", "content": "偏好居家徒手训练"}], "n1", NOW)

    store.update_note_fields("n1", {"title": "新标题"})

    item = store.memory_candidates(status="pending")[0]
    assert item["source_kind"] == "note"
    assert item["source_title"] == "新标题"


def test_legacy_row_without_a_snapshot_is_marked_unknown(tmp_path):
    """老数据既没有快照、来源也早被删了：如实标 unknown，不伪造标题。"""
    db_path = tmp_path / "page.sqlite3"
    store = NoteStore(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute("""INSERT INTO memory_candidates
            (type, content, content_hash, model, source_note_id, created_at)
            VALUES ('preference', '老候选', 'hash-old', 'm', '6aa7e7a20000000027014556', ?)""",
            (NOW,))

    item = store.memory_candidates(status="pending")[0]
    assert item["source_kind"] == "unknown"
    assert item["source_title"] == ""


def test_existing_database_gains_the_snapshot_column(tmp_path):
    """老库第一次打开时自动补上快照列，现有候选与记忆一条都不丢。"""
    db_path = tmp_path / "old.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute("""CREATE TABLE memory_candidates (
            id INTEGER PRIMARY KEY,
            type TEXT NOT NULL CHECK(type IN ('preference', 'habit', 'correction')),
            content TEXT NOT NULL, content_hash TEXT NOT NULL,
            model TEXT NOT NULL, source_note_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'confirmed', 'dismissed')),
            memory_id INTEGER, created_at TEXT NOT NULL, reviewed_at TEXT,
            UNIQUE(source_note_id, content_hash)
        )""")
        db.execute("""INSERT INTO memory_candidates
            (type, content, content_hash, model, source_note_id, status, created_at)
            VALUES ('preference', '旧候选', 'hash-old', 'm', 'note0', 'pending', ?)""",
            (NOW,))

    store = NoteStore(db_path)   # 打开即迁移

    with sqlite3.connect(store.path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(memory_candidates)")}
    assert "source_label" in columns
    item = store.memory_candidates(status="pending")[0]
    assert item["content"] == "旧候选"
    assert item["source_kind"] == "unknown"   # 来源 id 早就不存在了，只能如实说
