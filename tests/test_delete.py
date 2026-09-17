"""删除笔记与任务的测试。

删除是本项目里唯一不可逆的写操作，所以这里测的不是"能不能删掉"，而是三件事：
1. **该删的都删了**——笔记是根，图片/分析/意图/收藏夹归属/检索索引（documents →
   chunks → embeddings）/建议全是派生的。只删 notes 一行会留下一堆孤儿行，检索还会
   继续命中已删除的内容，所以断言直接数表，而不是通过接口问（接口看不见的行最危险）。
2. **不该删的没动**——收藏夹本身、计划本身、其他笔记的数据都要原样留着。
3. **部分失败不拖垮整批**——用户选了 20 篇，其中 1 篇早被别处删了，没理由让他重选。
"""
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import indexing
from app.database import NoteStore
from app.embeddings import EXPECTED_DIMENSION
from app.main import create_app

NOW = "2026-01-01T00:00:00+00:00"


@pytest.fixture(autouse=True)
def _no_memory_extraction(monkeypatch):
    """确认意图后的记忆提炼保持静默，避免单测触发真实模型调用。"""

    async def fake_extract(note, confirmation, existing):
        return [], None

    monkeypatch.setattr("app.main.extract_memories_safe", fake_extract)


def fake_embedder(texts):
    """假向量：删除测的是关联关系，向量本身是什么值无关紧要，但维度必须合规。"""
    return [[0.01] * EXPECTED_DIMENSION for _ in texts]


def make_note(db_path, note_id="n1", title="网易 Agent 一面面经", images=()):
    store = NoteStore(db_path)
    store.save({
        "note_id": note_id, "title": title,
        "content": f"{title}：一面问了 RAG 的召回策略与 Agent 的工具编排。",
        "images": list(images),
    })
    # save() 只写 notes 一行；assets 是导入流程单独建的，这里得显式补，
    # 否则"删除时图片行有没有清掉"根本无从验证。
    if images:
        store.ensure_assets(store.get(note_id))
    return store


def seed_derived(db_path, note_id="n1", with_images=True):
    """把一篇笔记能派生的数据全都造出来。

    手工造而不是调 analyze / suggest 那几个接口：这里要验证的是删除有没有漏表，
    派生数据越全越好，不该被各条上游链路的门禁（比如"先确认意图才能建待办"）绑住。
    """
    images = ["https://example.com/1.jpg"] if with_images else []
    store = make_note(db_path, note_id, images=images)
    store.save_analysis(note_id, "test-model", {"summary": "面经摘要"}, NOW)
    store.save_intent_confirmation(note_id, "confirmed", "秋招与面试", "想复盘一面", [], NOW)
    folder = store.create_folder(f"夹子-{note_id}", "", None, NOW)
    store.add_note_to_folder(note_id, folder["id"], "manual", NOW)
    store.save_folder_suggestion(note_id, "test-model", {"suggestions": []}, NOW)
    # 索引：documents → chunks；向量挂在 chunk 上，是最容易漏掉的一层。
    indexing.index_note(store, note_id)
    document = store.document_for_note(note_id)
    indexing.embed_document(store, document["id"], embedder=fake_embedder)
    # 计划步骤引用的来源笔记（plan_item_sources）：删笔记只解除引用，行必须清掉。
    store.create_plan("一周面试冲刺", "把面经吃透", [note_id], [{
        "title": "复盘一面", "details": "", "scheduled_at": "2099-09-18T20:00:00+00:00",
        "reminder_at": None, "timezone": "Asia/Shanghai", "source_note_ids": [note_id],
    }], NOW)
    # 待办：默认跟着笔记一起删（delete_tasks=false 时才保留）。
    store.create_task(note_id, "复盘一面", "", "2099-09-18T20:00:00+00:00", None,
                      "Asia/Shanghai", "复盘一面", NOW)
    return store, folder, document


def counts(db_path, note_id="n1"):
    """数出这篇笔记在各表里的残留行数。"""
    with sqlite3.connect(db_path) as db:
        document_ids = [r[0] for r in db.execute(
            "SELECT id FROM documents WHERE note_id = ?", (note_id,)).fetchall()]
        chunks = embeddings = 0
        if document_ids:
            marks = ",".join("?" * len(document_ids))
            chunks = db.execute(
                f"SELECT COUNT(*) FROM chunks WHERE document_id IN ({marks})",
                document_ids).fetchone()[0]
            embeddings = db.execute(
                f"""SELECT COUNT(*) FROM embeddings WHERE chunk_id IN
                    (SELECT id FROM chunks WHERE document_id IN ({marks}))""",
                document_ids).fetchone()[0]

        def one(table):
            return db.execute(
                f"SELECT COUNT(*) FROM {table} WHERE note_id = ?", (note_id,)).fetchone()[0]

        return {
            "notes": one("notes"), "assets": one("assets"), "analyses": one("analyses"),
            "intent_confirmations": one("intent_confirmations"),
            "folder_suggestions": one("folder_suggestions"), "note_folders": one("note_folders"),
            "plan_item_sources": one("plan_item_sources"),
            "tasks": one("tasks"), "documents": len(document_ids),
            "chunks": chunks, "embeddings": embeddings,
        }


def test_delete_note_purges_every_derived_table(tmp_path):
    db_path = tmp_path / "del.sqlite3"
    store, folder, document = seed_derived(db_path)

    before = counts(db_path)
    # 前置断言：派生数据真的造出来了。否则"删完是 0"可能只是因为本来就没有。
    for table, n in before.items():
        assert n > 0, f"{table} 没有造出数据，这条用例证明不了删除是否漏表"

    with TestClient(create_app(db_path)) as client:
        response = client.delete("/notes/n1")
        assert response.status_code == 200
        body = response.json()
        assert body["notes"] == 1
        assert body["documents"] == 1 and body["chunks"] > 0 and body["embeddings"] > 0

        # 删完之后接口层面也查不到了。
        assert client.get("/notes/n1").status_code == 404
        assert client.get("/notes").json()["items"] == []
        # 再删一次是 404，不是静默成功——调用方需要知道"这次什么都没删"。
        assert client.delete("/notes/n1").status_code == 404

    assert counts(db_path) == {key: 0 for key in before}

    # 收藏夹本身是用户资产，不能跟着笔记一起消失，只是里面的笔记数归零。
    with TestClient(create_app(db_path)) as client:
        kept = client.get(f"/folders/{folder['id']}").json()
        assert kept["name"] == folder["name"] and kept["note_count"] == 0


def test_delete_note_removes_local_image_files(tmp_path):
    db_path = tmp_path / "del.sqlite3"
    media_root = tmp_path / "media"
    store, _, _ = seed_derived(db_path)

    # 在 media 目录里放一个真文件，并把它登记成这篇笔记的图片。
    target = media_root / "n1" / "1.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"fake-jpeg-bytes")
    asset = store.assets("n1")[0]
    store.complete_asset(asset["id"], {
        "storage_key": "n1/1.jpg", "mime_type": "image/jpeg", "size_bytes": 15,
        "width": 10, "height": 10, "sha256": "x" * 64,
    })
    assert target.exists()

    with TestClient(create_app(db_path, media_root=media_root)) as client:
        body = client.delete("/notes/n1").json()
        assert body["removed_files"] == 1 and body["failed_files"] == []

    # 库删干净了，磁盘上也不该留下无主的图片。
    assert not target.exists()


def test_missing_image_file_is_not_counted_as_removed(tmp_path):
    """库里登记了图片、磁盘上却没有文件时，不能虚报"清理了 1 个文件"。

    换过机器、手清过 media 目录都会造成这种状态。用 unlink(missing_ok=True) 会把
    "本来就不存在"也算成删成功，用户看到"删除 2 个文件"会以为磁盘真的干净了。
    """
    db_path = tmp_path / "del.sqlite3"
    store, _, _ = seed_derived(db_path)
    asset = store.assets("n1")[0]
    store.complete_asset(asset["id"], {
        "storage_key": "n1/never-downloaded.jpg", "mime_type": "image/jpeg",
        "size_bytes": 15, "width": 10, "height": 10, "sha256": "x" * 64,
    })

    with TestClient(create_app(db_path, media_root=tmp_path / "media")) as client:
        body = client.delete("/notes/n1").json()
        assert body["removed_files"] == 0
        # 文件不存在不算失败：数据库里那行确实清掉了，目标已经达成。
        assert body["failed_files"] == []


def test_delete_note_can_keep_its_tasks(tmp_path):
    db_path = tmp_path / "del.sqlite3"
    store = make_note(db_path)
    store.save_intent_confirmation("n1", "confirmed", "秋招与面试", "想复盘一面", [], NOW)

    with TestClient(create_app(db_path)) as client:
        created = client.post("/notes/n1/tasks", json={
            "title": "复盘一面", "scheduled_at": "2099-09-18T20:00:00+08:00",
            "timezone": "Asia/Shanghai",
        })
        assert created.status_code == 200, created.text
        task_id = created.json()["id"]

        # 默认连待办一起删：待办内容就是从这篇笔记推出来的，笔记没了它们是无源之水。
        assert client.delete("/notes/n1").json()["tasks"] == 1
        assert client.get(f"/tasks/{task_id}").status_code == 404

    # delete_tasks=false 时保留：已经排进日历的待办是用户的时间承诺。
    other_path = tmp_path / "keep.sqlite3"
    other = make_note(other_path)
    other.save_intent_confirmation("n1", "confirmed", "秋招与面试", "想复盘一面", [], NOW)
    with TestClient(create_app(other_path)) as client:
        kept_id = client.post("/notes/n1/tasks", json={
            "title": "保留的待办", "scheduled_at": "2099-09-18T20:00:00+08:00",
            "timezone": "Asia/Shanghai",
        }).json()["id"]
        assert client.delete("/notes/n1", params={"delete_tasks": "false"}).json()["tasks"] == 0
        assert client.get(f"/tasks/{kept_id}").status_code == 200


def test_delete_note_keeps_the_plan_that_referenced_it(tmp_path):
    db_path = tmp_path / "del.sqlite3"
    seed_derived(db_path, note_id="n1")
    seed_derived(db_path, note_id="n2")

    with TestClient(create_app(db_path)) as client:
        plan = client.post("/plans", json={
            "title": "一周面试冲刺", "goal": "把面经吃透", "source_note_ids": ["n1"],
            "items": [{"title": "复盘一面", "scheduled_at": "2099-09-18T20:00:00+08:00",
                       "timezone": "Asia/Shanghai", "source_note_ids": ["n1"]}],
        }).json()

        client.delete("/notes/n1")

        # 计划是用户确认过的承诺，去掉一篇参考来源不该让整份计划凭空消失；
        # 但指向已删笔记的来源行必须清掉，否则会留下悬空引用。
        assert client.get(f"/plans/{plan['id']}").status_code == 200
        assert counts(db_path, "n1")["plan_item_sources"] == 0
        # 另一篇笔记的数据一行都不该动。
        assert counts(db_path, "n2")["notes"] == 1
        assert counts(db_path, "n2")["chunks"] > 0


def test_bulk_delete_reports_missing_without_failing_batch(tmp_path):
    db_path = tmp_path / "del.sqlite3"
    for note_id in ("n1", "n2", "n3"):
        seed_derived(db_path, note_id=note_id, with_images=False)

    with TestClient(create_app(db_path)) as client:
        body = client.post("/notes/bulk-delete", json={
            "note_ids": ["n1", "n2", "nope", "n1"],   # 中间夹一个不存在的，末尾重复一次
        }).json()
        assert body["deleted_count"] == 2
        assert sorted(item["note_id"] for item in body["deleted"]) == ["n1", "n2"]
        assert body["failed"] == [{"note_id": "nope", "reason": "笔记不存在。"}]

        # 成功的两篇真删干净了，没被那个失败 ID 拖累。
        assert counts(db_path, "n1")["notes"] == 0
        assert counts(db_path, "n2")["documents"] == 0
        assert counts(db_path, "n3")["notes"] == 1
        assert [n["note_id"] for n in client.get("/notes").json()["items"]] == ["n3"]

        # 空列表在接口层就被挡下：一次都不删的请求大概率是调用方写错了。
        assert client.post("/notes/bulk-delete", json={"note_ids": []}).status_code == 422


def test_bulk_delete_tasks_removes_only_existing(tmp_path):
    db_path = tmp_path / "del.sqlite3"
    store = make_note(db_path)
    store.save_intent_confirmation("n1", "confirmed", "秋招与面试", "想复盘一面", [], NOW)

    with TestClient(create_app(db_path)) as client:
        ids = [client.post("/notes/n1/tasks", json={
            "title": f"待办{i}", "scheduled_at": "2099-09-18T20:00:00+08:00",
            "timezone": "Asia/Shanghai",
        }).json()["id"] for i in range(3)]

        body = client.post("/tasks/bulk-delete", json={
            "task_ids": [ids[0], ids[1], 9999],
        }).json()
        assert body["deleted"] == [ids[0], ids[1]]
        assert body["missing"] == [9999]
        assert body["deleted_count"] == 2

        left = client.get("/tasks").json()["items"]
        assert [t["id"] for t in left] == [ids[2]]


def register_local_image(store, media_root, note_id="n1", name="1.jpg"):
    """在 media 目录里放一个真文件，并把它登记成这篇笔记的图片。"""
    target = media_root / note_id / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"fake-jpeg-bytes")
    asset = store.assets(note_id)[0]
    store.complete_asset(asset["id"], {
        "storage_key": f"{note_id}/{name}", "mime_type": "image/jpeg", "size_bytes": 15,
        "width": 10, "height": 10, "sha256": "x" * 64,
    })
    return asset, target


def test_image_delete_blocked_by_outer_layer_still_returns_200(tmp_path, monkeypatch):
    """外层删除拦截层抛 SystemExit 时，接口必须返回 200，而不是 500。

    真实事故（2026-09-17，批量删 8 篇）：图片清理那一步被运行环境的"安全删除"
    拦截层拒绝，它抛的是 SystemExit——**不是 Exception**，所以 `except OSError` 完全
    兜不住，一路穿过 FastAPI 变成 500。而此刻数据库里那几行**已经删掉且提交了**
    （purge_note 是先删库、再删文件），于是用户看到"请求失败"，以为没删成功，
    界面也因为没刷新一直显示着已经不存在的笔记。

    所以这里钉两件事：① 返回 200 并把残留文件如实列进 failed_files；
    ② 库里那行确实删了，不能让 500 把"已经完成的事"说成失败。
    """
    import pathlib

    db_path = tmp_path / "del.sqlite3"
    media_root = tmp_path / "media"
    store, _, _ = seed_derived(db_path)
    asset, target = register_local_image(store, media_root)
    assert target.exists()

    original = pathlib.Path.unlink

    def deny(self, *args, **kwargs):
        if self.exists() and self.name == "1.jpg":
            raise SystemExit(1)      # 拦截层的拒绝方式：非 Exception，且不做任何删除
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "unlink", deny)

    with TestClient(create_app(db_path, media_root=media_root)) as client:
        response = client.delete("/notes/n1")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["notes"] == 1                 # 库里那行确实删掉了
        assert body["removed_files"] == 0
        assert body["failed_files"] == ["n1/1.jpg"]   # 没删成的文件如实上报
        assert client.get("/notes/n1").status_code == 404

    # 文件还在——不假装删掉了。残留由用户可见地列出来，而不是静默虚报。
    assert target.exists()


def test_bulk_delete_marks_unexpected_failure_without_dropping_the_rest(tmp_path, monkeypatch):
    """某篇在存储层炸了（不是 404）时，剩下那些已经删掉的结果不能被一起丢掉。"""
    db_path = tmp_path / "del.sqlite3"
    for note_id in ("n1", "n2", "n3"):
        seed_derived(db_path, note_id=note_id, with_images=False)

    original = NoteStore.delete_note

    def boom(self, note_id, delete_tasks=True):
        if note_id == "n2":
            raise RuntimeError("storage exploded")
        return original(self, note_id, delete_tasks)

    monkeypatch.setattr(NoteStore, "delete_note", boom)

    with TestClient(create_app(db_path)) as client:
        response = client.post("/notes/bulk-delete", json={"note_ids": ["n1", "n2", "n3"]})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["deleted_count"] == 2
        assert body["failed"] == [{"note_id": "n2", "reason": "删除失败：RuntimeError"}]
        # 失败那篇原样留着（不是被删一半），成功的两篇真没了。
        assert counts(db_path, "n2")["notes"] == 1
        assert [n["note_id"] for n in client.get("/notes").json()["items"]] == ["n2"]
