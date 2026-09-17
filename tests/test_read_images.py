"""补读图片端点 + 前端静态托管测试。

离线原则：视觉模型与本地文件校验都在边界处替换，只验证"读图结果是否正确并入
笔记 payload、是否触发重建索引、没有本地图的笔记是否被正确跳过"这组业务约定。
"""
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def make_note(note_id, *, with_image_text=False, with_images=True):
    note = {
        "note_id": note_id,
        "title": f"测试笔记 {note_id}",
        "content": "这是正文",
        "warnings": [],
    }
    if with_images:
        note["images"] = ["https://sns-webpic.xhscdn.com/a.jpg"]
    if with_image_text:
        note["image_text"] = "已有图片文字"
    return note


async def fake_transcribe(paths, **_kwargs):
    # 端点把本地路径列表传进来；这里我们不真读盘，只确认它收到了路径。
    return {"status": "ok", "text": "识别出的图片文字", "model": "fake",
            "error": None, "read": len(paths), "dropped": 0}


def test_read_images_single_note_writes_payload(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.transcribe_local_files", fake_transcribe)
    # 假装所有已落盘资产都校验通过；storage_path 返回占位路径（不真读盘）。
    monkeypatch.setattr("app.main.local_file_valid", lambda root, asset: True)
    monkeypatch.setattr("app.main.storage_path", lambda root, key: Path("/tmp/fake.png"))

    with TestClient(create_app(tmp_path / "db.sqlite3")) as client:
        store = client.app.state.store
        note = make_note("n1")
        store.save(note)
        store.ensure_assets(note)

        r = client.post("/notes/n1/read-images")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ok"
        assert body["read"] == 1

        # 图片文字必须落库，且警告同步更新。
        saved = client.get("/notes/n1").json()
        assert saved["image_text"] == "识别出的图片文字"
        assert saved["warnings"]
        # 重建索引后该笔记可被检索。
        assert body["index"]["searchable"] is True


def test_read_images_skips_when_image_text_exists(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.transcribe_local_files", fake_transcribe)
    monkeypatch.setattr("app.main.local_file_valid", lambda root, asset: True)
    monkeypatch.setattr("app.main.storage_path", lambda root, key: Path("/tmp/fake.png"))

    with TestClient(create_app(tmp_path / "db.sqlite3")) as client:
        store = client.app.state.store
        store.save(make_note("n2", with_image_text=True))
        store.ensure_assets(make_note("n2"))

        # 默认跳过已有图片文字的笔记。
        r = client.post("/notes/n2/read-images")
        assert r.json()["status"] == "skipped"

        # force=true 时仍然会重读。
        r2 = client.post("/notes/n2/read-images?force=true")
        assert r2.json()["status"] == "ok"
        assert client.get("/notes/n2").json()["image_text"] == "识别出的图片文字"


def test_read_images_no_local_images(tmp_path, monkeypatch):
    # 没有任何本地图片时不应调用视觉模型，直接返回 no_local_images。
    called = {"n": 0}

    async def counting_transcribe(paths, **_kwargs):
        called["n"] += 1
        return {"status": "ok", "text": "x", "model": "fake", "error": None, "read": 1, "dropped": 0}
    monkeypatch.setattr("app.main.transcribe_local_files", counting_transcribe)
    monkeypatch.setattr("app.main.local_file_valid", lambda root, asset: False)

    with TestClient(create_app(tmp_path / "db.sqlite3")) as client:
        store = client.app.state.store
        note = make_note("n3", with_images=False)
        store.save(note)
        store.ensure_assets(note)

        r = client.post("/notes/n3/read-images")
        assert r.status_code == 200
        assert r.json()["status"] == "no_local_images"
        assert called["n"] == 0


def test_read_images_batch_only_processes_local(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.transcribe_local_files", fake_transcribe)
    # 只让 b1 的本地图"校验通过"，b2 视为无本地图。
    monkeypatch.setattr("app.main.local_file_valid",
                        lambda root, asset: asset.get("note_id") == "b1")
    monkeypatch.setattr("app.main.storage_path", lambda root, key: Path("/tmp/x.png"))

    with TestClient(create_app(tmp_path / "db.sqlite3")) as client:
        store = client.app.state.store
        store.save(make_note("b1"))
        store.ensure_assets(make_note("b1"))
        store.save(make_note("b2"))
        store.ensure_assets(make_note("b2"))

        r = client.post("/notes/read-images", json={"all": True})
        assert r.status_code == 200
        items = {i["note_id"]: i["status"] for i in r.json()["items"]}
        assert items["b1"] == "ok"
        assert items["b2"] == "no_local_images"
        assert client.get("/notes/b1").json()["image_text"] == "识别出的图片文字"


def test_read_images_batch_requires_target(tmp_path):
    with TestClient(create_app(tmp_path / "db.sqlite3")) as client:
        r = client.post("/notes/read-images", json={})
        assert r.status_code == 400


def test_web_static_served(tmp_path):
    # /ui 由后端托管，与 API 同源；index.html 与静态资源都应能取到。
    with TestClient(create_app(tmp_path / "db.sqlite3")) as client:
        r = client.get("/ui/")
        assert r.status_code == 200
        assert "小红书知识库" in r.text
        js = client.get("/ui/app.js")
        assert js.status_code == 200
        assert "api" in js.text
        css = client.get("/ui/style.css")
        assert css.status_code == 200
