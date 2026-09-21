"""图片测试：使用微型合成图片和模拟网络，不下载用户资料。"""
from io import BytesIO
import json
import sqlite3

import httpx
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app import indexing
from app.database import NoteStore
from app.extractor import ImportProblem
from app.main import create_app
from app.media import (check_image_url, stash_image, storage_path,
                       take_stashed_image)

CDN = "https://sns-webpic-qc.xhscdn.com/image"
NOTE_URL = "https://www.xiaohongshu.com/explore/abc123"


def picture_bytes():
    output = BytesIO()
    Image.new("RGB", (3, 2), "red").save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def mock_network(monkeypatch):
    # 只替换网络传输，不替换下载和图片校验逻辑。
    real_client = httpx.Client
    monkeypatch.setattr("app.media.socket.getaddrinfo", lambda *args: [(2, 1, 6, "", ("1.1.1.1", 443))])
    def install(handler):
        monkeypatch.setattr("app.media.httpx.Client", lambda **kwargs: real_client(
            transport=httpx.MockTransport(handler), **kwargs))
    return install


def setup_note(tmp_path, urls=None):
    path = tmp_path / "test.sqlite3"
    NoteStore(path).save({"note_id": "abc123", "title": "test", "images": urls or [CDN]})
    return path


def note_page(image_url):
    """构造一篇只带图片、正文是空壳的笔记页面。"""
    state = {"note": {"noteDetailMap": {"abc123": {"note": {
        "title": "测试标题", "desc": "#标签 #内推", "type": "normal",
        "imageList": [{"urlDefault": image_url}],
    }}}}}
    return "<script>window.__INITIAL_STATE__=" + json.dumps(state) + ";</script>"


def test_download_open_skip_and_repair(tmp_path, mock_network):
    calls = []
    data = picture_bytes()
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=data)
    mock_network(handler)
    db = setup_note(tmp_path)
    with TestClient(create_app(db)) as client:
        result = client.post("/notes/abc123/assets/download").json()
        assert (result["downloaded"], result["failed"]) == (1, 0)
        asset = result["items"][0]
        assert asset["mime_type"] == "image/png"  # 链接没有后缀，仍按真实图片格式识别。
        assert (asset["width"], asset["height"]) == (3, 2)
        assert client.get(asset["file_url"]).content == data
        again = client.post("/notes/abc123/assets/download").json()
        assert again["skipped"] == 1 and len(calls) == 1
        # 这里只故意损坏测试临时目录里的文件，验证损坏后可重新下载。
        file = tmp_path / "media" / asset["storage_key"]
        file.write_bytes(b"broken")
        assert client.get(asset["file_url"]).status_code == 404
        assert client.post("/notes/abc123/assets/download").json()["downloaded"] == 1
        assert len(client.get("/notes/abc123/assets").json()["items"]) == 1
    with TestClient(create_app(db)) as client:
        assert client.get(asset["file_url"]).content == data  # 新应用仍能读出文件和记录。


def test_partial_failure_and_retry(tmp_path, mock_network):
    failed = True
    def handler(request):
        if str(request.url).endswith("2") and failed:
            return httpx.Response(403)
        return httpx.Response(200, content=picture_bytes())
    mock_network(handler)
    with TestClient(create_app(setup_note(tmp_path, [CDN + "1", CDN + "2"]))) as client:
        result = client.post("/notes/abc123/assets/download").json()
        assert result["status"] == "partial" and result["failed"] == 1
        assert [item["position"] for item in result["items"]] == [1, 2]
        assert result["items"][1]["file_url"] is None
        failed = False
        result = client.post("/notes/abc123/assets/download").json()
        assert (result["downloaded"], result["skipped"], result["failed"]) == (1, 1, 0)


@pytest.mark.parametrize("mode", ["html", "oversize", "pixels", "redirect", "timeout"])
def test_failures_do_not_publish_files(tmp_path, monkeypatch, mock_network, mode):
    data = picture_bytes()
    if mode == "oversize":
        monkeypatch.setattr("app.media.MAX_BYTES", 10)
    if mode == "pixels":
        monkeypatch.setattr("app.media.MAX_PIXELS", 1)
    calls = []
    def handler(request):
        calls.append(str(request.url))
        if mode == "timeout":
            raise httpx.ReadTimeout("test")
        if mode == "redirect":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})
        return httpx.Response(200, content=b"<html>login</html>" if mode == "html" else data)
    mock_network(handler)
    with TestClient(create_app(setup_note(tmp_path))) as client:
        result = client.post("/notes/abc123/assets/download").json()
        assert result["failed"] == 1 and result["items"][0]["error"]
        assert client.get(f"/assets/{result['items'][0]['id']}/file").status_code == 404
        assert not list((tmp_path / "media").rglob("*.part"))
        assert len(calls) == 1


def test_paths_hosts_private_dns_and_lock(tmp_path, monkeypatch):
    with pytest.raises(ImportProblem):
        storage_path(tmp_path, "../outside")
    with pytest.raises(ImportProblem):
        check_image_url("https://sns-webpic-qc.xhscdn.com.evil.com/a")
    monkeypatch.setattr("app.media.socket.getaddrinfo", lambda *args: [(2, 1, 6, "", ("127.0.0.1", 443))])
    with pytest.raises(ImportProblem):
        check_image_url(CDN)
    with TestClient(create_app(setup_note(tmp_path))) as client:
        assert client.get("/assets/999/file").status_code == 404
        assert client.post("/notes/missing/assets/download").status_code == 404
        lock = client.app.state.media_lock
        lock.acquire()
        try:
            assert client.post("/notes/abc123/assets/download").status_code == 409
        finally:
            lock.release()


def test_refresh_source_updates_failed_urls_and_preserves_downloaded_file(tmp_path, monkeypatch):
    """刷新只替换来源数据；成功文件保留，旧的多余位置只软标记。"""
    old_urls = [CDN + "-old-1", CDN + "-old-2", CDN + "-old-3"]
    db_path = setup_note(tmp_path, old_urls)
    store = NoteStore(db_path)
    store.ensure_assets(store.get("abc123"))
    store.asset_status(1, "failed", "HTTP 403")
    # 用数据库元数据模拟第 2 张已经下载成功；刷新逻辑不应清空这些字段。
    with sqlite3.connect(db_path) as db:
        db.execute("""UPDATE assets SET status='success', storage_key='abc123/kept.jpg',
            mime_type='image/jpeg', size_bytes=10, width=1, height=1, sha256='kept'
            WHERE position=2""")

    async def fake_fetch(_url):
        return "fresh html", "https://www.xiaohongshu.com/explore/abc123"

    def fake_parse(_html, _url, requested_url=None):
        return {
            "note_id": "abc123", "title": "刷新后的标题", "content": "新正文",
            "source_url": "https://www.xiaohongshu.com/explore/abc123",
            "images": [CDN + "-fresh-1", CDN + "-fresh-2"],
        }

    monkeypatch.setattr("app.main.fetch_page", fake_fetch)
    monkeypatch.setattr("app.main.parse_note", fake_parse)
    with TestClient(create_app(db_path)) as client:
        response = client.post("/notes/abc123/refresh-source", json={
            "share_text": "https://www.xiaohongshu.com/explore/abc123"
        })
        assert response.status_code == 200
        assert response.json()["note"]["title"] == "刷新后的标题"
        assets = client.get("/notes/abc123/assets").json()["items"]

    assert assets[0]["source_url"].endswith("fresh-1") and assets[0]["status"] == "pending"
    assert assets[0]["error"] is None
    assert assets[1]["source_url"].endswith("fresh-2") and assets[1]["status"] == "success"
    assert assets[1]["storage_key"] == "abc123/kept.jpg"
    assert assets[2]["status"] == "obsolete"


def test_refresh_source_rebuilds_the_index(tmp_path, monkeypatch):
    """正文刷新后旧索引就过期了，接口应当顺手重建，别让用户再点一次。"""
    db_path = setup_note(tmp_path)

    async def fake_fetch(_url):
        return "fresh html", "https://www.xiaohongshu.com/explore/abc123"

    def fake_parse(_html, _url, requested_url=None):
        return {"note_id": "abc123", "title": "刷新后的标题", "content": "新正文",
                "source_url": "https://www.xiaohongshu.com/explore/abc123", "images": []}

    calls = []

    def recording_index(_store, note_id, **_kwargs):
        calls.append(note_id)
        return {"status": "indexed", "note_id": note_id, "searchable": True}

    monkeypatch.setattr("app.main.fetch_page", fake_fetch)
    monkeypatch.setattr("app.main.parse_note", fake_parse)
    monkeypatch.setattr(indexing, "DEFAULT_INDEXER", recording_index)

    with TestClient(create_app(db_path)) as client:
        body = client.post("/notes/abc123/refresh-source", json={
            "share_text": "https://www.xiaohongshu.com/explore/abc123"}).json()

    assert calls == ["abc123"]
    assert body["index"]["searchable"] is True
    # 已经自动做了，就不该再提示用户手动点一次建索引。
    assert not any(step.endswith("/index") for step in body["next_steps"])


def test_refresh_rejects_a_different_note(tmp_path, monkeypatch):
    db_path = setup_note(tmp_path)

    async def fake_fetch(_url):
        return "html", "https://www.xiaohongshu.com/explore/other"

    monkeypatch.setattr("app.main.fetch_page", fake_fetch)
    monkeypatch.setattr("app.main.parse_note", lambda *_, **__: {
        "note_id": "other", "title": "其他笔记", "content": "内容", "images": []
    })
    with TestClient(create_app(db_path)) as client:
        response = client.post("/notes/abc123/refresh-source", json={
            "share_text": "https://www.xiaohongshu.com/explore/abc123"
        })
    assert response.status_code == 409
    assert NoteStore(db_path).get("abc123")["title"] == "test"


def test_stashed_bytes_are_saved_without_a_second_request(tmp_path, mock_network):
    """读图时已经下载过的字节应当直接落盘。

    图床地址带签名、很快过期，落盘再发一次请求就是必然的 403——而这批字节
    本来就在手上。这条用例把网络层设成"只要被碰到就记一笔"，用 0 次调用来证明。
    """
    data = picture_bytes()
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(403)
    mock_network(handler)
    stash_image(CDN, data, "image/png")
    with TestClient(create_app(setup_note(tmp_path))) as client:
        result = client.post("/notes/abc123/assets/download").json()
        assert (result["downloaded"], result["reused"], result["failed"]) == (1, 1, 0)
        asset = result["items"][0]
        assert asset["mime_type"] == "image/png"  # 类型仍以解码结果为准，不信 MIME 参数。
        assert (asset["width"], asset["height"]) == (3, 2)
        assert client.get(asset["file_url"]).content == data
    assert calls == []
    assert take_stashed_image(CDN) is None  # 取走即删除，不会被下一批误用。


def test_stashed_bytes_that_are_not_images_fail_honestly(tmp_path, mock_network):
    """暂存内容不是图片时要如实失败，不能因为"来自本机"就跳过校验。"""
    mock_network(lambda request: httpx.Response(200, content=picture_bytes()))
    stash_image(CDN, b"<html>login</html>", "image/webp")
    with TestClient(create_app(setup_note(tmp_path))) as client:
        result = client.post("/notes/abc123/assets/download").json()
        assert result["failed"] == 1 and result["reused"] == 0
        assert "有效图片" in result["items"][0]["error"]
        assert not list((tmp_path / "media").rglob("*.part"))


def test_image_stash_is_bounded(tmp_path, monkeypatch):
    """一批长图笔记可能几十兆，没被取走的暂存要能自己让位。"""
    monkeypatch.setattr("app.media.STASH_MAX_BYTES", 10)
    stash_image(CDN + "1", b"x" * 8, "image/png")
    stash_image(CDN + "2", b"y" * 8, "image/png")
    assert take_stashed_image(CDN + "1") is None
    assert take_stashed_image(CDN + "2") is not None


def test_batch_import_downloads_images_in_the_same_round(tmp_path, monkeypatch, mock_network):
    """批量导入必须自己把图片存下来。

    它以前只存正文，图片要等用户手点「下载图片」——那时签名地址早已过期，
    界面上却只显示"下载完成 0 张"，用户既存不到图也不知道为什么。
    """
    data = picture_bytes()
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=data)
    mock_network(handler)

    async def fake_fetch(_url):
        return note_page(CDN), NOTE_URL

    monkeypatch.setattr("app.main.fetch_page", fake_fetch)
    with TestClient(create_app(tmp_path / "test.db")) as client:
        result = client.post("/notes/import-batch", json={"share_text": NOTE_URL}).json()
        assert result["items"][0]["status"] == "imported"
        assert result["items"][0]["images"] == {"total": 1, "saved": 1, "failed": 0}
        items = client.get("/notes/abc123/assets").json()["items"]
        assert items[0]["status"] == "success"
        assert client.get(items[0]["file_url"]).content == data
    assert calls == [CDN]


def test_batch_import_reports_image_failure_without_losing_the_note(tmp_path, monkeypatch, mock_network):
    """图片存不下来只影响图片：笔记本身照样导入，且失败张数如实回报。"""
    mock_network(lambda request: httpx.Response(403))
    async def fake_fetch(_url):
        return note_page(CDN), NOTE_URL

    monkeypatch.setattr("app.main.fetch_page", fake_fetch)
    with TestClient(create_app(tmp_path / "test.db")) as client:
        result = client.post("/notes/import-batch", json={"share_text": NOTE_URL}).json()
        assert result["items"][0]["status"] == "imported"
        assert result["items"][0]["images"] == {"total": 1, "saved": 0, "failed": 1}
        assert client.get("/notes/abc123").status_code == 200
