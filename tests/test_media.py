"""图片测试：使用微型合成图片和模拟网络，不下载用户资料。"""
from io import BytesIO
import sqlite3

import httpx
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.database import NoteStore
from app.extractor import ImportProblem
from app.main import create_app
from app.media import check_image_url, storage_path

CDN = "https://sns-webpic-qc.xhscdn.com/image"


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

    def fake_parse(_html, _url):
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


def test_refresh_rejects_a_different_note(tmp_path, monkeypatch):
    db_path = setup_note(tmp_path)

    async def fake_fetch(_url):
        return "html", "https://www.xiaohongshu.com/explore/other"

    monkeypatch.setattr("app.main.fetch_page", fake_fetch)
    monkeypatch.setattr("app.main.parse_note", lambda *_: {
        "note_id": "other", "title": "其他笔记", "content": "内容", "images": []
    })
    with TestClient(create_app(db_path)) as client:
        response = client.post("/notes/abc123/refresh-source", json={
            "share_text": "https://www.xiaohongshu.com/explore/abc123"
        })
    assert response.status_code == 409
    assert NoteStore(db_path).get("abc123")["title"] == "test"
