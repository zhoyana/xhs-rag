# pytest 自动发现 test_ 开头的函数。测试使用合成网页，不会抓取真实小红书。
# 测试通过只证明这些场景符合预期，不能证明任意真实笔记都能导入。
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.extractor import ImportProblem, share_url, parse_note, fetch_page
from app.main import create_app

URL = "https://www.xiaohongshu.com/explore/abc123"


def fixture_html(content="测试正文", kind="normal"):
    # 构造固定的模拟页面；可传入不同正文来测试空内容或特殊字符串。
    return "<script>window.__INITIAL_STATE__=" + json.dumps({"note": {"noteDetailMap": {
        "abc123": {"note": {"title": "测试标题", "desc": content, "type": kind,
                                "imageList": [{"urlDefault": "https://example.com/test.jpg"}]}}
    }}}) + ";</script>"


def test_parse_and_keep_original_content():
    # assert 表示断言：结果不符合预期就让这项测试失败。
    note = parse_note(fixture_html("undefined #标签 正文"), URL)
    assert note["content"] == "undefined #标签 正文"
    assert note["note_id"] == "abc123"
    assert note["warnings"]


# 参数化测试：同一段测试逻辑分别使用这两种输入执行。
@pytest.mark.parametrize("html", ["请登录", fixture_html("")])
def test_missing_data_is_not_success(html):
    # pytest.raises 表示“这里应该抛出指定异常”；没有抛出反而说明测试失败。
    with pytest.raises(ImportProblem):
        parse_note(html, URL)


def test_parser_returns_safe_stage_codes_and_does_not_import_recommendation():
    # 结构变更要能定位，同时绝不能拿页面中的第一条推荐笔记充当目标笔记。
    with pytest.raises(ImportProblem) as missing_map:
        parse_note('<script>window.__INITIAL_STATE__={"note":{}};</script>', URL)
    assert missing_map.value.code == "note_detail_map_missing"

    other_html = fixture_html().replace("abc123", "other456")
    with pytest.raises(ImportProblem) as missing_id:
        parse_note(other_html, URL)
    assert missing_id.value.code == "requested_note_id_missing"


def test_parser_can_use_matching_internal_note_id_when_map_key_changes():
    state = {"note": {"noteDetailMap": {"different-key": {"note": {
        "noteId": "abc123", "title": "目标标题", "desc": "目标正文", "imageList": []
    }}}}}
    html = '<script>window.__INITIAL_STATE__=' + json.dumps(state) + ';</script>'
    assert parse_note(html, URL)["title"] == "目标标题"


def test_intake_job_exposes_only_safe_error_code(tmp_path, monkeypatch):
    async def fake_fetch(_url):
        return '<script>window.__INITIAL_STATE__={"note":{}};</script>', URL

    monkeypatch.setattr("app.main.fetch_page", fake_fetch)
    with TestClient(create_app(tmp_path / "intake.sqlite3")) as client:
        queued = client.post("/notes/intake", json={"share_text": URL})
        assert queued.status_code == 202
        job = client.get(f"/jobs/{queued.json()['job_id']}").json()
        assert job["status"] == "failed"
        assert job["progress"]["error_code"] == "note_detail_map_missing"
        assert job["note_id"] is None
        assert client.get("/notes").json()["items"] == []


@pytest.mark.parametrize("url", ["http://127.0.0.1", "https://evil.com", "https://www.xiaohongshu.com.evil.com/a",
                                 "https://user:pass@www.xiaohongshu.com/a", "https://www.xiaohongshu.com:8000/a"])
def test_reject_unapproved_urls(url):
    with pytest.raises(ImportProblem):
        share_url(url)


def test_share_text():
    assert share_url("分享笔记 " + URL + "?xsec_token=abc\\_123\\&xsec_source=pc_feed") == URL + "?xsec_token=abc_123&xsec_source=pc_feed"


def test_api_dedup_and_persistence(tmp_path, monkeypatch):
    # tmp_path 和 monkeypatch 是 pytest 提供的工具（fixture），不是自己传的参数。
    # tmp_path 隔离测试数据；monkeypatch 暂时替换函数，测试结束后自动恢复。
    async def fake_fetch(url):
        return fixture_html(), URL
    monkeypatch.setattr("app.main.fetch_page", fake_fetch)
    # 替换调用方实际引用的 fetch_page，使接口测试不依赖外部网络。
    db_path = tmp_path / "test.db"
    with TestClient(create_app(db_path)) as client:
        # TestClient 在进程内调用接口，不需要先启动端口；with 会执行应用生命周期。
        assert client.get("/health").json() == {"status": "ok"}
        first = client.post("/notes/import", json={"share_text": URL})
        assert first.status_code == 200
        assert first.json()["status"] == "imported"
        assert client.post("/notes/import", json={"share_text": URL}).json()["status"] == "existing"
        assert len(client.get("/notes").json()["items"]) == 1
        assert client.get("/notes/missing").status_code == 404
        assert client.post("/notes/import", json={"share_text": "oops"}).status_code == 400
    with TestClient(create_app(db_path)) as client:
        # 使用同一个文件启动新应用，验证数据不是只存在于上一次应用的内存里。
        assert client.get("/notes/abc123").status_code == 200


def test_failed_import_not_saved(tmp_path, monkeypatch):
    # 让下载返回登录页，验证接口报错之后数据库仍为空。
    async def fake_fetch(url):
        return "登录页面", URL
    monkeypatch.setattr("app.main.fetch_page", fake_fetch)
    with TestClient(create_app(tmp_path / "test.db")) as client:
        assert client.post("/notes/import", json={"share_text": URL}).status_code == 422
        assert client.get("/notes").json() == {"items": []}


def test_batch_import_isolated_results_and_duplicates(tmp_path, monkeypatch):
    calls = []

    async def fake_fetch(url):
        # 根据链接末尾构造不同笔记，模拟同一批中导入两篇真实笔记。
        note_id = url.rstrip("/").split("/")[-1]
        calls.append(note_id)
        html = fixture_html(f"{note_id} 的正文").replace("abc123", note_id)
        final_url = f"https://www.xiaohongshu.com/explore/{note_id}"
        return html, final_url

    monkeypatch.setattr("app.main.fetch_page", fake_fetch)
    pasted = "\n".join([
        "第一篇 " + URL,
        URL,  # 同一批完全重复，不应再次请求网站。
        "https://evil.com/not-allowed",
        "第二篇 https://www.xiaohongshu.com/explore/def456",
    ])

    with TestClient(create_app(tmp_path / "test.db")) as client:
        response = client.post("/notes/import-batch", json={"share_text": pasted})
        assert response.status_code == 200
        result = response.json()
        assert result["total"] == 4
        assert (result["succeeded"], result["failed"], result["duplicates"]) == (2, 1, 1)
        assert [item["status"] for item in result["items"]] == [
            "imported", "duplicate", "failed", "imported"
        ]
        assert calls == ["abc123", "def456"]
        assert len(client.get("/notes").json()["items"]) == 2

        # 下一批再次导入已经保存的笔记，应返回 existing。
        again = client.post("/notes/import-batch", json={"share_text": URL}).json()
        assert again["items"][0]["status"] == "existing"


def test_batch_import_limits(tmp_path, monkeypatch):
    calls = []

    async def fake_fetch(url):
        calls.append(url)
        return fixture_html(), URL

    monkeypatch.setattr("app.main.fetch_page", fake_fetch)
    too_many = "\n".join(
        f"https://www.xiaohongshu.com/explore/n{i}" for i in range(21)
    )
    with TestClient(create_app(tmp_path / "test.db")) as client:
        assert client.post("/notes/import-batch", json={"share_text": "没有链接"}).status_code == 400
        assert client.post("/notes/import-batch", json={"share_text": too_many}).status_code == 400
    assert calls == []


def test_batch_network_error_is_classified_without_leaking_url(tmp_path, monkeypatch, caplog):
    """诊断信息应能定位错误类别，但不能泄露分享链接中的 token。"""
    secret_token = "secret-token-must-not-appear"
    url = f"https://www.xiaohongshu.com/explore/abc123?xsec_token={secret_token}"

    async def failed_fetch(_url):
        # 用底层 ConnectionResetError 模拟来源站点主动断开连接。
        raise httpx.ReadError("request failed", request=httpx.Request("GET", _url)) \
            from ConnectionResetError()

    monkeypatch.setattr("app.main.fetch_page", failed_fetch)
    with TestClient(create_app(tmp_path / "test.db")) as client:
        response = client.post("/notes/import-batch", json={"share_text": url})

    assert response.status_code == 200
    result = response.json()
    assert result["request_id"]
    assert result["items"][0]["error_code"] == "connection_reset"
    assert secret_token not in response.text
    assert secret_token not in caplog.text


def test_redirect_target_checked(monkeypatch):
    # 模拟一个跳往本机地址的恶意重定向，确保只请求第一跳，不请求受限目标。
    import asyncio
    real_client = httpx.AsyncClient
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})
    monkeypatch.setattr("app.extractor.httpx.AsyncClient", lambda **kwargs: real_client(
        # **kwargs 收集/转发命名参数；MockTransport 用 handler 代替真实网络传输。
        transport=httpx.MockTransport(handler), **kwargs))
    with pytest.raises(ImportProblem):
        asyncio.run(fetch_page(URL))
    assert len(calls) == 1


def test_intake_persists_job_and_downloads_images_immediately(tmp_path, monkeypatch):
    """统一收件应先返回 job，再自动完成导入和图片下载。"""
    from app.database import NoteStore

    async def fake_fetch(_url):
        return fixture_html("收件正文"), URL

    download_calls = []

    def fake_download(store, note, media_root):
        download_calls.append(note["note_id"])
        return {"status": "complete", "total": 1, "downloaded": 1, "skipped": 0, "failed": 0}

    monkeypatch.setattr("app.main.fetch_page", fake_fetch)
    monkeypatch.setattr("app.main.download_note_assets", fake_download)
    db_path = tmp_path / "intake.sqlite3"
    with TestClient(create_app(db_path)) as client:
        response = client.post("/notes/intake", json={
            "share_text": URL + "?xsec_token=secret-intake-token"
        })
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        job = client.get(f"/jobs/{job_id}").json()

    assert job["status"] == "completed"
    assert job["progress"] == {"import": "completed", "download_assets": "completed"}
    assert job["note_id"] == "abc123"
    assert job["result"]["assets"]["downloaded"] == 1
    assert download_calls == ["abc123"]
    assert "secret-intake-token" not in str(NoteStore(db_path).job(job_id))


def test_intake_failure_is_visible_in_persistent_job(tmp_path, monkeypatch):
    async def failed_fetch(_url):
        raise httpx.ConnectError("could not connect")

    monkeypatch.setattr("app.main.fetch_page", failed_fetch)
    with TestClient(create_app(tmp_path / "intake.sqlite3")) as client:
        response = client.post("/notes/intake", json={"share_text": URL})
        job_id = response.json()["job_id"]
        job = client.get(f"/jobs/{job_id}").json()

    assert job["status"] == "failed"
    assert job["progress"]["import"] == "failed"
    assert job["error"] == "无法连接来源站点。"


def test_jobs_left_running_are_marked_interrupted_after_restart(tmp_path):
    from app.database import NoteStore

    store = NoteStore(tmp_path / "jobs.sqlite3")
    store.create_job("job-running", "note_intake", "2026-09-14T00:00:00+00:00", {
        "import": "completed", "download_assets": "running"
    })
    with TestClient(create_app(store.path)) as client:
        job = client.get("/jobs/job-running").json()

    assert job["status"] == "failed"
    assert job["current_step"] == "interrupted"
    assert job["progress"]["interrupted"] is True
