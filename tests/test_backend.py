# pytest 自动发现 test_ 开头的函数。测试使用合成网页，不会抓取真实小红书。
# 测试通过只证明这些场景符合预期，不能证明任意真实笔记都能导入。
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.extractor import ImportProblem, share_url, parse_note, fetch_page
from app import indexing
from app.main import create_app

URL = "https://www.xiaohongshu.com/explore/abc123"


def fixture_html(content="测试正文", kind="normal", note_id="abc123"):
    # 构造固定的模拟页面；可传入不同正文来测试空内容或特殊字符串。
    # note_id 可变，批量导入要用它造出"几条不同笔记"的页面。
    return "<script>window.__INITIAL_STATE__=" + json.dumps({"note": {"noteDetailMap": {
        note_id: {"note": {"title": "测试标题", "desc": content, "type": kind,
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


def test_parser_keeps_original_note_id_after_non_detail_redirect():
    # 最终跳到首页时，原始分享直链仍能提供 ID；但网页必须含对应笔记。
    assert parse_note(fixture_html(), "https://www.xiaohongshu.com/", requested_url=URL)["note_id"] == "abc123"
    with pytest.raises(ImportProblem) as missing:
        parse_note('<script>window.__INITIAL_STATE__={"note":{}};</script>',
                   "https://www.xiaohongshu.com/", requested_url=URL)
    assert missing.value.code == "note_detail_map_missing"


def test_parser_rejects_conflicting_redirect_note_id():
    with pytest.raises(ImportProblem) as conflict:
        parse_note(fixture_html(), "https://www.xiaohongshu.com/explore/other456",
                   requested_url=URL)
    assert conflict.value.code == "redirect_note_id_mismatch"


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


def test_short_link_remembers_detail_id_across_later_redirect(monkeypatch):
    # 用本地模拟响应复现：短链 -> 详情页 -> 首页，不访问真实网站。
    import asyncio
    real_client = httpx.AsyncClient
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/short":
            return httpx.Response(302, headers={"location": URL})
        if request.url.path == "/explore/abc123":
            return httpx.Response(302, headers={"location": "/"})
        return httpx.Response(200, text=fixture_html())

    monkeypatch.setattr("app.extractor.httpx.AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(handler), **kwargs))
    html, detail_url = asyncio.run(fetch_page("https://xhslink.com/short"))
    assert paths == ["/short", "/explore/abc123", "/"]
    assert parse_note(html, detail_url)["note_id"] == "abc123"


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
    assert job["progress"] == {"import": "completed", "download_assets": "completed",
                               "index": "completed"}
    assert job["note_id"] == "abc123"
    assert job["result"]["assets"]["downloaded"] == 1
    # 收件箱进来的笔记必须当场可检索，而不是等用户想起来再手动点一次建索引。
    assert job["result"]["index"]["searchable"] is True
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


async def _no_mcp(_url):
    """模拟"这条通道用不了"，用来验证收件箱的降级链路。"""
    return None


def test_intake_prefers_mcp_and_records_source(tmp_path, monkeypatch):
    """统一收件优先走 MCP 采集，并把采集来源写进任务，便于排查空壳笔记。"""
    async def fake_collect(_url, **_kwargs):
        return {"note_id": "abc123", "title": "MCP 标题", "content": "MCP 采到的正文内容",
                "source_url": URL, "images": [], "kind": "normal",
                "warnings": []}, "mcp"

    async def exploding_fetch(_url):
        raise AssertionError("MCP 已采到时不应再请求网页")

    monkeypatch.setattr("app.xhs_source.collect_note", fake_collect)
    monkeypatch.setattr("app.main.fetch_page", exploding_fetch)

    with TestClient(create_app(tmp_path / "intake-mcp.sqlite3")) as client:
        response = client.post("/notes/intake", json={"share_text": URL})
        job = client.get(f"/jobs/{response.json()['job_id']}").json()

    assert job["status"] == "completed"
    assert job["result"]["source"] == "mcp"
    assert job["note_id"] == "abc123"


def test_intake_falls_back_to_html_when_mcp_is_unusable(tmp_path, monkeypatch):
    """MCP 用不了时必须继续用网页解析，收件箱不能因此整单失败。"""
    async def fake_fetch(_url):
        return fixture_html("回退正文"), URL

    def fake_download(_store, _note, _root):
        return {"status": "complete", "total": 0, "downloaded": 0, "skipped": 0, "failed": 0}

    monkeypatch.setattr("app.xhs_source.collect_via_mcp", _no_mcp)
    monkeypatch.setattr("app.main.fetch_page", fake_fetch)
    monkeypatch.setattr("app.main.download_note_assets", fake_download)

    with TestClient(create_app(tmp_path / "intake-fallback.sqlite3")) as client:
        response = client.post("/notes/intake", json={"share_text": URL})
        job = client.get(f"/jobs/{response.json()['job_id']}").json()

    assert job["status"] == "completed"
    assert job["result"]["source"] == "html"
    assert job["note_id"] == "abc123"


def test_single_import_reports_its_collection_source(tmp_path, monkeypatch):
    """单条导入接口同样回报来源，方便前端和日志对照。"""
    async def fake_fetch(_url):
        return fixture_html("导入正文"), URL

    monkeypatch.setattr("app.main.fetch_page", fake_fetch)

    with TestClient(create_app(tmp_path / "import-source.sqlite3")) as client:
        response = client.post("/notes/import", json={"share_text": URL})

    assert response.status_code == 200
    assert response.json()["source"] == "html"


def test_single_import_indexes_the_note_immediately(tmp_path, monkeypatch):
    """导入完就能语义检索，是响应必须回答的问题，不能只回报"保存成功"。"""
    async def fake_fetch(_url):
        return fixture_html("可检索的正文"), URL

    calls = []

    def recording_index(_store, note_id, **_kwargs):
        calls.append(note_id)
        return {"status": "indexed", "note_id": note_id, "chunk_count": 2,
                "vector_total": 2, "searchable": True}

    monkeypatch.setattr("app.main.fetch_page", fake_fetch)
    monkeypatch.setattr(indexing, "DEFAULT_INDEXER", recording_index)

    with TestClient(create_app(tmp_path / "import-index.sqlite3")) as client:
        body = client.post("/notes/import", json={"share_text": URL}).json()

    # 索引必须建立在保存之后，且作用在这次导入的笔记上。
    assert calls == ["abc123"]
    assert body["index"]["searchable"] is True
    assert body["index"]["vector_total"] == 2


def test_index_failure_does_not_break_the_import(tmp_path, monkeypatch):
    """索引失败只让这篇暂时搜不到，不能把已经采到手的笔记一起丢掉。"""
    async def fake_fetch(_url):
        return fixture_html("索引会失败的正文"), URL

    def broken_index(_store, _note_id, **_kwargs):
        raise RuntimeError("向量模型不可用")

    monkeypatch.setattr("app.main.fetch_page", fake_fetch)
    monkeypatch.setattr(indexing, "DEFAULT_INDEXER", broken_index)

    with TestClient(create_app(tmp_path / "import-index-fail.sqlite3")) as client:
        response = client.post("/notes/import", json={"share_text": URL})
        saved = client.get("/notes/abc123")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "imported"
    assert body["index"]["status"] == "failed"
    assert body["index"]["searchable"] is False
    # 笔记确实落库了——派生步骤失败不该回滚主流程。
    assert saved.status_code == 200


def test_batch_import_marks_each_note_searchable(tmp_path, monkeypatch):
    """批量导入要能看出哪几条已经能搜到，而不是只回报保存成功。"""
    urls = ["https://www.xiaohongshu.com/explore/aaa111",
            "https://www.xiaohongshu.com/explore/bbb222"]

    async def fake_fetch(url):
        note_id = url.rstrip("/").rsplit("/", 1)[-1]
        return fixture_html("批量正文", note_id=note_id), url

    monkeypatch.setattr("app.main.fetch_page", fake_fetch)

    with TestClient(create_app(tmp_path / "batch-index.sqlite3")) as client:
        body = client.post("/notes/import-batch",
                           json={"share_text": "\n".join(urls)}).json()

    assert body["succeeded"] == 2
    assert [item["note_id"] for item in body["items"]] == ["aaa111", "bbb222"]
    assert all(item["searchable"] is True for item in body["items"])
    assert all(item["index_status"] == "indexed" for item in body["items"])


def test_intake_indexes_a_note_whose_content_lives_in_images(tmp_path, monkeypatch):
    """正文极短、内容都在长图里的笔记，从收件箱进来也必须当场可检索。

    真机验收时正是这条路径暴露过问题：收件箱不做图片转写，剥掉话题标签后
    正文为空，索引被直接跳过，用户收完一篇却一个字都搜不到。
    """
    from app import xhs_source
    from app.database import NoteStore
    from app.embeddings import EXPECTED_DIMENSION
    from app.indexing import index_and_embed

    async def fake_fetch(_url):
        # 正文只有"面经"两个字加话题标签——真正的内容在长图里。
        return fixture_html("面经 #秋招[话题]# #AI[话题]#"), URL

    async def fake_transcribe(images, **_kwargs):
        return {"status": "ok", "text": "面试官问了大模型 Agent 的记忆机制如何设计。",
                "model": "fake", "error": None, "read": len(images), "dropped": 0}

    def fake_embedder(texts):
        return [[1.0] + [0.0] * (EXPECTED_DIMENSION - 1) for _ in texts]

    def fake_download(_store, _note, _root):
        return {"status": "complete", "total": 1, "downloaded": 1, "skipped": 0, "failed": 0}

    # 用真实索引实现、只替换向量计算：这里要验证的正是"图片文字真的进了索引正文"。
    def real_index(store, note_id, **_kwargs):
        return index_and_embed(store, note_id, embedder=fake_embedder)

    monkeypatch.setattr("app.main.fetch_page", fake_fetch)
    monkeypatch.setattr("app.main.download_note_assets", fake_download)
    monkeypatch.setattr(xhs_source, "DEFAULT_TRANSCRIBE", fake_transcribe)
    monkeypatch.setattr(indexing, "DEFAULT_INDEXER", real_index)

    db_path = tmp_path / "image-content.sqlite3"
    with TestClient(create_app(db_path)) as client:
        queued = client.post("/notes/intake", json={"share_text": URL})
        job = client.get(f"/jobs/{queued.json()['job_id']}").json()

    assert job["status"] == "completed"
    assert job["result"]["index"]["status"] == "indexed"
    assert job["result"]["index"]["searchable"] is True

    store = NoteStore(db_path)
    document = store.document_for_note("abc123")
    assert document is not None
    indexed_text = "\n".join(chunk["content"] for chunk in store.document_chunks(document["id"]))
    assert "记忆机制" in indexed_text


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
