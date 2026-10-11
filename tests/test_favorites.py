"""收藏夹同步：解析、链接往返，以及两个后台任务端点的对外契约。

用例不启 18060 上的采集服务、也不访问小红书：`xhs_mcp.fetch_my_favorites` 与
`xhs_source.fetch_feed_detail` / `xhs_source.fetch_page` 三层都在用例里替换掉。

收藏列表的真实形状取自 2026-10-11 真机 dump：`get_my_profile(tab="fav")` 返回
`{userBasicInfo, interactions, feeds[]}`，每条 feed 带 `id` + `xsecToken`，展示信息
在 `noteCard` 里（实测 20 条，无翻页）。
"""
import pytest
from fastapi.testclient import TestClient

from app import favorites, xhs_mcp, xhs_source
from app.main import create_app
from app.xhs_link import parse_xhs_link

FEED_ID = "6ab2412a000000003703c455"
OTHER_ID = "6ab2412a000000003703c456"
TOKEN = "ABIOjBXzgMip-Ka1BwWUKH47OzULHVONZxapoF3ej1x7w="


def feed_entry(feed_id=FEED_ID, token=TOKEN, title="珀莱雅AI面试高频题中文50+追问47道+回答",
               cover="http://sns-webpic-qc.xhscdn.com/a/cover!nd_dft_wlteh_webp_3"):
    return {
        "xsecToken": token, "id": feed_id, "index": 0,
        "noteCard": {
            "type": "normal",
            "displayTitle": title,
            "user": {"userId": "5c28c3180000000006031a26", "nickname": "Offer学姐柠檬"},
            "interactInfo": {"liked": True, "likedCount": "28"},
            "cover": {"urlDefault": cover},
        },
    }


def profile_payload(feeds=None):
    return {
        "userBasicInfo": {"nickname": "。", "ipLocation": "浙江"},
        "interactions": [
            {"type": "fans", "name": "粉丝", "count": "41"},
            {"type": "interaction", "name": "获赞与收藏", "count": "344"},
        ],
        "feeds": [feed_entry()] if feeds is None else feeds,
    }


def detail_payload(feed_id=FEED_ID, title="收藏导入的笔记"):
    """get_feed_detail 的返回；imageList 留空，用例就不必伪造图床网络。"""
    note = {
        "noteId": feed_id, "title": title,
        "desc": "一面问了 ReAct 与 Function Calling 的执行链路，二面考多 Agent 协作与工具失败处理。",
        "type": "normal", "user": {"nickname": "Offer学姐柠檬"}, "imageList": [],
    }
    return {"feed_id": feed_id, "data": {"note": note}}


def use_mcp(monkeypatch, *, detail=None, favorites_payload=None, favorites_error=None):
    """把两层站外调用换成替身；默认两条通道都返回正常数据。"""
    async def fake_detail(feed_id, xsec_token, **_kwargs):
        return (detail or detail_payload)(feed_id)

    async def fake_favorites(**_kwargs):
        if favorites_error is not None:
            raise favorites_error
        return favorites_payload if favorites_payload is not None else profile_payload()

    monkeypatch.setattr(xhs_source, "fetch_feed_detail", fake_detail)
    monkeypatch.setattr(xhs_mcp, "fetch_my_favorites", fake_favorites)


# ---------------------------- 链接往返 ----------------------------
# 这是整条链路的第一个接缝：拼出来的链接必须能被现有解析器还原。
# 拼错了（少个参数、令牌没编码）不会报错，只会表现为"每条都采集失败"。

@pytest.mark.parametrize("token", [TOKEN, "ab+cd/ef=gh==", "a=b&c=d"])
def test_note_link_round_trips_through_the_link_parser(token):
    ref = parse_xhs_link(favorites.note_link(FEED_ID, token))
    assert ref["feed_id"] == FEED_ID
    # 令牌里的 = + / 必须原样还原：`+` 在 query 里不编码会被解成空格，令牌当场作废。
    assert ref["xsec_token"] == token
    assert ref["short_link"] is False


def test_note_link_uses_the_explore_path():
    """解析器只认 /explore/<id> 或 /discovery/item/<id>，用别的路径会拿不到编号。"""
    assert "/explore/" + FEED_ID in favorites.note_link(FEED_ID, TOKEN)


# ---------------------------- 列表归一 ----------------------------

def test_parse_profile_normalizes_the_list():
    parsed = favorites.parse_profile(profile_payload())
    assert parsed["skipped"] == 0
    assert parsed["profile"]["nickname"] == "。"
    assert parsed["profile"]["interactions"] == {"粉丝": "41", "获赞与收藏": "344"}
    item = parsed["items"][0]
    assert item["feed_id"] == FEED_ID
    assert item["xsec_token"] == TOKEN
    assert item["title"].startswith("珀莱雅")
    assert item["author"] == "Offer学姐柠檬"
    assert item["kind"] == "normal"
    assert item["liked_count"] == "28"
    assert item["cover"].startswith("http://sns-webpic-qc.xhscdn.com/")


def test_parse_profile_counts_entries_it_cannot_collect():
    """缺令牌的条目列出来只会让人点了没反应，所以不显示——但必须计数，
    否则"收藏 20 条只列出 18 条"就成了静默丢数据。"""
    parsed = favorites.parse_profile(profile_payload(feeds=[
        feed_entry(),
        feed_entry(feed_id=OTHER_ID, token=""),
        feed_entry(feed_id="x", token=TOKEN),
    ]))
    assert [i["feed_id"] for i in parsed["items"]] == [FEED_ID]
    assert parsed["skipped"] == 2


def test_parse_profile_accepts_an_empty_favorites_list():
    """收藏夹是空的是一种正常结果，不能报错——否则用户会以为通道坏了。"""
    parsed = favorites.parse_profile(profile_payload(feeds=[]))
    assert parsed["items"] == [] and parsed["skipped"] == 0


def test_parse_profile_rejects_a_payload_without_feeds():
    with pytest.raises(favorites.FavoritesError) as missing:
        favorites.parse_profile({"userBasicInfo": {}})
    assert missing.value.code == "feeds_missing"


def test_parse_profile_rejects_a_non_object_payload():
    with pytest.raises(favorites.FavoritesError) as bad:
        favorites.parse_profile("未登录")
    assert bad.value.code == "payload_invalid"


def test_cover_url_only_accepts_the_image_host_whitelist():
    """封面是外部返回的地址，直接塞进页面等于给页面开了个口子。"""
    parsed = favorites.parse_profile(profile_payload(feeds=[
        feed_entry(cover="http://127.0.0.1:8011/notes"),
        feed_entry(feed_id=OTHER_ID, cover="javascript:alert(1)"),
    ]))
    assert [i["cover"] for i in parsed["items"]] == ["", ""]


# ---------------------------- 读收藏（后台任务） ----------------------------

def test_sync_lists_favorites_and_marks_ones_already_in_the_library(tmp_path, monkeypatch):
    use_mcp(monkeypatch)
    with TestClient(create_app(tmp_path / "fav.sqlite3")) as client:
        # 先导入一篇同编号的笔记，用来验证"已在库"的标注。
        link = favorites.note_link(FEED_ID, TOKEN)
        assert client.post("/notes/import", json={"share_text": link}).status_code == 200

        queued = client.post("/favorites/sync")
        assert queued.status_code == 202
        job = client.get(queued.json()["poll_url"]).json()

        assert job["status"] == "completed"
        assert job["progress"]["fetch"] == "completed"
        result = job["result"]
        assert result["profile"]["nickname"] == "。"
        assert result["items"][0]["feed_id"] == FEED_ID
        # 图像地址与会话令牌都在返回值里，前端勾选后要原样带回来。
        assert result["items"][0]["xsec_token"] == TOKEN
        assert result["items"][0]["in_library"] is True
        # 20 条的能力边界要如实转达给前端，不能让它自己猜。
        assert "20" in result["limit_note"]


def test_sync_failure_says_the_collector_is_down(tmp_path, monkeypatch):
    use_mcp(monkeypatch, favorites_error=xhs_mcp.XhsMcpUnavailable(
        "本地采集服务不可用：请确认 xiaohongshu-mcp 已启动（端口 18060）。"))
    with TestClient(create_app(tmp_path / "fav2.sqlite3")) as client:
        job = client.get(client.post("/favorites/sync").json()["poll_url"]).json()
    assert job["status"] == "failed"
    assert job["progress"]["fetch"] == "failed"
    assert "18060" in job["error"]
    assert job["result"] is None


def test_sync_reports_an_unreadable_payload_as_its_own_reason(tmp_path, monkeypatch):
    """服务返回的不是收藏列表（未登录、页面没渲染出子 tab）→ 说清原因，
    不要冒充"你的收藏是空的"。"""
    use_mcp(monkeypatch, favorites_payload={"something": "else"})
    with TestClient(create_app(tmp_path / "fav3.sqlite3")) as client:
        job = client.get(client.post("/favorites/sync").json()["poll_url"]).json()
    assert job["status"] == "failed"
    assert job["progress"]["error_code"] == "feeds_missing"
    assert "收藏" in job["error"]


# ---------------------------- 导入勾选项（后台任务） ----------------------------

def test_import_brings_the_selected_favorites_into_the_library(tmp_path, monkeypatch):
    use_mcp(monkeypatch)
    body = {"items": [{"feed_id": FEED_ID, "xsec_token": TOKEN}]}
    with TestClient(create_app(tmp_path / "imp.sqlite3")) as client:
        queued = client.post("/favorites/import", json=body)
        assert queued.status_code == 202
        assert queued.json()["total"] == 1
        job = client.get(queued.json()["poll_url"]).json()

        assert job["status"] == "completed"
        assert job["result"]["imported"] == 1
        assert job["result"]["failed"] == 0
        item = job["result"]["items"][0]
        assert item["status"] == "imported"
        assert item["note_id"] == FEED_ID
        assert item["title"] == "收藏导入的笔记"
        assert item["searchable"] is True
        # 笔记真的进库了，不是只回报了一句"成功"。
        assert client.get(f"/notes/{FEED_ID}").status_code == 200
        assert len(client.get("/notes").json()["items"]) == 1


def test_import_keeps_going_when_one_note_fails(tmp_path, monkeypatch):
    """一条失败不能带走整批，也不能让 job 永远停在 running（前端会一直转圈）。

    这里让坏笔记走"通道是通的、但这条的结果不能用"（内容受限 / 笔记已删除）——
    它不会触发熔断，所以下一条还能正常走 MCP。反过来，如果失败是**通道级**的
    （服务挂了），熔断器会立刻打开，整批快速失败并在 `collector_mcp` 里统一报因，
    那是有意为之：别对着一个死掉的服务连打二十次。
    """
    use_mcp(monkeypatch)
    # 网页解析也拿不到内容，这条才会彻底失败而不是"换条通道又成功了"。
    async def unusable_page(_url):
        return "<html>请登录</html>", _url
    monkeypatch.setattr(xhs_source, "fetch_page", unusable_page)

    async def flaky_detail(feed_id, xsec_token, **_kwargs):
        if feed_id == OTHER_ID:
            # 采集通了但返回里没有笔记内容 → collect_via_mcp 判定"这次结果不合规"，
            # 不熔断，回退网页解析；网页解析也拿不到，于是这条失败。
            return {"feed_id": feed_id, "data": {}}
        return detail_payload(feed_id)
    monkeypatch.setattr(xhs_source, "fetch_feed_detail", flaky_detail)

    body = {"items": [{"feed_id": OTHER_ID, "xsec_token": TOKEN},
                      {"feed_id": FEED_ID, "xsec_token": TOKEN}]}
    with TestClient(create_app(tmp_path / "imp2.sqlite3")) as client:
        job = client.get(client.post("/favorites/import", json=body).json()["poll_url"]).json()
        assert job["status"] == "partial"
        assert job["result"]["failed"] == 1
        assert job["result"]["imported"] == 1
        by_id = {i["feed_id"]: i for i in job["result"]["items"]}
        assert by_id[OTHER_ID]["status"] == "failed"
        assert by_id[OTHER_ID]["error"]
        assert by_id[FEED_ID]["status"] == "imported"
        # 坏的那条不该留在库里。
        assert client.get(f"/notes/{OTHER_ID}").status_code == 404


def test_import_dedupes_repeated_feed_ids(tmp_path, monkeypatch):
    use_mcp(monkeypatch)
    body = {"items": [{"feed_id": FEED_ID, "xsec_token": TOKEN},
                      {"feed_id": FEED_ID, "xsec_token": TOKEN}]}
    with TestClient(create_app(tmp_path / "imp3.sqlite3")) as client:
        queued = client.post("/favorites/import", json=body)
        assert queued.json()["total"] == 1
        job = client.get(queued.json()["poll_url"]).json()
    assert len(job["result"]["items"]) == 1


@pytest.mark.parametrize("body", [
    {"items": []},                                                   # 什么都没勾
    {"items": [{"feed_id": "short", "xsec_token": TOKEN}]},          # 编号太短
    {"items": [{"feed_id": "../../notes.sqlite3", "xsec_token": TOKEN}]},  # 编号会被拼进链接
    {"items": [{"feed_id": FEED_ID, "xsec_token": "x"}]},            # 令牌为空壳
    {"items": [{"feed_id": FEED_ID}]},                               # 缺令牌
])
def test_import_rejects_bad_input(tmp_path, body):
    """编号只允许字母数字：它会被拼进采集链接，放任任意字符串等于把
    "服务端替客户端访问任意地址"这条口子留在接口上。"""
    with TestClient(create_app(tmp_path / "bad.sqlite3")) as client:
        assert client.post("/favorites/import", json=body).status_code == 422


def test_import_caps_the_batch_size(tmp_path):
    items = [{"feed_id": f"{i:024d}", "xsec_token": TOKEN} for i in range(21)]
    with TestClient(create_app(tmp_path / "cap.sqlite3")) as client:
        assert client.post("/favorites/import", json={"items": items}).status_code == 422
