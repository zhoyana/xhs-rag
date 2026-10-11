"""Agent 侧的收藏夹工具：list_my_favorites（读）与 import_my_favorites（写）。

用例不启 18060 上的采集服务、也不访问小红书：`fetch_my_favorites` 与
`get_feed_detail` 两层都在用例里换成替身，图片下载换成不落盘的替身。

这里最要紧的一条不变量是**会话令牌不进模型上下文**：`xsec_token` 是 44 字符
base64，必须逐字符正确，让模型抄一遍就是给幻觉留口子。所以列表工具返回里不该
出现它，导入工具的入参里也不该出现它——两个方向各有一条用例守着。
"""
import asyncio
import json

from app import favorites, xhs_mcp
from app.database import NoteStore

FEED_A = "6ab2412a000000003703c455"
FEED_B = "6ab2412a000000003703c456"
FEED_C = "6ab2412a000000003703c457"
TOKEN = "ABIOjBXzgMip-Ka1BwWUKH47OzULHVONZxapoF3ej1x7w="
NEVER_LISTED = "6ab2412a000000003703ffff"


def run(coro):
    return asyncio.run(coro)


def feed_entry(feed_id, title, token=TOKEN):
    return {
        "xsecToken": token, "id": feed_id, "index": 0,
        "noteCard": {
            "type": "normal", "displayTitle": title,
            "user": {"nickname": "Offer学姐柠檬"},
            "interactInfo": {"likedCount": "28"},
            "cover": {"urlDefault": "http://sns-webpic-qc.xhscdn.com/a/cover!nd_dft_wlteh_webp_3"},
        },
    }


def profile_payload(*pairs):
    """pairs: (feed_id, title)；不传就是空收藏夹。"""
    return {
        "userBasicInfo": {"nickname": "。", "ipLocation": "浙江"},
        "interactions": [{"type": "interaction", "name": "获赞与收藏", "count": "344"}],
        "feeds": [feed_entry(fid, title) for fid, title in pairs],
    }


def detail_text(feed_id, title="收藏导入的笔记", images=0):
    note = {
        "noteId": feed_id, "title": title,
        "desc": "一面问了 ReAct 与 Function Calling 的执行链路。",
        "type": "normal",
        "imageList": [{"urlDefault": f"http://sns-webpic-qc.xhscdn.com/a/{i}!nd_dft_wlteh_webp_3"}
                      for i in range(1, images + 1)],
    }
    return json.dumps({"feed_id": feed_id, "data": {"note": note, "comments": {"list": []}}})


class FakeDetailTool:
    """get_feed_detail 替身：按 feed_id 回内容，也可以回一个异常。

    `default` 可以是一个接收 args 的函数——采集回来的 note 编号必须与被请求的
    编号一致（真实服务就是这么回的），固定返回某一篇会让"导 A 存成了 B"这种
    错误在用例里看不出来。
    """

    def __init__(self, replies=None, default=None):
        self.replies = dict(replies or {})
        self.default = default
        self.calls = []

    async def ainvoke(self, args):
        self.calls.append(args)
        reply = self.replies.get(args["feed_id"], self.default)
        if callable(reply):
            reply = reply(args)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def echo_feed_ids(title="收藏导入的笔记", images=0):
    """默认替身：回一篇与被请求编号同名同编号的笔记。"""
    return lambda args: detail_text(args["feed_id"], title, images=images)


async def fake_transcribe(images, title="", **_kwargs):
    return {"status": "ok", "text": "（测试替身转写）", "images": [], "model": "fake",
            "read": len(images), "dropped": 0, "error": None}


def fake_download(store, note, root):
    total = len(note.get("images", []))
    return {"status": "complete", "total": total, "downloaded": total,
            "skipped": 0, "failed": 0}


def ok_indexer(store, note_id):
    return {"status": "indexed", "note_id": note_id, "searchable": True, "chunk_count": 1}


def build(store, media_root, *, fetch=None, detail=None, indexer=ok_indexer, cache=None):
    """返回 {工具名: 工具} 与用到的假采集器，方便断言调用参数。

    默认给一个全新的令牌缓存：模块级那个是**跨对话**共享的会话状态，用例之间
    共用它会让"没读过列表就导入"这类断言随着执行顺序随机通过或失败。
    """
    fake_detail = detail if detail is not None else FakeDetailTool(default=echo_feed_ids())
    built = xhs_mcp.build_favorites_tools(
        fake_detail, store, media_root, fetch=fetch, indexer=indexer,
        transcribe=fake_transcribe, downloader=fake_download,
        cache=cache if cache is not None else favorites.TokenCache())
    return {tool.name: tool for tool in built}, fake_detail


async def favorites_fetch(payload):
    return payload


def fetch_payload(payload):
    """构造一个「读到就返回这份 payload」的替身。"""
    async def fetch():
        return payload
    return fetch


def fetch_ok(*pairs):
    async def fetch():
        return profile_payload(*pairs)
    return fetch


async def store_save(store, feed_id, title):
    """把一篇笔记直接写进库，用来构造"已经在知识库里"的前置状态。"""
    from app.xhs_link import note_from_feed_detail
    note = note_from_feed_detail(json.loads(detail_text(feed_id, title)))
    store.save(note)
    return note


# ---------------------------- 读收藏 ----------------------------

def test_list_returns_the_list_but_never_the_token(tmp_path):
    """令牌必须留在服务端：它逐字符正确才有用，抄错一个字符表现成"采集失败"。"""
    store = NoteStore(tmp_path / "list.sqlite3")
    tools, _detail = build(store, tmp_path / "media",
                           fetch=fetch_ok((FEED_A, "珀莱雅AI面试高频题"), (FEED_B, "企业知识库怎么搭")))
    result = run(tools["list_my_favorites"].ainvoke({}))

    assert result["ok"] is True
    data = result["data"]
    assert [item["feed_id"] for item in data["items"]] == [FEED_A, FEED_B]
    assert data["items"][0]["title"] == "珀莱雅AI面试高频题"
    assert data["items"][0]["author"] == "Offer学姐柠檬"
    assert data["items"][0]["in_library"] is False
    assert data["in_library"] == 0
    # 唯一一句真正重要的话：整份返回值里不该出现令牌。
    assert TOKEN not in json.dumps(data, ensure_ascii=False)
    assert "xsec_token" not in json.dumps(data, ensure_ascii=False)


def test_list_marks_what_is_already_in_the_library(tmp_path):
    """已经入库的条目要标出来：模型才知道"同步一下"其实没有新东西可导。"""
    store = NoteStore(tmp_path / "marked.sqlite3")
    tools, _detail = build(store, tmp_path / "media",
                           fetch=fetch_ok((FEED_A, "旧的面经"), (FEED_B, "新的面经")))
    run(store_save(store, FEED_A, "旧的面经"))

    data = run(tools["list_my_favorites"].ainvoke({}))["data"]
    by_id = {item["feed_id"]: item for item in data["items"]}
    assert by_id[FEED_A]["in_library"] is True
    assert by_id[FEED_B]["in_library"] is False
    assert data["in_library"] == 1


def test_list_carries_the_capability_limit_into_the_result(tmp_path):
    """读不全这件事必须跟着结果回到模型手里，否则它会替服务端吹牛。"""
    store = NoteStore(tmp_path / "limit.sqlite3")
    tools, _detail = build(store, tmp_path / "media", fetch=fetch_ok((FEED_A, "面经")))
    data = run(tools["list_my_favorites"].ainvoke({}))["data"]

    assert "20" in data["limit_note"]
    assert "收藏" in data["limit_note"]


def test_list_accepts_an_empty_favorites_list(tmp_path):
    """收藏夹是空的是一种正常结果，不能报错——否则模型会说通道坏了。"""
    store = NoteStore(tmp_path / "empty.sqlite3")
    tools, _detail = build(store, tmp_path / "media", fetch=fetch_ok())
    result = run(tools["list_my_favorites"].ainvoke({}))

    assert result["ok"] is True
    assert result["data"]["items"] == []


def test_list_reports_the_collector_being_down_as_its_own_code(tmp_path):
    store = NoteStore(tmp_path / "down.sqlite3")

    async def fetch():
        raise xhs_mcp.XhsMcpUnavailable(
            "本地采集服务不可用：请确认 xiaohongshu-mcp 已启动（端口 18060）。")

    tools, _detail = build(store, tmp_path / "media", fetch=fetch)
    result = run(tools["list_my_favorites"].ainvoke({}))

    assert result["ok"] is False
    assert result["error"]["code"] == "favorites_unavailable"
    assert "18060" in result["error"]["message"]


def test_list_does_not_pretend_an_unreadable_payload_is_an_empty_list(tmp_path):
    """服务没渲染出子 tab 时返回的不是列表；说成"你没有收藏"就是在骗用户。"""
    store = NoteStore(tmp_path / "bad.sqlite3")
    tools, _detail = build(store, tmp_path / "media", fetch=fetch_payload({"something": "else"}))
    result = run(tools["list_my_favorites"].ainvoke({}))

    assert result["ok"] is False
    assert result["error"]["code"] == "feeds_missing"


# ---------------------------- 导入收藏 ----------------------------

def test_import_without_a_recent_list_is_refused(tmp_path):
    """没有列表就没有令牌；宁可让模型重读一次，也不要拿空令牌去采集。"""
    store = NoteStore(tmp_path / "nolist.sqlite3")
    tools, _detail = build(store, tmp_path / "media", fetch=fetch_ok((FEED_A, "面经")))
    result = run(tools["import_my_favorites"].ainvoke({}))

    assert result["ok"] is False
    assert result["error"]["code"] == "favorites_not_listed"
    assert store.get(FEED_A) is None


def test_import_takes_everything_not_in_the_library_yet(tmp_path):
    """feed_ids 留空 = 全都要：这是"把新收藏同步一下"的默认路径。"""
    store = NoteStore(tmp_path / "all.sqlite3")
    tools, detail = build(
        store, tmp_path / "media",
        fetch=fetch_ok((FEED_A, "旧的面经"), (FEED_B, "新的面经")),
        detail=FakeDetailTool(replies={FEED_A: detail_text(FEED_A, "旧的面经"),
                                       FEED_B: detail_text(FEED_B, "新的面经")}))
    run(store_save(store, FEED_A, "旧的面经"))
    run(tools["list_my_favorites"].ainvoke({}))

    result = run(tools["import_my_favorites"].ainvoke({}))

    assert result["ok"] is True
    data = result["data"]
    assert data["requested"] == 2
    assert data["imported"] == 1
    assert data["existing"] == 1
    assert data["failed"] == 0
    assert data["not_attempted"] == 0
    # 已在库的那条不该再去采集一次（白跑一次采集 + 一次视觉调用）。
    assert [call["feed_id"] for call in detail.calls] == [FEED_B]
    assert store.get(FEED_B)["title"] == "新的面经"


def test_import_only_the_ids_the_model_asked_for(tmp_path):
    """模型只报编号，令牌由服务端接上——这是"让 Agent 挑着导"的关键一步。"""
    store = NoteStore(tmp_path / "pick.sqlite3")
    tools, detail = build(
        store, tmp_path / "media",
        fetch=fetch_ok((FEED_A, "面经"), (FEED_B, "转租"), (FEED_C, "小沙发")),
        detail=FakeDetailTool(default=echo_feed_ids()))

    run(tools["list_my_favorites"].ainvoke({}))
    result = run(tools["import_my_favorites"].ainvoke({"feed_ids": [FEED_B]}))

    assert result["data"]["requested"] == 1
    assert result["data"]["imported"] == 1
    assert store.get(FEED_B) is not None
    assert store.get(FEED_A) is None and store.get(FEED_C) is None
    assert [call["feed_id"] for call in detail.calls] == [FEED_B]


def test_import_rejects_ids_that_were_never_listed(tmp_path):
    """编出来的编号要当场拒绝：放它过去只会得到一句含糊的"采集失败"，
    而真实原因是模型把编号记错了，那是最难查的一类故障。"""
    store = NoteStore(tmp_path / "bogus.sqlite3")
    tools, detail = build(store, tmp_path / "media", fetch=fetch_ok((FEED_A, "面经")))

    run(tools["list_my_favorites"].ainvoke({}))
    result = run(tools["import_my_favorites"].ainvoke({"feed_ids": [NEVER_LISTED]}))

    assert result["ok"] is False
    assert result["error"]["code"] == "favorites_unknown_id"
    assert NEVER_LISTED in result["error"]["message"]
    assert detail.calls == []
    assert store.get(NEVER_LISTED) is None


def test_import_loads_the_token_it_remembered(tmp_path):
    """真正采到的是列表那一次拿到的令牌，而不是模型（或别处）编的。"""
    store = NoteStore(tmp_path / "token.sqlite3")
    tools, detail = build(store, tmp_path / "media", fetch=fetch_ok((FEED_A, "面经")))

    run(tools["list_my_favorites"].ainvoke({}))
    run(tools["import_my_favorites"].ainvoke({}))

    assert detail.calls[0]["xsec_token"] == TOKEN
    assert detail.calls[0]["feed_id"] == FEED_A


def test_import_reports_unsearchable_notes_without_calling_them_failures(tmp_path):
    """纯图片且没识别出文字 → 存下来了、但搜不到。这是 skipping，不是 failure。"""
    store = NoteStore(tmp_path / "skip.sqlite3")

    def skipping_indexer(store_, note_id):
        return {"status": "skipped", "note_id": note_id, "searchable": False,
                "reason": "没有可检索的正文。"}

    tools, _detail = build(store, tmp_path / "media", fetch=fetch_ok((FEED_A, "纯图笔记")),
                           indexer=skipping_indexer)
    run(tools["list_my_favorites"].ainvoke({}))
    data = run(tools["import_my_favorites"].ainvoke({}))["data"]

    assert data["imported"] == 1
    assert data["failed"] == 0
    item = data["items"][0]
    assert item["status"] == "imported"
    assert item["searchable"] is False
    assert item["index_status"] == "skipped"
    assert store.get(FEED_A) is not None


def test_import_reports_the_image_count(tmp_path):
    """图片有没有落盘是用户最关心的事之一，必须回传给模型如实汇报。"""
    store = NoteStore(tmp_path / "img.sqlite3")
    tools, _detail = build(store, tmp_path / "media", fetch=fetch_ok((FEED_A, "图文笔记")),
                           detail=FakeDetailTool(default=echo_feed_ids(images=3)))
    run(tools["list_my_favorites"].ainvoke({}))
    item = run(tools["import_my_favorites"].ainvoke({}))["data"]["items"][0]

    assert item["images"] == {"total": 3, "saved": 3, "failed": 0}


def test_import_keeps_going_when_one_note_is_unusable(tmp_path):
    """一条采不到不能带走整批——但通道级故障是另一回事（见下一个用例）。"""
    store = NoteStore(tmp_path / "partial.sqlite3")
    tools, _detail = build(
        store, tmp_path / "media",
        fetch=fetch_ok((FEED_A, "坏的"), (FEED_B, "好的")),
        # 采集通了但返回里没有笔记内容 → 不熔断，判定"这次结果不合规"。
        detail=FakeDetailTool(replies={FEED_A: json.dumps({"feed_id": FEED_A, "data": {}}),
                                       FEED_B: detail_text(FEED_B, "好的")}))
    run(tools["list_my_favorites"].ainvoke({}))
    data = run(tools["import_my_favorites"].ainvoke({}))["data"]

    assert data["imported"] == 1
    assert data["failed"] == 1
    by_id = {item["feed_id"]: item for item in data["items"]}
    assert by_id[FEED_A]["status"] == "failed" and by_id[FEED_A]["error"]
    assert by_id[FEED_B]["status"] == "imported"
    assert store.get(FEED_A) is None


def test_import_stops_when_the_channel_dies_midway(tmp_path):
    """通道挂了就停：对着一个死掉的服务连打十几次，用户只会白等几分钟。

    停的时候必须把「还剩几条没试」一起报回去，否则模型会把"试了 1 条"讲成
    "收藏里只有 1 条"。
    """
    store = NoteStore(tmp_path / "channel.sqlite3")
    tools, _detail = build(
        store, tmp_path / "media",
        fetch=fetch_ok((FEED_A, "第一条"), (FEED_B, "第二条"), (FEED_C, "第三条")),
        detail=FakeDetailTool(replies={
            FEED_B: RuntimeError("connection reset"),  # ainvoke 抛错 → 通道级失败
        }, default=echo_feed_ids("第一条")))
    run(tools["list_my_favorites"].ainvoke({}))
    data = run(tools["import_my_favorites"].ainvoke({}))["data"]

    assert data["imported"] == 1
    assert data["failed"] == 1
    assert data["attempted"] == 2
    assert data["not_attempted"] == 1
    assert data["channel_error"]
    assert store.get(FEED_A) is not None
    assert store.get(FEED_C) is None


def test_import_refuses_a_stale_list_instead_of_using_an_expired_token(tmp_path):
    """列表读于半小时前 → 令牌可能已失效，重读一次比拿过期令牌碰运气更省事。"""
    store = NoteStore(tmp_path / "stale.sqlite3")
    now = [1000.0]
    cache = favorites.TokenCache(ttl=60, clock=lambda: now[0])
    tools, _detail = build(store, tmp_path / "media", fetch=fetch_ok((FEED_A, "面经")),
                           cache=cache)

    run(tools["list_my_favorites"].ainvoke({}))
    now[0] += 61
    result = run(tools["import_my_favorites"].ainvoke({}))

    assert result["ok"] is False
    assert result["error"]["code"] == "favorites_not_listed"
    assert cache.feed_ids() == []


# ---------------------------- 接口面 ----------------------------

def test_the_import_tool_has_no_token_parameter():
    """把"令牌不进模型"钉在 schema 上：以后谁想加个 token 参数，这条用例会拦下来。"""
    schema = xhs_mcp.ImportFavoritesInput.model_json_schema()
    assert set(schema["properties"]) == {"feed_ids"}


def test_the_list_tool_takes_no_parameters():
    """读的就是当前登录账号自己的收藏，没有可调的开关——省得模型发明 tab 参数。"""
    assert xhs_mcp.ListFavoritesInput.model_json_schema()["properties"] == {}


def test_favorites_tools_are_mounted_only_when_there_is_a_library(monkeypatch):
    """写工具随 store 一起挂载：没有知识库就没有"导入收藏夹"这个入口。"""
    from langchain_core.tools import StructuredTool

    async def fake_cached():
        # load_xhs_tools 按名字找 get_feed_detail，所以这里得是个真的工具对象。
        return [StructuredTool.from_function(
            func=lambda feed_id, xsec_token="": "", name="get_feed_detail",
            description="测试替身")]

    monkeypatch.setattr(xhs_mcp, "_cached_tools", fake_cached)
    read_only = run(xhs_mcp.load_xhs_tools())
    with_library = run(xhs_mcp.load_xhs_tools(store=NoteStore(":memory:"), media_root=None))

    assert "import_my_favorites" not in {tool.name for tool in read_only}
    assert "list_my_favorites" not in {tool.name for tool in read_only}
    names = {tool.name for tool in with_library}
    assert {"list_my_favorites", "import_my_favorites"} <= names


def test_reset_cache_forgets_remembered_tokens(tmp_path):
    """令牌是进程内的会话状态：不清掉的话，"没读过列表就导入"会随执行顺序随机通过。"""
    store = NoteStore(tmp_path / "reset.sqlite3")
    tools, _detail = build(store, tmp_path / "media", fetch=fetch_ok((FEED_A, "面经")),
                           cache=xhs_mcp._favorites_cache)  # 显式用模块级那个

    run(tools["list_my_favorites"].ainvoke({}))
    assert xhs_mcp._favorites_cache.feed_ids() == [FEED_A]

    xhs_mcp.reset_cache()
    assert xhs_mcp._favorites_cache.feed_ids() == []


# ---------------------------- 辅助 ----------------------------
