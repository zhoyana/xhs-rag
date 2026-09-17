"""会话列表 / 重命名 / 删除 + SSE 流式对话的接口测试。

流式接口是前端工作台的命脉：既要按 SSE 帧把工具步骤与文本增量推出去，
又要在收尾时把助手消息落库。这两件事坏任何一件，界面就会表现成
"一直转圈不出字"，或者"聊完了但历史里没有"。所以两者都在这里钉住。
"""
import json

from fastapi.testclient import TestClient

from app.main import create_app


def parse_sse(text):
    """把 SSE 响应体拆成事件列表，顺带保证帧格式是 data: + 空行。"""
    events = []
    for frame in text.split("\n\n"):
        line = next((l for l in frame.split("\n") if l.startswith("data:")), None)
        if line:
            events.append(json.loads(line[5:].strip()))
    return events


def fake_agent(content="收到"):
    """非流式入口的替身；只看编排层怎么用返回值，不碰真实模型。"""
    async def _agent(history, tools, model=None, max_rounds=6, required_tool=None):
        return {"content": content, "tool_calls": [], "retrieved_note_ids": [],
                "cited_note_ids": [], "source_note_ids": [], "plan_id": None}
    return _agent


def fake_stream(events, captured=None):
    """流式入口的替身：原样吐事件，并把上下文留给用例断言。"""
    async def _stream(history, tools, model=None, max_rounds=6, required_tool=None):
        if captured is not None:
            captured["history"] = history
            captured["required_tool"] = required_tool
            captured["tool_names"] = [getattr(tool, "name", None) for tool in tools]
        for event in events:
            yield event
    return _stream


def test_list_threads_carries_message_count_and_preview(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.run_agent_turn", fake_agent("第一轮回答"))
    with TestClient(create_app(tmp_path / "list.sqlite3")) as client:
        first = client.post("/threads", json={"title": "会话一"}).json()["id"]
        client.post(f"/threads/{first}/messages", json={"content": "第一个问题"})
        second = client.post("/threads", json={"title": "会话二"}).json()["id"]

        items = client.get("/threads").json()["items"]
        by_id = {item["id"]: item for item in items}
        assert set(by_id) == {first, second}
        assert by_id[first]["message_count"] == 2
        assert by_id[first]["last_message"] == "第一轮回答"
        assert by_id[first]["last_role"] == "assistant"
        assert by_id[second]["message_count"] == 0
        assert by_id[second]["last_message"] == ""
        # 最近活跃的排最前：前端侧边栏直接按这个顺序渲染。
        assert items[0]["id"] == second


def test_last_message_preview_is_single_line_and_capped(tmp_path, monkeypatch):
    messy = "第一行\n第二行\t带制表符 " + "很长的内容" * 30
    monkeypatch.setattr("app.main.run_agent_turn", fake_agent(messy))
    with TestClient(create_app(tmp_path / "preview.sqlite3")) as client:
        thread_id = client.post("/threads", json={"title": "预览"}).json()["id"]
        client.post(f"/threads/{thread_id}/messages", json={"content": "写点长的"})
        item = client.get("/threads").json()["items"][0]
        assert "\n" not in item["last_message"] and "\t" not in item["last_message"]
        assert len(item["last_message"]) == 80


def test_rename_and_delete_thread(tmp_path):
    with TestClient(create_app(tmp_path / "crud.sqlite3")) as client:
        thread_id = client.post("/threads", json={"title": "旧标题"}).json()["id"]

        renamed = client.patch(f"/threads/{thread_id}", json={"title": "新标题"})
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "新标题"
        assert client.get(f"/threads/{thread_id}").json()["title"] == "新标题"

        assert client.delete(f"/threads/{thread_id}").status_code == 200
        assert client.get(f"/threads/{thread_id}").status_code == 404
        # 删过的会话不应再出现在列表里，重复删除要明确 404。
        assert client.get("/threads").json()["items"] == []
        assert client.delete(f"/threads/{thread_id}").status_code == 404
        assert client.patch(f"/threads/{thread_id}", json={"title": "x"}).status_code == 404


def test_delete_thread_also_removes_messages(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.run_agent_turn", fake_agent("答复"))
    db_path = tmp_path / "cascade.sqlite3"
    with TestClient(create_app(db_path)) as client:
        thread_id = client.post("/threads", json={"title": "级联"}).json()["id"]
        client.post(f"/threads/{thread_id}/messages", json={"content": "在吗"})
        client.delete(f"/threads/{thread_id}")

    # 重建 app 再查：消息表里的孤儿行应随会话一起删掉，否则会泄漏到下次统计。
    with TestClient(create_app(db_path)) as client:
        assert client.get("/threads").json()["items"] == []
        new_id = client.post("/threads", json={"title": "新的"}).json()["id"]
        assert client.get(f"/threads/{new_id}").json()["messages"] == []


def test_message_limit_window_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.run_agent_turn", fake_agent("收到"))
    with TestClient(create_app(tmp_path / "window.sqlite3")) as client:
        thread_id = client.post("/threads", json={"title": "窗口"}).json()["id"]
        for i in range(3):
            client.post(f"/threads/{thread_id}/messages", json={"content": f"问题 {i}"})

        # 不传参数沿用旧的 20 条窗口，行为不变。
        assert len(client.get(f"/threads/{thread_id}").json()["messages"]) == 6
        recent = client.get(f"/threads/{thread_id}?message_limit=2").json()["messages"]
        assert [m["content"] for m in recent] == ["问题 2", "收到"]


def test_stream_endpoint_emits_events_and_persists_reply(tmp_path, monkeypatch):
    events = [
        {"type": "tool_start", "name": "search_notes", "args": {"query": "面试"}},
        {"type": "tool_end", "name": "search_notes", "args": {"query": "面试"}, "ok": True,
         "duration_ms": 12.0, "result_count": 3, "top_score": 0.81, "error_code": None},
        {"type": "delta", "text": "你收藏的"},
        {"type": "delta", "text": "面经共 3 篇。"},
        {"type": "done", "content": "你收藏的面经共 3 篇。",
         "tool_calls": [{"name": "search_notes", "args": {"query": "面试"}, "ok": True,
                         "duration_ms": 12.0, "result_count": 3, "top_score": 0.81,
                         "error_code": None}],
         "retrieved_note_ids": ["n1"], "cited_note_ids": ["n1"], "plan_id": None,
         # 「检索」视图并进对话后，前端展开的「原始命中」就靠这份片段。
         "retrieved_hits": [{"note_id": "n1", "title": "面经", "score": 0.81,
                             "snippet": "一面问了 RAG 的召回策略。"}]},
    ]
    captured = {}
    monkeypatch.setattr("app.main.run_agent_turn_stream", fake_stream(events, captured))

    with TestClient(create_app(tmp_path / "stream.sqlite3")) as client:
        thread_id = client.post("/threads", json={"title": "流式"}).json()["id"]
        response = client.post(f"/threads/{thread_id}/messages/stream",
                               json={"content": "面经里考什么"})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        got = parse_sse(response.text)
        # start 由接口自己发，其余按 agent 吐出的事件顺序原样透传。
        assert [e["type"] for e in got] == [
            "start", "tool_start", "tool_end", "delta", "delta", "done"]
        assert got[-1]["message"]["content"] == "你收藏的面经共 3 篇。"
        # 模型看到的历史必须含刚发出的提问，否则这一轮就是"没听见用户说话"。
        assert captured["history"][-1]["content"] == "面经里考什么"
        # 普通咨询看不到写入型工具，权限收口在编排层。
        assert "draft_plan" not in captured["tool_names"]

        thread = client.get(f"/threads/{thread_id}").json()
        assert [m["role"] for m in thread["messages"]] == ["user", "assistant"]
        assert thread["messages"][1]["metadata"]["cited_note_ids"] == ["n1"]
        assert thread["messages"][1]["metadata"]["tool_calls"][0]["result_count"] == 3
        # 命中片段随回答一起落库：重新打开会话时「原始命中」还能展开，
        # 不需要为此再跑一次检索（也就不会再产生一次 embedding 费用）。
        assert thread["messages"][1]["metadata"]["retrieved_hits"][0]["snippet"] == \
            "一面问了 RAG 的召回策略。"


def test_stream_plan_request_grants_draft_tool(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.main.run_agent_turn_stream",
        fake_stream([{"type": "done", "content": "草稿已生成", "tool_calls": [],
                      "retrieved_note_ids": [], "cited_note_ids": [], "plan_id": 7}], captured),
    )
    with TestClient(create_app(tmp_path / "plan.sqlite3")) as client:
        thread_id = client.post("/threads", json={"title": "计划"}).json()["id"]
        response = client.post(f"/threads/{thread_id}/messages/stream",
                               json={"content": "帮我制定一版一周面试准备计划"})
        assert response.status_code == 200
        # 明确要计划时才把 draft_plan 放进工具面，并要求模型必须调用它。
        assert "draft_plan" in captured["tool_names"]
        assert captured["required_tool"] == "draft_plan"
        assert parse_sse(response.text)[-1]["message"]["metadata"]["plan_id"] == 7


def test_stream_failure_keeps_user_message(tmp_path, monkeypatch):
    async def broken(history, tools, model=None, max_rounds=6, required_tool=None):
        yield {"type": "delta", "text": "话说了一半"}
        raise RuntimeError("模型挂了")

    monkeypatch.setattr("app.main.run_agent_turn_stream", broken)
    with TestClient(create_app(tmp_path / "broken.sqlite3")) as client:
        thread_id = client.post("/threads", json={"title": "失败"}).json()["id"]
        response = client.post(f"/threads/{thread_id}/messages/stream",
                               json={"content": "在吗"})

        assert response.status_code == 200
        got = parse_sse(response.text)
        assert got[-1]["type"] == "error"
        # 流已经开始就改不了状态码，但用户的提问必须留下、失败轮不写助手消息。
        thread = client.get(f"/threads/{thread_id}").json()
        assert [m["role"] for m in thread["messages"]] == ["user"]
        assert thread["messages"][0]["content"] == "在吗"


def test_stream_unknown_thread_returns_404(tmp_path):
    with TestClient(create_app(tmp_path / "missing.sqlite3")) as client:
        response = client.post("/threads/not-a-thread/messages/stream",
                               json={"content": "在吗"})
        assert response.status_code == 404


def test_plan_tool_stays_granted_only_across_the_clarifying_turn(tmp_path, monkeypatch):
    """计划工具的授权要跟着"会话状态"走，而不是只看最后一句。

    agent 追问目标后，用户的回答里不再有"计划"二字。此时若把工具收回，计划就
    只能停在聊天记录里；但也不能反过来一直开着，否则模型会自己冒出来起草计划。
    """
    captured = []
    turns = [
        # 明确要计划 → 授权
        {"user": "帮我制定一份一周面试准备计划", "reply": "你每天能投入多少时间？"},
        # 回答追问（不含"计划"）→ 仍要授权，否则草案落不了库
        {"user": "每天两小时，从周五开始", "reply": "草案已经保存，请确认后再写入日历。"},
        # 无关提问 → 必须收回工具，避免模型自作主张起草
        {"user": "RAG 的召回策略有哪些？", "reply": "常见的是向量召回加关键词召回。"},
    ]

    def make_stream(reply):
        async def _stream(history, tools, model=None, max_rounds=6, required_tool=None):
            captured.append({
                "reply": reply,
                "required_tool": required_tool,
                "tools": [getattr(tool, "name", None) for tool in tools],
            })
            yield {"type": "done", "content": reply, "tool_calls": [],
                   "retrieved_note_ids": [], "cited_note_ids": [], "plan_id": None}
        return _stream

    with TestClient(create_app(tmp_path / "plan.sqlite3")) as client:
        thread_id = client.post("/threads", json={"title": "商量计划"}).json()["id"]
        for turn in turns:
            monkeypatch.setattr("app.main.run_agent_turn_stream", make_stream(turn["reply"]))
            client.post(f"/threads/{thread_id}/messages/stream", json={"content": turn["user"]})

    assert [item["required_tool"] for item in captured[:2]] == ["draft_plan", "draft_plan"]
    assert "draft_plan" in captured[0]["tools"]
    assert "draft_plan" in captured[1]["tools"]
    # 第三轮是普通提问：既不强制，模型也根本看不见这个工具。
    assert captured[2]["required_tool"] is None
    assert "draft_plan" not in captured[2]["tools"]
