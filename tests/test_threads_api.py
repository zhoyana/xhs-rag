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
         "retrieved_note_ids": ["n1"], "cited_note_ids": ["n1"], "plan_id": None},
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
