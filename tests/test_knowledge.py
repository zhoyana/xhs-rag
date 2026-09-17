"""LangChain 知识库层测试：验证中文分块、幂等索引和持久化。"""
from fastapi.testclient import TestClient

from app.database import NoteStore
from app.knowledge import CHUNK_SIZE, document_hash, split_note
from app.main import create_app


def sample_note(note_id="note-kb", repeat=80):
    return {
        "note_id": note_id,
        "title": "Agent 工程学习路线",
        "content": "。".join(["先理解检索，再完成评测，然后学习工具调用"] * repeat),
        "source_url": f"https://www.xiaohongshu.com/explore/{note_id}",
        "images": [],
    }


def test_langchain_chinese_splitter_keeps_metadata_and_order():
    # 秋招知识点：单元测试关注纯函数，不访问数据库，失败时更容易定位。
    document, chunks = split_note(sample_note())
    assert document.metadata["source_type"] == "xiaohongshu"
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= CHUNK_SIZE for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all("start_index" in chunk.metadata for chunk in chunks)
    assert not document.page_content.startswith("标题：")


def test_topics_become_metadata_instead_of_standalone_chunks():
    note = sample_note(repeat=2)
    note["content"] += "\n#时间序列[话题]##机器学习[话题]#"
    document, chunks = split_note(note)

    assert document.metadata["topics"] == ["时间序列", "机器学习"]
    assert "[话题]" not in document.page_content
    assert all("[话题]" not in chunk.page_content for chunk in chunks)
    assert all(chunk.page_content.strip() for chunk in chunks)
    # 标题仍计入幂等指纹：只改标题也必须重新建立向量。
    changed = sample_note(repeat=2)
    changed["title"] = "新的标题"
    assert document_hash(document) != document_hash(split_note(changed)[0])


def test_note_with_only_topics_falls_back_to_them_as_text():
    """正文只剩话题标签时，标签是这篇笔记仅有的文字线索，不能判成"正文为空"。

    小红书的面经、教程常把内容全画在长图里，desc 只剩标签。这种情况若直接
    判为空，索引会被跳过，用户收藏完一个字都搜不到。
    """
    note = {
        "note_id": "topics-only",
        "title": "淘宝 AI Agent 开发面经",
        "content": "#秋招[话题]# #面试[话题]# #AI产品经理[话题]#",
        "source_url": "https://www.xiaohongshu.com/explore/topics-only",
        "images": [],
    }

    document, chunks = split_note(note)

    assert "秋招" in document.page_content and "AI产品经理" in document.page_content
    # 标签仍然进 metadata，只是同时兜底当正文用。
    assert document.metadata["topics"] == ["秋招", "面试", "AI产品经理"]
    assert chunks


def test_image_text_is_indexed_but_not_the_header_alone():
    """图片文字并入可检索正文——这是图片承载型笔记能被搜到的前提。"""
    note = sample_note(repeat=2)
    note["content"] = "#面经[话题]#"
    note["image_text"] = "一面问了 ReAct 与 Function Calling 的执行链路。"

    document, chunks = split_note(note)

    assert "ReAct" in document.page_content
    assert "【图片文字】" in document.page_content
    assert any("ReAct" in chunk.page_content for chunk in chunks)


def test_index_is_idempotent_and_persistent(tmp_path):
    # 秋招知识点：这是集成测试，同时覆盖 HTTP、业务逻辑和临时 SQLite。
    db_path = tmp_path / "knowledge.sqlite3"
    NoteStore(db_path).save(sample_note())

    with TestClient(create_app(db_path)) as client:
        first = client.post("/notes/note-kb/index").json()
        second = client.post("/notes/note-kb/index").json()
        assert first["status"] == "indexed"
        assert second["status"] == "existing"
        assert first["document"]["id"] == second["document"]["id"]
        # 秋招知识点：重复请求 ID 不变，是接口幂等性的直接证据。
        chunks = client.get(
            f"/documents/{first['document']['id']}/chunks"
        ).json()["items"]
        assert len(chunks) == first["document"]["chunk_count"]

    # 新建应用模拟服务重启，SQLite 中的文档和 chunks 仍应存在。
    # 秋招知识点：持久化测试可防止误把关键状态只放在进程内存中。
    with TestClient(create_app(db_path)) as client:
        saved = client.get("/notes/note-kb/document")
        assert saved.status_code == 200
        assert saved.json()["chunk_count"] == len(chunks)


def test_missing_index_returns_404(tmp_path):
    db_path = tmp_path / "knowledge.sqlite3"
    NoteStore(db_path).save(sample_note(note_id="not-indexed", repeat=2))
    with TestClient(create_app(db_path)) as client:
        assert client.get("/notes/not-indexed/document").status_code == 404


def test_image_text_is_indexed_alongside_the_content():
    """图片承载型笔记：视觉转写结果必须进正文，否则收藏完根本搜不到。"""
    note = sample_note(repeat=2)
    note["image_text"] = "一面：讲讲 Redis 的持久化方案。"
    document, chunks = split_note(note)

    assert "Redis 的持久化方案" in document.page_content
    assert any("Redis 的持久化方案" in chunk.page_content for chunk in chunks)
    # 正文本身仍要保留，转写结果只是补充而不是替换。
    assert "先理解检索" in document.page_content


def test_image_only_note_can_still_be_documented():
    """正文只剩话题标签时，只要转写出文字就该能建文档，不再判为"正文为空"。"""
    note = sample_note(repeat=1)
    note["content"] = "#面试[话题]#"
    note["image_text"] = "面试问题：如何设计一个检索评测集？"
    document, chunks = split_note(note)

    assert document.page_content.strip()
    assert chunks
    assert all(chunk.page_content.strip() for chunk in chunks)
