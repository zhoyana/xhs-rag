"""全局测试约定。

这个项目的测试必须离线且可重复：单测不依赖本机 18060 端口上有没有
xiaohongshu-mcp 在跑，也不依赖小红书页面当前长什么样。站外依赖统一在
网络边界处替换掉，业务逻辑本身仍然被完整执行。
"""
import pytest

from app import indexing, media, xhs_mcp, xhs_source


@pytest.fixture(autouse=True)
def _offline_mcp_channel(monkeypatch):
    """默认把 MCP 采集通道关掉。

    替换的是最内层的网络调用 fetch_feed_detail，而不是 collect_via_mcp 本身：
    这样"先试 MCP、失败回退网页解析"的判定逻辑在每条用例里都真的跑了一遍。
    需要验证 MCP 成功路径的用例，自己重新 patch 这个函数即可。
    """
    xhs_source.reset_breaker()

    async def unavailable(*_args, **_kwargs):
        raise xhs_mcp.XhsMcpUnavailable("测试环境不连站外 MCP。")

    monkeypatch.setattr(xhs_source, "fetch_feed_detail", unavailable)
    yield
    xhs_source.reset_breaker()


@pytest.fixture(autouse=True)
def _offline_indexer(monkeypatch):
    """默认不加载真实的 Embedding 模型。

    所有写路径（收件箱导入、批量导入、刷新正文、Agent 收藏）保存后都会
    "建索引 + 生成向量"，而真实模型要下载权重、推理耗时数秒，单测必须离线
    且秒级完成。这里替换的是 app.indexing 里的可替换默认索引器——替换点只有
    一个，所以四条路径一次全覆盖，不会出现某条路径漏了替身而偷偷加载模型。

    需要验证真实索引逻辑的用例直接测 app.indexing（注入假向量函数），
    或自己给工具传 indexer。
    """
    def fake_index(store, note_id, **_kwargs):
        return {"status": "indexed", "note_id": note_id, "chunk_count": 1,
                "vector_total": 1, "searchable": True}

    monkeypatch.setattr(indexing, "DEFAULT_INDEXER", fake_index)
    yield


@pytest.fixture(autouse=True)
def _clean_image_stash():
    """图片字节暂存是进程级的，用例之间必须互不影响。

    它跨越"读图"和"落盘"两个阶段，一旦某条用例留下的字节被下一条取走，
    就会出现"这条用例明明没走网络却成功了"的假象。
    """
    media.clear_stashed_images()
    yield
    media.clear_stashed_images()


@pytest.fixture(autouse=True)
def _offline_transcriber(monkeypatch):
    """默认不调用视觉模型。

    采集层会给"内容全在图片里"的笔记补上图片文字，那要调真实视觉 API 并下载
    图片。单测必须离线，这里换成固定替身；需要验证转写落地格式的用例直接测
    xhs_vision.apply_image_text，或自己给 collect_note 传 transcribe。
    """
    async def fake_transcribe(images, **_kwargs):
        return {"status": "ok", "text": "图片里识别出的文字", "model": "fake",
                "error": None, "read": len(images), "dropped": 0}

    monkeypatch.setattr(xhs_source, "DEFAULT_TRANSCRIBE", fake_transcribe)
    yield
