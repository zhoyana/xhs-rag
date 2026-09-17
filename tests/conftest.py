"""全局测试约定。

这个项目的测试必须离线且可重复：单测不依赖本机 18060 端口上有没有
xiaohongshu-mcp 在跑，也不依赖小红书页面当前长什么样。站外依赖统一在
网络边界处替换掉，业务逻辑本身仍然被完整执行。
"""
import pytest

from app import xhs_mcp, xhs_source


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
