"""xiaohongshu-mcp 桥接：把外部 MCP 服务的读类工具挂进本地 Agent。

xiaohongshu-mcp（github.com/xpzouying/xiaohongshu-mcp）以 Streamable HTTP
协议提供 MCP 服务（默认 http://localhost:18060/mcp）。本模块只做三件事：
连接、按白名单过滤、失败降级。MCP 工具与本地工具在 LangChain 里是同一
抽象（BaseTool / StructuredTool），因此 Agent 循环（run_agent_turn）零改动。
"""
import logging
import os

logger = logging.getLogger(__name__)

# 白名单：只暴露读类工具。发布/评论/点赞是真实账号操作，一旦挂给模型，
# 提示词注入或模型幻觉都可能触发真实发布——写权限永远不进 Agent。
# 秋招知识点：外部工具的 description 本身也是不可信输入，可能诱导模型
# 调用危险工具；白名单在"工具进入模型视野之前"收敛，比靠提示词约束可靠。
XHS_MCP_ALLOWED = ("search_feeds", "get_feed_detail", "list_feeds", "get_my_profile")

# 服务地址可用环境变量覆盖；与本仓库 tools 目录下的启动配置保持一致。
XHS_MCP_URL = os.getenv("XHS_MCP_URL", "http://localhost:18060/mcp")

_cache: list | None = None


def reset_cache():
    """测试与配置变更时清空缓存；生产进程内无需调用。"""
    global _cache
    _cache = None


async def load_xhs_tools() -> list:
    """加载白名单内的 MCP 工具；服务不可用时返回空列表并记日志。

    秋招知识点（外部依赖的降级设计）：xiaohongshu-mcp 是增强路径——
    它挂掉只意味着 agent 少几个站外工具，绝不能让一次聊天请求 500。
    成功加载后做进程内缓存：MCP 工具 schema 在服务运行期间是稳定的，
    不必每条聊天消息都做一次 tools/list 往返。
    """
    global _cache
    if _cache is not None:
        return _cache
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({
            "xiaohongshu": {"transport": "streamable_http", "url": XHS_MCP_URL},
        })
        tools = [t for t in await client.get_tools() if t.name in XHS_MCP_ALLOWED]
        _cache = tools
        logger.info("xhs_mcp_tools_loaded count=%d url=%s", len(tools), XHS_MCP_URL)
        return tools
    except Exception:  # noqa: BLE001 - 外部服务任何故障都降级为"没有这些工具"
        logger.warning("xhs_mcp_tools_unavailable url=%s", XHS_MCP_URL, exc_info=True)
        return []
