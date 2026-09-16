"""xiaohongshu-mcp 桥接：把外部 MCP 服务的读类工具挂进本地 Agent。

xiaohongshu-mcp（github.com/xpzouying/xiaohongshu-mcp）以 Streamable HTTP
协议提供 MCP 服务（默认 http://localhost:18060/mcp）。本模块只做四件事：
连接、按白名单过滤、把"粘链接采集"补成一个本地工具、失败降级。MCP 工具与
本地工具在 LangChain 里是同一抽象（BaseTool / StructuredTool），因此 Agent
循环（run_agent_turn）零改动。
"""
import json
import logging
import os

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from .xhs_link import XhsLinkError, expand_short_link, note_from_feed_detail, parse_xhs_link

logger = logging.getLogger(__name__)

# 白名单：只暴露读类工具。发布/评论/点赞是真实账号操作，一旦挂给模型，
# 提示词注入或模型幻觉都可能触发真实发布——写权限永远不进 Agent。
# 秋招知识点：外部工具的 description 本身也是不可信输入，可能诱导模型
# 调用危险工具；白名单在"工具进入模型视野之前"收敛，比靠提示词约束可靠。
XHS_MCP_ALLOWED = ("search_feeds", "get_feed_detail", "list_feeds", "get_my_profile")

# 服务地址可用环境变量覆盖；与本仓库 tools 目录下的启动配置保持一致。
XHS_MCP_URL = os.getenv("XHS_MCP_URL", "http://localhost:18060/mcp")

LINK_TOOL_NAME = "collect_note_by_link"
LINK_TOOL_DESCRIPTION = (
    "按小红书链接采集笔记原文。用户直接粘贴小红书链接或分享文案时用它，"
    "比 search_feeds 更准确。链接必须带 xsec_token（从 App 或网页复制的分享链接都带），"
    "令牌会过期，失败就说明链接已失效，应请用户重新复制，不要反复重试。"
)

_cache: list | None = None


class CollectNoteInput(BaseModel):
    """模型只需提供一个链接；是否加载评论由模型按用户意图决定。"""

    link: str = Field(min_length=8, max_length=2000,
                      description="小红书笔记链接，或包含链接的整段分享文案")
    with_comments: bool = Field(default=False, description="是否加载更多评论，会更慢")


def reset_cache():
    """测试与配置变更时清空缓存；生产进程内无需调用。"""
    global _cache
    _cache = None


def _failure(code: str, message: str) -> dict:
    return {"ok": False, "data": None, "error": {"code": code, "message": message}}


def _tool_text(raw) -> str | None:
    """取出 MCP 工具返回的文本内容。

    秋招知识点（协议到框架的类型落差）：langchain-mcp-adapters 不会把 MCP 的
    content 数组拍平成字符串，而是原样转成 LangChain 的 content blocks
    —— [{ "type": "text", "text": "..." }]。直接 json.loads 会拿到 list 而报错；
    这里统一压平，顺便兼容旧版本的纯字符串返回。
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict) and raw.get("type") == "text":
        return str(raw.get("text") or "")
    if isinstance(raw, (list, tuple)):
        parts = []
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        # 文本块是同一份内容的切片，必须直接相连；加换行会切断 JSON 字符串。
        return "".join(parts)
    return None


def build_link_collect_tool(detail_tool) -> StructuredTool:
    """把 MCP 的 get_feed_detail 包成"粘链接即可采集"的工具。

    秋招知识点：MCP 侧只有 feed_id + xsec_token 的接口，但用户手里只有链接。
    参数形态的差异不该由模型去感知——由编排层补齐，模型只看到"能按链接采集"。
    """

    async def collect(link: str, with_comments: bool = False) -> dict:
        try:
            ref = parse_xhs_link(link)
            if ref["short_link"]:
                # 短链不含笔记编号，先跟随 302 再解析。
                ref = parse_xhs_link(await expand_short_link(ref["url"]))
        except XhsLinkError as exc:
            return _failure(exc.code, str(exc))
        except Exception:  # noqa: BLE001 - 短链跳转失败不该让整轮对话失败
            logger.warning("xhs_short_link_expand_failed", exc_info=True)
            return _failure("short_link_unreachable", "短链展开失败，请改用完整的小红书链接。")

        if not ref["feed_id"]:
            return _failure("note_id_missing", "链接里没有笔记编号，请复制笔记详情页的链接。")
        if not ref["xsec_token"]:
            return _failure(
                "xsec_token_missing",
                "链接缺少 xsec_token，无法采集；请从 App 或网页重新复制完整分享链接。",
            )

        args = {"feed_id": ref["feed_id"], "xsec_token": ref["xsec_token"]}
        if with_comments:
            args.update({"load_all_comments": True, "limit": 20})
        try:
            raw = await detail_tool.ainvoke(args)
        except Exception:  # noqa: BLE001 - 站外失败只影响这一个工具
            logger.warning("xhs_feed_detail_call_failed", exc_info=True)
            return _failure("collect_unavailable", "站外采集服务暂时不可用，请稍后再试。")

        text = _tool_text(raw)
        if text is None:
            return _failure("payload_invalid", "采集返回不是预期的结构。")
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            # MCP 采集失败时返回的是纯文本提示，不是 JSON；原样不暴露给模型。
            return _failure("collect_failed", "采集失败，链接可能已过期或笔记不可见。")

        try:
            note = note_from_feed_detail(payload)
        except XhsLinkError as exc:
            return _failure(exc.code, str(exc))
        return {"ok": True, "data": note, "error": None}

    return StructuredTool.from_function(
        name=LINK_TOOL_NAME,
        description=LINK_TOOL_DESCRIPTION,
        coroutine=collect,
        args_schema=CollectNoteInput,
    )


async def load_xhs_tools() -> list:
    """加载站外工具：白名单 MCP 读工具 + 一个本地链接采集工具。

    服务不可用时返回空列表并记日志。

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
        # 只有拿得到 get_feed_detail 时才补链接采集工具，否则它无从执行。
        detail_tool = next((t for t in tools if t.name == "get_feed_detail"), None)
        if detail_tool is not None:
            tools.append(build_link_collect_tool(detail_tool))
        _cache = tools
        logger.info("xhs_mcp_tools_loaded count=%d url=%s", len(tools), XHS_MCP_URL)
        return tools
    except Exception:  # noqa: BLE001 - 外部服务任何故障都降级为"没有这些工具"
        logger.warning("xhs_mcp_tools_unavailable url=%s", XHS_MCP_URL, exc_info=True)
        return []
