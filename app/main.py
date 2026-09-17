"""接口入口：以后安卓 App 就通过这些 HTTP 接口调用后端。"""
# 阅读路线：请求 JSON → ImportRequest 校验 → import_note → 提取器 → 数据库 → 响应 JSON。
# import 用于引入别的模块；以点开头的 .database 表示当前 app 包里的 database.py。
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
import logging
import os
import re
import sqlite3
import asyncio
from typing import Literal
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .database import NoteStore, DEFAULT_DB
from .extractor import ImportProblem, share_url, fetch_page, parse_note
from .media import download_note_assets, local_file_valid, storage_path
from .ai import AIProblem, analyze_note, suggest_folders, draft_plan, extract_memories_safe, extract_chat_memories_safe
from .retrieval import search_knowledge
from .chat_agent import plan_requested, run_agent_turn
from .plan_quality import check_plan_quality
from .tools import build_langchain_tools
from .xhs_mcp import load_xhs_tools
from . import indexing, xhs_source
from .embeddings import (
    MODEL_NAME as EMBEDDING_MODEL,
    blob_to_vector,
    cosine_similarity,
    embed_query as embed_search_query,
    embed_texts,
)


# 使用固定名称的日志器，方便以后接入结构化日志和可观测性平台。
# 日志内容只包含请求编号、批次位置和异常类型，绝不记录分享链接或 xsec_token。
logger = logging.getLogger("xhs.import")


def classify_network_error(exc: httpx.HTTPError) -> tuple[str, str]:
    """把 HTTPX 异常转换为可安全返回的错误码和中文提示。

    不直接返回 str(exc)，因为 HTTPX 的原始错误信息可能包含完整请求 URL，
    而小红书分享 URL 中可能带有临时访问 token。
    """
    if isinstance(exc, httpx.TimeoutException):
        return "timeout", "读取来源网页超时，请稍后重试。"
    if isinstance(exc, httpx.ProxyError):
        return "proxy_error", "无法通过当前网络代理访问来源站点。"

    # Windows/Linux 对同一网络故障可能使用不同的底层异常类。
    # 这里只查看异常类型名称，不读取可能含 URL 的异常文本。
    cause_names = set()
    current: BaseException | None = exc
    while current is not None and len(cause_names) < 8:
        cause_names.add(type(current).__name__.lower())
        current = current.__cause__ or current.__context__

    if "gaierror" in cause_names:
        return "dns_error", "无法解析来源站点域名，请检查网络或 DNS。"
    if any("ssl" in name or "certificate" in name for name in cause_names):
        return "tls_error", "与来源站点建立安全连接失败。"
    if "connectionrefusederror" in cause_names:
        return "connection_refused", "来源站点拒绝了连接。"
    if "connectionreseterror" in cause_names or isinstance(exc, (httpx.ReadError, httpx.RemoteProtocolError)):
        return "connection_reset", "来源站点中断了连接，请稍后重试。"
    return "connect_error", "无法连接来源站点。"


class ImportRequest(BaseModel):
    # BaseModel 是 Pydantic 的数据模型，不是数据库表。
    # share_text: str 声明字段类型；Field 约束字符串长度并提供接口文档说明。
    # FastAPI 会先校验请求；不符合要求时自动返回 422，不会执行导入函数。
    share_text: str = Field(min_length=1, max_length=10000, description="完整分享文本或链接")


class IntakeRequest(BaseModel):
    share_text: str = Field(min_length=1, max_length=10000, description="小红书分享文案或链接")


class BatchImportRequest(BaseModel):
    # 一个字符串最适合“整段粘贴”的使用方式；后端负责从里面找出所有链接。
    share_text: str = Field(
        min_length=1,
        max_length=100000,
        description="批量粘贴内容，可每行一个链接，也可包含小红书分享文案",
    )


class RefreshSourceRequest(BaseModel):
    share_text: str | None = Field(
        default=None, max_length=10000,
        description="可选的新分享文案或链接；不填时使用笔记已保存的来源地址",
    )


class IntentConfirmationRequest(BaseModel):
    # decision 表示用户怎样处理模型建议，而不是让模型替用户做决定。
    decision: Literal["confirmed", "edited", "dismissed"] = Field(
        description="confirmed=直接确认，edited=修改后确认，dismissed=暂不采用"
    )
    # category 和 intent 留空时沿用模型建议；edited 时接口会要求填写它们。
    category: str | None = Field(default=None, max_length=50, description="用户确认的分类")
    intent: str | None = Field(default=None, max_length=500, description="用户真实的收藏目的")
    selected_actions: list[str] | None = Field(
        default=None, max_length=20, description="用户选中的行动；以后可据此创建提醒"
    )


class TaskCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200, description="要执行的任务")
    details: str = Field(default="", max_length=2000, description="补充说明")
    scheduled_at: datetime = Field(description="计划执行时间，必须带时区，例如 2026-09-05T20:00:00+08:00")
    reminder_at: datetime | None = Field(default=None, description="提醒时间；不填表示暂不提醒")
    timezone: str = Field(default="Asia/Shanghai", max_length=100, description="IANA 时区名称")
    source_action: str | None = Field(default=None, max_length=500, description="来自哪条 AI 行动建议")


class TaskUpdateRequest(BaseModel):
    # 所有字段都可选，只提交需要修改的部分。
    title: str | None = Field(default=None, min_length=1, max_length=200)
    details: str | None = Field(default=None, max_length=2000)
    scheduled_at: datetime | None = None
    reminder_at: datetime | None = None
    timezone: str | None = Field(default=None, max_length=100)


class PlanItemRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200, description="计划步骤")
    details: str = Field(default="", max_length=2000)
    scheduled_at: datetime = Field(description="步骤执行时间，必须带时区")
    reminder_at: datetime | None = None
    timezone: str = Field(default="Asia/Shanghai", max_length=100)
    source_note_ids: list[str] = Field(default_factory=list, max_length=10)


class PlanCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    goal: str = Field(min_length=1, max_length=1000)
    items: list[PlanItemRequest] = Field(min_length=1, max_length=30)
    source_note_ids: list[str] = Field(default_factory=list, max_length=20)


class PlanUpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    goal: str | None = Field(default=None, min_length=1, max_length=1000)


class PlanItemUpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    details: str | None = Field(default=None, max_length=2000)
    scheduled_at: datetime | None = None
    reminder_at: datetime | None = None
    timezone: str | None = Field(default=None, max_length=100)


class PlanItemsReorderRequest(BaseModel):
    item_ids: list[int] = Field(min_length=1, max_length=30)


class AIDraftPlanRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=1000)
    source_note_ids: list[str] = Field(min_length=1, max_length=10)
    start_date: str = Field(description="计划开始日期，例如 2026-09-16")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="自然语言检索问题")
    top_k: int = Field(default=5, ge=1, le=20, description="返回最相关的 Chunk 数量")
    deduplicate_notes: bool = Field(
        default=True, description="是否限制每篇笔记最多返回一个最高分 Chunk"
    )


class ThreadCreateRequest(BaseModel):
    title: str = Field(default="新对话", min_length=1, max_length=100)


class ChatMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000, description="用户发送给知识库助手的消息")


def memory_triggered(content: str) -> bool:
    """只用明确的记忆指令或稳定偏好触发提炼；一次性问题不触发模型调用。"""
    text = content.strip()
    if re.search(r"(?:别|不要|不用)记住", text):
        return False
    if re.search(r"(?:请|帮我|你要|以后)?记住|以后(?:都|请|按这个)|下次(?:也|都|请)", text):
        return True
    return bool(re.search(
        r"我(?:一直|长期|通常|平时|习惯|偏好|更喜欢|不喜欢|每天(?:只能|只有|都)|"
        r"每周(?:只能|只有|都)|是在职备考)", text
    ))


class FolderCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=50, description="收藏夹名称")
    description: str = Field(default="", max_length=500, description="收藏夹用途说明")
    color: str | None = Field(default=None, max_length=30, description="客户端展示颜色，例如 #4CAF50")


class FolderUpdateRequest(BaseModel):
    # 所有字段可选，PUT 时只修改用户实际提交的字段。
    name: str | None = Field(default=None, min_length=1, max_length=50)
    description: str | None = Field(default=None, max_length=500)
    color: str | None = Field(default=None, max_length=30)


class FolderSuggestionConfirmRequest(BaseModel):
    folder_ids: list[int] = Field(
        min_length=1, max_length=3,
        description="用户最终选择的 1 到 3 个收藏夹 ID，可与 AI 推荐不同",
    )


class MemoryUpdateRequest(BaseModel):
    """修正长期记忆：请求模型本身就是字段白名单。

    秋招知识点（批量赋值 / Mass Assignment 防护）：model 与 source_note_id
    这类由服务端掌控的字段刻意不在这里声明——否则请求体就能决定"能改哪些
    字段"，用户可以伪造记忆的产出模型与来源笔记，破坏可审计性和可追溯性。
    声明式白名单还有第二层好处：Pydantic 直接丢弃未声明的键，接口层就完成了
    过滤；存储层的 allowed 集合再兜一次底，两层叠加即纵深防御。
    """

    type: Literal["preference", "habit", "correction"] | None = Field(
        default=None, description="记忆类型：内容偏好 / 行为习惯 / 纠正记录")
    content: str | None = Field(
        default=None, min_length=1, max_length=200,
        description="记忆内容，用第三人称“用户……”陈述；留空表示不改")
    status: Literal["active", "superseded", "dismissed"] | None = Field(
        default=None,
        description="active 生效；dismissed 已删除，改回 active 即恢复；"
                    "superseded 是系统内部状态，手动设置会被 400 拒绝")


class UserMemoryCreateRequest(BaseModel):
    type: Literal["preference", "habit", "correction"]
    content: str = Field(min_length=1, max_length=200, description="你自己确认的长期偏好或习惯")


class MemoryCandidateReviewRequest(BaseModel):
    decision: Literal["confirmed", "dismissed"]
    type: Literal["preference", "habit", "correction"] | None = None
    content: str | None = Field(default=None, min_length=1, max_length=200)


def create_app(db_path=DEFAULT_DB, media_root=None):
    # 从项目根目录的 .env 读取本地配置。这个文件已被 .gitignore 排除，
    # 因而不会在提交代码时意外上传 API Key。
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    # “工厂函数”：每调用一次就创建一个应用，测试时可以传入临时数据库路径。
    # db_path=DEFAULT_DB 是默认参数，不传时使用项目的正式数据文件。
    @asynccontextmanager
    async def lifespan(app):
        # lifespan 管理服务的启动和停止。yield 前的代码在启动时运行。
        # app.state 是应用共享的数据容器，后面的接口从这里取数据库操作对象。
        app.state.store = NoteStore(db_path)
        # BackgroundTasks 暂存在当前进程；重启后明确标记中断任务，避免进度假性停滞。
        app.state.store.mark_interrupted_jobs(datetime.now(timezone.utc).isoformat())
        app.state.media_root = Path(media_root) if media_root else Path(db_path).parent / "media"
        # 第一版使用单进程锁，避免重复点击同时下载；不能当作多进程任务队列使用。
        app.state.media_lock = Lock()
        # yield 把控制权交回框架，让服务开始处理请求；关闭时会继续执行其后的代码。
        yield

    app = FastAPI(title="小红书知识库后端", version="0.4.0", lifespan=lifespan,
                  description="本地单用户学习版：支持导入、图片、AI 分析、意图确认和任务；无登录，请勿公开到互联网。")

    def utc_text(value: datetime, field_name: str):
        # 带时区时间统一换算为 UTC 保存，避免安卓手机或服务器处于不同时区时错点提醒。
        if value.tzinfo is None or value.utcoffset() is None:
            raise HTTPException(400, detail=f"{field_name} 必须包含时区，例如 +08:00。")
        return value.astimezone(timezone.utc).isoformat()

    def valid_timezone(name: str):
        try:
            ZoneInfo(name)
        except ZoneInfoNotFoundError:
            raise HTTPException(400, detail="timezone 不是有效的 IANA 时区名称。") from None

    @app.get("/health", summary="检查后端是否启动")
    def health():
        # @app.get 是装饰器，把 URL 和函数关联起来。GET 主要用于读取。
        # 返回 Python 字典后，FastAPI 会将它转换为客户端能读取的 JSON。
        return {"status": "ok"}

    @app.post("/threads", status_code=201, summary="创建知识库对话")
    def create_thread(body: ThreadCreateRequest):
        now = datetime.now(timezone.utc).isoformat()
        return app.state.store.create_thread(str(uuid4()), body.title.strip(), now)

    @app.get("/threads/{thread_id}", summary="查看对话及最近消息")
    def get_thread(thread_id: str):
        thread = app.state.store.thread(thread_id)
        if thread is None:
            raise HTTPException(404, detail="对话不存在。")
        return thread

    @app.post("/threads/{thread_id}/messages", summary="与知识库 Agent 对话")
    async def send_chat_message(thread_id: str, body: ChatMessageRequest,
                                background_tasks: BackgroundTasks):
        if app.state.store.thread(thread_id) is None:
            raise HTTPException(404, detail="对话不存在。")
        now = datetime.now(timezone.utc).isoformat()
        app.state.store.add_chat_message(thread_id, "user", body.content.strip(), {}, now)

        def recall(query, limit):
            # Workbuddy 完成向量记忆召回后会自动走新版；当前版本安全退化为最近记忆。
            if hasattr(app.state.store, "recall_memories"):
                return app.state.store.recall_memories(query=query, limit=limit)
            return app.state.store.active_memories(limit=limit)

        async def create_draft(goal, source_note_ids, start_date):
            # 复用已有计划接口的校验和持久化逻辑，避免 Agent 另写一套业务规则。
            return await create_ai_plan_draft(AIDraftPlanRequest(
                goal=goal, source_note_ids=source_note_ids, start_date=start_date
            ))

        tools = build_langchain_tools(
            searcher=lambda query, limit: search_knowledge(
                app.state.store, query, limit, query_embedder=embed_search_query
            ),
            getter=app.state.store.get,
            recaller=recall,
            drafter=create_draft,
        )
        # 站外能力（小红书搜索/笔记详情/按链接采集/按链接收藏）来自外部 MCP 服务，
        # 按白名单挂载；服务不可用时静默缩编为纯本地工具，不影响聊天主链路。
        # 传入 store 才会多出"收藏进知识库"这个写工具：没有库就没有写入口。
        tools = tools + await load_xhs_tools(
            store=app.state.store, media_root=app.state.media_root
        )
        # 代码层权限控制比 Prompt 更可靠：普通咨询时模型根本看不到写入型工具。
        wants_plan = plan_requested(body.content)
        if not wants_plan:
            tools = [tool for tool in tools if tool.name != "draft_plan"]
        # 只发送最近 20 条消息：这是滑动窗口短期记忆，避免上下文无限增长。
        history = app.state.store.thread(thread_id, message_limit=20)["messages"]
        try:
            result = await run_agent_turn(
                history, tools, required_tool="draft_plan" if wants_plan else None
            )
        except Exception as exc:
            logger.warning("chat_agent_failed thread_id=%s exception_type=%s", thread_id, type(exc).__name__)
            raise HTTPException(502, detail="知识库助手暂时无法回答，请稍后重试。") from None

        assistant = app.state.store.add_chat_message(
            thread_id, "assistant", result["content"], {
                "tool_calls": result["tool_calls"],
                "retrieved_note_ids": result["retrieved_note_ids"],
                "cited_note_ids": result["cited_note_ids"],
                "source_note_ids": result["source_note_ids"],
                "plan_id": result["plan_id"],
            }, datetime.now(timezone.utc).isoformat()
        )
        # 响应写好后再提炼：用户先拿到正常回复，候选记忆随后进入审核区。
        if memory_triggered(body.content):
            background_tasks.add_task(consolidate_thread_memories, thread_id)
        return {"thread_id": thread_id, "message": assistant}

    async def consolidate_thread_memories(thread_id: str):
        """会话 → 长期记忆的沉淀通道（episodic → semantic 固化）。

        三层记忆里，工作记忆是 prompt 本身、会话记忆是 chat_messages 表，
        但会话层的内容会随对话结束而消散。这里把对话中暴露的持久偏好提炼成
        记忆候选，交给用户审核后进入长期记忆层——固化走的是"AI 提议、
        用户拍板"的候选审核流程，不让模型静默改写用户画像。
        """
        store = app.state.store
        thread = store.thread(thread_id)
        if thread is None:
            return
        messages = thread["messages"]
        user_turns = sum(1 for m in messages if m.get("role") == "user")
        # 明确说“记住”或表达长期习惯时，一轮也可以提炼；普通短对话仍跳过。
        latest_user = next((m["content"] for m in reversed(messages)
                            if m.get("role") == "user"), "")
        if (user_turns < 2 or len(messages) < 4) and not memory_triggered(latest_user):
            logger.info("memory_consolidate_skipped thread_id=%s messages=%d",
                        thread_id, len(messages))
            return
        try:
            for attempt in range(2):
                # 待审核候选也用于去重，但仍不能参与 Agent 的 recall_memories()。
                known_memories = store.active_memories(limit=10) + store.memory_candidates(
                    status="pending", source_note_id=thread_id, limit=10
                )
                memories, model = await extract_chat_memories_safe(
                    messages, known_memories, thread_id
                )
                if model is not None:
                    break  # 空数组也可能是正常结果，不因此重试。
                if attempt == 0:
                    await asyncio.sleep(0.5)
            if memories and model:
                candidates = store.save_memory_candidates(
                    model, memories, thread_id, datetime.now(timezone.utc).isoformat()
                )
                logger.info("memory_candidates_from_chat thread_id=%s count=%d",
                            thread_id, len(candidates))
        except Exception as exc:
            # 后台附加功能失败只记录类型；不会影响已经生成的对话回复。
            logger.warning("memory_consolidate_failed thread_id=%s exception_type=%s",
                           thread_id, type(exc).__name__)

    @app.post("/threads/{thread_id}/consolidate",
              summary="从会话消息中提炼长期记忆候选（会话结束或每 N 轮触发）")
    def consolidate_thread(thread_id: str, background_tasks: BackgroundTasks):
        """沉淀入口只做两件事：验证会话存在、把慢操作丢给后台任务。

        秋招知识点：读操作（存在性检查、404 快速失败）在请求内同步完成，
        LLM 提炼放后台——与笔记意图确认的记忆触发保持同一模式，调用方
        （会话结束钩子或前端"保存记忆"按钮）拿到的永远是毫秒级响应。
        """
        if app.state.store.thread(thread_id) is None:
            raise HTTPException(404, detail="会话不存在。")
        background_tasks.add_task(consolidate_thread_memories, thread_id)
        return {"thread_id": thread_id, "consolidation": "queued"}

    @app.get("/ai/config", summary="检查视觉模型配置；不会显示密钥")
    def ai_config():
        # 这里只判断密钥是否存在，绝不把密钥内容返回给浏览器或手机。
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        return {
            "provider": "deepseek",
            "configured": bool(api_key),
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash-vision-exp"),
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        }

    async def _html_note(url: str) -> dict:
        """网页解析通道。

        刻意引用模块级 fetch_page / parse_note（而不是从别的模块导入函数对象），
        这样接口测试替换 app.main.fetch_page 依旧生效，不必联网。
        """
        html, final_url = await fetch_page(url)
        return xhs_source.parse_html_note(parse_note, html, final_url, url)

    async def collect_note(url: str) -> tuple[dict, str]:
        """统一采集入口：优先 MCP（真实浏览器，结构完整），失败回退网页解析。

        秋招知识点（降级链路）：MCP 更准但依赖本地服务与登录态，网页解析更独立
        但怕页面改版。两条通道产出同一个 note 结构，谁成功都能继续走入库和下载，
        调用方不需要分支处理。

        采集层顺带补齐图片文字：内容全在长图里的笔记正文只剩话题标签，不读图
        的话入库后索引是空的、收藏完搜不到。正文够长的笔记不会触发读图。
        """
        return await xhs_source.collect_note(url, html_note=_html_note)

    async def import_from_url(url: str):
        """单条与批量接口共用的核心步骤：采集、保存、建索引。"""
        note, source = await collect_note(url)
        note["created_at"] = datetime.now(timezone.utc).isoformat()
        saved, created = app.state.store.save(note)
        # 索引读的是入库后的正文（含图片转写文字），所以必须排在 save 之后；
        # 它是阻塞的模型推理，放进线程避免堵住事件循环。
        index = await asyncio.to_thread(
            indexing.index_safely, app.state.store, saved["note_id"]
        )
        return {"status": "imported" if created else "existing", "note": saved,
                "source": source, "index": index}

    async def process_intake(job_id: str, url: str):
        """后台执行导入和图片下载，并在每个阶段写入 SQLite 进度。"""
        store = app.state.store
        now = datetime.now(timezone.utc).isoformat()
        progress = {"import": "running", "download_assets": "pending", "index": "pending"}
        store.update_job(job_id, status="running", current_step="import",
                         progress=progress, now=now)
        try:
            note, source = await collect_note(url)
            logger.info("intake_collected job_id=%s source=%s", job_id, source)
            existing = store.get(note["note_id"])
            if existing is None:
                note["created_at"] = datetime.now(timezone.utc).isoformat()
                saved, _created = store.save(note)
                store.ensure_assets(saved)
                import_status = "imported"
            else:
                # 重复收藏时也刷新临时图片地址；成功下载的本地文件会保留。
                saved = store.refresh_note_source(
                    note["note_id"], note, datetime.now(timezone.utc).isoformat()
                )
                import_status = "existing"

            progress["import"] = "completed"
            progress["download_assets"] = "running"
            store.update_job(job_id, status="running", current_step="download_assets",
                             progress=progress, note_id=saved["note_id"], now=datetime.now(timezone.utc).isoformat())
            # 下载器与 SQLite/Pillow 操作为同步函数，放入线程避免堵住 FastAPI 事件循环。
            download_result = await asyncio.to_thread(
                download_note_assets, store, saved, app.state.media_root
            )
            progress["download_assets"] = (
                "completed" if download_result["failed"] == 0 else "partial"
            )
            progress["index"] = "running"
            store.update_job(job_id, status="running", current_step="index",
                             progress=progress, now=datetime.now(timezone.utc).isoformat())
            # 入库后立即建索引并生成向量，收件箱进来的笔记马上可被语义检索。
            index = await asyncio.to_thread(
                indexing.index_safely, store, saved["note_id"]
            )
            # skipped（纯图片且未识别出文字）不是故障，只有 failed 才算这一步失败。
            progress["index"] = "failed" if index.get("status") == "failed" else "completed"
            final_status = "completed" if download_result["failed"] == 0 else "partial"
            store.update_job(
                job_id, status=final_status, current_step="completed",
                progress=progress,
                result={"import_status": import_status, "source": source,
                        "index": index, "assets": {
                    "total": download_result["total"],
                    "downloaded": download_result["downloaded"],
                    "skipped": download_result["skipped"],
                    "failed": download_result["failed"],
                }},
                now=datetime.now(timezone.utc).isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
        except ImportProblem as exc:
            progress["import"] = "failed"
            progress["error_code"] = exc.code
            # 只记录错误阶段和 job_id，不记录带临时授权的链接或网页正文。
            logger.warning("intake_import_failed job_id=%s error_code=%s", job_id, exc.code)
            store.update_job(job_id, status="failed", current_step="failed",
                             progress=progress, error=str(exc),
                             now=datetime.now(timezone.utc).isoformat(),
                             finished_at=datetime.now(timezone.utc).isoformat())
        except httpx.HTTPError as exc:
            _error_code, message = classify_network_error(exc)
            progress["import"] = "failed"
            store.update_job(job_id, status="failed", current_step="failed",
                             progress=progress, error=message,
                             now=datetime.now(timezone.utc).isoformat(),
                             finished_at=datetime.now(timezone.utc).isoformat())
        except Exception:
            # 日志只记录 job 编号和异常类型；不打印 URL 或原始异常文本。
            logger.exception("intake_job_failed job_id=%s", job_id)
            progress["error"] = "internal_error"
            store.update_job(job_id, status="failed", current_step="failed",
                             progress=progress, error="处理失败，请查看服务端日志。",
                             now=datetime.now(timezone.utc).isoformat(),
                             finished_at=datetime.now(timezone.utc).isoformat())

    @app.post("/notes/intake", status_code=202, summary="统一收件：导入笔记并立即下载图片")
    async def intake_note(body: IntakeRequest, background_tasks: BackgroundTasks):
        try:
            url = share_url(body.share_text)
        except ImportProblem as exc:
            raise HTTPException(400, detail=str(exc)) from None
        job_id = uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        app.state.store.create_job(
            job_id, "note_intake", now,
            {"import": "pending", "download_assets": "pending", "index": "pending"},
        )
        # URL 只存在于当前进程的任务参数中，不写入 jobs 表或日志。
        background_tasks.add_task(process_intake, job_id, url)
        return {"job_id": job_id, "status": "queued", "poll_url": f"/jobs/{job_id}"}

    @app.get("/jobs/{job_id}", summary="查询收件处理进度")
    def get_job(job_id: str):
        job = app.state.store.job(job_id)
        if job is None:
            raise HTTPException(404, detail="处理任务不存在。")
        return job

    @app.post("/notes/import", summary="导入一篇小红书笔记")
    async def import_note(body: ImportRequest):
        # POST 用于提交数据。body 是已经通过校验的请求对象。
        # async/await 让等待网络响应时可以让出执行权；不代表自动启动后台任务。
        # 当前请求仍需等提取完成才返回，数据库操作也仍是同步执行。
        try:
            # 第一步：从分享文本中找出链接，并检查是否属于允许的域名。
            url = share_url(body.share_text)
        except ImportProblem as exc:
            # try/except 捕获预期错误；exc 是这次异常的对象。
            # HTTPException 将业务错误转换为 HTTP 错误响应。400 表示输入有问题。
            # from None 隐藏隐式异常链，不表示忽略错误；客户端仍会收到 detail。
            raise HTTPException(400, detail=str(exc)) from None
        try:
            # 第二步：调用公共导入函数。单条和批量接口会走完全相同的解析规则。
            return await import_from_url(url)
        except ImportProblem as exc:
            # 422：链接已接受，但网页内容不满足我们的导入要求。
            raise HTTPException(422, detail=str(exc)) from None
        except httpx.TimeoutException:
            # 504：上游请求超时；放在 HTTPError 前，因为它属于更具体的错误类型。
            raise HTTPException(504, detail="读取来源网页超时，请稍后重试。") from None
        except httpx.HTTPError:
            # 502：与来源站点通信失败；不把可能含敏感 URL 的原始异常返回给用户。
            raise HTTPException(502, detail="无法连接来源站点；未保存笔记。") from None

    @app.post("/notes/import-batch", summary="批量粘贴小红书链接到收件箱")
    async def import_notes_batch(body: BatchImportRequest):
        # 正则只负责找出 http(s) 片段；域名和安全性仍交给 share_url 严格校验。
        normalized = body.share_text.replace("\\_", "_").replace("\\&", "&")
        candidates = re.findall(r"https?://[^\s]+", normalized)
        if not candidates:
            raise HTTPException(400, detail="没有找到任何 http 或 https 链接。")
        if len(candidates) > 20:
            raise HTTPException(400, detail="每次最多粘贴 20 条链接，请分批导入。")

        # request_id 用来关联同一批次的日志和响应，但不包含任何用户输入。
        request_id = uuid4().hex
        items = []
        seen_urls = set()
        for position, candidate in enumerate(candidates, 1):
            try:
                url = share_url(candidate)
                if url in seen_urls:
                    # 同一批中重复粘贴不再次访问来源网站。
                    items.append({"position": position, "status": "duplicate", "error": None})
                    continue
                seen_urls.add(url)
                result = await import_from_url(url)
                items.append({
                    "position": position,
                    "status": result["status"],
                    "note_id": result["note"]["note_id"],
                    "title": result["note"]["title"],
                    # 批量导入最关心的就是"哪几条现在能搜到"，直接放在每条结果里。
                    "searchable": bool(result["index"].get("searchable")),
                    "index_status": result["index"].get("status"),
                    "error": None,
                })
            except ImportProblem as exc:
                items.append({"position": position, "status": "failed", "error": str(exc)})
            except httpx.HTTPError as exc:
                error_code, message = classify_network_error(exc)
                logger.warning(
                    "batch_import_failed request_id=%s position=%d error_code=%s exception_type=%s",
                    request_id,
                    position,
                    error_code,
                    type(exc).__name__,
                )
                items.append({
                    "position": position,
                    "status": "failed",
                    "error_code": error_code,
                    "error": message,
                })

        succeeded = sum(item["status"] in {"imported", "existing"} for item in items)
        failed = sum(item["status"] == "failed" for item in items)
        duplicates = sum(item["status"] == "duplicate" for item in items)
        return {
            "request_id": request_id,
            "total": len(items),
            "succeeded": succeeded,
            "failed": failed,
            "duplicates": duplicates,
            "items": items,
        }

    @app.get("/notes", summary="查看已保存的笔记")
    def list_notes(limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
        # 查询参数写在 URL 的 ? 后，例如 /notes?limit=10&offset=20。
        # limit 控制返回条数，offset 控制跳过条数；ge 是 >=，le 是 <=。
        return {"items": app.state.store.list(limit, offset)}

    @app.get("/notes/{note_id}", summary="查看单篇笔记")
    def get_note(note_id: str):
        # {note_id} 是路径参数，例如 /notes/abc123 会传入字符串 "abc123"。
        note = app.state.store.get(note_id)
        if note is None:
            # None 表示没有查到；404 表示请求的资源不存在。
            raise HTTPException(404, detail="笔记不存在。")
        return note

    @app.post("/plans", status_code=201, summary="创建学习计划草案")
    def create_plan(body: PlanCreateRequest):
        for note_id in body.source_note_ids:
            get_note(note_id)
        normalized = []
        for item in body.items:
            for note_id in item.source_note_ids:
                get_note(note_id)
            valid_timezone(item.timezone)
            scheduled = utc_text(item.scheduled_at, "scheduled_at")
            reminder = utc_text(item.reminder_at, "reminder_at") if item.reminder_at else None
            if reminder and reminder > scheduled:
                raise HTTPException(400, detail="提醒时间不能晚于步骤执行时间。")
            normalized.append({"title": item.title.strip(), "details": item.details.strip(),
                               "scheduled_at": scheduled, "reminder_at": reminder,
                               "timezone": item.timezone, "source_note_ids": item.source_note_ids})
        now = datetime.now(timezone.utc).isoformat()
        return app.state.store.create_plan(
            body.title.strip(), body.goal.strip(), body.source_note_ids, normalized, now
        )

    @app.post("/plans/draft-from-ai", status_code=201, summary="让 AI 生成学习计划草案")
    async def create_ai_plan_draft(body: AIDraftPlanRequest):
        try:
            datetime.strptime(body.start_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(400, detail="start_date 必须是 YYYY-MM-DD 格式。") from None
        notes = [get_note(note_id) for note_id in body.source_note_ids]
        intents = [app.state.store.intent_confirmation(note_id)
                   for note_id in body.source_note_ids]
        intents = [item for item in intents if item is not None]
        try:
            result, model = await draft_plan(body.goal, notes, intents, body.start_date)
            items = []
            for item in result["items"]:
                for note_id in item.get("source_note_ids", body.source_note_ids):
                    if note_id not in body.source_note_ids:
                        raise AIProblem("AI 返回了未授权的来源笔记。")
                valid_timezone("Asia/Shanghai")
                scheduled = utc_text(datetime.fromisoformat(item["scheduled_at"]), "scheduled_at")
                reminder = (utc_text(datetime.fromisoformat(item["reminder_at"]), "reminder_at")
                            if item.get("reminder_at") else None)
                if reminder and reminder > scheduled:
                    raise AIProblem("计划提醒时间晚于执行时间。")
                items.append({"title": item["title"].strip(), "details": item.get("details", "").strip(),
                              "scheduled_at": scheduled, "reminder_at": reminder,
                              "timezone": "Asia/Shanghai", "source_note_ids": item.get("source_note_ids", body.source_note_ids)})
        except (AIProblem, ValueError) as exc:
            raise HTTPException(502, detail=str(exc)) from None
        plan = app.state.store.create_plan(
            result["title"].strip(), result["goal"].strip(), body.source_note_ids,
            items, datetime.now(timezone.utc).isoformat()
        )
        return {"status": "draft", "model": model, "plan": plan}

    @app.get("/plans", summary="查看学习计划")
    def list_plans(status: Literal["draft", "confirmed", "active", "completed"] | None = None):
        return {"items": app.state.store.plans(status)}

    @app.get("/plans/{plan_id}", summary="查看计划详情")
    def get_plan(plan_id: int):
        plan = app.state.store.plan(plan_id)
        if plan is None:
            raise HTTPException(404, detail="计划不存在。")
        return plan

    @app.put("/plans/{plan_id}", summary="修改计划草案")
    def update_plan(plan_id: int, body: PlanUpdateRequest):
        plan = get_plan(plan_id)
        if plan["status"] != "draft":
            raise HTTPException(409, detail="只有 draft 状态的计划可以修改。")
        fields = body.model_dump(exclude_unset=True)
        if not fields:
            return plan
        assignments = ", ".join(f"{key}=?" for key in fields)
        values = [value.strip() for value in fields.values()]
        with sqlite3.connect(app.state.store.path) as db:
            db.execute(f"UPDATE plans SET {assignments}, updated_at=? WHERE id=?",
                       [*values, datetime.now(timezone.utc).isoformat(), plan_id])
        return get_plan(plan_id)

    @app.get("/plans/{plan_id}/quality-check", summary="确认前检查计划时间和工作量")
    def plan_quality_check(plan_id: int):
        # 只读预检：草案和已经执行的计划都能查看，不会修改现有任务。
        return check_plan_quality(get_plan(plan_id))

    @app.post("/plans/{plan_id}/confirm", summary="确认学习计划草案")
    def confirm_plan(plan_id: int):
        plan = get_plan(plan_id)
        if plan["status"] != "draft":
            raise HTTPException(409, detail="只有 draft 状态的计划可以确认。")
        quality = check_plan_quality(plan)
        if not quality["can_confirm"]:
            # 返回结构化结果，Swagger 能直接指出哪些步骤需要改时间。
            raise HTTPException(409, detail=quality)
        return app.state.store.update_plan_status(
            plan_id, "confirmed", datetime.now(timezone.utc).isoformat()
        )

    def require_draft_item(plan_id: int, item_id: int):
        plan = get_plan(plan_id)
        if plan["status"] != "draft":
            raise HTTPException(409, detail="只有 draft 状态的计划可以编辑步骤。")
        item = next((item for item in plan["items"] if item["id"] == item_id), None)
        if item is None:
            raise HTTPException(404, detail="计划步骤不存在。")
        return item

    @app.put("/plans/{plan_id}/items/{item_id}", summary="修改计划步骤")
    def update_plan_item(plan_id: int, item_id: int, body: PlanItemUpdateRequest):
        require_draft_item(plan_id, item_id)
        fields = body.model_dump(exclude_unset=True)
        if "title" in fields:
            fields["title"] = fields["title"].strip()
        if "details" in fields:
            fields["details"] = fields["details"].strip()
        current = app.state.store.plan_item(item_id)
        timezone_name = fields.get("timezone", current["timezone"])
        valid_timezone(timezone_name)
        scheduled = fields.get("scheduled_at")
        reminder = fields.get("reminder_at")
        scheduled_text = utc_text(scheduled, "scheduled_at") if scheduled else current["scheduled_at"]
        # 显式传 null 表示清空提醒；完全不传才表示保持原值。
        if "reminder_at" in fields:
            reminder_text = utc_text(reminder, "reminder_at") if reminder is not None else None
        else:
            reminder_text = current["reminder_at"]
        if reminder_text and reminder_text > scheduled_text:
            raise HTTPException(400, detail="提醒时间不能晚于步骤执行时间。")
        if "scheduled_at" in fields:
            fields["scheduled_at"] = scheduled_text
        if "reminder_at" in fields:
            fields["reminder_at"] = reminder_text
        return app.state.store.update_plan_item(
            item_id, fields, datetime.now(timezone.utc).isoformat()
        )

    @app.delete("/plans/{plan_id}/items/{item_id}", summary="删除计划草案步骤")
    def delete_plan_item(plan_id: int, item_id: int):
        require_draft_item(plan_id, item_id)
        return app.state.store.delete_plan_item(item_id, datetime.now(timezone.utc).isoformat())

    @app.post("/plans/{plan_id}/items/reorder", summary="调整计划步骤顺序")
    def reorder_plan_items(plan_id: int, body: PlanItemsReorderRequest):
        plan = get_plan(plan_id)
        if plan["status"] != "draft":
            raise HTTPException(409, detail="只有 draft 状态的计划可以调整步骤。")
        try:
            return app.state.store.reorder_plan_items(
                plan_id, body.item_ids, datetime.now(timezone.utc).isoformat()
            )
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from None

    @app.post("/plans/{plan_id}/create-tasks", summary="将已确认计划转换为任务")
    def plan_create_tasks(plan_id: int):
        plan = get_plan(plan_id)
        if plan["status"] not in {"confirmed", "active"}:
            raise HTTPException(409, detail="请先确认计划，再创建任务。")
        if not plan["source_note_ids"]:
            raise HTTPException(409, detail="计划至少需要一个来源笔记，才能关联现有任务。")
        result, task_ids = app.state.store.create_plan_tasks(
            plan_id, plan["source_note_ids"][0], datetime.now(timezone.utc).isoformat()
        )
        return {"status": "created" if task_ids else "existing", "task_ids": task_ids, "plan": result}

    @app.post("/notes/{note_id}/refresh-source", summary="刷新笔记正文和临时图片地址")
    async def refresh_note_source(note_id: str, body: RefreshSourceRequest | None = None):
        current = get_note(note_id)
        # 新分享文本通常带有更新的临时授权参数；接口和日志均不会回显它。
        candidate = body.share_text if body and body.share_text else current.get("source_url", "")
        try:
            url = share_url(candidate)
            html, final_url = await fetch_page(url)
            fresh_note = parse_note(html, final_url, requested_url=url)
            refreshed = app.state.store.refresh_note_source(
                note_id, fresh_note, datetime.now(timezone.utc).isoformat()
            )
        except ValueError as exc:
            raise HTTPException(409, detail=str(exc)) from None
        except ImportProblem as exc:
            raise HTTPException(422, detail=str(exc)) from None
        except httpx.HTTPError as exc:
            _error_code, message = classify_network_error(exc)
            raise HTTPException(502, detail=message) from None
        # 正文变了，旧的分块与向量就过期了；刷新后立刻重建，
        # 避免"刷新过正文反而搜不到"——用户刚从来源页更新过内容。
        index = await asyncio.to_thread(indexing.index_safely, app.state.store, note_id)
        return {
            "status": "refreshed",
            "note": refreshed,
            "index": index,
            "next_steps": [
                f"POST /notes/{note_id}/assets/download",
                f"POST /notes/{note_id}/analyze?force=true",
            ],
        }

    def require_folder(folder_id: int, active_only: bool = False):
        folder = app.state.store.folder(folder_id)
        if folder is None or (active_only and folder["status"] != "active"):
            raise HTTPException(404, detail="收藏夹不存在或已归档。")
        return folder

    @app.post("/folders", status_code=201, summary="创建收藏夹")
    def create_folder(body: FolderCreateRequest):
        now = datetime.now(timezone.utc).isoformat()
        try:
            return app.state.store.create_folder(
                body.name.strip(), body.description.strip(), body.color, now
            )
        except sqlite3.IntegrityError:
            # 秋招知识点：业务冲突使用 409，与参数格式错误 422 区分。
            raise HTTPException(409, detail="同名收藏夹已经存在。") from None

    @app.get("/folders", summary="查看收藏夹列表")
    def list_folders(include_archived: bool = False):
        return {"items": app.state.store.folders(include_archived)}

    @app.get("/folders/{folder_id}", summary="查看收藏夹")
    def get_folder(folder_id: int):
        return require_folder(folder_id)

    @app.put("/folders/{folder_id}", summary="修改收藏夹")
    def update_folder(folder_id: int, body: FolderUpdateRequest):
        require_folder(folder_id, active_only=True)
        fields = body.model_dump(exclude_unset=True)
        if "name" in fields:
            fields["name"] = fields["name"].strip()
        if "description" in fields:
            fields["description"] = fields["description"].strip()
        try:
            return app.state.store.update_folder(
                folder_id, fields, datetime.now(timezone.utc).isoformat()
            )
        except sqlite3.IntegrityError:
            raise HTTPException(409, detail="同名收藏夹已经存在。") from None

    @app.delete("/folders/{folder_id}", summary="归档收藏夹（软删除）")
    def archive_folder(folder_id: int):
        require_folder(folder_id, active_only=True)
        return app.state.store.update_folder(
            folder_id, {"status": "archived"}, datetime.now(timezone.utc).isoformat()
        )

    @app.post("/notes/{note_id}/folders/{folder_id}", summary="把笔记加入收藏夹")
    def add_note_folder(note_id: str, folder_id: int):
        get_note(note_id)
        require_folder(folder_id, active_only=True)
        created = app.state.store.add_note_to_folder(
            note_id, folder_id, "manual", datetime.now(timezone.utc).isoformat()
        )
        return {"status": "added" if created else "existing", "folder_id": folder_id}

    @app.get("/notes/{note_id}/folders", summary="查看笔记所在的收藏夹")
    def get_note_folders(note_id: str):
        get_note(note_id)
        return {"items": app.state.store.folders_for_note(note_id)}

    @app.delete("/notes/{note_id}/folders/{folder_id}", summary="将笔记移出收藏夹")
    def remove_note_folder(note_id: str, folder_id: int):
        get_note(note_id)
        require_folder(folder_id)
        removed = app.state.store.remove_note_from_folder(note_id, folder_id)
        if not removed:
            raise HTTPException(404, detail="这篇笔记不在该收藏夹中。")
        return {"status": "removed", "folder_id": folder_id}

    @app.get("/folders/{folder_id}/notes", summary="查看收藏夹中的笔记")
    def get_folder_notes(folder_id: int, limit: int = Query(50, ge=1, le=100),
                         offset: int = Query(0, ge=0)):
        require_folder(folder_id)
        return {"items": app.state.store.notes_in_folder(folder_id, limit, offset)}

    @app.post("/notes/{note_id}/folder-suggestions", summary="让 AI 推荐收藏夹")
    async def create_folder_suggestion(
        note_id: str, force: bool = Query(False, description="是否忽略缓存并重新调用模型")
    ):
        note = get_note(note_id)
        cached = app.state.store.folder_suggestion(note_id)
        if cached is not None and not force:
            # operation 描述本次请求；status 描述建议处于 pending/confirmed 哪个业务阶段。
            return {"operation": "cached", **cached}

        analysis_record = app.state.store.analysis(note_id)
        if analysis_record is None:
            raise HTTPException(409, detail="请先完成这篇笔记的 AI 分析。")
        folders = app.state.store.folders(include_archived=False)
        if not folders:
            raise HTTPException(409, detail="请先创建至少一个启用中的收藏夹。")
        confirmation = app.state.store.intent_confirmation(note_id)
        try:
            result, model = await suggest_folders(
                note, analysis_record["analysis"], confirmation, folders,
                # 同一篇笔记在不同用户手里应该进不同收藏夹，靠的就是这份记忆。
                app.state.store.active_memories(),
            )
        except AIProblem as exc:
            raise HTTPException(502, detail=str(exc)) from None
        saved = app.state.store.save_folder_suggestion(
            note_id, model, result, datetime.now(timezone.utc).isoformat()
        )
        return {"operation": "suggested", **saved}

    @app.get("/notes/{note_id}/folder-suggestions", summary="查看 AI 收藏夹建议")
    def get_folder_suggestion(note_id: str):
        get_note(note_id)
        suggestion = app.state.store.folder_suggestion(note_id)
        if suggestion is None:
            raise HTTPException(404, detail="这篇笔记还没有收藏夹建议。")
        return suggestion

    @app.post("/notes/{note_id}/folder-suggestions/confirm", summary="确认或修改 AI 收藏夹建议")
    def confirm_folder_suggestion(note_id: str, body: FolderSuggestionConfirmRequest):
        get_note(note_id)
        if app.state.store.folder_suggestion(note_id) is None:
            raise HTTPException(409, detail="请先生成收藏夹建议。")
        # Pydantic 校验数量，业务层再校验重复项和收藏夹状态。
        if len(set(body.folder_ids)) != len(body.folder_ids):
            raise HTTPException(400, detail="folder_ids 不能包含重复收藏夹。")
        for folder_id in body.folder_ids:
            require_folder(folder_id, active_only=True)
        saved = app.state.store.confirm_folder_suggestion(
            note_id, body.folder_ids, datetime.now(timezone.utc).isoformat()
        )
        return {"status": "confirmed", **saved}

    @app.post("/notes/{note_id}/index", summary="用 LangChain 建立文档和中文分块")
    def index_note(note_id: str):
        # 秋招知识点：API 层只做编排；切分在 knowledge.py，持久化在 database.py，
        # 组合流程（含收藏时的"索引 + 向量"）在 indexing.py，端点与工具共用一份实现。
        get_note(note_id)
        try:
            result = indexing.index_note(app.state.store, note_id)
        except indexing.EmptyContent as exc:
            # 秋招知识点：422 表示请求格式可解析，但业务内容无法处理。
            raise HTTPException(422, detail=str(exc)) from None
        return {"status": result["status"], "document": result["document"]}

    @app.get("/notes/{note_id}/document", summary="查看笔记对应的知识库文档")
    def get_document(note_id: str):
        # 秋招知识点：404 表示资源不存在，与参数校验失败（400/422）语义不同。
        get_note(note_id)
        document = app.state.store.document_for_note(note_id)
        if document is None:
            raise HTTPException(404, detail="这篇笔记尚未建立知识库索引。")
        return document

    @app.get("/documents/{document_id}/chunks", summary="按顺序查看 LangChain 分块")
    def get_document_chunks(document_id: int):
        chunks = app.state.store.document_chunks(document_id)
        if not chunks:
            raise HTTPException(404, detail="文档不存在或尚无分块。")
        return {"items": chunks}

    @app.post("/documents/{document_id}/embed", summary="为尚未向量化的 Chunk 生成 Embedding")
    def embed_document(document_id: int):
        # 与收藏流程共用 app.indexing 的实现，避免"手动能搜、收藏后搜不到"的逻辑漂移。
        # 这里显式传入本模块的 embed_texts：测试习惯替换 app.main.embed_texts，
        # 显式传参比依赖 indexing 模块的默认值更稳定。
        result = indexing.embed_document(app.state.store, document_id, embedder=embed_texts)
        if result["status"] == "missing":
            raise HTTPException(404, detail="文档不存在或尚无分块。")
        return result

    @app.post("/search", summary="使用 Embedding 余弦相似度检索知识库")
    def semantic_search(body: SearchRequest):
        selected = search_knowledge(
            app.state.store, body.query.strip(), body.top_k, body.deduplicate_notes,
            query_embedder=embed_search_query,
        )
        if not selected:
            raise HTTPException(409, detail="知识库中还没有可检索向量，请先建立索引并生成 Embedding。")
        return {
            "query": body.query.strip(),
            "model": EMBEDDING_MODEL,
            "total_candidates": len(app.state.store.searchable_embeddings(EMBEDDING_MODEL)),
            "deduplicated_notes": body.deduplicate_notes,
            "items": selected,
        }

    def asset_items(note_id):
        # 只在文件存在且指纹一致时提供访问地址，不暴露本机绝对路径。
        items = app.state.store.assets(note_id)
        for item in items:
            ready = local_file_valid(app.state.media_root, item)
            item["file_available"] = ready
            item["file_url"] = f"/assets/{item['id']}/file" if ready else None
        return items

    @app.post("/notes/{note_id}/assets/download", summary="下载图片；重复执行会跳过已有完整文件")
    def download_assets(note_id: str):
        note = get_note(note_id)
        # 普通 def 接口由框架放在线程池执行，不阻塞异步事件循环。
        # 仍是同步响应：浏览器要等本批下载结束。生产版需持久任务队列。
        if not app.state.media_lock.acquire(blocking=False):
            raise HTTPException(409, detail="已有图片下载进行中，请稍后重试。")
        try:
            result = download_note_assets(app.state.store, note, app.state.media_root)
            return {**result, "items": asset_items(note_id)}
        finally:
            app.state.media_lock.release()

    @app.get("/notes/{note_id}/assets", summary="按原笔记顺序查看图片保存状态")
    def list_assets(note_id: str):
        get_note(note_id)  # 笔记不存在时返回 404；查询不触发下载或建记录。
        return {"items": asset_items(note_id)}

    @app.get("/assets/{asset_id}/file", summary="打开已保存的图片文件")
    def asset_file(asset_id: int):
        asset = app.state.store.asset(asset_id)
        if asset is None or not local_file_valid(app.state.media_root, asset):
            raise HTTPException(404, detail="图片未下载、文件缺失或已损坏，请重新下载。")
        return FileResponse(storage_path(app.state.media_root, asset["storage_key"]),
                            media_type=asset["mime_type"], headers={"X-Content-Type-Options": "nosniff"})

    @app.post("/notes/{note_id}/analyze", summary="用 DeepSeek 分析文字和图片")
    async def analyze(note_id: str, force: bool = Query(False, description="是否忽略缓存并重新付费分析")):
        note = get_note(note_id)
        # 默认优先返回已保存结果，避免刷新页面或重复点击造成重复费用。
        cached = app.state.store.analysis(note_id)
        if cached is not None and not force:
            return {"status": "cached", **cached}
        try:
            result, model = await analyze_note(
                note, app.state.store.assets(note_id), app.state.media_root,
                # 注入用户确认过的长期记忆：同一篇笔记，带记忆的分析会更贴近
                # 这位用户自己的分类习惯；没有记忆时该参数为空列表，提示词不变。
                app.state.store.active_memories(),
            )
        except AIProblem as exc:
            raise HTTPException(502, detail=str(exc)) from None
        created_at = datetime.now(timezone.utc).isoformat()
        saved = app.state.store.save_analysis(note_id, model, result, created_at)
        return {"status": "analyzed", **saved}

    @app.get("/notes/{note_id}/analysis", summary="读取已经保存的 AI 分析")
    def get_analysis(note_id: str):
        get_note(note_id)
        result = app.state.store.analysis(note_id)
        if result is None:
            raise HTTPException(404, detail="这篇笔记还没有进行 AI 分析。")
        return result

    @app.get("/notes/{note_id}/intent", summary="查看待确认或已确认的收藏意图")
    def get_intent(note_id: str):
        get_note(note_id)
        analysis_record = app.state.store.analysis(note_id)
        if analysis_record is None:
            raise HTTPException(404, detail="请先完成 AI 分析，再确认收藏意图。")

        analysis = analysis_record["analysis"]
        confirmation = app.state.store.intent_confirmation(note_id)
        return {
            "note_id": note_id,
            # 没有确认记录就是 pending，方便手机端显示“等待确认”。
            "status": confirmation["decision"] if confirmation else "pending",
            "proposal": {
                "category": analysis.get("category", "其他"),
                "intent": analysis.get("inferred_intent", ""),
                "confidence": analysis.get("intent_confidence"),
                "suggested_actions": analysis.get("suggested_actions", []),
            },
            "confirmation": confirmation,
        }

    async def extract_memory_after_confirmation(note_id: str):
        """确认意图之后的附加动作：提炼长期记忆并落库。

        秋招知识点（后台任务的边界）：这里刻意不写 jobs 表。记忆提炼是"尽力而为"
        的副产品，没有客户端轮询它的进度；而 BackgroundTasks 只存在于当前进程
        的内存里，重启即丢。为一次可丢失的后台动作再建任务表、状态机和轮询接口
        属于过度设计——真正需要可靠投递时应该上持久队列（Celery/RQ），
        而不是自己搭半个队列。
        """
        store = app.state.store
        note = store.get(note_id)
        confirmation = store.intent_confirmation(note_id)
        if note is None or confirmation is None:
            return
        # 已有记忆一起传进去，让模型在提示词内自查重复。
        memories, model = await extract_memories_safe(
            note, confirmation, store.active_memories()
        )
        if not memories or not model:
            # 降级路径：safe 包装已经写过 memory_extract_failed 日志。
            return
        candidates = store.save_memory_candidates(
            model, memories, note_id, datetime.now(timezone.utc).isoformat()
        )
        logger.info("memory_candidates_ready note_id=%s count=%d", note_id, len(candidates))

    @app.put("/notes/{note_id}/intent", summary="确认、修改或忽略模型推测的收藏意图")
    def confirm_intent(note_id: str, body: IntentConfirmationRequest,
                       background_tasks: BackgroundTasks):
        state = get_intent(note_id)
        proposal = state["proposal"]

        if body.decision == "edited" and (not body.category or not body.intent):
            raise HTTPException(400, detail="修改后确认时，category 和 intent 都不能为空。")

        # confirmed 必须完整采用模型建议。即使 Swagger 自动填了 "string"，也不会污染数据。
        # 只有 edited 才读取用户填写的文字；dismissed 不会产生待执行行动。
        if body.decision == "confirmed":
            category = proposal["category"].strip()
            intent = proposal["intent"].strip()
            actions = proposal["suggested_actions"]
        elif body.decision == "edited":
            category = body.category.strip()
            intent = body.intent.strip()
            actions = body.selected_actions if body.selected_actions is not None else proposal["suggested_actions"]
        else:
            category = proposal["category"].strip()
            intent = proposal["intent"].strip()
            actions = []
        # 清理空行动并限制单条长度，避免把异常大的文本存进数据库。
        actions = [action.strip()[:500] for action in actions if action.strip()]

        if not category or not intent:
            raise HTTPException(400, detail="分类或收藏意图为空，请填写后再确认。")
        updated_at = datetime.now(timezone.utc).isoformat()
        confirmation = app.state.store.save_intent_confirmation(
            note_id, body.decision, category, intent, actions, updated_at
        )
        # 只有用户真正采用（confirmed / edited）时才值得学习：dismissed 表示
        # "模型猜错了"，这条信息本身没有可复用的偏好，也就没必要多花一次调用。
        memory_job = "queued" if body.decision != "dismissed" else "skipped"
        if memory_job == "queued":
            # 先返回确认结果，提炼在响应之后执行：提炼失败不会影响这次确认。
            background_tasks.add_task(extract_memory_after_confirmation, note_id)
        return {**state, "status": body.decision, "confirmation": confirmation,
                "memory_job": memory_job}

    @app.post("/notes/{note_id}/tasks", summary="从已确认的收藏意图创建任务")
    def create_task(note_id: str, body: TaskCreateRequest):
        get_note(note_id)
        confirmation = app.state.store.intent_confirmation(note_id)
        if confirmation is None or confirmation["decision"] not in {"confirmed", "edited"}:
            raise HTTPException(409, detail="请先确认或修改收藏意图，再创建任务。")
        valid_timezone(body.timezone)
        scheduled_at = utc_text(body.scheduled_at, "scheduled_at")
        reminder_at = utc_text(body.reminder_at, "reminder_at") if body.reminder_at else None
        if reminder_at and reminder_at > scheduled_at:
            raise HTTPException(400, detail="提醒时间不能晚于计划执行时间。")
        now = datetime.now(timezone.utc).isoformat()
        return app.state.store.create_task(
            note_id, body.title.strip(), body.details.strip(), scheduled_at,
            reminder_at, body.timezone, body.source_action, now
        )

    @app.get("/tasks", summary="查看任务列表")
    def list_tasks(status: Literal["pending", "completed", "cancelled"] | None = None,
                   limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
        return {"items": app.state.store.tasks(status, limit, offset)}

    def require_task(task_id: int):
        task = app.state.store.task(task_id)
        if task is None:
            raise HTTPException(404, detail="任务不存在。")
        return task

    @app.get("/tasks/{task_id}", summary="查看单个任务")
    def get_task(task_id: int):
        return require_task(task_id)

    @app.put("/tasks/{task_id}", summary="修改任务内容或时间")
    def update_task(task_id: int, body: TaskUpdateRequest):
        current = require_task(task_id)
        fields = body.model_dump(exclude_unset=True)
        timezone_name = fields.get("timezone", current["timezone"])
        valid_timezone(timezone_name)
        if "scheduled_at" in fields:
            fields["scheduled_at"] = utc_text(fields["scheduled_at"], "scheduled_at")
        if "reminder_at" in fields and fields["reminder_at"] is not None:
            fields["reminder_at"] = utc_text(fields["reminder_at"], "reminder_at")
        scheduled_at = fields.get("scheduled_at", current["scheduled_at"])
        reminder_at = fields.get("reminder_at", current["reminder_at"])
        if reminder_at and reminder_at > scheduled_at:
            raise HTTPException(400, detail="提醒时间不能晚于计划执行时间。")
        if "title" in fields:
            fields["title"] = fields["title"].strip()
        if "details" in fields:
            fields["details"] = fields["details"].strip()
        return app.state.store.update_task(
            task_id, fields, datetime.now(timezone.utc).isoformat()
        )

    @app.post("/tasks/{task_id}/complete", summary="将任务标记为已完成")
    def complete_task(task_id: int):
        require_task(task_id)
        return app.state.store.update_task(
            task_id, {"status": "completed"}, datetime.now(timezone.utc).isoformat()
        )

    @app.delete("/tasks/{task_id}", summary="取消任务（保留记录，可追溯）")
    def cancel_task(task_id: int):
        require_task(task_id)
        return app.state.store.update_task(
            task_id, {"status": "cancelled"}, datetime.now(timezone.utc).isoformat()
        )

    # ------------------------------ 长期记忆管理 ------------------------------
    # 记忆是模型自动提炼出来的，可能过时、也可能理解偏了，所以必须给用户
    # 查看、修正、删除的出口。能被纠正的记忆才是资产，只能被动接受的记忆是负担。

    def require_memory(memory_id: int):
        memory = app.state.store.memory(memory_id)
        if memory is None:
            raise HTTPException(404, detail="记忆不存在。")
        return memory

    @app.get("/memory-candidates", summary="查看 AI 提议、等待你确认的记忆")
    def list_memory_candidates(
        status: Literal["pending", "confirmed", "dismissed"] | None = None,
        source_note_id: str | None = None,
        limit: int = Query(50, ge=1, le=100),
    ):
        return {"items": app.state.store.memory_candidates(status, source_note_id, limit)}

    @app.post("/memory-candidates/{candidate_id}/review", summary="确认、修改或忽略记忆候选")
    def review_memory_candidate(candidate_id: int, body: MemoryCandidateReviewRequest):
        content = body.content.strip() if body.content is not None else None
        if body.content is not None and not content:
            raise HTTPException(400, detail="记忆内容不能为空。")
        try:
            result = app.state.store.review_memory_candidate(
                candidate_id, body.decision, datetime.now(timezone.utc).isoformat(),
                body.type, content,
            )
        except ValueError as exc:
            raise HTTPException(409, detail=str(exc)) from None
        if result is None:
            raise HTTPException(404, detail="记忆候选不存在。")
        return result

    @app.post("/memories", status_code=201, summary="主动保存你确认的长期偏好")
    def create_user_memory(body: UserMemoryCreateRequest):
        content = body.content.strip()
        if not content:
            raise HTTPException(400, detail="记忆内容不能为空。")
        try:
            return app.state.store.create_user_memory(
                body.type, content, datetime.now(timezone.utc).isoformat()
            )
        except sqlite3.IntegrityError:
            raise HTTPException(409, detail="相同内容的记忆已存在。") from None

    @app.get("/memories", summary="查看长期记忆列表")
    def list_memories(status: Literal["active", "superseded", "dismissed"] | None = None,
                      limit: int = Query(50, ge=1, le=100),
                      offset: int = Query(0, ge=0)):
        # 默认不筛 status：用户要能看到"被删掉的记忆"，否则无从判断误删。
        return {"items": app.state.store.memories(status, limit, offset)}

    @app.get("/memories/{memory_id}", summary="查看单条记忆")
    def get_memory(memory_id: int):
        return require_memory(memory_id)

    @app.put("/memories/{memory_id}", summary="修正记忆内容、类型，或恢复已删除的记忆")
    def update_memory(memory_id: int, body: MemoryUpdateRequest):
        require_memory(memory_id)
        # exclude_unset=True：只处理请求体里真正出现的字段，没提交的保持原值。
        fields = body.model_dump(exclude_unset=True)
        # superseded 由系统在“新记忆替换旧口径”时自动打标，用户手动设置
        # 会破坏状态机语义（superseded 必须有替换它的那条新记忆可追溯）。
        if fields.get("status") == "superseded":
            raise HTTPException(
                400, detail="superseded 是系统内部状态，不能手动设置；要下线一条记忆请删除它。")
        if isinstance(fields.get("content"), str):
            fields["content"] = fields["content"].strip()
            # min_length 能拦住空串，但拦不住纯空格，这里补一次业务校验。
            if not fields["content"]:
                raise HTTPException(400, detail="记忆内容不能为空。")
        try:
            return app.state.store.update_memory(
                memory_id, fields, datetime.now(timezone.utc).isoformat()
            )
        except sqlite3.IntegrityError:
            # content_hash 的 UNIQUE 约束兜底：改成了另一条记忆已有的内容。
            raise HTTPException(409, detail="已存在相同内容的记忆。") from None

    @app.delete("/memories/{memory_id}", summary="删除记忆（软删除，保留记录）")
    def dismiss_memory(memory_id: int):
        require_memory(memory_id)
        # 与任务取消保持一致：只改状态、不物理删除，之后还能改回 active 恢复。
        return app.state.store.update_memory(
            memory_id, {"status": "dismissed"}, datetime.now(timezone.utc).isoformat()
        )

    return app


# 启动命令中的 app.main:app = app 文件夹 / main.py 模块 / 名为 app 的应用对象。
app = create_app()
