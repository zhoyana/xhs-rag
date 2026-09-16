# 第一课：小红书 Python 后端

本地单用户学习原型，不是可公开部署的产品。不接收 Cookie、不绕过登录或验证码。

## 1. 环境和运行（PowerShell）

在本文件所在目录打开终端。有 uv 时：

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

若已安装标准 Python，也可以用 `py -3 -m venv .venv` 创建环境，再用 `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` 安装依赖。

## 配置 DeepSeek 视觉模型

1. 复制 `.env.example`，把副本命名为 `.env`。
2. 用记事本或代码编辑器打开 `.env`，只替换 `DEEPSEEK_API_KEY=` 后面的示例文字。
3. 保存文件并重新启动后端。
4. 打开 `/docs`，执行 `GET /ai/config`。看到 `configured: true` 表示后端已读到密钥。

`.env` 仅保存在本机，并已加入 `.gitignore`。不要截图、上传或把密钥写进安卓 App。

图片下载完成后，在 `/docs` 执行 `POST /notes/{note_id}/analyze` 即可进行首次分析。
默认 `force=false`，已有结果会直接从 SQLite 返回，不重复调用模型；只有确实需要更新时才使用
`force=true`。使用 `GET /notes/{note_id}/analysis` 可以随时读取保存的分析结果。

## 批量收件箱

在 `/docs` 执行 `POST /notes/import-batch`，把多条小红书链接粘贴到同一个
`share_text` 字符串中即可。支持每行一个链接，也支持夹带普通分享文案；每批最多 20 条。
接口会逐条返回 `imported`、`existing`、`duplicate` 或 `failed`，单条失败不会回滚其他成功项。
批量收件箱只导入笔记，不自动下载图片或调用 AI，避免一次操作产生过多请求和费用。

## 确认收藏意图

分析完成后，先用 `GET /notes/{note_id}/intent` 查看模型建议。它最初显示 `pending`。
再通过 `PUT /notes/{note_id}/intent` 提交用户选择：`confirmed` 直接采用建议，`edited`
保存用户修改后的分类和意图，`dismissed` 表示暂不采用。确认数据和模型分析分表保存，
后续创建提醒时只使用用户确认过的结果。确认成功后，后台会尝试提炼长期记忆候选
（`dismissed` 除外）；候选需要用户再次审阅才会生效，详见后面的「长期记忆」一节。

## 任务接口

收藏意图确认后，通过 `POST /notes/{note_id}/tasks` 创建任务。时间必须带时区，后端会
统一转换成 UTC 保存，同时保留 `Asia/Shanghai` 这样的原时区名称。`GET /tasks` 查看列表，
`PUT /tasks/{task_id}` 修改任务，`POST /tasks/{task_id}/complete` 标记完成。
`DELETE /tasks/{task_id}` 使用软删除，只把状态改为 `cancelled`，不会清除历史记录。

打开 http://127.0.0.1:8000/docs 。停止服务按 Ctrl+C。

## 2. 按顺序测试

1. 展开 GET /health → Try it out → Execute，应看到 status 为 ok。
2. 展开 POST /notes/import → Try it out，将请求内容填写如下（替换为实际分享链接）：

```json
{"share_text": "粘贴小红书的完整分享内容或链接"}
```

3. Execute 后，成功返回 imported；重复导入同一笔记返回 existing。
4. GET /notes 查看已保存笔记。重启后笔记仍保留。
5. 失败返回真实错误，不生成虚构正文。422 可能是内容受限、无正文或解析器不适配；502 为网络错误；504 为超时。

接口测试页面使用外部 CDN 加载界面资源；若 /docs 空白，可先打开 /health 确认后端正常，再用 PowerShell 的 Invoke-RestMethod 测试。

## 3. 代码阅读顺序

- app/main.py：路由和请求校验。安卓客户端以后调用这些接口。
- app/extractor.py：分享文本 → URL → 网页 → 笔记数据。JSON5 解析，不执行页面代码。
- app/database.py：SQLite 存储，note_id 主键去重。
- tests/test_backend.py：使用合成网页验证流程，不依赖小红书在线可用性。

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 4. 当前边界

- 仅实现公开网页内嵌数据解析，真实链接必须单独测试；本地或测试成功不能证明部署服务器也能抓取。
- 无正文的图集会拒绝入库，不将空内容当完整笔记。
- 导入笔记时 images 仍是原始远程链接；0.2.0 新增独立的图片下载接口（见下节）。没有 OCR、视频转写、模型调用或 Wiki 编排。
- 导入为有超时限制的同步请求；后续再改为持久任务队列。
- 去重不刷新原笔记；失败尚无持久任务记录。
- 来源链接不存访问 token，token 只用于当次请求；不要在公共日志中记录完整分享链接。
- 无鉴权、用户隔离或限流，只绑定本机地址。不要改成公网监听；上线前还需 SSRF/DNS/出站网络约束、安全审查等。
- 保留数据目录 data/，删除会丢失本地笔记。不要把它提交到公共 Git 仓库。

## 5. 参考（不是已验证的官方小红书接口）

- FastAPI 入门：https://fastapi.tiangolo.com/tutorial/first-steps/
- 内嵌笔记数据结构参考：https://github.com/bnchiang96/xiaohongshu-importer

下一课：用真实的几篇笔记测试提取完整性，再连接最小安卓界面。

## 第二课：保存图片（0.2.0）

更新依赖并重新启动后端后，刷新 /docs：

1. POST /notes/{note_id}/assets/download：输入已保存的笔记编号，点击 Execute，无需请求正文。
2. 返回 total / downloaded / skipped / failed。HTTP 200 只表示批处理已返回，必须检查 failed 和每项 status。
3. GET /notes/{note_id}/assets：按 position 原始顺序查看元数据；第一次点击下载前列表为空。
4. 把成功项 file_url 拼在 http://127.0.0.1:8000 后，可在浏览器打开本地图片。
5. 重复 POST：完整文件会 skipped，失败、缺失或被修改的文件会尝试重新下载。

图片保存在 data/media/笔记编号/序号-文件指纹.扩展名；数据库新增 assets 表，原 notes 表不变。
原笔记 warnings 描述导入时的范围，不会随下载自动修改；当前图片保存状态以 assets 为准。
文件按原始字节保存，不重新压缩；Pillow 检查结构并解码，扩展名取真实格式。

限制：每张最多20 MB、2500万像素，仅静态 JPEG/PNG/WebP；只允许已知小红书图片 CDN。
下载使用单进程互斥锁，正在下载时第二个请求返回409；只运行一个 Uvicorn worker。
当前不是后台队列，需要等本批结束。下载中退出应用时可留下 downloading 状态，再次 POST 会重新尝试。
异常退出可能留下 .part 临时文件或未登记文件；不会通过图片接口暴露，不自动清理历史文件。
不修改系统代理或禁用 TLS；链接过期时需要重新获取来源，不通过绕过登录解决。
未实现多用户权限、存储配额或完整 DNS 重绑定防护，仍只能本机使用。

如果在受限沙箱启动服务会出现 WinError 10013，需要在经授权可联网的终端运行。

阅读 app/media.py 学习下载与校验，tests/test_media.py 学习模拟网络和失败重试测试。

## 秋招知识点地图：LangChain 知识库层

当前新增的数据链路是 `notes → LangChain Document → chunks`。建议面试前能脱离代码解释：

- **RAG 分层**：加载、转换/切分、Embedding、存储、检索、生成各自解决什么问题。
- **Document**：`page_content` 保存正文，`metadata` 保存来源、标题、笔记 ID 等过滤和溯源信息。
- **Chunking**：块太大降低召回精度，块太小丢失上下文；overlap 缓解边界信息损失但增加成本。
- **中文切分**：中文不依赖空格分词，因此为递归切分器配置段落、句号、问号等分隔符。
- **可追溯性**：保存 `source_url`、`note_id`、`chunk_index` 和 `start_index`，回答才能展示证据来源。
- **幂等性**：通过 SHA-256 内容指纹识别未变化文档，重复索引不会重复写入或重复调用 Embedding。
- **事务**：文档更新和 chunks 重建必须原子提交，避免只写入半套索引。
- **原始数据与派生数据**：notes、图片是原始资产；documents、chunks 是可重新生成的索引。
- **约束与参数化 SQL**：UNIQUE 防重复，`?` 参数避免 SQL 注入，排序字段保证分块顺序。
- **测试分层**：纯切分逻辑使用单元测试；接口、SQLite 和重启持久化使用集成测试。
- **HTTP 语义**：404 表示资源不存在，422 表示内容无法处理，重复索引返回 `existing`。
- **下一阶段评测**：选择 Embedding 前先准备查询、相关笔记和预期 chunk，避免只凭演示判断 RAG 效果。

对应阅读顺序：`app/knowledge.py` → `NoteStore.index_document` → `POST /notes/{note_id}/index`
→ `tests/test_knowledge.py`。

## Embedding 模型冒烟测试

安装依赖：

```powershell
cd C:\xhs\xhs-backend
.\.venv-local\Scripts\python.exe -m pip install -r requirements.txt
```

运行独立冒烟测试（首次运行会下载本地模型，不需要 API Key）：

```powershell
.\.venv-local\Scripts\python.exe scripts\embedding_smoke.py
```

该脚本验证 512 维向量、L2 归一化和最基本的语义排序，但不代替真实检索评测集。

### Embedding 持久化

执行 `POST /documents/{document_id}/embed`，后端只为缺失或内容已变化的 Chunk 生成向量。
向量以 `float32` 二进制保存在 SQLite `embeddings.vector` BLOB 中；512 维约占 2 KB。
重复执行返回 `existing`，不会再次加载计算相同内容。表同时保存 model、dimension 和
content_hash，禁止把不同模型或过期内容的向量混在一起比较。

## 最小语义检索

`POST /search` 将自然语言问题编码成查询向量，与 SQLite 中全部有效 Chunk 向量计算
余弦相似度，并返回 Top-K。当前面向几十篇本地笔记，使用 NumPy 全量扫描便于理解和评测；
数据量增大后再迁移到 FAISS、Chroma 或 Qdrant。接口不会返回原始 512 维向量。

人工检索评测集位于 `tests/data/retrieval_eval.jsonl`。第一版按笔记级标注相关答案，
包含直接事实、同义改写、多答案与无答案问题；不要根据当前搜索排名反向修改标签。
后续评测脚本将据此计算 Hit@K、MRR，并记录无答案问题的最高相似度。

运行自动检索评测：

```powershell
.\.venv-local\Scripts\python.exe scripts\retrieval_eval.py
```

加 `--save-json` 可把逐题排名、分数和耗时保存到 `data/eval_reports/`。脚本默认只读；
若人工标注的相关笔记尚未生成 Embedding，会停止并列出缺失 note_id，避免错误地拉低指标。
评测默认调用正在运行的 `http://127.0.0.1:8000/search`，复用后端已加载的模型，避免两个
Python 进程各加载一份 PyTorch/BGE 导致内存不足，同时覆盖真实 HTTP 端到端链路。

## 长期记忆：让建议越用越准

前面的章节解决的是「把一篇笔记读懂」，这一节解决的是「把用户读懂」。系统会把用户在
确认意图时做出的选择先生成记忆候选；只有用户审阅后才成为长期记忆，后续分析才能使用。
旧版本已保存的 active 记忆不会被迁移或删除。

### 闭环

1. **提议**：`PUT /notes/{note_id}/intent` 确认成功后（`dismissed` 除外），后台从
   「笔记 + 用户确认的意图」提炼 0~2 条，存进 `memory_candidates`，状态为 `pending`。
   `memory_job=queued` 只表示安排提炼，不保证模型一定找到稳定偏好。
2. **确认**：`GET /memory-candidates?status=pending` 查看候选；使用
   `POST /memory-candidates/{id}/review` 提交 `confirmed`（可修改内容/类型）或 `dismissed`。
   只有 confirmed 会进入 active 长期记忆。用户也可 `POST /memories` 主动保存真实偏好。
3. **注入**：`POST /notes/{note_id}/analyze` 和 `POST /notes/{note_id}/folder-suggestions`
   会把最近 10 条生效中的记忆拼进提示词，并在提示词里明确声明"仅供参考、不要盲从"——
   记忆是先验而非硬约束，避免历史偏好压制当前内容。
4. **管理**：已生效的记忆仍可修正、停用或恢复：

| 接口 | 作用 |
|------|------|
| `GET /memory-candidates` | 查看待确认、已确认或已忽略的 AI 候选 |
| `POST /memory-candidates/{id}/review` | 确认/修改后确认或忽略候选 |
| `POST /memories` | 用户主动保存自己确认的长期偏好 |
| `GET /memories` | 查看记忆列表，默认包含已删除的（便于判断误删），可用 `status` 过滤 |
| `GET /memories/{memory_id}` | 查看单条 |
| `PUT /memories/{memory_id}` | 修正内容或类型；提交 `{"status": "active"}` 可恢复已删除的记忆 |
| `DELETE /memories/{memory_id}` | 软删除（`status=dismissed`），保留记录可追溯 |

### 三类记忆

`preference`（内容偏好）、`habit`（行为习惯）、`correction`（纠正记录），
取值由数据库 `CHECK` 约束兜底。

### 设计要点

- **幂等**：同一笔记、同一内容的候选不会重复创建；active 记忆用内容指纹去重。
- **人工确认**：候选和长期记忆分表，AI 的推测不会未经确认就影响下一次回答。
- **可追溯**：每条记忆记下 `source_note_id`（来自哪篇笔记）与 `model`（哪次模型调用产出）。
- **不可信输入**：笔记正文、意图、已有记忆、注入的记忆，进提示词前统一清洗
  （剥离控制字符、折叠空白、按长度截断），防止内容里的文字伪造出新的指令边界。
- **降级**：提炼失败只记日志（`memory_extract_failed`），绝不影响意图确认这条主链路。

### 秋招知识点

| 主题 | 位置 |
|------|------|
| CHECK / UNIQUE 约束做数据正确性兜底 | `app/database.py` 的 `memories` 建表语句 |
| 幂等写入与去重的区别 | `NoteStore.save_memory_candidates`、`NoteStore.save_memories` |
| Human-in-the-loop 与候选状态机 | `memory_candidates`、`NoteStore.review_memory_candidate` |
| 软删除与可审计性 | `NoteStore.update_memory`、`DELETE /memories/{id}` |
| 事务可见性：跨连接的读必须等提交 | `NoteStore.save_memories`（先收集 id，出事务再回读） |
| 提示词注入防御（控制字符 / 截断 / 指令边界） | `app/ai.py` 的 `_sanitize_inline`、`_memory_block` |
| 模型输出不可信与三段式校验 | `app/ai.py` 的 `_valid_memories` |
| 主链路与副产品解耦、静默降级与 fail fast 的边界 | `app/ai.py` 的 `extract_memories_safe` |
| BackgroundTasks 的边界、何时才该上持久队列 | `app/main.py` 的 `extract_memory_after_confirmation` |
| 记忆是先验而非硬约束（上下文偏置） | `app/ai.py` 的 `_memory_block` |
| 批量赋值 / 字段级越权防护 | `MemoryUpdateRequest` + `NoteStore.update_memory` 的 `allowed` 白名单 |
| 上下文预算与成本控制 | `MEMORY_*` 常量、`active_memories(limit=10)` |

### 测试

```powershell
.\.venv-local\Scripts\python.exe -m pytest tests/test_memories.py tests/test_memory_extract.py tests/test_memory_trigger.py tests/test_memory_candidates.py tests/test_memory_inject.py tests/test_memory_api.py -q
```

五个测试文件全部用替身替换真实模型调用，不联网、不产生费用：分别覆盖存储层约束、
提示词清洗与逐条校验、确认后触发落库、记忆注入接线、管理接口的越权与冲突处理。
