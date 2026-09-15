# 记忆层（Memory Layer）设计文档

状态：设计稿，待确认后编码 | 分支：`feature/memory` | 日期：2026-09-14

## 1. 为什么需要记忆

当前系统里 AI 的每一次分析都是"失忆"的：

- `analyses` 表只服务单篇笔记，模型不知道这个用户是谁、偏好什么；
- 用户在 `PUT /notes/{id}/intent` 里做过的 **confirmed / edited / dismissed** 决定，
  只停留在单条笔记上，对后续笔记的建议没有任何影响；
- 收藏夹建议（folder-suggestions）只能看到"现有文件夹列表"，看不到"用户习惯怎么归档"。

记忆层要补的就是这条**反馈闭环**：用户每次确认意图 → 系统沉淀可复用偏好 → 下次分析时注入提示词 → 建议越用越准。这也是本项目区别于普通"收藏夹工具"的核心差异化（Knowledge → Action 之外，再加 Personalization）。

## 2. 目标与非目标

**目标**

1. 意图确认后自动提取 0~2 条可复用记忆（DeepSeek 提取）；
2. 记忆注入 `analyze` 与 `folder-suggestions` 的提示词；
3. 用户可查看、修正、删除记忆（记忆错了必须能改）；
4. 全程本地 SQLite 存储，不新增外部依赖。

**非目标（本期不做）**

- 不做向量检索式记忆（记忆条数少，直接全量注入前 N 条即可；数据量大了再升级）；
- 不做多用户隔离（沿用现有"本地单用户"边界）；
- 不做记忆自动过期/衰减（先靠用户手动 dismiss）。

## 3. 数据流总览

```
用户确认意图 PUT /notes/{id}/intent  (confirmed / edited)
        │  确认先落库，不受提取失败影响
        ▼
后台提取：DeepSeek 读「笔记标题/正文 + 用户确认的分类/意图 + 已有 active 记忆」
        │  输出 0~2 条新记忆；与已有记忆重复则不输出
        ▼
memories 表（status=active）
        │
        ▼
下次调用 POST /notes/{id}/analyze 或 /notes/{id}/folder-suggestions 时，
把 active 记忆（最多 10 条，按 updated_at 倒序）拼进提示词
        │
        ▼
建议更贴合用户 → 用户确认率上升 → 沉淀更多记忆（反馈闭环）
```

## 4. 数据库设计（database.py 新增）

```sql
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('preference', 'habit', 'correction')),
    content TEXT NOT NULL,
    source_note_id TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'dismissed')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

- **type 三类**：
  - `preference`：内容偏好，如"用户偏好居家徒手训练，不去健身房"；
  - `habit`：行为习惯，如"用户习惯把做饭笔记归到健身（减脂餐）"；
  - `correction`：用户纠正过模型的地方，如"模型曾把减脂餐分类为做饭，用户改成了健身"。
- `source_note_id` 可空：便于追溯"这条记忆从哪篇笔记来"（类似 RAG 的可追溯性）。
- `status` 软删除：`DELETE /memories/{id}` 只改 `dismissed`，保留历史可审计——与现有 tasks 的取消策略一致。
- `updated_at`：用户编辑过记忆后更新，注入时按它倒序（用户最近整理过的最可信）。

`NoteStore` 新增方法：`save_memories()`、`memories(status, limit, offset)`、
`memory(id)`、`update_memory(id, fields)`、`active_memories(limit=10)`。

## 5. API 设计

### 新增接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/memories` | 查询记忆，`?status=active\|dismissed&limit&offset`，默认只看 active |
| PUT | `/memories/{id}` | 修正内容或类型；`content` 1~500 字，`type` 三选一 |
| DELETE | `/memories/{id}` | 软删除（status→dismissed），响应 `{status: dismissed}` |

### 修改的既有接口（向后兼容）

- `PUT /notes/{note_id}/intent`：确认/修改成功后，后台触发记忆提取。
  响应新增 `memory_job: "queued" | "skipped"` 字段（dismissed 不触发，返回 skipped）。
  **提取失败只记日志，绝不影响确认本身的成功响应**——意图确认是主链路，记忆是副产品。
- `POST /notes/{note_id}/analyze`：提示词新增"已知用户偏好"段落（见 §6）。
- `POST /notes/{note_id}/folder-suggestions`：同上注入。

## 6. DeepSeek 提取设计（ai.py 新增 `extract_memories()`）

**输入拼装**：笔记标题 + 正文摘要（截断至 800 字）+ 用户确认的 category/intent/selected_actions
+ 已有 active 记忆列表（用于去重）。

**提示词要点**：

```
你在为一位小红书收藏用户维护长期记忆。
根据这篇笔记的内容和用户刚刚确认的收藏意图，判断是否产生了值得长期记住的
新信息。只输出以下三类之一的内容：preference（内容偏好）、habit（行为习惯）、
correction（用户纠正过模型的分类或理解）。
规则：
1. 最多 2 条，每条不超过 100 字，用第三人称陈述（"用户……"）；
2. 只记可复用的稳定信息，不记单篇笔记的事实细节；
3. 若与已有记忆语义重复，直接跳过；
4. 没有值得记的就输出空数组。
输出严格 JSON：{"memories": [{"type": "...", "content": "..."}]}
```

**工程细节**：

- 去重分两层：提示词内让模型自查语义重复 + 代码层 exact-match 比对 `content`；
- 输出校验：复用现有 `AIProblem` 错误通道；type/content 不合法的条目直接丢弃；
- 调用失败（超时/5xx）静默降级：写日志 `memory_extract_failed note_id=...`，本次不沉淀。

## 7. 注入策略（main.py / ai.py）

- 注入内容：`active_memories(limit=10)`，格式 `[-] type: content` 每行一条；
- analyze 提示词开头新增段落：

```
已知用户偏好（来自历史确认记录，供参考但不要盲从）：
- preference: 用户偏好居家徒手训练
- correction: 用户习惯把减脂餐归为健身而非做饭
```

- 上限 10 条是成本护栏：记忆条目短（≤100 字），10条约 1000 字，对 token 成本影响可忽略；
- "不要盲从"一句是防 prompt 偏置：记忆是先验不是硬约束，模型仍以当前笔记内容为准。

## 8. 风险与边界

| 风险 | 对策 |
|------|------|
| 提取调用增加成本 | 每次确认仅 1 次小请求（输入 <2K token）；dismissed 不触发 |
| 错误记忆污染后续建议 | 用户可 PUT 修正 / DELETE 软删；提示词声明"供参考不盲从" |
| 记忆内容注入提示词的安全 | 内容截断 500 字、剥离控制字符；不做模板拼接之外的任何解释 |
| 提取失败拖慢确认响应 | 后台 BackgroundTasks 执行，失败仅记日志 |
| 与现有测试冲突 | 所有新逻辑可注入假 client；既有接口响应只加字段不改字段 |

## 9. 实施步骤（预计 3 个提交）

1. **commit 1**：`memories` 表 + NoteStore 方法 + `extract_memories()` + 意图确认触发（含单元测试）；
2. **commit 2**：analyze / folder-suggestions 提示词注入（含注入测试）；
3. **commit 3**：`/memories` 管理接口 + 集成测试 + README 更新（新增"记忆层"章节与秋招知识点地图）。

## 10. 秋招知识点标注（面试讲法）

- **反馈闭环 / 数据飞轮**：用户行为（confirm/edit/dismiss）变成模型先验，产品越用越准；
  能讲清"记忆如何产生、如何验证有效（对比注入前后建议采纳率）"。
- **记忆分层**：短期（单次会话上下文）vs 长期（结构化偏好）；为什么第一期选结构化
  而不是向量库——条数少、需人工可读可改，向量检索是过早优化（数据量大了再迁移）。
- **软删除与可追溯**：memories 带 source_note_id，删除只是打标，与 tasks 的取消策略一致。
- **成本与幂等**：dismissed 不触发、重复内容跳过、注入条数设上限。
- **Prompt 工程**：记忆作为先验注入但声明"不盲从"，避免历史偏好压制当前内容。
