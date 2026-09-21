"""持久化模块：SQLite 是本地文件数据库，无需另外安装服务。"""
import hashlib
import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path

# __file__ 是当前文件路径；resolve() 转为绝对路径；parents[1] 取项目目录。
# Path 可以用 / 拼接路径，避免手动处理 Windows 的反斜杠。
DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "notes.sqlite3"


class NoteStore:
    # 类把相关的数据和操作组织在一起。self 表示当前的 NoteStore 实例。
    def __init__(self, path=DEFAULT_DB):
        # __init__ 在创建实例时运行：准备目录，并确保表存在。
        self.path = Path(path)
        # parents=True 创建缺失的父目录；exist_ok=True 允许目录已经存在。
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            # sqlite3 的 with 管理事务：正常结束提交，异常时回滚。
            # 注意它不保证在退出 with 时关闭连接；生产版本应显式管理连接生命周期。
            # PRIMARY KEY 保证 note_id 唯一；payload 用 JSON 文本保存整篇笔记。
            db.execute("CREATE TABLE IF NOT EXISTS notes (note_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        # CREATE IF NOT EXISTS 只增加新表，不清空以前已保存的笔记。
        with closing(sqlite3.connect(self.path)) as db, db:
            # jobs 保存后台处理的状态，不保存分享原文/临时 URL，避免把 xsec_token 长期落盘。
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, job_type TEXT NOT NULL,
                status TEXT NOT NULL, current_step TEXT NOT NULL,
                progress_json TEXT NOT NULL, result_json TEXT,
                note_id TEXT, error TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, finished_at TEXT
            )""")
            # 计划先保存为 draft；确认后才允许批量写入现有 tasks 表。
            db.execute("""CREATE TABLE IF NOT EXISTS plans (
                id INTEGER PRIMARY KEY, title TEXT NOT NULL, goal TEXT NOT NULL,
                source_note_ids TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'draft',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, confirmed_at TEXT
            )""")
            # 对话是短期记忆：thread 保存会话，message 保存按顺序发生的消息。
            # 与 memories（长期偏好）分表，避免把一句临时追问误当成长期画像。
            db.execute("""CREATE TABLE IF NOT EXISTS chat_threads (
                id TEXT PRIMARY KEY, title TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY, thread_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            )""")
            db.execute("""CREATE INDEX IF NOT EXISTS idx_chat_messages_thread
                ON chat_messages(thread_id, id)""")
            db.execute("""CREATE TABLE IF NOT EXISTS plan_items (
                id INTEGER PRIMARY KEY, plan_id INTEGER NOT NULL,
                item_index INTEGER NOT NULL, title TEXT NOT NULL, details TEXT NOT NULL DEFAULT '',
                scheduled_at TEXT NOT NULL, reminder_at TEXT, timezone TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', task_id INTEGER,
                UNIQUE(plan_id, item_index)
            )""")
            # 一个步骤可以引用多篇笔记；关系表比把列表塞进 plan_items 更易查询。
            db.execute("""CREATE TABLE IF NOT EXISTS plan_item_sources (
                plan_item_id INTEGER NOT NULL, note_id TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                PRIMARY KEY(plan_item_id, note_id)
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS assets (
                id INTEGER PRIMARY KEY, note_id TEXT NOT NULL, position INTEGER NOT NULL,
                source_url TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                storage_key TEXT, mime_type TEXT, size_bytes INTEGER, width INTEGER,
                height INTEGER, sha256 TEXT, error TEXT,
                UNIQUE(note_id, position)
            )""")
            # AI 分析和原笔记分开保存。重新分析只覆盖分析结果，不会修改原文和图片。
            db.execute("""CREATE TABLE IF NOT EXISTS analyses (
                note_id TEXT PRIMARY KEY, model TEXT NOT NULL,
                payload TEXT NOT NULL, created_at TEXT NOT NULL
            )""")
            # 用户的最终选择独立保存，避免把模型的“推测”误当成用户真实意图。
            db.execute("""CREATE TABLE IF NOT EXISTS intent_confirmations (
                note_id TEXT PRIMARY KEY, decision TEXT NOT NULL,
                category TEXT NOT NULL, intent TEXT NOT NULL,
                selected_actions TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")
            # 记忆层：用户确认意图后沉淀的长期偏好，供后续 AI 分析注入提示词。
            # type 用 CHECK 约束兜底：应用层校验管体验，数据库约束管数据正确性。
            # status 三态：active 生效 / superseded 被新记忆替换（系统专用）/
            # dismissed 用户删除。三者都保留记录，可审计可恢复。
            # embedding 三列缓存记忆的向量：embedding_model 记录生成模型，
            # embedding_hash 记录生成向量时的内容指纹（内容改过向量就过期）。
            db.execute("""CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                type TEXT NOT NULL CHECK(type IN ('preference', 'habit', 'correction')),
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE,
                model TEXT NOT NULL,
                source_note_id TEXT, source_label TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK(status IN ('active', 'superseded', 'dismissed')),
                embedding BLOB, embedding_model TEXT, embedding_hash TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")
            # AI 提议与用户确认分开：候选不进入 active_memories，也不会注入提示词。
            # 老的 active 记忆原样保留；新流程只改变之后提炼出的记忆。
            # source_label 是提炼那一刻的来源标题快照，见下方迁移处的说明。
            db.execute("""CREATE TABLE IF NOT EXISTS memory_candidates (
                id INTEGER PRIMARY KEY,
                type TEXT NOT NULL CHECK(type IN ('preference', 'habit', 'correction')),
                content TEXT NOT NULL, content_hash TEXT NOT NULL,
                model TEXT NOT NULL, source_note_id TEXT NOT NULL,
                source_label TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'confirmed', 'dismissed')),
                memory_id INTEGER, created_at TEXT NOT NULL, reviewed_at TEXT,
                UNIQUE(source_note_id, content_hash)
            )""")
            # 旧版本表的 status 只有 (active, dismissed) 两态且没有向量列。
            # SQLite 不能修改 CHECK 约束，只能“建新表 → 搬数据 → 换名”三步重建。
            # 秋招知识点：SQLite 的 ALTER TABLE 能力有限，schema 演进的惯例是
            # create-new / copy / drop / rename，全程在同一事务里，要么旧要么新。
            columns = {row[1] for row in db.execute("PRAGMA table_info(memories)").fetchall()}
            if "embedding" not in columns:
                db.execute("""CREATE TABLE memories_new (
                    id INTEGER PRIMARY KEY,
                    type TEXT NOT NULL CHECK(type IN ('preference', 'habit', 'correction')),
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    model TEXT NOT NULL,
                    source_note_id TEXT,
                    status TEXT NOT NULL DEFAULT 'active'
                        CHECK(status IN ('active', 'superseded', 'dismissed')),
                    embedding BLOB, embedding_model TEXT, embedding_hash TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                )""")
                db.execute("""INSERT INTO memories_new
                    (id, type, content, content_hash, model, source_note_id,
                     status, created_at, updated_at)
                    SELECT id, type, content, content_hash, model, source_note_id,
                           status, created_at, updated_at FROM memories""")
                db.execute("DROP TABLE memories")
                db.execute("ALTER TABLE memories_new RENAME TO memories")
            # 记忆与候选都记一份"来源标题快照"。来源 id 指向的笔记/会话随时可能被删，
            # 一旦删掉，解析来源就只剩一串 id，审核界面上用户根本判断不了该不该信这条
            # 记忆。快照在提炼那一刻写下、之后不随源变化，所以源没了依据也还在。
            # 这里加的是普通列，ALTER TABLE ADD COLUMN 就够，不需要重建表。
            for table in ("memories", "memory_candidates"):
                fields = {row[1] for row in db.execute(
                    f"PRAGMA table_info({table})").fetchall()}
                if "source_label" not in fields:
                    db.execute(
                        f"ALTER TABLE {table} ADD COLUMN source_label TEXT NOT NULL DEFAULT ''")
            # 秋招知识点：content_hash 是内容 SHA-256 指纹 + UNIQUE 约束，重复记忆在
            # 数据库层直接插不进去——幂等去重从“代码约定”升级为“硬约束”（对齐 chunks）。
            # 任务使用普通关系表保存，安卓 App 以后只需读取这些字段即可展示待办。
            db.execute("""CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY, note_id TEXT NOT NULL,
                title TEXT NOT NULL, details TEXT NOT NULL DEFAULT '',
                scheduled_at TEXT NOT NULL, reminder_at TEXT,
                timezone TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                source_action TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")
            # 收藏夹是用户可维护的业务数据，不属于可随时重建的检索索引。
            # status 使用软删除：归档收藏夹时不物理删除历史关系，便于恢复与审计。
            db.execute("""CREATE TABLE IF NOT EXISTS folders (
                id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL DEFAULT '', color TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS note_folders (
                note_id TEXT NOT NULL, folder_id INTEGER NOT NULL,
                assigned_by TEXT NOT NULL DEFAULT 'manual', created_at TEXT NOT NULL,
                PRIMARY KEY(note_id, folder_id),
                FOREIGN KEY(folder_id) REFERENCES folders(id)
            )""")
            # 秋招知识点：联合主键天然实现幂等，同一笔记重复加入同一收藏夹不会产生重复行。
            # AI 建议与最终归档分表保存：模型输出是“推测”，确认结果才是用户事实。
            db.execute("""CREATE TABLE IF NOT EXISTS folder_suggestions (
                note_id TEXT PRIMARY KEY, model TEXT NOT NULL,
                payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                selected_folder_ids TEXT, created_at TEXT NOT NULL, confirmed_at TEXT
            )""")
            # documents/chunks 是从 notes 派生的知识库索引，不修改原始笔记。
            # 秋招知识点：原始数据与派生索引分层，索引损坏时可以从 notes 安全重建。
            db.execute("""CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY, note_id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL, content TEXT NOT NULL,
                source_type TEXT NOT NULL, source_url TEXT NOT NULL,
                content_hash TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL, content TEXT NOT NULL,
                content_hash TEXT NOT NULL, metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(document_id, chunk_index)
                -- 秋招知识点：联合唯一约束在数据库层保证同一文档的分块序号不重复。
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY, chunk_id INTEGER NOT NULL,
                model TEXT NOT NULL, dimension INTEGER NOT NULL,
                vector BLOB NOT NULL, content_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(chunk_id, model)
            )""")
            # 秋招知识点：BLOB 保存原始字节；SQLite 不理解向量含义，维度和 dtype 由应用保证。

    def ensure_assets(self, note):
        # enumerate(..., 1) 从 1 开始编号，保留原笔记图片顺序。
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executemany("INSERT OR IGNORE INTO assets (note_id, position, source_url) VALUES (?, ?, ?)",
                           [(note["note_id"], i, url) for i, url in enumerate(note.get("images", []), 1)])

    def assets(self, note_id):
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row  # 按列名读取，再转换为普通字典。
            return [dict(row) for row in db.execute(
                "SELECT * FROM assets WHERE note_id = ? ORDER BY position", (note_id,))]

    def asset(self, asset_id):
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()
            return dict(row) if row else None

    def asset_status(self, asset_id, status, error=None):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE assets SET status = ?, error = ? WHERE id = ?", (status, error, asset_id))

    def complete_asset(self, asset_id, info):
        # 只有文件成功保存后才写 success；数据库中不会把半张图片标成完成。
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""UPDATE assets SET status='success', error=NULL, storage_key=?,
                mime_type=?, size_bytes=?, width=?, height=?, sha256=? WHERE id=?""",
                (info["storage_key"], info["mime_type"], info["size_bytes"], info["width"],
                 info["height"], info["sha256"], asset_id))

    def save(self, note):
        with sqlite3.connect(self.path) as db:
            # ? 是 SQL 参数占位符，数据单独传入，不用拼接 SQL，避免 SQL 注入。
            # INSERT OR IGNORE 在主键重复时跳过插入；不会覆盖已有笔记。
            # dumps：字典 → JSON 字符串；ensure_ascii=False 让中文保持可读。
            cursor = db.execute("INSERT OR IGNORE INTO notes VALUES (?, ?)",
                                (note["note_id"], json.dumps(note, ensure_ascii=False)))
            created = cursor.rowcount == 1  # rowcount 是这次语句影响的行数。
        # 返回数据库实际保存的内容和“是否新建”；重复时读回原来的记录。
        return self.get(note["note_id"]), created

    def update_note_fields(self, note_id, fields):
        """就地更新已存在笔记的部分字段并写回。

        save() 是 INSERT OR IGNORE，对已经入库的笔记无效——这正是"补读图"
        这类事后修正需要的入口：读回 payload、改几个字段、整体写回，保证
        图片文字、警告、图片计数这些派生数据跟原文一起留在同一份 JSON 里。
        字段缺失时跳过，不强制调用方补齐整篇笔记。
        """
        note = self.get(note_id)
        if note is None:
            return None
        for key, value in fields.items():
            if value is not None:
                note[key] = value
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE notes SET payload = ? WHERE note_id = ?",
                       (json.dumps(note, ensure_ascii=False), note_id))
        return note

    def create_plan(self, title, goal, source_note_ids, items, now):
        """在一个事务中创建计划和所有步骤，避免只保存半个计划。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            cursor = db.execute("""INSERT INTO plans
                (title, goal, source_note_ids, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)""",
                (title, goal, json.dumps(source_note_ids), now, now))
            plan_id = cursor.lastrowid
            db.executemany("""INSERT INTO plan_items
                (plan_id, item_index, title, details, scheduled_at, reminder_at, timezone)
                VALUES (?, ?, ?, ?, ?, ?, ?)""", [
                (plan_id, index, item["title"], item["details"], item["scheduled_at"],
                 item["reminder_at"], item["timezone"])
                for index, item in enumerate(items)
            ])
            for index, item in enumerate(items):
                item_row = db.execute("SELECT id FROM plan_items WHERE plan_id=? AND item_index=?",
                                      (plan_id, index)).fetchone()
                db.executemany("INSERT OR IGNORE INTO plan_item_sources (plan_item_id, note_id, reason) VALUES (?, ?, ?)",
                               [(item_row[0], note_id, "AI 或用户指定的参考笔记")
                                for note_id in item.get("source_note_ids", [])])
        return self.plan(plan_id)

    def plan(self, plan_id):
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM plans WHERE id=?", (plan_id,)).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["source_note_ids"] = json.loads(result["source_note_ids"])
            result["items"] = [dict(item) for item in db.execute(
                "SELECT * FROM plan_items WHERE plan_id=? ORDER BY item_index", (plan_id,)
            ).fetchall()]
            for item in result["items"]:
                item["source_note_ids"] = [row[0] for row in db.execute(
                    "SELECT note_id FROM plan_item_sources WHERE plan_item_id=?", (item["id"],)
                ).fetchall()]
            return result

    def plans(self, status=None):
        with closing(sqlite3.connect(self.path)) as db:
            ids = ([row[0] for row in db.execute(
                "SELECT id FROM plans WHERE status=? ORDER BY id DESC", (status,)
            ).fetchall()] if status else [row[0] for row in db.execute(
                "SELECT id FROM plans ORDER BY id DESC").fetchall()])
        return [self.plan(plan_id) for plan_id in ids]

    def update_plan_status(self, plan_id, status, now):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE plans SET status=?, updated_at=?, confirmed_at=? WHERE id=?",
                       (status, now, now if status == "confirmed" else None, plan_id))
        return self.plan(plan_id)

    def update_plan_item(self, item_id, fields, now):
        """更新草案步骤；调用方先确认该步骤所属计划仍是 draft。"""
        allowed = {"title", "details", "scheduled_at", "reminder_at", "timezone"}
        changes = {key: value for key, value in fields.items() if key in allowed}
        if not changes:
            return self.plan_item(item_id)
        assignments = ", ".join(f"{key}=?" for key in changes)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(f"UPDATE plan_items SET {assignments} WHERE id=?",
                       [*changes.values(), item_id])
            db.execute("UPDATE plans SET updated_at=? WHERE id=(SELECT plan_id FROM plan_items WHERE id=?)",
                       (now, item_id))
        return self.plan_item(item_id)

    def plan_item(self, item_id):
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM plan_items WHERE id=?", (item_id,)).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["source_note_ids"] = [item[0] for item in db.execute(
                "SELECT note_id FROM plan_item_sources WHERE plan_item_id=?", (item_id,)
            ).fetchall()]
            return result

    def delete_plan_item(self, item_id, now):
        """删除草案步骤和对应来源关系，并重新编号剩余步骤。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            row = db.execute("SELECT plan_id, item_index FROM plan_items WHERE id=?", (item_id,)).fetchone()
            if row is None:
                return None
            plan_id, old_index = row
            db.execute("DELETE FROM plan_item_sources WHERE plan_item_id=?", (item_id,))
            db.execute("DELETE FROM plan_items WHERE id=?", (item_id,))
            db.execute("UPDATE plan_items SET item_index=item_index-1 WHERE plan_id=? AND item_index>?",
                       (plan_id, old_index))
            db.execute("UPDATE plans SET updated_at=? WHERE id=?", (now, plan_id))
        return self.plan(plan_id)

    def reorder_plan_items(self, plan_id, item_ids, now):
        """按用户给出的完整 ID 顺序重排；事务保证不会出现重复序号。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            rows = db.execute("SELECT id FROM plan_items WHERE plan_id=?", (plan_id,)).fetchall()
            actual = {row[0] for row in rows}
            if set(item_ids) != actual or len(item_ids) != len(actual):
                raise ValueError("item_ids 必须包含该计划的全部步骤且不能重复。")
            # 临时加偏移，避免 UNIQUE(plan_id, item_index) 的交换冲突。
            db.executemany("UPDATE plan_items SET item_index=? WHERE id=?",
                           [(index + 1000, item_id) for index, item_id in enumerate(item_ids)])
            db.executemany("UPDATE plan_items SET item_index=? WHERE id=?",
                           [(index, item_id) for index, item_id in enumerate(item_ids)])
            db.execute("UPDATE plans SET updated_at=? WHERE id=?", (now, plan_id))
        return self.plan(plan_id)

    def create_plan_tasks(self, plan_id, note_id, now):
        """确认后的每个步骤只生成一次任务，重复调用不会重复创建。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.row_factory = sqlite3.Row
            plan = db.execute("SELECT status FROM plans WHERE id=?", (plan_id,)).fetchone()
            rows = db.execute("""SELECT * FROM plan_items
                WHERE plan_id=? AND task_id IS NULL ORDER BY item_index""", (plan_id,)).fetchall()
            task_ids = []
            for row in rows:
                source_row = db.execute("SELECT note_id FROM plan_item_sources WHERE plan_item_id=? ORDER BY note_id LIMIT 1", (row["id"],)).fetchone()
                task_note_id = source_row[0] if source_row else note_id
                cursor = db.execute("""INSERT INTO tasks
                    (note_id, title, details, scheduled_at, reminder_at, timezone,
                     source_action, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (task_note_id, row["title"], row["details"], row["scheduled_at"],
                     row["reminder_at"], row["timezone"], f"plan:{plan_id}", now, now))
                task_ids.append(cursor.lastrowid)
                db.execute("UPDATE plan_items SET task_id=?, status='scheduled' WHERE id=?",
                           (cursor.lastrowid, row["id"]))
            db.execute("UPDATE plans SET status='active', updated_at=? WHERE id=?", (now, plan_id))
        return self.plan(plan_id), task_ids

    def delete_plan(self, plan_id):
        """丢弃计划草案：步骤与来源关系一并删掉。

        只有 draft 允许删（接口层把关）：已确认的计划已经生成任务，删掉计划会让
        那些任务失去来源，收尾方式应该是取消任务，而不是抹掉计划本身。
        """
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""DELETE FROM plan_item_sources WHERE plan_item_id IN
                (SELECT id FROM plan_items WHERE plan_id = ?)""", (plan_id,))
            db.execute("DELETE FROM plan_items WHERE plan_id = ?", (plan_id,))
            cursor = db.execute("DELETE FROM plans WHERE id = ?", (plan_id,))
            deleted = cursor.rowcount
        return deleted > 0

    def create_thread(self, thread_id, title, now):
        """创建一次对话；UUID 由接口层生成，数据库只负责持久化。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""INSERT INTO chat_threads(id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)""", (thread_id, title, now, now))
        return self.thread(thread_id)

    def thread(self, thread_id, message_limit=20):
        """读取会话及最近消息；窗口上限用于控制模型上下文成本。"""
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM chat_threads WHERE id=?", (thread_id,)).fetchone()
            if row is None:
                return None
            # 子查询先取最近 N 条，再按正序返回，保证模型看到自然对话顺序。
            messages = db.execute("""SELECT * FROM (
                SELECT * FROM chat_messages WHERE thread_id=? ORDER BY id DESC LIMIT ?
            ) ORDER BY id""", (thread_id, message_limit)).fetchall()
        result = dict(row)
        result["messages"] = []
        for message in messages:
            item = dict(message)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            result["messages"].append(item)
        return result

    def add_chat_message(self, thread_id, role, content, metadata, now):
        """追加一条不可变消息，并更新会话活跃时间。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            cursor = db.execute("""INSERT INTO chat_messages
                (thread_id, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)""",
                (thread_id, role, content, json.dumps(metadata, ensure_ascii=False), now))
            db.execute("UPDATE chat_threads SET updated_at=? WHERE id=?", (now, thread_id))
            message_id = cursor.lastrowid
        return next(item for item in self.thread(thread_id)["messages"] if item["id"] == message_id)

    def list_threads(self, limit=50):
        """列出会话，附消息条数与最后一条消息预览，供前端侧边栏展示历史。

        侧边栏要一眼看出"这个会话聊到哪了"，所以预览压成一行并截断到 80 字；
        用相关子查询取最后一条消息，避免把所有消息都读进内存。
        """
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("""
                SELECT t.id, t.title, t.created_at, t.updated_at,
                    (SELECT COUNT(*) FROM chat_messages m
                      WHERE m.thread_id = t.id) AS message_count,
                    (SELECT m.role FROM chat_messages m
                      WHERE m.thread_id = t.id ORDER BY m.id DESC LIMIT 1) AS last_role,
                    (SELECT m.content FROM chat_messages m
                      WHERE m.thread_id = t.id ORDER BY m.id DESC LIMIT 1) AS last_message
                FROM chat_threads t
                ORDER BY t.updated_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            preview = item.pop("last_message") or ""
            item["last_message"] = " ".join(preview.split())[:80]
            items.append(item)
        return items

    def rename_thread(self, thread_id, title, now):
        """改标题只动 chat_threads；顺便刷新 updated_at，让侧边栏顺序跟手。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            cursor = db.execute("UPDATE chat_threads SET title=?, updated_at=? WHERE id=?",
                                (title, now, thread_id))
            updated = cursor.rowcount
        return self.thread(thread_id) if updated else None

    def delete_thread(self, thread_id):
        """删除会话及其消息，返回是否真的删到了记录（接口层据此决定 404）。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DELETE FROM chat_messages WHERE thread_id=?", (thread_id,))
            cursor = db.execute("DELETE FROM chat_threads WHERE id=?", (thread_id,))
            deleted = cursor.rowcount
        return deleted > 0

    def create_job(self, job_id, job_type, now, progress):
        """先持久化 queued 记录，再安排后台工作，避免任务没有可查询的 ID。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""INSERT INTO jobs
                (job_id, job_type, status, current_step, progress_json, created_at, updated_at)
                VALUES (?, ?, 'queued', 'queued', ?, ?, ?)""",
                (job_id, job_type, json.dumps(progress, ensure_ascii=False), now, now))
        return self.job(job_id)

    def update_job(self, job_id, *, status=None, current_step=None, progress=None,
                   result=None, note_id=None, error=None, finished_at=None, now=None):
        """只更新后台处理状态；None 表示本次不改该字段。"""
        fields = {"updated_at": now}
        for key, value in (("status", status), ("current_step", current_step),
                           ("note_id", note_id), ("error", error),
                           ("finished_at", finished_at)):
            if value is not None:
                fields[key] = value
        if progress is not None:
            fields["progress_json"] = json.dumps(progress, ensure_ascii=False)
        if result is not None:
            fields["result_json"] = json.dumps(result, ensure_ascii=False)
        assignments = ", ".join(f"{key}=?" for key in fields)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(f"UPDATE jobs SET {assignments} WHERE job_id=?",
                       [*fields.values(), job_id])
        return self.job(job_id)

    def job(self, job_id):
        """读取一个 job，并解码 JSON 状态字段。"""
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["progress"] = json.loads(result.pop("progress_json"))
        raw_result = result.pop("result_json")
        result["result"] = json.loads(raw_result) if raw_result else None
        return result

    def mark_interrupted_jobs(self, now):
        """服务重启时将无法续跑的进程内任务明确标记，避免状态永久卡住。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            rows = db.execute(
                "SELECT job_id, progress_json FROM jobs WHERE status IN ('queued', 'running')"
            ).fetchall()
            for job_id, raw_progress in rows:
                progress = json.loads(raw_progress)
                progress["interrupted"] = True
                db.execute("""UPDATE jobs SET status='failed', current_step='interrupted',
                    progress_json=?, error='服务重启，任务未完成；请重新提交分享内容。',
                    updated_at=?, finished_at=? WHERE job_id=?""",
                    (json.dumps(progress, ensure_ascii=False), now, now, job_id))
        return len(rows)

    def refresh_note_source(self, note_id, fresh_note, refreshed_at):
        """原子更新笔记来源数据，并同步图片地址但保留已经下载的文件。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            row = db.execute("SELECT payload FROM notes WHERE note_id=?", (note_id,)).fetchone()
            if row is None:
                return None
            current = json.loads(row[0])
            # URL 最终解析出的 ID 必须仍是当前笔记，避免把另一篇内容覆盖进来。
            if fresh_note.get("note_id") != note_id:
                raise ValueError("刷新链接对应的不是当前笔记。")
            refreshed = {
                **current,
                **fresh_note,
                "note_id": note_id,
                "created_at": current.get("created_at", fresh_note.get("created_at")),
                "refreshed_at": refreshed_at,
            }
            db.execute(
                "UPDATE notes SET payload=? WHERE note_id=?",
                (json.dumps(refreshed, ensure_ascii=False), note_id),
            )

            new_urls = fresh_note.get("images", [])
            existing_rows = db.execute(
                "SELECT id, position, status FROM assets WHERE note_id=?", (note_id,)
            ).fetchall()
            existing = {row[1]: {"id": row[0], "status": row[2]} for row in existing_rows}
            for position, source_url in enumerate(new_urls, 1):
                asset = existing.get(position)
                if asset is None:
                    db.execute("""INSERT INTO assets
                        (note_id, position, source_url, status) VALUES (?, ?, ?, 'pending')""",
                        (note_id, position, source_url))
                elif asset["status"] == "success":
                    # 本地文件已经通过校验时只刷新远端地址，不触碰文件元数据。
                    db.execute("UPDATE assets SET source_url=? WHERE id=?", (source_url, asset["id"]))
                else:
                    # 失败项换成最新临时地址，并清除旧错误，允许下一次下载重新尝试。
                    db.execute("""UPDATE assets SET source_url=?, status='pending', error=NULL,
                        storage_key=NULL, mime_type=NULL, size_bytes=NULL, width=NULL,
                        height=NULL, sha256=NULL WHERE id=?""", (source_url, asset["id"]))

            # 来源页面减少图片时只标记旧记录，不删除数据库行或磁盘文件。
            db.execute("""UPDATE assets SET status='obsolete'
                WHERE note_id=? AND position>?""", (note_id, len(new_urls)))
        return self.get(note_id)

    def get(self, note_id):
        with sqlite3.connect(self.path) as db:
            # (note_id,) 是只含一个元素的元组，末尾逗号不能省略。
            # fetchone() 返回一行（元组），没有结果则返回 None。
            row = db.execute("SELECT payload FROM notes WHERE note_id = ?", (note_id,)).fetchone()
        # row[0] 取第一列；loads：JSON 字符串 → Python 字典。
        return json.loads(row[0]) if row else None

    def list(self, limit=50, offset=0):
        with sqlite3.connect(self.path) as db:
            # DESC 倒序排列，使最近插入的笔记排在前面；LIMIT/OFFSET 实现分页。
            rows = db.execute("SELECT payload FROM notes ORDER BY rowid DESC LIMIT ? OFFSET ?",
                              (limit, offset)).fetchall()
        # 列表推导式：对每行做 JSON 解码，再把结果收集为列表。
        return [json.loads(row[0]) for row in rows]

    @staticmethod
    def _folder_dict(row):
        return dict(row) if row else None

    def create_folder(self, name, description, color, now):
        """创建收藏夹；名称由数据库 UNIQUE 约束保证不重复。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            cursor = db.execute("""INSERT INTO folders
                (name, description, color, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)""", (name, description, color, now, now))
            folder_id = cursor.lastrowid
        return self.folder(folder_id)

    def folder(self, folder_id):
        """读取收藏夹，并实时统计其中有多少篇笔记。"""
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("""SELECT f.*, COUNT(nf.note_id) AS note_count
                FROM folders f LEFT JOIN note_folders nf ON nf.folder_id=f.id
                WHERE f.id=? GROUP BY f.id""", (folder_id,)).fetchone()
        return self._folder_dict(row)

    def folders(self, include_archived=False):
        """列出收藏夹；默认隐藏已经软删除（归档）的收藏夹。"""
        where = "" if include_archived else "WHERE f.status='active'"
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(f"""SELECT f.*, COUNT(nf.note_id) AS note_count
                FROM folders f LEFT JOIN note_folders nf ON nf.folder_id=f.id
                {where} GROUP BY f.id ORDER BY f.id""").fetchall()
        return [self._folder_dict(row) for row in rows]

    def update_folder(self, folder_id, fields, now):
        """只更新白名单字段，避免把用户输入直接拼进 SQL。"""
        allowed = {"name", "description", "color", "status"}
        changes = {key: value for key, value in fields.items() if key in allowed}
        changes["updated_at"] = now
        assignments = ", ".join(f"{key}=?" for key in changes)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                f"UPDATE folders SET {assignments} WHERE id=?",
                [*changes.values(), folder_id],
            )
        return self.folder(folder_id)

    def add_note_to_folder(self, note_id, folder_id, assigned_by, now):
        """建立多对多关系；重复调用返回已有关系而不重复插入。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            cursor = db.execute("""INSERT OR IGNORE INTO note_folders
                (note_id, folder_id, assigned_by, created_at) VALUES (?, ?, ?, ?)""",
                (note_id, folder_id, assigned_by, now))
        return cursor.rowcount == 1

    def remove_note_from_folder(self, note_id, folder_id):
        """只删除归档关系，不删除笔记或收藏夹本身。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            cursor = db.execute(
                "DELETE FROM note_folders WHERE note_id=? AND folder_id=?",
                (note_id, folder_id),
            )
        return cursor.rowcount == 1

    def folders_for_note(self, note_id):
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("""SELECT f.*, nf.assigned_by, nf.created_at AS assigned_at
                FROM note_folders nf JOIN folders f ON f.id=nf.folder_id
                WHERE nf.note_id=? ORDER BY f.id""", (note_id,)).fetchall()
        return [dict(row) for row in rows]

    def notes_in_folder(self, folder_id, limit=50, offset=0):
        """通过关系表找到笔记，再解码 notes.payload。"""
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute("""SELECT n.payload FROM note_folders nf
                JOIN notes n ON n.note_id=nf.note_id WHERE nf.folder_id=?
                ORDER BY nf.created_at DESC LIMIT ? OFFSET ?""",
                (folder_id, limit, offset)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def folder_suggestion(self, note_id):
        """读取模型建议和用户确认状态，并把 JSON 字段还原为 Python 对象。"""
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT * FROM folder_suggestions WHERE note_id=?", (note_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["suggestion"] = json.loads(result.pop("payload"))
        raw_selected = result.pop("selected_folder_ids")
        result["selected_folder_ids"] = json.loads(raw_selected) if raw_selected else None
        return result

    def save_folder_suggestion(self, note_id, model, result, now):
        """重新推荐会覆盖旧建议并回到 pending，但不会移除已经归档的笔记。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""INSERT INTO folder_suggestions
                (note_id, model, payload, status, selected_folder_ids, created_at, confirmed_at)
                VALUES (?, ?, ?, 'pending', NULL, ?, NULL)
                ON CONFLICT(note_id) DO UPDATE SET model=excluded.model,
                    payload=excluded.payload, status='pending', selected_folder_ids=NULL,
                    created_at=excluded.created_at, confirmed_at=NULL""",
                (note_id, model, json.dumps(result, ensure_ascii=False), now))
        return self.folder_suggestion(note_id)

    def confirm_folder_suggestion(self, note_id, folder_ids, now):
        """在一个事务中保存确认结果并建立归档关系，避免只完成其中一半。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""UPDATE folder_suggestions SET status='confirmed',
                selected_folder_ids=?, confirmed_at=? WHERE note_id=?""",
                (json.dumps(folder_ids), now, note_id))
            db.executemany("""INSERT OR IGNORE INTO note_folders
                (note_id, folder_id, assigned_by, created_at) VALUES (?, ?, 'ai_confirmed', ?)""",
                [(note_id, folder_id, now) for folder_id in folder_ids])
        return self.folder_suggestion(note_id)

    def index_document(self, note, document, chunks, content_hash, now):
        """原子保存文档及分块；内容没变时直接复用已有索引。"""
        # 秋招知识点：幂等性表示同一输入重复执行仍得到同一业务结果，不制造重复数据。
        with closing(sqlite3.connect(self.path)) as db, db:
            # 秋招知识点：事务保证 document 与所有 chunks 要么一起提交，要么一起回滚。
            db.row_factory = sqlite3.Row
            existing = db.execute(
                "SELECT id, content_hash FROM documents WHERE note_id = ?", (note["note_id"],)
            ).fetchone()
            if existing and existing["content_hash"] == content_hash:
                # 秋招知识点：内容指纹命中时跳过重建，可减少后续 Embedding 调用和费用。
                return self.document_for_note(note["note_id"]), False

            if existing:
                document_id = existing["id"]
                db.execute("""UPDATE documents SET title=?, content=?, source_type=?, source_url=?,
                    content_hash=?, updated_at=? WHERE id=?""",
                    (note["title"], document.page_content, document.metadata["source_type"],
                     document.metadata["source_url"], content_hash, now, document_id))
                # chunks 是可重建的派生数据；只替换当前文档的旧分块。
                # Chunk 即将替换，先清理依赖旧 chunk_id 的派生向量，避免孤儿记录。
                db.execute("""DELETE FROM embeddings WHERE chunk_id IN
                    (SELECT id FROM chunks WHERE document_id = ?)""", (document_id,))
                db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                # 秋招知识点：这里删除的仅是可重建派生数据，不会删除 notes 原文或图片。
            else:
                cursor = db.execute("""INSERT INTO documents
                    (note_id, title, content, source_type, source_url, content_hash, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (note["note_id"], note["title"], document.page_content,
                     document.metadata["source_type"], document.metadata["source_url"],
                     content_hash, now, now))
                document_id = cursor.lastrowid

            db.executemany("""INSERT INTO chunks
                (document_id, chunk_index, content, content_hash, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""", [
                (document_id, chunk.metadata["chunk_index"], chunk.page_content,
                 chunk.metadata["content_hash"], json.dumps(chunk.metadata, ensure_ascii=False), now)
                for chunk in chunks
            ])
        return self.document_for_note(note["note_id"]), True

    def document_for_note(self, note_id):
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM documents WHERE note_id = ?", (note_id,)).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["chunk_count"] = db.execute(
                "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (row["id"],)
            ).fetchone()[0]
            return result

    def document_chunks(self, document_id):
        # 秋招知识点：ORDER BY chunk_index 保证检索后需要拼接上下文时仍可恢复阅读顺序。
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index", (document_id,)
            ).fetchall()
        return [{**dict(row), "metadata": json.loads(row["metadata_json"])} for row in rows]

    def chunks_needing_embeddings(self, document_id, model):
        """只返回尚无向量或内容指纹已变化的 Chunk。"""
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute("""SELECT c.*, d.title FROM chunks c
                JOIN documents d ON d.id=c.document_id
                LEFT JOIN embeddings e ON e.chunk_id=c.id AND e.model=?
                WHERE c.document_id=? AND (e.id IS NULL OR e.content_hash != c.content_hash)
                ORDER BY c.chunk_index""", (model, document_id))]

    def save_embeddings(self, records, now):
        """批量插入或更新向量；chunk_id + model 决定唯一记录。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executemany("""INSERT INTO embeddings
                (chunk_id, model, dimension, vector, content_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id, model) DO UPDATE SET
                    dimension=excluded.dimension, vector=excluded.vector,
                    content_hash=excluded.content_hash, created_at=excluded.created_at""", [
                (record["chunk_id"], record["model"], record["dimension"],
                 record["vector"], record["content_hash"], now)
                for record in records
            ])

    def embedding_count(self, document_id, model):
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute("""SELECT COUNT(*) FROM embeddings e
                JOIN chunks c ON c.id=e.chunk_id
                WHERE c.document_id=? AND e.model=?""", (document_id, model)).fetchone()[0]

    def searchable_embeddings(self, model):
        """读取当前模型下仍与 Chunk 内容指纹一致的可检索向量。"""
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("""SELECT e.vector, e.dimension, c.id AS chunk_id,
                c.chunk_index, c.content, c.metadata_json,
                d.id AS document_id, d.note_id, d.title, d.source_url
                FROM embeddings e
                JOIN chunks c ON c.id=e.chunk_id
                JOIN documents d ON d.id=c.document_id
                WHERE e.model=? AND e.content_hash=c.content_hash""", (model,)).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            results.append(item)
        return results

    def analysis(self, note_id):
        """读取某篇笔记已有的 AI 分析；从未分析时返回 None。"""
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT model, payload, created_at FROM analyses WHERE note_id = ?", (note_id,)
            ).fetchone()
        if row is None:
            return None
        return {"note_id": note_id, "model": row["model"],
                "analysis": json.loads(row["payload"]), "created_at": row["created_at"]}

    def save_analysis(self, note_id, model, result, created_at):
        """保存分析；ON CONFLICT 让用户明确要求重新分析时可以更新旧结果。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""INSERT INTO analyses (note_id, model, payload, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(note_id) DO UPDATE SET
                    model=excluded.model, payload=excluded.payload, created_at=excluded.created_at""",
                (note_id, model, json.dumps(result, ensure_ascii=False), created_at))
        return self.analysis(note_id)

    def intent_confirmation(self, note_id):
        """读取用户的最终选择；尚未处理时返回 None。"""
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("""SELECT decision, category, intent, selected_actions, updated_at
                FROM intent_confirmations WHERE note_id = ?""", (note_id,)).fetchone()
        if row is None:
            return None
        return {
            "decision": row["decision"],
            "category": row["category"],
            "intent": row["intent"],
            "selected_actions": json.loads(row["selected_actions"]),
            "updated_at": row["updated_at"],
        }

    def save_intent_confirmation(self, note_id, decision, category, intent, actions, updated_at):
        """新增或更新用户选择；用户以后改变想法时可以再次提交。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""INSERT INTO intent_confirmations
                (note_id, decision, category, intent, selected_actions, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(note_id) DO UPDATE SET decision=excluded.decision,
                    category=excluded.category, intent=excluded.intent,
                    selected_actions=excluded.selected_actions, updated_at=excluded.updated_at""",
                (note_id, decision, category, intent,
                 json.dumps(actions, ensure_ascii=False), updated_at))
        return self.intent_confirmation(note_id)

    # ---- 收件箱（Inbox）：导入之后的整理队列 ----

    def inbox_notes(self, limit=20):
        """列出还没有整理结论的笔记。

        判据是"intent_confirmations 里有没有记录"，而不是新增一个队列状态字段：
        确认、修改、忽略三种结果都会留下记录，所以"有没有记录"天然等价于
        "用户处理过没有"，不需要再维护第二份状态、也就不会和它不同步。
        """
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("""
                SELECT n.payload,
                       (a.note_id IS NOT NULL) AS has_analysis,
                       (fs.note_id IS NOT NULL) AS has_suggestion
                FROM notes n
                LEFT JOIN intent_confirmations ic ON ic.note_id = n.note_id
                LEFT JOIN analyses a ON a.note_id = n.note_id
                LEFT JOIN folder_suggestions fs ON fs.note_id = n.note_id
                WHERE ic.note_id IS NULL
                ORDER BY n.rowid DESC LIMIT ?""", (limit,)).fetchall()
        items = []
        for row in rows:
            note = json.loads(row["payload"])
            items.append({
                "note_id": note.get("note_id"),
                "title": note.get("title") or "",
                "created_at": note.get("created_at"),
                "excerpt": " ".join((note.get("content") or "").split())[:200],
                "has_analysis": bool(row["has_analysis"]),
                # 有分析不一定有建议：可能建收藏夹之前就分析过了。
                "has_suggestion": bool(row["has_suggestion"]),
            })
        return items

    def inbox_count(self):
        """待整理条数，供前端在标签上显示徽标。"""
        with sqlite3.connect(self.path) as db:
            row = db.execute("""
                SELECT COUNT(*) FROM notes n
                LEFT JOIN intent_confirmations ic ON ic.note_id = n.note_id
                WHERE ic.note_id IS NULL""").fetchone()
        return int(row[0])

    def apply_triage(self, note_id, decision, category, intent, actions,
                     folder_ids, now):
        """在一个事务里落定整理结论：意图确认 + 归档关系。

        拆成两次写会留下"意图确认了、笔记却没进任何收藏夹"的半成品——卡片从
        收件箱消失，用户以为整理好了，实际笔记不在任何收藏夹里，且没有任何
        界面能暴露这件事。同连接同事务提交，要么都成，要么都不变。
        """
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""INSERT INTO intent_confirmations
                (note_id, decision, category, intent, selected_actions, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(note_id) DO UPDATE SET decision=excluded.decision,
                    category=excluded.category, intent=excluded.intent,
                    selected_actions=excluded.selected_actions, updated_at=excluded.updated_at""",
                (note_id, decision, category, intent,
                 json.dumps(actions, ensure_ascii=False), now))
            if folder_ids:
                # 建议行可能不存在（用户直接确认而不走 AI 推荐），UPDATE 落空也没关系：
                # 归档关系本身写在 note_folders 里，那才是检索收藏夹内容时的数据源。
                db.execute("""UPDATE folder_suggestions SET status='confirmed',
                    selected_folder_ids=?, confirmed_at=? WHERE note_id=?""",
                    (json.dumps(folder_ids), now, note_id))
                db.executemany("""INSERT OR IGNORE INTO note_folders
                    (note_id, folder_id, assigned_by, created_at)
                    VALUES (?, ?, 'ai_confirmed', ?)""",
                    [(note_id, folder_id, now) for folder_id in folder_ids])
        return self.intent_confirmation(note_id)

    # ---- 记忆层（Memory Layer）----
    # 记忆是意图确认的副产品：确认是主链路，沉淀失败不影响确认接口。

    @staticmethod
    def memory_content_hash(content):
        """内容指纹：同一句话重复沉淀时在数据库层直接被 UNIQUE 拒绝。"""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def save_memory_candidates(self, model, records, source_note_id, now):
        """保存 AI 候选，不激活；重复提炼同一来源和内容不会刷出新候选。

        来源标题在这里写下快照：审核一条候选时"它从哪儿来"决定用户信不信它，
        而来源笔记随时可能被删除——等界面要显示时再查就查不到了。
        """
        with closing(sqlite3.connect(self.path)) as db, db:
            db.row_factory = sqlite3.Row   # 解析来源标题要按列名取
            label = self._source_title(db, source_note_id)
            db.executemany("""INSERT OR IGNORE INTO memory_candidates
                (type, content, content_hash, model, source_note_id, source_label, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""", [
                (record["type"], record["content"],
                 self.memory_content_hash(record["content"]), model, source_note_id,
                 label, now)
                for record in records
            ])
        return self.memory_candidates(source_note_id=source_note_id)

    def memory_candidate(self, candidate_id):
        """单条候选，字段与列表一致（含来源解析）——审核接口的回包也走这里。"""
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM memory_candidates WHERE id=?", (candidate_id,)).fetchone()
            labels = self._source_labels(db, [row["source_note_id"]]) if row else {}
        if row is None:
            return None
        return self._attach_source(dict(row), labels)

    def memory_candidates(self, status=None, source_note_id=None, limit=50):
        """按状态/来源查看待确认信息；不与长期记忆混在同一列表。"""
        conditions, params = [], []
        if status:
            conditions.append("status=?")
            params.append(status)
        if source_note_id:
            conditions.append("source_note_id=?")
            params.append(source_note_id)
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                f"SELECT * FROM memory_candidates {where} ORDER BY id DESC LIMIT ?",
                [*params, limit],
            ).fetchall()
            labels = self._source_labels(db, [row["source_note_id"] for row in rows])
        return [self._attach_source(dict(row), labels) for row in rows]

    def review_memory_candidate(self, candidate_id, decision, now, memory_type=None, content=None):
        """确认/忽略在同一事务中完成，避免候选已确认但记忆未落库。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM memory_candidates WHERE id=?", (candidate_id,)).fetchone()
            if row is None:
                return None
            if row["status"] != "pending":
                raise ValueError("这条候选已经处理过。")
            if decision == "dismissed":
                db.execute("""UPDATE memory_candidates SET status='dismissed', reviewed_at=?
                    WHERE id=?""", (now, candidate_id))
            else:
                chosen_type = memory_type or row["type"]
                chosen_content = content or row["content"]
                content_hash = self.memory_content_hash(chosen_content)
                existing = db.execute("SELECT id, status FROM memories WHERE content_hash=?",
                                      (content_hash,)).fetchone()
                if existing and existing["status"] != "active":
                    raise ValueError("相同记忆已被停用，请在记忆管理中恢复，而非重复创建。")
                if existing:
                    memory_id = existing["id"]
                else:
                    # 来源快照跟着候选一起进记忆：候选审核完，来源笔记还可能被删，
                    # 那时记忆的依据标签仍然要能说清"它当初从哪儿来"。
                    cursor = db.execute("""INSERT INTO memories
                        (type, content, content_hash, model, source_note_id,
                         source_label, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
                        (chosen_type, chosen_content, content_hash, row["model"],
                         row["source_note_id"], row["source_label"] or "", now, now))
                    memory_id = cursor.lastrowid
                db.execute("""UPDATE memory_candidates
                    SET status='confirmed', memory_id=?, reviewed_at=? WHERE id=?""",
                    (memory_id, now, candidate_id))
        return {"candidate": self.memory_candidate(candidate_id),
                "memory": self.memory(memory_id) if decision == "confirmed" else None}

    def create_user_memory(self, memory_type, content, now):
        """用户主动提供的偏好直接视为已确认；model 标记来源，不伪装成 AI 产物。"""
        with closing(sqlite3.connect(self.path)) as db, db:
            cursor = db.execute("""INSERT INTO memories
                (type, content, content_hash, model, source_note_id,
                 status, created_at, updated_at)
                VALUES (?, ?, ?, 'user', NULL, 'active', ?, ?)""",
                (memory_type, content, self.memory_content_hash(content), now, now))
            memory_id = cursor.lastrowid
        return self.memory(memory_id)

    def save_memories(self, model, records, source_note_id, now):
        """批量保存提取出的记忆；返回真正新建的部分（重复内容被数据库跳过）。

        秋招知识点：INSERT OR IGNORE + UNIQUE 是常用的幂等写入组合——
        重复执行同一批提取结果不会产生重复记忆，调用方无需先查询再插入。
        """
        saved_ids = []
        with closing(sqlite3.connect(self.path)) as db, db:
            db.row_factory = sqlite3.Row   # 解析来源标题要按列名取
            label = self._source_title(db, source_note_id)
            for record in records:
                cursor = db.execute("""INSERT OR IGNORE INTO memories
                    (type, content, content_hash, model, source_note_id,
                     source_label, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
                    (record["type"], record["content"],
                     self.memory_content_hash(record["content"]),
                     model, source_note_id, label, now, now))
                if cursor.rowcount == 1:
                    saved_ids.append(cursor.lastrowid)
        # 秋招知识点：回读必须放在事务提交之后——另一个连接看不到未提交的写入。
        return [self.memory(memory_id) for memory_id in saved_ids]

    def memory(self, memory_id):
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            labels = self._source_labels(db, [row["source_note_id"]]) if row else {}
        return self._memory_public(row, labels) if row else None

    def memories(self, status=None, limit=50, offset=0):
        """列出记忆；默认包含 dismissed，管理接口可用 status 过滤。"""
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            if status:
                rows = db.execute("""SELECT * FROM memories WHERE status = ?
                    ORDER BY updated_at DESC LIMIT ? OFFSET ?""",
                    (status, limit, offset)).fetchall()
            else:
                rows = db.execute("""SELECT * FROM memories
                    ORDER BY updated_at DESC LIMIT ? OFFSET ?""",
                    (limit, offset)).fetchall()
            labels = self._source_labels(db, [row["source_note_id"] for row in rows])
        return [self._memory_public(row, labels) for row in rows]

    def active_memories(self, limit=10):
        """注入提示词用：只取 active，按 updated_at 倒序（最近整理过的最可信）。

        秋招知识点：注入条数设上限是上下文预算管理——记忆条目短（约 100 字），
        10 条对 token 成本影响可忽略，同时避免无限增长的提示词。
        """
        return self.memories(status="active", limit=limit)

    def update_memory(self, memory_id, fields, now):
        """只更新白名单字段；content 变化时同步重算 content_hash。"""
        allowed = {"type", "content", "status"}
        changes = {key: value for key, value in fields.items() if key in allowed}
        if "content" in changes:
            changes["content_hash"] = self.memory_content_hash(changes["content"])
        changes["updated_at"] = now
        assignments = ", ".join(f"{key} = ?" for key in changes)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(f"UPDATE memories SET {assignments} WHERE id = ?",
                       [*changes.values(), memory_id])
        return self.memory(memory_id)

    # ---- 记忆召回（recall）----
    # 统一入口：调用方（分析 / 收藏夹建议 / 未来的问答与计划）只依赖
    # recall_memories()，内部从“时间排序”升级到“向量检索”对调用方透明。

    def _source_labels(self, db, source_ids):
        """把来源 id 解析成 {id: (kind, title)}。

        记忆与候选的来源有两种形态：确认笔记意图时提炼的挂在 note_id 上，
        对话里固化的挂在 thread_id 上。两者都是字符串，形态不可靠地区分
        （笔记 id 是十六进制串、会话 id 是 uuid，但这是巧合不是契约），
        所以两个表都查一遍，用"哪个查得到"来定类型。

        审核一条记忆时，"它从哪儿来"直接决定用户要不要信它——这个信息必须由
        存储层给准，不能让前端看着一串 id 猜。
        """
        labels = {}
        ids = [item for item in dict.fromkeys(source_ids) if item]
        if not ids:
            return labels
        placeholders = ",".join("?" * len(ids))
        for row in db.execute(
                f"SELECT note_id, payload FROM notes WHERE note_id IN ({placeholders})", ids):
            try:
                title = json.loads(row["payload"]).get("title") or ""
            except (ValueError, TypeError):
                title = ""
            labels[row["note_id"]] = ("note", title)
        for row in db.execute(
                f"SELECT id, title FROM chat_threads WHERE id IN ({placeholders})", ids):
            labels[row["id"]] = ("thread", row["title"] or "")
        return labels

    def _source_title(self, db, source_id):
        """现查来源标题；写快照时用它（那时源一定还在）。"""
        if not source_id:
            return ""
        return self._source_labels(db, [source_id]).get(source_id, ("", ""))[1]

    def _attach_source(self, item, labels):
        """给一条记忆/候选补上 source_kind / source_title（人话版），并去掉快照列。

        标题以落库时的快照为准：审核界面要回答的是"这条当初是从哪儿提炼出来的"，
        快照记录下来就不再改变，源笔记被删掉也还在。老数据没有快照，才回退到现查。
        三者都拿不到（源已删除的老数据）时标 unknown，由界面说明依据缺失，
        而不是甩一串 id 让用户猜。
        """
        snapshot = (item.pop("source_label", None) or "").strip()
        source_id = item.get("source_note_id")
        if not source_id:
            # 没有来源 id 说明是用户自己加的，与"AI 从某处提炼"必须能区分开。
            item["source_kind"], item["source_title"] = "user", ""
            return item
        kind, live_title = (labels or {}).get(source_id, (None, ""))
        if kind:
            # 源还在库里就按现查的显示（标题改过则以新的为准，点进去要对得上）；
            # 只有现查连标题都拿不到时才用快照兜底。
            item["source_kind"], item["source_title"] = kind, live_title or snapshot
        elif snapshot:
            item["source_kind"], item["source_title"] = "deleted", snapshot
        else:
            item["source_kind"], item["source_title"] = "unknown", ""
        return item

    def _memory_public(self, row, labels=None):
        """去掉向量本体，但保留"这条的向量是不是最新的"这个可观测状态。

        embedding / embedding_model 是实现细节，不进接口响应；而
        embedding_fresh 是用户看得懂的状态——内容改过、向量还没重算时它是
        False。不暴露这个字段的话，向量过期就成了一件只有日志知道的事：
        用户明明改了记忆，回答却还按旧口径，界面上看不出任何异常。

        source_kind / source_title 由 _attach_source 解析：没有来源 id 说明这条
        是用户自己加的（kind=user），与"AI 从某处提炼出来"必须能区分开。
        """
        fresh = False
        if "embedding" in row.keys():
            fresh = bool(row["embedding"]) and row["embedding_hash"] == row["content_hash"]
        public = dict(row)
        public.pop("embedding", None)
        public.pop("embedding_model", None)
        public.pop("embedding_hash", None)
        public["embedding_fresh"] = fresh
        return self._attach_source(public, labels)

    def memory_stats(self, model=None):
        """记忆管理页的顶部计数：各状态条数 + 有多少条向量待重建。

        单列一个方法，而不是让前端拉全量列表再自己数：active 一旦超过列表上限，
        "注入预算"和"向量过期数"就会算错——而这两个数字恰好是用户最需要准的
        （前者决定哪些记忆真进了提示词，后者决定他要不要点重建）。
        """
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            memories = {row["status"]: row["n"] for row in db.execute(
                "SELECT status, COUNT(*) AS n FROM memories GROUP BY status")}
            candidates = {row["status"]: row["n"] for row in db.execute(
                "SELECT status, COUNT(*) AS n FROM memory_candidates GROUP BY status")}
            if model:
                stale = db.execute("""SELECT COUNT(*) FROM memories
                    WHERE status='active' AND (embedding IS NULL
                        OR embedding_model != ? OR embedding_hash != content_hash)""",
                    (model,)).fetchone()[0]
            else:
                stale = db.execute("""SELECT COUNT(*) FROM memories
                    WHERE status='active'
                      AND (embedding IS NULL OR embedding_hash != content_hash)"""
                    ).fetchone()[0]
        return {"memories": memories, "candidates": candidates,
                "stale_embeddings": stale}

    def memories_missing_embeddings(self, model, limit=200):
        """active 且向量缺失或已过期的记忆（内容改过，指纹就对不上）。"""
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("""SELECT * FROM memories
                WHERE status = 'active'
                  AND (embedding IS NULL OR embedding_model != ?
                       OR embedding_hash != content_hash)
                ORDER BY updated_at DESC LIMIT ?""", (model, limit)).fetchall()
        return [dict(row) for row in rows]

    def save_memory_embeddings(self, records):
        """把向量写回记忆行；embedding_hash 记录生成向量时的内容指纹。

        秋招知识点：刻意不更新 updated_at——它是“记忆编辑时间”，
        向量回填属于内部缓存维护，不该改变业务上的排序语义。
        """
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executemany("""UPDATE memories
                SET embedding = ?, embedding_model = ?, embedding_hash = ?
                WHERE id = ?""", [
                (record["vector"], record["model"], record["content_hash"],
                 record["id"])
                for record in records
            ])

    def _active_rows_with_embeddings(self, model, limit=200):
        """内部查询：带向量缓存列的 active 记忆（只取向量已就绪的行）。"""
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("""SELECT * FROM memories
                WHERE status = 'active' AND embedding IS NOT NULL
                  AND embedding_model = ? AND embedding_hash = content_hash
                ORDER BY updated_at DESC LIMIT ?""", (model, limit)).fetchall()
        return [dict(row) for row in rows]

    def _recall_by_similarity(self, query, limit):
        """按相关度召回：缺失向量先补算，再与查询向量算余弦取 top-k。"""
        # 函数内延迟 import：database 层保持不依赖 AI 模块；这里也避免了
        # embeddings 模块将来反向引用存储层时的循环 import。
        from . import embeddings as emb

        model_name = emb.MODEL_NAME
        # 秋招知识点：lazy backfill（惰性回填）——向量在第一次被需要时补算，
        # 而不是写入时同步计算。写入链路保持轻快，模型加载成本分摊到读路径。
        pending = self.memories_missing_embeddings(model_name)
        if pending:
            vectors = emb.embed_texts([row["content"] for row in pending])
            self.save_memory_embeddings([
                {"id": row["id"], "vector": emb.vector_to_blob(vector),
                 "model": model_name, "content_hash": row["content_hash"]}
                for row, vector in zip(pending, vectors)
            ])

        query_vector = emb.embed_query(query)
        scored = []
        for row in self._active_rows_with_embeddings(model_name):
            vector = emb.blob_to_vector(row.pop("embedding"), emb.EXPECTED_DIMENSION)
            row.pop("embedding_model")
            row.pop("embedding_hash")
            scored.append({**row, "score": emb.cosine_similarity(query_vector, vector)})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]

    def recall_memories(self, query=None, limit=10):
        """统一取记忆入口：query 为空按时间取最新，非空按相关度召回 top-k。

        秋招知识点：接口先于实现稳定——调用方今天传 query=None 拿到时间排序，
        明天同一签名自动升级为向量检索。向量算不了（模型缺失/推理异常）时
        降级回时间排序并记日志：召回是增强路径，绝不能拖垮业务主链路。
        """
        if not query or not query.strip():
            return self.active_memories(limit=limit)
        try:
            return self._recall_by_similarity(query.strip(), limit)
        except Exception:  # noqa: BLE001 - 任何向量链路故障都降级，不向上抛
            logging.getLogger(__name__).warning(
                "memory_recall_fallback", exc_info=True)
            return self.active_memories(limit=limit)

    @staticmethod
    def _task_dict(row):
        """sqlite3.Row 转普通字典，集中维护返回给接口的字段。"""
        return dict(row) if row else None

    def create_task(self, note_id, title, details, scheduled_at, reminder_at,
                    timezone_name, source_action, now):
        with closing(sqlite3.connect(self.path)) as db, db:
            cursor = db.execute("""INSERT INTO tasks
                (note_id, title, details, scheduled_at, reminder_at, timezone,
                 source_action, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (note_id, title, details, scheduled_at, reminder_at,
                 timezone_name, source_action, now, now))
            task_id = cursor.lastrowid
        return self.task(task_id)

    def task(self, task_id):
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return self._task_dict(row)

    def tasks(self, status=None, limit=50, offset=0, date_from=None, date_to=None):
        """按状态与时间区间查询任务。

        date_from / date_to 是已归一到 UTC 的 ISO 文本，与库里 scheduled_at 同格式，
        直接做字符串比较即可。日历必须带区间：不带区间时"按时间升序 + limit"会先把
        最早的历史任务取出来，当月反而被挤掉——任务攒多了日历就永远是空的。
        """
        clauses, params = [], []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if date_from:
            clauses.append("scheduled_at >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("scheduled_at < ?")
            params.append(date_to)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                f"SELECT * FROM tasks {where} ORDER BY scheduled_at LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return [self._task_dict(row) for row in rows]

    def update_task(self, task_id, fields, updated_at):
        # 字段名来自下面的白名单而不是用户原文，避免动态 SQL 注入。
        allowed = {"title", "details", "scheduled_at", "reminder_at", "timezone", "status"}
        changes = {key: value for key, value in fields.items() if key in allowed}
        changes["updated_at"] = updated_at
        assignments = ", ".join(f"{key} = ?" for key in changes)
        values = [*changes.values(), task_id]
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(f"UPDATE tasks SET {assignments} WHERE id = ?", values)
        return self.task(task_id)

    def delete_task(self, task_id):
        """彻底删除任务行（硬删除）。

        与 update_task(status='cancelled') 是两种不同语义：取消表示"这事我不做了"，
        记录本身有意义，要留着；删除表示"这条记录本来就是错的"，留下只会污染列表。
        误建的任务（例如调试时刷出来的空标题）只能靠删除清掉。
        """
        with closing(sqlite3.connect(self.path)) as db, db:
            cursor = db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return cursor.rowcount == 1

    def delete_tasks(self, task_ids):
        """批量硬删除任务，返回真正被删掉的 ID。

        IN (?, ?, ...) 的占位符按数量动态拼；只拼问号不拼值，值仍走参数绑定。
        输入先按出现顺序去重，返回时也保持这个顺序，方便前端把结果一一对上。
        """
        ids = list(dict.fromkeys(int(i) for i in task_ids))
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        with closing(sqlite3.connect(self.path)) as db, db:
            found = {row[0] for row in db.execute(
                f"SELECT id FROM tasks WHERE id IN ({marks})", ids).fetchall()}
            if found:
                db.execute(f"DELETE FROM tasks WHERE id IN ({marks})", ids)
        return [i for i in ids if i in found]

    # 以 note_id 直接关联的派生表。表名是类常量而不是用户输入，可以安全拼进 SQL。
    NOTE_CHILD_TABLES = (
        "assets", "analyses", "intent_confirmations", "folder_suggestions", "note_folders",
    )

    def delete_note(self, note_id, delete_tasks=True):
        """彻底删除一篇笔记及其全部派生数据，不存在时返回 None。

        笔记是这一坨数据的根：图片、AI 分析、收藏意图、收藏夹归属、检索索引
        （documents → chunks → embeddings）、收藏夹建议都是从它派生的。只删 notes 那一行
        会留下一堆指向不存在笔记的孤儿行，检索还会继续命中已删除的内容。

        documents/chunks/embeddings 是三层嵌套，必须从最里层往外删：先按 note_id 找到
        document，再删它的 chunks，最后按 chunk_id 删 embeddings。

        返回各类被删行数，外加 storage_keys（图片文件名）。磁盘文件交给调用方删除——
        数据库层不该知道 media_root 在哪。delete_tasks=False 时保留该笔记派生的待办：
        已经排进日历的待办是用户的时间承诺，是否一起清掉由接口层决定。
        """
        stats = {"note_id": note_id}
        with closing(sqlite3.connect(self.path)) as db, db:
            row = db.execute("SELECT 1 FROM notes WHERE note_id=?", (note_id,)).fetchone()
            if row is None:
                return None
            document_ids = [r[0] for r in db.execute(
                "SELECT id FROM documents WHERE note_id=?", (note_id,)).fetchall()]
            if document_ids:
                marks = ",".join("?" * len(document_ids))
                stats["embeddings"] = db.execute(
                    f"""DELETE FROM embeddings WHERE chunk_id IN
                        (SELECT id FROM chunks WHERE document_id IN ({marks}))""",
                    document_ids).rowcount
                stats["chunks"] = db.execute(
                    f"DELETE FROM chunks WHERE document_id IN ({marks})",
                    document_ids).rowcount
                stats["documents"] = db.execute(
                    f"DELETE FROM documents WHERE id IN ({marks})", document_ids).rowcount
            storage_keys = [r[0] for r in db.execute(
                "SELECT storage_key FROM assets WHERE note_id=? AND storage_key IS NOT NULL",
                (note_id,)).fetchall()]
            for table in self.NOTE_CHILD_TABLES:
                stats[table] = db.execute(
                    f"DELETE FROM {table} WHERE note_id=?", (note_id,)).rowcount
            # 计划步骤引用的来源笔记只解除引用，不删计划本身：计划是用户确认过的承诺，
            # 去掉一篇参考来源不该让整份计划凭空消失。
            stats["plan_item_sources"] = db.execute(
                "DELETE FROM plan_item_sources WHERE note_id=?", (note_id,)).rowcount
            stats["tasks"] = db.execute(
                "DELETE FROM tasks WHERE note_id=?", (note_id,)).rowcount if delete_tasks else 0
            stats["notes"] = db.execute(
                "DELETE FROM notes WHERE note_id=?", (note_id,)).rowcount
        stats["storage_keys"] = storage_keys
        return stats
