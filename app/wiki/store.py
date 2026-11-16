"""
wiki/store.py —— Wiki 知识页存储

【作用与功能】
存放“面向问题”的知识页:每页由一个标题、一组别名、若干小节构成，
每个小节必须携带可回溯的原文出处(Span: 文档 + 页码 + 原文摘句)。
页面落在 SQLite(data/wiki.db)，租户为空串表示内置公共页。

【主要组成】
- `Span`: 原文出处(文档 ID、文档名、页码、摘句)
- `Section`: 小节(标题即问题、结论正文、出处列表)
- `Page`: 知识页(标题、别名、摘要、小节列表、审核状态)
- `WikiStore`: 建表、读写、按租户列出页面
- `get_store`: 进程内单例入口

【适用场景】
- 问答链路第一级命中 Wiki 页时直接取页面正文作上下文
- 管理端查看与维护知识页
"""

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(_PROJECT_ROOT, "data", "wiki.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS wiki_pages (
    id         TEXT PRIMARY KEY,
    tenant_id  TEXT NOT NULL DEFAULT '',
    title      TEXT NOT NULL,
    aliases    TEXT NOT NULL DEFAULT '[]',
    summary    TEXT NOT NULL DEFAULT '',
    sections   TEXT NOT NULL DEFAULT '[]',
    status     TEXT NOT NULL DEFAULT 'draft',
    updated_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_wiki_tenant ON wiki_pages(tenant_id, status);

CREATE TABLE IF NOT EXISTS wiki_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    target_type TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    action      TEXT NOT NULL,
    reviewer    TEXT NOT NULL DEFAULT '',
    note        TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL DEFAULT 0
);
"""

# 老库升级:补审批字段(ALTER TABLE 是幂等的，按列名判断)
_MIGRATION_COLUMNS = [
    ("content_hash", "TEXT NOT NULL DEFAULT ''"),
    ("approved_by", "TEXT NOT NULL DEFAULT ''"),
    ("approved_at", "REAL NOT NULL DEFAULT 0"),
    ("review_note", "TEXT NOT NULL DEFAULT ''"),
]


@dataclass
class Span:
    """原文出处:知识页里每句结论都要能点回这一条。"""

    doc_id: str = ""
    doc_name: str = ""
    page: int = 0
    quote: str = ""

    def ref(self) -> str:
        """出处的人类可读描述，用于拼接引用行。"""
        return f"{self.doc_name} 第{self.page}页"


@dataclass
class Section:
    """知识页小节:标题写成用户会问的问题，正文给结论，spans 给依据。"""

    heading: str = ""
    body: str = ""
    spans: list = field(default_factory=list)


@dataclass
class Page:
    """Wiki 知识页。status 为 draft 表示尚未人工审核，不应进入问答。"""

    id: str = ""
    title: str = ""
    aliases: list = field(default_factory=list)
    summary: str = ""
    sections: list = field(default_factory=list)
    status: str = "approved"
    tenant_id: str = ""
    updated_at: float = 0.0
    approved_by: str = ""
    approved_at: float = 0.0
    review_note: str = ""

    def content_hash(self) -> str:
        """内容指纹:标题+别名+摘要+小节。内容变化即需要重新审定。"""
        payload = json.dumps(
            {
                "title": self.title,
                "aliases": self.aliases,
                "summary": self.summary,
                "sections": [{"h": s.heading, "b": s.body} for s in self.sections],
            },
            ensure_ascii=False, sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def to_row(self) -> dict:
        """转成 SQLite 行字典(sections 序列化为 JSON)。"""
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "title": self.title,
            "aliases": json.dumps(self.aliases, ensure_ascii=False),
            "summary": self.summary,
            "sections": json.dumps(
                [
                    {
                        "heading": s.heading,
                        "body": s.body,
                        "spans": [
                            {"doc_id": sp.doc_id, "doc_name": sp.doc_name,
                             "page": sp.page, "quote": sp.quote}
                            for sp in s.spans
                        ],
                    }
                    for s in self.sections
                ],
                ensure_ascii=False,
            ),
            "status": self.status,
            "updated_at": self.updated_at,
            "content_hash": self.content_hash(),
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "review_note": self.review_note,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Page":
        """从 SQLite 行还原 Page 对象。"""
        sections = [
            Section(
                heading=s.get("heading", ""),
                body=s.get("body", ""),
                spans=[Span(**sp) for sp in s.get("spans", [])],
            )
            for s in json.loads(row["sections"] or "[]")
        ]
        return cls(
            id=row["id"],
            title=row["title"],
            aliases=json.loads(row["aliases"] or "[]"),
            summary=row["summary"],
            sections=sections,
            status=row["status"],
            tenant_id=row["tenant_id"],
            updated_at=row["updated_at"],
            approved_by=row["approved_by"] if "approved_by" in row.keys() else "",
            approved_at=row["approved_at"] if "approved_at" in row.keys() else 0.0,
            review_note=row["review_note"] if "review_note" in row.keys() else "",
        )

    def render(self, sections: list = None) -> str:
        """渲染为可直接放入 Prompt 的上下文文本。

        参数:
            sections (list, 可选): 只渲染命中的小节；为空时渲染整页。

        返回:
            str: 标题 + 摘要 + 小节正文 + 依据行。
        """
        picked = sections if sections else self.sections
        parts = [self.title, self.summary.strip()]
        for sec in picked:
            block = f"{sec.heading}\n{sec.body.strip()}"
            if sec.spans:
                refs = "；".join(sp.ref() for sp in sec.spans)
                block += f"\n（依据：{refs}）"
            parts.append(block)
        return "\n\n".join(p for p in parts if p)


class WikiStore:
    """知识页的 SQLite 读写入口(每次操作独立连接，避免跨线程共享)。"""

    def __init__(self, db_path: str = DB_PATH):
        """初始化存储并确保表结构与种子页面就绪。

        参数:
            db_path (str): SQLite 文件路径，默认 data/wiki.db。
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
        self._migrate()
        self._sync_seed()

    def _migrate(self) -> None:
        """老库升级:补齐审批相关列(按列名判断，可重复执行)。"""
        with self._conn() as conn:
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(wiki_pages)").fetchall()}
            for name, ddl in _MIGRATION_COLUMNS:
                if name not in cols:
                    conn.execute(f"ALTER TABLE wiki_pages ADD COLUMN {name} {ddl}")

    def _conn(self) -> sqlite3.Connection:
        """打开一个新连接(行工厂为 sqlite3.Row)。"""
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def _sync_seed(self) -> None:
        """按种子版本写入内置页面。

        - 版本变化(seed.SEED_VERSION 递增)时覆盖全部内置页，保证代码里的
          答复口径更新能生效；
        - 版本未变时只补缺失的 id，不影响新增页与人工数据。
        """
        from app.wiki.seed import SEED_PAGES, SEED_VERSION

        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM wiki_meta WHERE key = 'seed_version'"
            ).fetchone()
        current = row["value"] if row else ""

        if current != str(SEED_VERSION):
            for page in SEED_PAGES:
                self.upsert(page)
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO wiki_meta(key, value) VALUES('seed_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(SEED_VERSION),),
                )
            self._clear_query_cache()
            return

        with self._conn() as conn:
            existing = {r["id"] for r in conn.execute("SELECT id FROM wiki_pages").fetchall()}
        for page in SEED_PAGES:
            if page.id not in existing:
                self.upsert(page)

    def _clear_query_cache(self) -> None:
        """页面口径变化后清除各租户的问答缓存。

        缓存只保存答案文本，不感知页面内容;补别名或改口径后若不清理，
        用户会继续拿到旧答案(实测:补完别名仍返回"资料中没有相关内容")。

        注意:内置页的 tenant_id 是空串，若按 wiki_pages 里的租户去枚举，
        真实租户目录 cache/<tenant> 一个都清不到(实测 before=7 / after=7)。
        因此直接遍历 cache 根目录及其所有子目录。
        """
        root = "cache"
        if not os.path.isdir(root):
            return
        directories = [root]
        try:
            for name in os.listdir(root):
                sub = os.path.join(root, name)
                if os.path.isdir(sub):
                    directories.append(sub)
        except OSError:
            return
        for directory in directories:
            try:
                names = os.listdir(directory)
            except OSError:
                continue
            for name in names:
                if not name.endswith(".json"):
                    continue
                try:
                    os.remove(os.path.join(directory, name))
                except OSError:
                    pass

    def upsert(self, page: Page) -> None:
        """写入或覆盖一页(按 id 主键)。

        内容指纹未变时保留运行时的审定状态与审批信息，避免版本升级或
        重新同步种子把已审定的页面打回草案；内容变化则以页面声明的
        初始状态为准(内容改了就要重审)。
        """
        row = page.to_row()
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT content_hash, status, approved_by, approved_at, review_note "
                "FROM wiki_pages WHERE id = ?",
                (page.id,),
            ).fetchone()
            if existing and existing["content_hash"] == row["content_hash"]:
                row["status"] = existing["status"]
                row["approved_by"] = existing["approved_by"]
                row["approved_at"] = existing["approved_at"]
                row["review_note"] = existing["review_note"]
            conn.execute(
                """
                INSERT INTO wiki_pages (id, tenant_id, title, aliases, summary, sections,
                                        status, updated_at, content_hash,
                                        approved_by, approved_at, review_note)
                VALUES (:id, :tenant_id, :title, :aliases, :summary, :sections,
                        :status, :updated_at, :content_hash,
                        :approved_by, :approved_at, :review_note)
                ON CONFLICT(id) DO UPDATE SET
                    tenant_id=excluded.tenant_id, title=excluded.title, aliases=excluded.aliases,
                    summary=excluded.summary, sections=excluded.sections, status=excluded.status,
                    updated_at=excluded.updated_at, content_hash=excluded.content_hash,
                    approved_by=excluded.approved_by, approved_at=excluded.approved_at,
                    review_note=excluded.review_note
                """,
                row,
            )

    def review(self, page_id: str, action: str, reviewer: str, note: str = "") -> str:
        """审定一页:approve 上线、reject 打回草案，并写审计日志。

        参数:
            page_id (str): 页面 ID。
            action (str): approve 或 reject。
            reviewer (str): 审定人。
            note (str): 审定意见，驳回时建议写明原因。

        返回:
            str: 变更后的状态(approved / draft)。

        异常:
            ValueError: action 非法。
            KeyError: 页面不存在。
        """
        if action not in ("approve", "reject"):
            raise ValueError("action must be approve or reject")

        status = "approved" if action == "approve" else "draft"
        now = time.time()
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE wiki_pages SET status = ?, approved_by = ?, approved_at = ?, "
                "review_note = ?, updated_at = ? WHERE id = ?",
                (status,
                 reviewer if action == "approve" else "",
                 now if action == "approve" else 0.0,
                 note, now, page_id),
            )
            if cur.rowcount == 0:
                raise KeyError(page_id)
        self.log_review("page", page_id, action, reviewer, note)
        return status

    def log_review(self, target_type: str, target_id: str, action: str,
                   reviewer: str = "", note: str = "") -> None:
        """写入一条审定审计记录(page / document)。"""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO review_log (target_type, target_id, action, reviewer, note, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (target_type, target_id, action, reviewer, note, time.time()),
            )

    def list_reviews(self, limit: int = 50) -> list:
        """最近的审定记录(新→旧)。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM review_log ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def list_pages(self, tenant_id: str = "", status: str = "approved") -> list:
        """列出该租户可见的知识页(内置页 tenant_id 为空串，对所有租户可见)。

        参数:
            tenant_id (str): 租户 ID。
            status (str): 只返回该状态；传 None 表示不限状态。

        返回:
            list[Page]: 知识页列表。
        """
        sql = "SELECT * FROM wiki_pages WHERE tenant_id IN ('', ?)"
        args = [tenant_id]
        if status:
            sql += " AND status = ?"
            args.append(status)
        sql += " ORDER BY id"
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [Page.from_row(r) for r in rows]

    def get(self, page_id: str, tenant_id: str = "") -> Page | None:
        """按 ID 取一页；不存在返回 None。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM wiki_pages WHERE id = ? AND tenant_id IN ('', ?)",
                (page_id, tenant_id),
            ).fetchone()
        return Page.from_row(row) if row else None


_STORE_LOCK = threading.Lock()
_STORES: dict = {}


def get_store(db_path: str = DB_PATH) -> WikiStore:
    """获取进程内单例存储(按路径缓存)。

    参数:
        db_path (str): SQLite 路径。

    返回:
        WikiStore: 已建表并完成种子写入的存储实例。
    """
    with _STORE_LOCK:
        if db_path not in _STORES:
            _STORES[db_path] = WikiStore(db_path)
        return _STORES[db_path]
