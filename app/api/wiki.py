"""app/api/wiki.py —— Wiki 知识页查询与路由调试接口

只读接口:页面内容由人工在 data/wiki.db 维护，这里负责按租户列出、
查看单页，以及用 /route 观察一个问题会被路由到哪一页(便于调别名)。
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.auth import require_user
from app.wiki.audit import DEFAULT_QUESTIONS, profile_document, trial_search
from app.wiki.doc_status import set_source_status
from app.wiki.router import WikiRouter
from app.wiki.store import get_store

router = APIRouter(prefix="/api/wiki", tags=["wiki"])


class RouteRequest(BaseModel):
    """路由调试请求体。"""

    query: str


class ReviewRequest(BaseModel):
    """页面审定请求体。"""

    action: str          # approve | reject
    note: str = ""


class DocumentReviewRequest(BaseModel):
    """文档审定请求体。"""

    source: str          # 文档名
    status: str = "approved"   # approved | draft
    note: str = ""


class CheckRequest(BaseModel):
    """上线前质检请求体。"""

    source: str
    questions: list[str] = []   # 留空则用默认代表性问题


def _require_super_admin(user: dict) -> None:
    """审定属于管理动作，仅 super_admin 可执行。"""
    if user.get("role") != "super_admin":
        raise HTTPException(403, "Only super_admin can review")


def _reviewer(user: dict) -> str:
    """记录审定人，便于审计追溯。"""
    return user.get("username") or "unknown"


def _page_summary(page) -> dict:
    """列表视图:不含正文，只给标题、摘要与出处数量。"""
    return {
        "id": page.id,
        "title": page.title,
        "summary": page.summary,
        "aliases": page.aliases,
        "status": page.status,
        "sections": len(page.sections),
        "sources": len({sp.doc_name for s in page.sections for sp in s.spans}),
        "approved_by": page.approved_by,
        "approved_at": page.approved_at,
    }


def _page_detail(page) -> dict:
    """详情视图:带小节正文与可溯源的原文出处。"""
    return {
        "id": page.id,
        "title": page.title,
        "summary": page.summary,
        "aliases": page.aliases,
        "status": page.status,
        "sections": [
            {
                "heading": s.heading,
                "body": s.body,
                "spans": [
                    {"doc_id": sp.doc_id, "doc_name": sp.doc_name,
                     "page": sp.page, "quote": sp.quote}
                    for sp in s.spans
                ],
            }
            for s in page.sections
        ],
    }


@router.get("/pages")
def list_pages(user: dict = Depends(require_user)):
    """列出当前租户可见的知识页(含内置公共页)。"""
    pages = get_store().list_pages(user["tenant_id"], status=None)
    return {"total": len(pages), "pages": [_page_summary(p) for p in pages]}


@router.get("/pages/{page_id}")
def get_page(page_id: str, user: dict = Depends(require_user)):
    """查看单个知识页的完整内容与原文出处。"""
    page = get_store().get(page_id, user["tenant_id"])
    if not page:
        raise HTTPException(404, "Wiki page not found")
    return _page_detail(page)


@router.post("/route")
def debug_route(req: RouteRequest, user: dict = Depends(require_user)):
    """调试接口:返回该问题是否命中 Wiki，命中哪一页/哪一节。

    未命中即表示这条问题会走 RAG 兜底，可据此补充别名或新建页面。
    """
    hit = WikiRouter(get_store(), user["tenant_id"]).route(req.query)
    if not hit:
        return {"hit": False, "page_id": None, "matched": None, "sections": []}
    return {
        "hit": True,
        "page_id": hit.page.id,
        "page_title": hit.page.title,
        "matched": hit.matched,
        "score": hit.score,
        "sections": [s.heading for s in hit.sections] or "整页",
    }


@router.post("/pages/{page_id}/review")
def review_page(page_id: str, req: ReviewRequest, user: dict = Depends(require_user)):
    """审定知识页:approve 上线参与问答，reject 打回草案。

    仅 super_admin 可执行；每次审定写入 review_log，记录审定人与意见。
    """
    _require_super_admin(user)
    store = get_store()
    try:
        status = store.review(page_id, req.action, _reviewer(user), req.note)
    except KeyError:
        raise HTTPException(404, "Wiki page not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"page_id": page_id, "status": status, "reviewer": _reviewer(user)}


@router.post("/documents/review")
def review_document(req: DocumentReviewRequest, user: dict = Depends(require_user)):
    """审定原始文档:approved 参与检索，draft 从问答中剔除。

    切片级 doc_status 立即生效，无需重建索引。
    """
    _require_super_admin(user)
    if req.status not in ("approved", "draft"):
        raise HTTPException(400, "status must be approved or draft")

    changed = set_source_status(user["tenant_id"], req.source, req.status)
    get_store().log_review("document", req.source, req.status, _reviewer(user), req.note)
    return {"source": req.source, "status": req.status, "updated_chunks": changed}


@router.post("/documents/check")
def check_document(req: CheckRequest, user: dict = Depends(require_user)):
    """上线前质检:切片质量 + 该文档范围内的试检索。

    审核人据此判断「这份文档切得对不对、能不能被召回」，再决定是否批准。
    试检索复用生产检索链路，仅把过滤条件换成该文档，因此结果与上线后一致。
    """
    _require_super_admin(user)
    try:
        profile = profile_document(user["tenant_id"], req.source)
    except KeyError:
        raise HTTPException(404, "Document not found in vector store")

    questions = (req.questions or DEFAULT_QUESTIONS)[:5]
    trial = trial_search(user["tenant_id"], req.source, questions)

    issues = []
    if profile["short_chunks"]:
        issues.append(f"{profile['short_chunks']} 个切片过短(<30 字)，检索价值低")
    if profile["duplicate_chunks"]:
        issues.append(f"{profile['duplicate_chunks']} 个重复切片")
    if profile["suspected_furniture"]:
        issues.append(f"疑似页眉页脚残留 {len(profile['suspected_furniture'])} 处")
    if profile["no_page_chunks"]:
        issues.append(f"{profile['no_page_chunks']} 个切片缺页码，引用无法回溯")
    missed = [t["question"] for t in trial if not t["hits"]]
    if missed:
        issues.append(f"{len(missed)} 个试检索问题零命中")

    return {
        "source": req.source,
        "profile": profile,
        "trial": trial,
        "review_ready": not issues,
        "issues": issues,
    }


@router.get("/reviews")
def list_reviews(limit: int = 50, user: dict = Depends(require_user)):
    """最近的审定记录(新→旧)，用于合规审计与追溯。"""
    _require_super_admin(user)
    return {"reviews": get_store().list_reviews(limit)}
