"""wiki/router.py —— 把用户问题路由到知识页

路由策略刻意保持确定性与零模型开销:
1. 查询里出现“第X条”这类明确法条引用时，直接交给 RAG，不进 Wiki；
2. 否则用页面别名与小节标题做归一化短语匹配，命中即返回该页(或该小节)；
3. 都未命中则返回 None，由调用方降级到混合检索。
"""

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

from app.wiki.store import Page, WikiStore

# 明确指向具体法条的查询(如“第三十九条”)交给 RAG，Wiki 页只讲内规与合规口径
_CLAUSE_RE = re.compile(r"第[一二三四五六七八九十百千零〇0-9]{1,6}条")
# 归一化时去掉空白与中英文标点，避免问法差异导致匹配失败
_STRIP_RE = re.compile(r"[\s\u3000，。？！、；：\u201c\u201d\u2018\u2019（）()《》【】,\.\?!;:\"'\-\u2014]+")
# 语义兜底阈值:实测正例 0.561~0.658、负例 0.332~0.487，取中间值
SEMANTIC_THRESHOLD = 0.55
# 页面向量缓存:(租户, 页面内容指纹) -> (页面列表, 向量矩阵)
_PAGE_VECTOR_CACHE = {}


def normalize(text: str) -> str:
    """归一化问题文本:小写 + 去空白与标点。"""
    return _STRIP_RE.sub("", (text or "").lower())


@dataclass
class WikiHit:
    """一次命中:页面 + 命中小节(空表示整页) + 命中短语 + 得分。"""

    page: Page
    sections: list = field(default_factory=list)
    matched: str = ""
    score: int = 0

    def render(self) -> str:
        """渲染为 Prompt 上下文。"""
        return self.page.render(self.sections)


class WikiRouter:
    """基于别名的确定性路由(不做向量检索，7 个页面量级足够)。"""

    def __init__(self, store: WikiStore, tenant_id: str = ""):
        """初始化并加载该租户可见的已审核页面。

        参数:
            store (WikiStore): 知识页存储。
            tenant_id (str): 租户 ID。
        """
        self._store = store
        self._tenant_id = tenant_id
        self._pages = store.list_pages(tenant_id)

    def route(self, query: str) -> WikiHit | None:
        """尝试把问题路由到知识页。

        参数:
            query (str): 用户问题(通常是重写后的查询)。

        返回:
            WikiHit | None: 命中时返回页面与小节；未命中返回 None(调用方降级到 RAG)。
        """
        q = normalize(query)
        if not q or _CLAUSE_RE.search(q):
            return None

        best = WikiHit(page=None, sections=[], matched="", score=0)
        for page in self._pages:
            # 别名与整页标题命中:匹配串越长越具体
            for alias in page.aliases + [page.title]:
                a = normalize(alias)
                if len(a) >= 2 and a in q and len(a) > best.score:
                    best = WikiHit(page=page, sections=[], matched=alias, score=len(a))
            # 小节标题命中:只带回该节，省 Prompt 预算
            for sec in page.sections:
                h = normalize(sec.heading)
                if len(h) >= 4 and h in q and len(h) + 2 > best.score:
                    best = WikiHit(page=page, sections=[sec], matched=sec.heading,
                                   score=len(h) + 2)
        return best if best.page else None

    def route_semantic(self, query: str, embedder, threshold: float = SEMANTIC_THRESHOLD):
        """短语未命中时的语义兜底:用问题向量匹配页面标题+摘要+别名+小节标题。

        字面别名无法穷举用户说法("司机多久被要求休息"不含别名"司机休息")，
        因此补一层语义匹配;但仍保持高阈值，宁可落回 RAG 也不误命中。

        参数:
            query (str): 用户原始问题。
            embedder: LangChain embeddings 实例(提供 embed_query/embed_documents)。
            threshold (float): 余弦相似度阈值，低于该值视为未命中。

        返回:
            WikiHit | None: 命中返回整页(不带小节)，否则 None。
        """
        if not self._pages or embedder is None:
            return None
        if not normalize(query) or _CLAUSE_RE.search(query):
            return None

        bundles = [_page_bundle(p) for p in self._pages]
        signature = (self._tenant_id,
                     hashlib.md5("\n".join(bundles).encode("utf-8")).hexdigest())
        matrix = _PAGE_VECTOR_CACHE.get(signature)
        if matrix is None:
            matrix = np.asarray(embedder.embed_documents(bundles), dtype=float)
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
            _PAGE_VECTOR_CACHE[signature] = matrix

        vector = np.asarray(embedder.embed_query(query), dtype=float)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return None
        sims = matrix @ (vector / norm)
        index = int(sims.argmax())
        if sims[index] < threshold:
            return None
        return WikiHit(page=self._pages[index], sections=[], matched="semantic",
                       score=int(sims[index] * 100))


def _page_bundle(page: Page) -> str:
    """把一页的可检索文本拼成一段，用于语义路由的页面向量。"""
    return "。".join([
        page.title,
        page.summary,
        "、".join(page.aliases),
        "、".join(sec.heading for sec in page.sections),
    ])
