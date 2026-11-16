"""
test_wiki_semantic.py —— 语义兜底路由的单元测试

【测试覆盖范围】
- 高相似度命中页面;低相似度落回 RAG;法条类查询不参与语义路由。
- 用假 embedder 提供确定性向量,不加载真实模型。

【依赖】
- app.wiki.router 与本地 wiki.db(允许为空库，只要是 WikiStore 实例即可)。
"""
import numpy as np

from app.wiki.router import WikiRouter
from app.wiki.store import get_store


class _FakeEmbedder:
    """按页面下标返回正交单位向量，便于精确构造相似度。"""

    def __init__(self, dim: int, query_vec: list):
        """初始化。

        参数:
            dim (int): 页面数量(向量维度)。
            query_vec (list): embed_query 固定返回的向量。
        """
        self._dim = dim
        self._query_vec = query_vec

    def embed_documents(self, texts: list) -> list:
        """第 i 篇文档返回第 i 个基向量。"""
        return [[1.0 if j == i else 0.0 for j in range(self._dim)]
                for i in range(len(texts))]

    def embed_query(self, text: str) -> list:
        """所有查询返回预设向量。"""
        return self._query_vec


def _router():
    return WikiRouter(get_store(), "test-tenant")


def test_semantic_hit_when_similar():
    router = _router()
    if not router._pages:
        return
    page = router._pages[0]
    vec = [1.0] + [0.0] * (len(router._pages) - 1)
    hit = router.route_semantic("完全不含别名的说法", _FakeEmbedder(len(router._pages), vec))
    assert hit is not None and hit.page.id == page.id


def test_semantic_miss_when_unrelated():
    router = _router()
    if not router._pages:
        return
    vec = [0.0] * len(router._pages)   # 与所有页面正交 -> 相似度为 0
    assert router.route_semantic("公司班车几点发车", _FakeEmbedder(len(router._pages), vec)) is None


def test_clause_query_not_routed():
    router = _router()
    if not router._pages:
        return
    vec = [1.0] + [0.0] * (len(router._pages) - 1)
    assert router.route_semantic("刑法第二百六十四条", _FakeEmbedder(len(router._pages), vec)) is None


def test_empty_query_not_routed():
    router = _router()
    assert router.route_semantic("", object()) is None
