"""
test_doc_status_filter.py —— 文档级状态(doc_status)过滤的单元测试

【测试覆盖范围】
- `_get_all_texts_with_meta`:未审定(draft)切片不进入稀疏通道语料。
- `HybridRetriever._dense_search`:稠密检索带上排除 draft 的过滤条件。

【依赖】
- 用假向量库与 Mock 替代真实 Chroma，不加载模型。
"""
from unittest.mock import Mock

from app.api.chat import _get_all_texts_with_meta
from app.retrieval.hybrid_retriever import HybridRetriever


class _FakeCollection:
    def get(self):
        return {
            "documents": ["已审定正文", "草案正文"],
            "metadatas": [
                {"source": "规范.pdf", "doc_status": "approved"},
                {"source": "手册.pdf", "doc_status": "draft"},
            ],
        }


class _FakeStore:
    _collection = _FakeCollection()


def test_draft_chunk_excluded_from_sparse_corpus():
    """草案切片不应进入 BM25 / BGE-M3 稀疏通道的语料。"""
    texts, text_meta = _get_all_texts_with_meta(_FakeStore())
    assert texts == ["已审定正文"]
    assert "草案正文" not in text_meta


def test_dense_search_passes_doc_filter():
    """稠密检索必须携带排除 draft 的过滤条件。"""
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.k = 5
    retriever.doc_filter = {"doc_status": {"$ne": "draft"}}
    retriever.dense_store = Mock()
    retriever.dense_store.similarity_search_with_score.return_value = []

    retriever._dense_search("能赔多少")

    kwargs = retriever.dense_store.similarity_search_with_score.call_args.kwargs
    assert kwargs["filter"] == {"doc_status": {"$ne": "draft"}}
