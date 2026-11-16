"""wiki/audit.py —— 文档上线前的质检与试检索

审核人批准一份文档前，需要先回答两个问题：
1. 这份文档切得对不对？（切片长度、空片、重复片、页眉页脚残留、页码覆盖）
2. 这份文档能不能被召回？（用代表性问题在该文档范围内做一次试检索）

`profile_document` 提供前者，`trial_search` 提供后者；两者都只读，
会同时包含 draft 切片——否则未审定文档永远无法被审核。
"""

import os
from collections import Counter

# 低于该长度的切片基本没有检索价值
MIN_CHUNK_CHARS = 30
# 页眉页脚判定：该短行出现在 40% 以上的切片里
FURNITURE_RATIO = 0.4

DEFAULT_QUESTIONS = ["这份文档主要内容是什么", "有哪些禁止或限制", "违反规定会有什么后果"]


def _collection(tenant_id: str):
    """打开租户向量库集合。"""
    import chromadb

    import app.core.config as cfg

    persist_dir = os.path.join(cfg.CHROMA_PERSIST_DIR, tenant_id)
    return chromadb.PersistentClient(path=persist_dir).get_or_create_collection("langchain")


def repeated_lines(docs: list, ratio: float = FURNITURE_RATIO, max_len: int = 40) -> list:
    """找出在多数切片里重复出现的短行(页眉页脚残留)。

    参数:
        docs (list[str]): 切片正文列表。
        ratio (float): 出现在多少比例的切片上才判定为版式噪声。
        max_len (int): 只统计不超过该长度的行。

    返回:
        list[str]: 疑似残留行，按出现次数降序。
    """
    if len(docs) < 3:
        return []
    counter = Counter()
    for d in docs:
        seen = {ln.strip() for ln in (d or "").split("\n") if 0 < len(ln.strip()) <= max_len}
        counter.update(seen)
    threshold = max(3, int(len(docs) * ratio))
    return [ln for ln, c in counter.most_common() if c >= threshold][:5]


def profile_document(tenant_id: str, source: str) -> dict:
    """统计一份文档的切片质量，供审核人判断是否值得上线。

    参数:
        tenant_id (str): 租户 ID。
        source (str): 文档名。

    返回:
        dict: 切片数、长度分布、异常切片数、页码覆盖、文档状态与指纹。

    异常:
        KeyError: 该文档没有任何切片。
    """
    got = _collection(tenant_id).get(where={"source": source},
                                     include=["documents", "metadatas"])
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    if not docs:
        raise KeyError(source)

    lengths = [len(d or "") for d in docs]
    statuses = Counter((m or {}).get("doc_status", "<缺失>") for m in metas)
    doc_ids = {(m or {}).get("doc_id", "") for m in metas}

    return {
        "source": source,
        "chunks": len(docs),
        "avg_chars": round(sum(lengths) / len(lengths)),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "short_chunks": sum(1 for n in lengths if n < MIN_CHUNK_CHARS),
        "duplicate_chunks": len(docs) - len({d for d in docs if d}),
        "no_page_chunks": sum(1 for m in metas if not (m or {}).get("page_number")),
        "suspected_furniture": repeated_lines(docs),
        "doc_status": statuses.most_common(1)[0][0] if statuses else "<缺失>",
        "doc_id": next(iter(doc_ids)) if len(doc_ids) == 1 else list(doc_ids),
    }


def trial_search(tenant_id: str, source: str, questions: list, k: int = 3) -> list:
    """在该文档范围内做试检索，验证「上线后能否被召回」。

    复用生产链路(稠密 + BM25 + RRF)，仅把过滤条件换成该文档，
    因此结果与上线后的实际召回一致；未加载零模型之外的重排模型。

    参数:
        tenant_id (str): 租户 ID。
        source (str): 文档名。
        questions (list[str]): 代表性问题。
        k (int): 每个问题返回的条数。

    返回:
        list[dict]: 每个问题及其命中片段(页码 / RRF 分数 / 摘要)。
    """
    from langchain_community.vectorstores import Chroma

    import app.core.config as cfg
    from app.retrieval.embedder_factory import create_embedder
    from app.retrieval.hybrid_retriever import HybridRetriever

    got = _collection(tenant_id).get(where={"source": source},
                                     include=["documents", "metadatas"])
    texts = [d for d in (got.get("documents") or []) if d]
    if not texts:
        raise KeyError(source)
    text_meta = {d: m for d, m in zip(got["documents"], got["metadatas"] or []) if d}

    persist_dir = os.path.join(cfg.CHROMA_PERSIST_DIR, tenant_id)
    store = Chroma(embedding_function=create_embedder(), persist_directory=persist_dir)
    retriever = HybridRetriever(
        store, texts, k=k, text_meta=text_meta, doc_filter={"source": source},
    )

    results = []
    for q in questions:
        docs = retriever.retrieve(q)
        results.append({
            "question": q,
            "hits": [
                {
                    "page": (d.metadata or {}).get("page_number"),
                    "rrf_score": round(float((d.metadata or {}).get("rrf_score", 0)), 4),
                    "text": d.page_content[:120].replace("\n", " "),
                }
                for d in docs
            ],
        })
    return results
