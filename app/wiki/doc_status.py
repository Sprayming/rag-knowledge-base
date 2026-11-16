"""wiki/doc_status.py —— 设置切片级文档审定状态

doc_status 取 approved / draft。draft 的切片不参与问答检索，
审定通过后改为 approved 即恢复。供 API 与命令行脚本共用。
"""

import os

import app.core.config as cfg

VALID_STATUS = ("approved", "draft")


def _collection(tenant_id: str):
    """打开租户向量库集合。"""
    import chromadb

    persist_dir = os.path.join(cfg.CHROMA_PERSIST_DIR, tenant_id)
    return chromadb.PersistentClient(path=persist_dir).get_or_create_collection("langchain")


def set_source_status(tenant_id: str, source: str, status: str) -> int:
    """把某份文档的全部切片改为指定审定状态。

    参数:
        tenant_id (str): 租户 ID。
        source (str): 文档名(与切片 metadata 的 source 一致)。
        status (str): approved 或 draft。

    返回:
        int: 实际更新的切片数。
    """
    if status not in VALID_STATUS:
        raise ValueError(f"status must be one of {VALID_STATUS}")

    col = _collection(tenant_id)
    got = col.get(where={"source": source}, include=["metadatas"])
    ids, metas = got["ids"], got["metadatas"]

    changed_ids, changed_metas = [], []
    for i, meta in zip(ids, metas):
        meta = dict(meta or {})
        if meta.get("doc_status") == status:
            continue
        meta["doc_status"] = status
        changed_ids.append(i)
        changed_metas.append(meta)

    if changed_ids:
        col.update(ids=changed_ids, metadatas=changed_metas)
    return len(changed_ids)


def fill_missing_status(tenant_id: str, status: str = "approved") -> int:
    """为缺少 doc_status 的历史切片补默认状态(不覆盖已有值)。

    返回:
        int: 实际更新的切片数。
    """
    col = _collection(tenant_id)
    got = col.get(include=["metadatas"])
    ids, metas = got["ids"], got["metadatas"]

    changed_ids, changed_metas = [], []
    for i, meta in zip(ids, metas):
        meta = dict(meta or {})
        if meta.get("doc_status"):
            continue
        meta["doc_status"] = status
        changed_ids.append(i)
        changed_metas.append(meta)

    if changed_ids:
        col.update(ids=changed_ids, metadatas=changed_metas)
    return len(changed_ids)
