"""backfill_chunk_metadata.py —— 为已入库切片回填 page_number 与 doc_id

[已废弃] 本脚本用「全文 find」定位页码，遇到重复段落会把页码带回更早的页
(实测刑法 540 片中有 90 处页码回跳)。请改用 scripts/cleanup_p3_metadata.py:
它按切片顺序逐页对齐，保证页码单调，并支持删除测试文档、清理问答缓存。

背景:早期入库的切片 metadata 只有 source / chunk，引用无法回显页码，
Wiki 知识页也无法绑定来源。本脚本不重新嵌入向量，而是按正文把每个
切片匹配回原 PDF 的页码，再原地更新 metadata。

用法:
    python scripts/backfill_chunk_metadata.py --tenant 7e990ab9
"""
import argparse
import bisect
import hashlib
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import fitz  # noqa: E402
import chromadb  # noqa: E402

import app.core.config as cfg  # noqa: E402
from app.processing.multimodal_pipeline import is_toc_noise  # noqa: E402


def _norm(s: str) -> str:
    """去空白与全角空格，保证与 PDF 文本层可比较。"""
    return re.sub(r"[\s\u3000]+", "", s)


def _probe(chunk: str) -> str:
    """取切片中最长的一段连续正文作为定位探针。

    多模态管线会给切片插入 [图片描述]/[图片文字] 等标记行，
    这些内容不在 PDF 文本层里，需跳过。
    """
    for line in chunk.split("\n"):
        line = line.strip()
        if not line or line.startswith("["):
            continue
        s = _norm(line)
        if len(s) >= 12:
            return s[:40]
    return ""


def _page_index(pdf_path: str):
    """构建该 PDF 的页码索引：(全文归一化串, 每页起始偏移, 页号列表)。"""
    with fitz.open(pdf_path) as doc:
        pages = [_norm(p.get_text()) for p in doc]
    offsets, cursor, parts = [], 0, []
    for i, t in enumerate(pages, 1):
        offsets.append(cursor)
        parts.append(t)
        cursor += len(t)
    return "".join(parts), offsets, list(range(1, len(pages) + 1))


def _locate(probe: str, full: str, offsets: list, page_nos: list):
    """在全文串中定位探针，返回所在页码；找不到返回 None。"""
    if not probe:
        return None
    pos = full.find(probe)
    if pos < 0:
        return None
    idx = bisect.bisect_right(offsets, pos) - 1
    return page_nos[idx] if 0 <= idx < len(page_nos) else None


def _find_source_file(tenant_id: str, source: str) -> str:
    """按优先级找源 PDF：上传目录 → wendang → wendang/pdf。"""
    candidates = [
        os.path.join(cfg.UPLOAD_DIR, tenant_id, source),
        os.path.join("wendang", source),
        os.path.join("wendang", "pdf", source),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", default="7e990ab9")
    args = ap.parse_args()

    persist_dir = os.path.join(cfg.CHROMA_PERSIST_DIR, args.tenant)
    col = chromadb.PersistentClient(path=persist_dir).get_or_create_collection("langchain")
    data = col.get(include=["documents", "metadatas"])
    ids, docs, metas = data["ids"], data["documents"], data["metadatas"]
    print(f"切片总数: {len(ids)}")

    by_source = {}
    for i, m in enumerate(metas):
        by_source.setdefault((m or {}).get("source", ""), []).append(i)

    total_mapped = 0
    toc_ids = []
    for source, idxs in sorted(by_source.items()):
        path = _find_source_file(args.tenant, source)
        if not path:
            print(f"- {source}: 找不到源文件，跳过 {len(idxs)} 片")
            continue

        doc_id = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
        full, offsets, page_nos = _page_index(path)
        if not full:
            print(f"- {source}: 无文本层(扫描件)，仅回填 doc_id {len(idxs)} 片")
            new_metas = [dict(metas[i] or {}, doc_id=doc_id) for i in idxs]
            col.update(ids=[ids[i] for i in idxs], metadatas=new_metas)
            continue

        new_metas, mapped, keep = [], 0, []
        for i in idxs:
            # 历史入库的目录页切片只有噪声，直接剔除
            if is_toc_noise(docs[i] or ""):
                toc_ids.append(ids[i])
                continue
            keep.append(i)
            page = _locate(_probe(docs[i] or ""), full, offsets, page_nos)
            if page:
                mapped += 1
            new_metas.append(dict(metas[i] or {}, doc_id=doc_id, page_number=page or 0))
        col.update(ids=[ids[i] for i in keep], metadatas=new_metas)
        total_mapped += mapped
        print(f"- {source}: {mapped}/{len(idxs)} 片定位到页码 (doc_id={doc_id})")

    if toc_ids:
        col.delete(ids=toc_ids)
        print(f"\n已剔除目录页切片: {len(toc_ids)} 片")

    print(f"完成: {total_mapped}/{len(ids)} 片回填了页码")


if __name__ == "__main__":
    main()
