"""
ingest_missing_pdf.py —— 把 uploads/{tenant}/ 下未入库的 PDF 走生产链路补入库

跳过已入库的 source,只处理缺失的。用法:
    python scripts/ingest_missing_pdf.py --tenant 7e990ab9
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.core import config as cfg
from app.processing.multimodal_pipeline import MultimodalPipeline
from app.retrieval.embedder_factory import create_embedder
from app.retrieval.bge_m3_embedder import BGEM3Embedder
from app.retrieval.sparse_store import save_sparse


def get_indexed_sources(tenant_id: str) -> set:
    """从 chroma 拉所有已入库的 source 名."""
    import chromadb
    persist_dir = os.path.join(cfg.CHROMA_PERSIST_DIR, tenant_id)
    if not os.path.isdir(persist_dir):
        return set()
    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_or_create_collection("langchain")
    got = col.get(include=["metadatas"])
    return {m.get("source") for m in (got.get("metadatas") or []) if m and m.get("source")}


def ingest_pdf(tenant_id: str, pdf_path: str, filename: str, embedder, store,
               status: str = "approved"):
    """复用 _run_indexing 的核心逻辑:抽取 -> 稀疏 -> 稠密入库."""
    print(f"\n[ingest] {filename}")
    pipeline = MultimodalPipeline()
    chunks = pipeline.process(pdf_path)
    if not chunks:
        print("  !! 无文本抽取结果,跳过")
        return 0

    # 切片正文去重(跨页重叠会切出完全相同的块)
    seen, kept = set(), []
    for c in chunks:
        key = c.text.strip()
        if key and key not in seen:
            seen.add(key)
            kept.append(c)
    chunks = kept
    texts = [c.text for c in chunks]
    print(f"  抽取到 {len(texts)} chunks(去重后)")

    # 稀疏向量
    if isinstance(embedder, BGEM3Embedder):
        sp_list = embedder.encode_sparse(texts)
        save_sparse(tenant_id, filename, [{"key": t[:200], "sp": sp} for t, sp in zip(texts, sp_list)])
        print(f"  稀疏向量已保存")

    # 稠密入库:metadata 带页码与文档指纹,供引用回溯与 Wiki 页绑定来源
    h = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()[:12]
    metadatas = [
        {"source": filename, "chunk": i, "page_number": c.page_number,
         "doc_id": h, "doc_status": status}
        for i, c in enumerate(chunks)
    ]
    store.add_texts(texts=texts, metadatas=metadatas)
    print(f"  稠密向量已入库")
    return len(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", default="7e990ab9")
    ap.add_argument("--status", default="approved", choices=["approved", "draft"],
                    help="写入切片的 doc_status；draft 表示未审定，不参与问答检索")
    args = ap.parse_args()

    upload_dir = os.path.join(cfg.UPLOAD_DIR, args.tenant)
    indexed = get_indexed_sources(args.tenant)
    print(f"已入库 sources: {indexed}")

    pdfs = [f for f in os.listdir(upload_dir) if f.lower().endswith(".pdf")]
    missing = [f for f in pdfs if f not in indexed]
    print(f"上传目录 PDFs: {pdfs}")
    print(f"待入库: {missing}")

    if not missing:
        print("无缺失,退出")
        return

    embedder = create_embedder()

    from langchain_chroma import Chroma
    persist_dir = os.path.join(cfg.CHROMA_PERSIST_DIR, args.tenant)
    store = Chroma(persist_directory=persist_dir, embedding_function=embedder)

    total = 0
    for fname in missing:
        n = ingest_pdf(args.tenant, os.path.join(upload_dir, fname), fname, embedder,
                       store, status=args.status)
        total += n
    print(f"\n完成,共新增 {total} chunks")


if __name__ == "__main__":
    main()
