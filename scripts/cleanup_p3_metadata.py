"""cleanup_p3_metadata.py —— P3 数据卫生清理(一次性维护脚本)

处理三件事:
1. 删除混入语料的测试文档(默认 流程验证_测试企业知识库.pdf)及其稀疏向量;
2. 为缺 page_number 的切片回填页码:
   - 有文本层的 PDF:按正文在页文本中定位;
   - 扫描件(无文本层):逐页 OCR 后再定位;
   - 仍定位不到的(图片描述类切片):按切片顺序用相邻已知页推断;
3. 为缺 doc_id 的切片补文档指纹(按源文件内容 SHA-256 前 12 位)。

数据变更完成后清理问答缓存，避免旧答案继续引用已删除内容。

用法:
    python scripts/cleanup_p3_metadata.py --tenant 7e990ab9            # 预演
    python scripts/cleanup_p3_metadata.py --tenant 7e990ab9 --execute  # 实际执行
"""
import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import chromadb  # noqa: E402
import fitz  # noqa: E402

import app.core.config as cfg  # noqa: E402
from app.processing.pdf_extractor import extract_pdf_pages  # noqa: E402

TEST_SOURCES = {"流程验证_测试企业知识库.pdf"}


def _norm(text: str) -> str:
    """去空白与全角空格，便于与 PDF 文本层/OCR 结果比较。"""
    return re.sub(r"[\s\u3000]+", "", text or "")


def _probe(chunk: str) -> str:
    """取切片中最长的一段连续正文作为定位探针(跳过图片描述等标记行)。"""
    for line in (chunk or "").split("\n"):
        line = line.strip()
        if not line or line.startswith("["):
            continue
        s = _norm(line)
        if len(s) >= 8:
            return s[:40]
    return ""


def _find_source_file(tenant_id: str, source: str) -> str:
    """按优先级找源 PDF:上传目录 → wendang → wendang/pdf。"""
    for path in (
        os.path.join(cfg.UPLOAD_DIR, tenant_id, source),
        os.path.join("wendang", source),
        os.path.join("wendang", "pdf", source),
    ):
        if os.path.exists(path):
            return path
    return ""


def _text_pages(path: str) -> list:
    """读取 PDF 文本层，返回每页归一化文本(无文本层则为空串)。"""
    with fitz.open(path) as doc:
        return [_norm(page.get_text()) for page in doc]


def _ocr_pages(path: str) -> list:
    """逐页 OCR，返回每页归一化文本(用于扫描件回填页码)。

    注意:PaddleOCR 在本机部分环境下会触发原生崩溃(0xC0000005)，
    因此仅在显式传入 --ocr 时调用，默认跳过扫描件。
    """
    from app.processing.ocr_engine import OCREngine

    pages = extract_pdf_pages(path)
    ocr = OCREngine()
    texts = []
    for page in pages:
        parts = [page.get("text") or ""]
        for img in page.get("images", []):
            try:
                parts.append(ocr.recognize(img["bytes"]) or "")
            except Exception:  # noqa: BLE001
                continue
        texts.append(_norm("\n".join(parts)))
    return texts


def _seq_pages(items: list, pages: list) -> dict:
    """按切片入库顺序逐页对齐，返回 {切片下标: 页码}。

    分块是按页独立进行的，切片顺序与页码严格单调，因此用游标逐页前进匹配，
    避免全文 find 被重复段落带到更早的页(旧回填逻辑出现过 90 处页码回跳)。
    未能匹配的图片描述类切片，按「下一个能匹配的切片所在页」回填；尾部切片
    落在最后一页。

    参数:
        items (list): [(chunk_no, 下标, 切片正文)]，需按 chunk_no 升序。
        pages (list): 每页归一化文本。

    返回:
        dict: {下标: 页码}
    """
    if not pages:
        return {}
    result, unmatched, cursor = {}, [], 0
    for _, index, text in items:
        probe = _probe(text)
        found = 0
        if probe:
            for page_index in range(cursor, len(pages)):
                if probe in pages[page_index]:
                    found = page_index + 1
                    break
        if found:
            result[index] = found
            cursor = found - 1
        else:
            unmatched.append(index)

    order = [index for _, index, _ in items]
    position = {index: pos for pos, index in enumerate(order)}
    for index in unmatched:
        later = [result[j] for j in order[position[index] + 1:] if j in result]
        if later:
            result[index] = later[0]
        elif result:
            result[index] = max(result.values())
        else:
            result[index] = len(pages)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", default="7e990ab9")
    ap.add_argument("--execute", action="store_true", help="实际写入;缺省只预演")
    ap.add_argument("--ocr", action="store_true",
                    help="对无文本层的扫描件逐页 OCR(本机 PaddleOCR 可能原生崩溃)")
    args = ap.parse_args()

    col = chromadb.PersistentClient(
        path=os.path.join(cfg.CHROMA_PERSIST_DIR, args.tenant)
    ).get_or_create_collection("langchain")
    data = col.get(include=["documents", "metadatas"])
    ids, docs, metas = data["ids"], data["documents"], data["metadatas"]
    print(f"切片总数: {len(ids)}")

    # ---- 1. 删除测试文档 ----
    test_ids = [
        ids[i] for i, meta in enumerate(metas)
        if (meta or {}).get("source") in TEST_SOURCES
    ]
    if test_ids:
        print(f"[删除] 测试文档切片 {len(test_ids)} 片")
        if args.execute:
            col.delete(ids=test_ids)
            try:
                from app.retrieval.sparse_store import delete_sparse
                for name in TEST_SOURCES:
                    delete_sparse(args.tenant, name)
            except Exception as e:  # noqa: BLE001
                print(f"  稀疏向量删除失败(忽略): {e}")

    kept = [i for i in range(len(ids)) if ids[i] not in set(test_ids)]

    # ---- 2. 按文档重排 page_number 并补 doc_id ----
    by_source = {}
    for i in kept:
        meta = metas[i] or {}
        by_source.setdefault(meta.get("source", ""), []).append(i)

    updated = 0
    for source, idxs in sorted(by_source.items()):
        path = _find_source_file(args.tenant, source)
        if not path:
            print(f"[跳过] {source}: 找不到源文件({len(idxs)} 片)")
            continue

        doc_id = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
        pages = _text_pages(path)
        if not any(pages):
            if not args.ocr:
                print(f"[跳过] {source}: 扫描件无文本层({len(idxs)} 片，加 --ocr 可尝试识别)")
                continue
            print(f"[OCR] {source}: 无文本层，开始逐页识别({len(pages)} 页)…")
            pages = _ocr_pages(path)
            if not any(pages):
                print(f"[跳过] {source}: OCR 未识别出文本")
                continue

        ordered = sorted(idxs, key=lambda i: int((metas[i] or {}).get("chunk", 0)))
        page_map = _seq_pages(
            [(int((metas[i] or {}).get("chunk", 0)), i, docs[i]) for i in ordered],
            pages,
        )

        target_ids, new_metas = [], []
        for i in ordered:
            meta = dict(metas[i] or {})
            page = page_map.get(i, meta.get("page_number") or 0)
            if meta.get("page_number") == page and meta.get("doc_id"):
                continue
            meta["doc_id"] = doc_id
            meta["page_number"] = page
            target_ids.append(ids[i])
            new_metas.append(meta)

        print(
            f"[回填] {source}: 更新 {len(target_ids)}/{len(ordered)} 片"
            f"(页码覆盖 {sum(1 for p in page_map.values() if p)}/{len(ordered)})"
        )
        if args.execute and target_ids:
            col.update(ids=target_ids, metadatas=new_metas)
            updated += len(target_ids)

    # ---- 3. 清理问答缓存 ----
    if args.execute:
        try:
            from app.wiki.store import get_store
            get_store()._clear_query_cache()
            print("[缓存] 已清理各租户问答缓存")
        except Exception as e:  # noqa: BLE001
            print(f"[缓存] 清理失败(忽略): {e}")

    print(f"完成: 更新 {updated} 片" if args.execute else "预演结束，加 --execute 实际写入")


if __name__ == "__main__":
    main()
