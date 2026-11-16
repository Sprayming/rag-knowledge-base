"""set_doc_status.py —— 命令行设置切片级文档审定状态

用法:
    # 1. 历史切片补默认值(只补缺失的，不覆盖已有状态)
    python scripts/set_doc_status.py --tenant 7e990ab9 --init-approved

    # 2. 把某份文档标记为草案
    python scripts/set_doc_status.py --tenant 7e990ab9 --source 客服手册.pdf --status draft

    # 3. 审定通过后恢复
    python scripts/set_doc_status.py --tenant 7e990ab9 --source 客服手册.pdf --status approved
"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import chromadb  # noqa: E402

import app.core.config as cfg  # noqa: E402
from app.wiki.doc_status import fill_missing_status, set_source_status  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", default="7e990ab9")
    ap.add_argument("--source", default="", help="文档名；留空表示处理全部切片")
    ap.add_argument("--status", default="approved", choices=["approved", "draft"])
    ap.add_argument("--init-approved", action="store_true",
                    help="只为缺少 doc_status 的切片补 approved，不覆盖已有状态")
    args = ap.parse_args()

    if args.init_approved:
        print(f"补齐默认状态 {fill_missing_status(args.tenant)} 片")
        return

    if not args.source:
        ap.error("非 --init-approved 模式必须指定 --source")

    changed = set_source_status(args.tenant, args.source, args.status)
    print(f"[{args.source}] 更新 {changed} 片 → {args.status}")

    persist_dir = os.path.join(cfg.CHROMA_PERSIST_DIR, args.tenant)
    metas = chromadb.PersistentClient(path=persist_dir).get_or_create_collection(
        "langchain").get(where={"source": args.source}, include=["metadatas"])["metadatas"]
    dist = Counter((m or {}).get("doc_status", "<缺失>") for m in metas)
    print(f"当前状态分布: {dict(dist)}")


if __name__ == "__main__":
    main()
