"""eval_wiki_vs_rag.py —— “Wiki 打头 / RAG 兜底”对照评测

默认只比较检索链路(确定性，不调用大模型):
同一个问题分别以
  A. Wiki 优先  —— _get_context 命中知识页即返回，不跑向量检索
  B. 纯 RAG     —— 关闭 Wiki，走稠密 + BM25 + BGE-M3 稀疏 + RRF
执行，统计上下文来源、片段数、上下文长度、金标证据是否命中、耗时。

加 --answer 会额外让大模型分别作答，便于人工对照或交给 RAGAS 二次评估。

用法:
    python scripts/eval_wiki_vs_rag.py --tenant 7e990ab9
    python scripts/eval_wiki_vs_rag.py --tenant 7e990ab9 --answer --out eval_wiki.json
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from langchain_community.vectorstores import Chroma  # noqa: E402

import app.core.config as cfg  # noqa: E402
from app.api.chat import _get_context  # noqa: E402
from app.retrieval.citation import CitationTracker  # noqa: E402
from app.retrieval.embedder_factory import create_embedder  # noqa: E402
from app.wiki.router import WikiRouter  # noqa: E402
from app.wiki.store import get_store  # noqa: E402

# 金标:expected 表示这条问题应当由哪一级回答，以及证据应出自哪份文档的哪一页
# 问法按客服/工作人员的实际咨询口径撰写，而不是司机自查口径
QUESTIONS = [
    {"id": "q1", "query": "司机证件过期了还能接单吗",
     "expected": "wiki", "doc": "网约车运营规范", "page": 1},
    {"id": "q2", "query": "乘客投诉司机绕路怎么办",
     "expected": "wiki", "doc": "网约车运营规范", "page": 1},
    {"id": "q3", "query": "乘客信息能不能提供给警方",
     "expected": "wiki", "doc": "民法典", "page": 15},
    {"id": "q4", "query": "例会迟到怎么处罚",
     "expected": "rag", "doc": "网约车管理制度", "page": 3},
    {"id": "q5", "query": "民法典第一百一十一条规定了什么",
     "expected": "rag", "doc": "民法典", "page": 15},
    {"id": "q6", "query": "乘客投诉司机绕路能赔多少",
     "expected": "wiki", "doc": "客服处置手册", "page": 8},
]


def build_pipeline(tenant_id: str, embedder, router):
    """构造与线上一致的最小管道(只保留评测用到的字段)。"""
    persist_dir = os.path.join(cfg.CHROMA_PERSIST_DIR, tenant_id)
    vector_store = Chroma(embedding_function=embedder, persist_directory=persist_dir)
    return SimpleNamespace(
        ct=CitationTracker(),
        mem=None,
        wiki=router,
        vector_store=vector_store,
        embedder=embedder,
    )


def wiki_evidence(router, query: str) -> list:
    """取知识页的出处(文档名, 页码)，用于判断金标证据是否被覆盖。"""
    hit = router.route(query)
    if not hit:
        return []
    sections = hit.sections or hit.page.sections
    return [(sp.doc_name, sp.page) for s in sections for sp in s.spans]


def run_once(query: str, tenant_id: str, pipeline, router, use_wiki: bool):
    """跑一次 _get_context，返回 (指标字典, 完整 Prompt)。"""
    pipeline.ct = CitationTracker()
    pipeline.wiki = router if use_wiki else None

    t0 = time.perf_counter()
    prompt = _get_context(query, pipeline, query, [], tenant_id)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    sources = [(s.filename, s.page_number) for s in pipeline.ct.get_sources()]
    hit_wiki = any(name.endswith("（内部知识页）") for name, _ in sources)
    evidence = wiki_evidence(router, query) if hit_wiki else sources

    result = {
        "mode": "wiki" if hit_wiki else "rag",
        "chunks": len(sources),
        "context_chars": len(pipeline.ct.format_context()),
        "evidence": [f"{d} 第{p}页" if p else (d or "未知来源") for d, p in evidence],
        "evidence_raw": evidence,
        "elapsed_ms": round(elapsed_ms, 1),
    }
    return result, prompt


def evidence_ok(result: dict, doc: str, page) -> bool:
    """金标证据是否命中:文档名包含即可，给了页码则要求页码一致。"""
    for name, pno in result["evidence_raw"]:
        if not name or doc not in name:
            continue
        if page is None or pno == page:
            return True
    return False


def ask_llm(prompt: str) -> str:
    """可选:让大模型基于该上下文作答(失败返回错误信息，不中断评测)。"""
    from app.llm.client import complete_chat

    try:
        return asyncio.run(complete_chat(
            [{"role": "user", "content": prompt}], temperature=0.1, max_tokens=512,
        ))
    except Exception as e:  # noqa: BLE001
        return f"[LLM 调用失败] {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", default="7e990ab9")
    ap.add_argument("--answer", action="store_true", help="额外调用大模型生成答案")
    ap.add_argument("--out", default="", help="把明细写入 JSON 文件")
    args = ap.parse_args()

    embedder = create_embedder()
    router = WikiRouter(get_store(), args.tenant)
    pipeline = build_pipeline(args.tenant, embedder, router)

    rows, detail = [], []
    for item in QUESTIONS:
        wiki_res, wiki_prompt = run_once(item["query"], args.tenant, pipeline, router, use_wiki=True)
        rag_res, rag_prompt = run_once(item["query"], args.tenant, pipeline, router, use_wiki=False)
        if args.answer:
            wiki_res["answer"] = ask_llm(wiki_prompt)
            rag_res["answer"] = ask_llm(rag_prompt)

        wiki_ok = evidence_ok(wiki_res, item["doc"], item["page"])
        rag_ok = evidence_ok(rag_res, item["doc"], item["page"])
        rows.append({
            "id": item["id"], "query": item["query"], "expected": item["expected"],
            "wiki_mode": wiki_res["mode"], "wiki_ok": wiki_ok,
            "rag_ok": rag_ok, "wiki_ms": wiki_res["elapsed_ms"],
            "rag_ms": rag_res["elapsed_ms"], "wiki_chars": wiki_res["context_chars"],
            "rag_chars": rag_res["context_chars"],
            "wiki_sources": wiki_res["evidence"], "rag_sources": rag_res["evidence"],
        })
        detail.append({"question": item, "wiki": wiki_res, "rag": rag_res})

    # 控制台表格
    print(f"\n{'问题':<28}{'期望':<6}{'Wiki模式':<10}{'Wiki证据':<10}{'RAG证据':<9}"
          f"{'Wiki ms':>9}{'RAG ms':>9}")
    print("-" * 92)
    for r in rows:
        print(f"{r['query'][:26]:<28}{r['expected']:<6}{r['wiki_mode']:<10}"
              f"{'OK' if r['wiki_ok'] else 'MISS':<10}{'OK' if r['rag_ok'] else 'MISS':<9}"
              f"{r['wiki_ms']:>9}{r['rag_ms']:>9}")

    total = len(rows)
    wiki_hit = sum(1 for r in rows if r["wiki_mode"] == "wiki")
    wiki_ok = sum(1 for r in rows if r["wiki_ok"])
    rag_ok = sum(1 for r in rows if r["rag_ok"])
    avg_wiki_ms = sum(r["wiki_ms"] for r in rows) / total
    avg_rag_ms = sum(r["rag_ms"] for r in rows) / total
    avg_wiki_chars = sum(r["wiki_chars"] for r in rows) / total
    avg_rag_chars = sum(r["rag_chars"] for r in rows) / total

    print("-" * 92)
    print(f"Wiki 命中率:        {wiki_hit}/{total}")
    print(f"金标证据覆盖率:     Wiki 优先管线 {wiki_ok}/{total}   纯 RAG {rag_ok}/{total}")
    print(f"平均上下文长度:     Wiki {avg_wiki_chars:.0f} 字   RAG {avg_rag_chars:.0f} 字")
    print(f"平均上下文构建耗时: Wiki {avg_wiki_ms:.0f} ms   RAG {avg_rag_ms:.0f} ms")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"rows": rows, "detail": detail}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n明细已写入 {args.out}")


if __name__ == "__main__":
    main()
