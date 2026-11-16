"""
evaluate_ablation.py —— 检索消融实验 + 回归对比管线

对比三种检索配置在同一份 Chroma 语料 + 同一份测试题下的 RAGAS 指标:
  A. dense_only      仅 BGE-M3 稠密向量
  B. dense_bm25      BGE-M3 稠密 + BM25 稀疏
  C. full_hybrid     BGE-M3 稠密 + BM25 稀疏 + BGE-M3 SPLADE 稀疏

使用方式:
    python scripts/evaluate_ablation.py [--tenant 7e990ab9] [--config all|dense_only|dense_bm25|full_hybrid]
    python scripts/evaluate_ablation.py --baseline evaluation_history/<run_id>.json   # 指定基线
    python scripts/evaluate_ablation.py --compare-only A.json B.json                # 干跑:只对比已有结果
    python scripts/evaluate_ablation.py --gate                                       # 回归 FAIL 时退出码 1(CI 卡口)

产出:
    evaluation_ablation.json                 汇总三组指标(保持旧格式,简历/报告引用它)
    evaluation_ablation_<config>.json        单组指标 + 题目级明细
    evaluation_history/<run_id>.json         每轮完整记录(指标 + 题目级明细),回归对比的数据源
    evaluation_regression_report.json        最近一次回归对比报告
    evaluation_regression_failures.jsonl     回退样本库(回退题目自动追加,供扩充题库/定向修复)
"""
import argparse
import json
import math
import os
import sys
import types
from datetime import datetime
from pathlib import Path

# 把项目根目录塞进 sys.path,便于以脚本方式直接运行
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # 让 ./chroma_db ./sparse_db ./model_cache 相对路径生效

# 与 app.evaluation.evaluator 相同:伪造 vertexai 避免 import 失败
_fake = types.ModuleType("langchain_community.chat_models.vertexai")
class _FakeDummy: pass
_fake.ChatVertexAI = _FakeDummy
sys.modules["langchain_community.chat_models.vertexai"] = _fake

from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
# 用旧版 ragas.metrics 单例(collections 里的新版 BaseMetric 类与 evaluate() 的 isinstance(Metric) 检查不兼容)
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import llm_factory
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas import evaluate

from app.core import config as cfg
from app.retrieval.embedder_factory import create_embedder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.sparse_store import load_sparse_lookup, save_sparse
from app.evaluation.evaluator import generate_answer

# 刑法测试题(共 25 题,ground_truth 均取自已入库的刑法 PDF 真实条文)
# 说明:劳动法 PDF 为扫描件且 OCR 后端不可用,只入库了图片描述占位符,
#       因此改用刑法作为评估语料(545 个真实条文 chunk)。
# 前 8 题为最初版本(保持可比),后 17 题为扩充题,覆盖总则与分则。
TEST_QUESTIONS = [
    # ---- 分则:刑罚判定(原 8 题,保持可比) ----
    {"question": "破坏火车、汽车足以发生倾覆危险但尚未造成严重后果的,应判处什么刑罚?", "ground_truth": "处三年以上十年以下有期徒刑"},
    {"question": "以非法占有为目的使用诈骗方法非法集资数额较大的,应判处什么刑罚?", "ground_truth": "处三年以上七年以下有期徒刑,并处罚金"},
    {"question": "在法律规定的国家考试中组织作弊的,应判处什么刑罚?", "ground_truth": "处三年以下有期徒刑或者拘役,并处或者单处罚金;情节严重的,处三年以上七年以下有期徒刑,并处罚金"},
    {"question": "国家机关工作人员滥用职权致使公共财产遭受重大损失的,应判处什么刑罚?", "ground_truth": "处三年以下有期徒刑或者拘役;情节特别严重的,处三年以上七年以下有期徒刑"},
    {"question": "司法工作人员徇私枉法对明知是无罪的人而使他受追诉的,应判处什么刑罚?", "ground_truth": "处五年以下有期徒刑或者拘役;情节严重的,处五年以上十年以下有期徒刑;情节特别严重的,处十年以上有期徒刑"},
    {"question": "伪造信用卡的,应判处什么刑罚?", "ground_truth": "处五年以下有期徒刑或者拘役,并处或者单处二万元以上二十万元以下罚金"},
    {"question": "煽动军人逃离部队情节严重的,应判处什么刑罚?", "ground_truth": "处三年以下有期徒刑、拘役或者管制"},
    {"question": "聚众冲击军事禁区严重扰乱秩序的首要分子,应判处什么刑罚?", "ground_truth": "处五年以上十年以下有期徒刑"},

    # ---- 分则:刑罚判定(扩充 5 题) ----
    {"question": "破坏轨道、桥梁、隧道、公路、机场,足以使火车发生倾覆危险但尚未造成严重后果的,应判处什么刑罚?", "ground_truth": "处三年以上十年以下有期徒刑"},
    {"question": "破坏交通工具造成严重后果的,应判处什么刑罚?", "ground_truth": "处十年以上有期徒刑、无期徒刑或者死刑;过失犯前款罪的,处三年以上七年以下有期徒刑;情节较轻的,处三年以下有期徒刑或者拘役"},
    {"question": "组织、领导恐怖活动组织的,应判处什么刑罚?", "ground_truth": "处十年以上有期徒刑或者无期徒刑,并处没收财产"},
    {"question": "伪造国库券数额巨大的,应判处什么刑罚?", "ground_truth": "处三年以上十年以下有期徒刑,并处五万元以上五十万元以下罚金"},
    {"question": "以非法占有为目的诈骗银行或者其他金融机构贷款数额较大的,应判处什么刑罚?", "ground_truth": "处五年以下有期徒刑或者拘役,并处二万元以上二十万元以下罚金"},

    # ---- 总则:犯罪构成与刑事责任(扩充 12 题) ----
    {"question": "什么是故意犯罪?", "ground_truth": "明知自己的行为会发生危害社会的结果,并且希望或者放任这种结果发生,因而构成犯罪的,是故意犯罪"},
    {"question": "什么是过失犯罪?", "ground_truth": "应当预见自己的行为可能发生危害社会的结果,因为疏忽大意而没有预见,或者已经预见而轻信能够避免,以致发生这种结果的,是过失犯罪"},
    {"question": "已满十四周岁不满十六周岁的人,犯哪些罪应当负刑事责任?", "ground_truth": "犯故意杀人、故意伤害致人重伤或者死亡、强奸、抢劫、贩卖毒品、放火、爆炸、投放危险物质罪的,应当负刑事责任"},
    {"question": "精神病人在不能辨认或者不能控制自己行为的时候造成危害结果的,是否负刑事责任?", "ground_truth": "不负刑事责任,但是应当责令他的家属或者监护人严加看管和医疗;在必要的时候,由政府强制医疗"},
    {"question": "又聋又哑的人或者盲人犯罪,应当如何处罚?", "ground_truth": "可以从轻、减轻或者免除处罚"},
    {"question": "正当防卫明显超过必要限度造成重大损害的,应如何处理?", "ground_truth": "应当负刑事责任,但是应当减轻或者免除处罚"},
    {"question": "紧急避险超过必要限度造成不应有的损害的,应如何处理?", "ground_truth": "应当负刑事责任,但是应当减轻或者免除处罚"},
    {"question": "什么是犯罪预备?", "ground_truth": "为了犯罪,准备工具、制造条件的,是犯罪预备;对于预备犯,可以比照既遂犯从轻、减轻处罚或者免除处罚"},
    {"question": "什么是犯罪未遂?", "ground_truth": "已经着手实行犯罪,由于犯罪分子意志以外的原因而未得逞的,是犯罪未遂;对于未遂犯,可以比照既遂犯从轻或者减轻处罚"},
    {"question": "犯罪中止没有造成损害的,应当如何处罚?", "ground_truth": "应当免除处罚;造成损害的,应当减轻处罚"},
    {"question": "在共同犯罪中起次要或者辅助作用的人如何处罚?", "ground_truth": "是从犯,对于从犯,应当从轻、减轻处罚或者免除处罚"},
    {"question": "教唆不满十八周岁的人犯罪的,应如何处罚?", "ground_truth": "应当从重处罚"},
]


TENANT_DEFAULT = "7e990ab9"
ABLATION_CONFIGS = {
    "dense_only": dict(
        dense_weight=1.0,
        sparse_weight=0.0,
        bge_sparse_weight=0.0,
        use_sparse_store=False,
    ),
    "dense_bm25": dict(
        dense_weight=1.0,
        sparse_weight=1.0,
        bge_sparse_weight=0.0,
        use_sparse_store=False,
    ),
    "full_hybrid": dict(
        dense_weight=1.0,
        sparse_weight=1.0,
        bge_sparse_weight=1.0,
        use_sparse_store=True,
    ),
}


# ────────────────────────────────────────────────────────────────
# 回归对比:把「评估脚本」升级为「评估管线」
# 每轮评估(指标 + 题目级明细)落盘 evaluation_history/,自动与上一轮
# (或 --baseline 指定)基线 diff,产出指标级涨跌与题目级回退清单;
# 回退题目自动追加进失败样本库,形成「评估 → 对比 → 样本回流」闭环。
# ────────────────────────────────────────────────────────────────
HISTORY_DIR = ROOT / "evaluation_history"
FAILURES_PATH = ROOT / "evaluation_regression_failures.jsonl"
REPORT_PATH = ROOT / "evaluation_regression_report.json"

METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
# 指标级判定: |delta| < 0.005 视为持平;跌幅 >0.02 记 ⚠ 显著回退
METRIC_FLAT_EPS = 0.005
METRIC_WARN_DROP = 0.02
# 题目级判定: 单题单指标跌幅 >0.05 计入回退清单(单题方差天然大于均值,阈值放宽)
QUESTION_DROP = 0.05

FLAG_MARK = {"up": "↑", "down": "↓", "warn": "⚠", "flat": "="}


def _git_commit() -> str | None:
    """记录当前代码版本,回归对不上时好定位是哪次改动引入的。"""
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=ROOT,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def save_run_record(summary_detail: dict) -> Path:
    """把本轮评估(指标 + 题目级明细)落盘到 evaluation_history/<run_id>.json。"""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    record = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "configs": summary_detail,
    }
    HISTORY_DIR.mkdir(exist_ok=True)
    path = HISTORY_DIR / f"{run_id}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_run_record(path) -> dict:
    """读取一轮评估记录;兼容旧的汇总格式({config: metrics},无题目级明细)。"""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "configs" in data:  # 新格式(带 run 元信息)
        return data
    # 旧汇总格式:包一层,per_question 置空(题目级对比自动跳过)
    return {
        "run_id": path.stem,
        "created_at": None,
        "git_commit": None,
        "configs": {
            name: {"metrics": m, "per_question": []} for name, m in data.items()
        },
    }


def latest_history_run(exclude: str | None = None) -> Path | None:
    """取上一轮评估记录(按文件名时间序);exclude 用于跳过刚落盘的本轮。"""
    if not HISTORY_DIR.is_dir():
        return None
    files = sorted(HISTORY_DIR.glob("*.json"))
    if exclude:
        files = [f for f in files if f.stem != exclude]
    return files[-1] if files else None


def compare_runs(baseline: dict, current: dict) -> dict:
    """对比两轮评估,产出指标级 diff、题目级回退清单与回归结论。"""
    metric_diffs: dict = {}
    question_regressions: list = []
    for cfg_name, cur in current["configs"].items():
        base = baseline["configs"].get(cfg_name)
        if not base:
            continue
        diffs = {}
        for k in METRIC_KEYS:
            b, c = base["metrics"].get(k), cur["metrics"].get(k)
            if b is None or c is None:
                continue
            delta = c - b
            if delta >= METRIC_FLAT_EPS:
                flag = "up"
            elif delta <= -METRIC_WARN_DROP:
                flag = "warn"
            elif delta <= -METRIC_FLAT_EPS:
                flag = "down"
            else:
                flag = "flat"
            diffs[k] = {
                "baseline": round(b, 4), "current": round(c, 4),
                "delta": round(delta, 4), "flag": flag,
            }
        metric_diffs[cfg_name] = diffs

        # 题目级:按 question 文本对齐,找单指标跌幅超阈值的题
        base_q = {q["question"]: q for q in base.get("per_question", [])}
        for q in cur.get("per_question", []):
            bq = base_q.get(q["question"])
            if not bq:
                continue
            for k in METRIC_KEYS:
                bv, cv = bq.get(k), q.get(k)
                if bv is None or cv is None:
                    continue
                d = cv - bv
                if d <= -QUESTION_DROP:
                    question_regressions.append({
                        "config": cfg_name, "question": q["question"],
                        "metric": k, "baseline": round(bv, 4),
                        "current": round(cv, 4), "delta": round(d, 4),
                    })

    warn_metrics = [
        (c, k) for c, ds in metric_diffs.items()
        for k, d in ds.items() if d["flag"] == "warn"
    ]
    down_metrics = [
        (c, k) for c, ds in metric_diffs.items()
        for k, d in ds.items() if d["flag"] == "down"
    ]
    if warn_metrics:
        verdict = "FAIL"
        suggestion = "存在显著回退指标(单项跌幅>0.02),不建议合并;先定位回退原因并重跑确认。"
    elif down_metrics or question_regressions:
        verdict = "WARN"
        suggestion = "有指标小幅下滑或题目级回退,建议人工 review 回退题目后再决定是否合并。"
    else:
        verdict = "PASS"
        suggestion = "回归全绿:指标持平或提升,可以合并。"
    return {
        "baseline_run_id": baseline["run_id"],
        "current_run_id": current["run_id"],
        "metric_diffs": metric_diffs,
        "question_regressions": question_regressions,
        "verdict": verdict,
        "suggestion": suggestion,
    }


def print_report(report: dict) -> None:
    """控制台打印回归报告(指标涨跌带 ↑/↓/⚠ 标注)。"""
    print(f"\n{'='*60}")
    print(f"回归对比: {report['baseline_run_id']} -> {report['current_run_id']}")
    print(f"{'='*60}")
    for cfg_name, diffs in report["metric_diffs"].items():
        print(f"[{cfg_name}]")
        for k, d in diffs.items():
            sign = "+" if d["delta"] >= 0 else ""
            print(
                f"  {FLAG_MARK[d['flag']]} {k:<18} "
                f"{d['baseline']:.4f} -> {d['current']:.4f}  ({sign}{d['delta']:.4f})"
            )
    regs = report["question_regressions"]
    if regs:
        print(f"\n题目级回退清单(共 {len(regs)} 条,阈值: 单指标跌幅>{QUESTION_DROP}):")
        for r in regs:
            print(
                f"  [{r['config']}] {r['question'][:36]}  "
                f"{r['metric']} {r['baseline']:.2f}->{r['current']:.2f} ({r['delta']:+.2f})"
            )
    else:
        print("\n题目级回退清单: 无(或基线缺少题目级明细)")
    print(f"\n结论: {report['verdict']} —— {report['suggestion']}")


def append_failure_samples(report: dict) -> int:
    """回退题目自动追加进失败样本库(JSONL),供后续扩充题库/定向修复。"""
    regs = report["question_regressions"]
    if not regs:
        return 0
    now = datetime.now().isoformat(timespec="seconds")
    with open(FAILURES_PATH, "a", encoding="utf-8") as f:
        for r in regs:
            f.write(json.dumps(
                {**r, "detected_at": now,
                 "current_run_id": report["current_run_id"],
                 "baseline_run_id": report["baseline_run_id"]},
                ensure_ascii=False,
            ) + "\n")
    return len(regs)



def load_chroma_store(tenant_id: str):
    """从已入库的 chroma_db/{tenant_id} 加载 dense_store,并抽出全部文本."""
    from langchain_chroma import Chroma
    import chromadb

    persist_dir = os.path.join(cfg.CHROMA_PERSIST_DIR, tenant_id)
    if not os.path.isdir(persist_dir):
        raise RuntimeError(f"chroma 目录不存在: {persist_dir}")

    embedder = create_embedder()
    store = Chroma(persist_directory=persist_dir, embedding_function=embedder)

    # 通过原生 chromadb 客户端拿全部文档文本(用于 BM25 / sparse 索引)
    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_or_create_collection("langchain")
    got = col.get(include=["documents", "metadatas"])
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    print(f"[load_chroma_store] tenant={tenant_id} chunks={len(docs)}")
    return store, docs, metas


def ensure_sparse_vectors(tenant_id: str, texts: list[str], metas: list[dict]):
    """若 sparse_db/{tenant_id} 还没有稀疏向量,则生成并落盘."""
    existing = load_sparse_lookup(tenant_id)
    if existing:
        print(f"[ensure_sparse_vectors] 已存在稀疏 lookup: {len(existing)} keys,跳过生成")
        return existing

    print(f"[ensure_sparse_vectors] 生成 BGE-M3 稀疏向量: {len(texts)} chunks ...")
    embedder = create_embedder()
    from app.retrieval.bge_m3_embedder import BGEM3Embedder
    if not isinstance(embedder, BGEM3Embedder):
        raise RuntimeError("Embedder 不是 BGEM3Embedder,无法生成稀疏向量")

    sp_list = embedder.encode_sparse(texts)
    # 按文件名分组保存,与生产 _run_indexing 行为一致
    groups: dict[str, list] = {}
    for text, meta, sp in zip(texts, metas, sp_list):
        fname = (meta or {}).get("source", "unknown")
        groups.setdefault(fname, []).append({"key": text[:200], "sp": sp})
    for fname, items in groups.items():
        save_sparse(tenant_id, fname, items)
        print(f"  saved sparse: {fname} items={len(items)}")

    return load_sparse_lookup(tenant_id)


def build_retriever(name: str, cfg_flags: dict, dense_store, texts, tenant_id, sparse_lookup):
    """根据消融配置构造 HybridRetriever."""
    sparse_store = sparse_lookup if cfg_flags["use_sparse_store"] else None
    return HybridRetriever(
        dense_store=dense_store,
        texts=texts,
        k=5,
        rrf_k=60,
        dense_weight=cfg_flags["dense_weight"],
        sparse_weight=cfg_flags["sparse_weight"],
        use_reranker=False,
        use_elasticsearch=False,
        tenant_id=tenant_id,
        sparse_store=sparse_store,
        bge_sparse_weight=cfg_flags["bge_sparse_weight"],
    )


def run_one_config(name: str, cfg_flags: dict, dense_store, texts, tenant_id, sparse_lookup):
    """对单个配置跑完整 RAGAS 评估,返回指标字典."""
    print(f"\n{'='*60}\n[config] {name}\n{'='*60}")
    retriever = build_retriever(name, cfg_flags, dense_store, texts, tenant_id, sparse_lookup)

    samples = []
    for i, item in enumerate(TEST_QUESTIONS, 1):
        q = item["question"]
        print(f"  [{i}/{len(TEST_QUESTIONS)}] {q[:30]}...")
        docs = retriever.retrieve(q, top_k=5)
        contexts = [d.page_content for d in docs]
        answer = generate_answer(q, contexts)
        print(f"      ctx_n={len(contexts)} ans[:60]={answer[:60]}")
        samples.append(
            {
                "user_input": q,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": item["ground_truth"],
            }
        )

    print(f"  运行 RAGAS ...")
    dataset = EvaluationDataset.from_list(samples)

    # 配置 RAGAS 使用的 LLM(DeepSeek,OpenAI 兼容协议) 与 embeddings(本地 BGE-M3)
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    llm_client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    )
    ragas_llm = llm_factory(
        os.getenv("LLM_MODEL", "deepseek-chat"),
        provider="openai",
        client=llm_client,
    )
    # 用项目的 BGEM3Embedder(已加载单例) 作为 RAGAS 的 embeddings,避免重复加载模型
    ragas_emb = LangchainEmbeddingsWrapper(create_embedder())

    # 旧版 Metric 单例:需要克隆后再赋值 llm/embeddings,避免污染全局
    import copy
    m_faith = copy.deepcopy(faithfulness); m_faith.llm = ragas_llm
    m_rel = copy.deepcopy(answer_relevancy); m_rel.llm = ragas_llm; m_rel.embeddings = ragas_emb
    m_prec = copy.deepcopy(context_precision); m_prec.llm = ragas_llm
    m_rec = copy.deepcopy(context_recall); m_rec.llm = ragas_llm

    result = evaluate(
        dataset,
        metrics=[m_faith, m_rel, m_prec, m_rec],
    )

    # EvaluationResult -> DataFrame -> 列均值;同时保留题目级明细供回归对比
    df = result.to_pandas()
    metrics = {}
    for k in METRIC_KEYS:
        if k in df.columns:
            try:
                metrics[k] = float(df[k].mean())
            except Exception:
                metrics[k] = str(df[k].iloc[0])
        else:
            metrics[k] = None

    per_question = []
    for _, row in df.iterrows():
        entry = {"question": row.get("user_input")}
        for k in METRIC_KEYS:
            try:
                fv = float(row.get(k))
                entry[k] = None if math.isnan(fv) else fv
            except (TypeError, ValueError):
                entry[k] = None
        per_question.append(entry)

    out_path = ROOT / f"evaluation_ablation_{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"config": name, "metrics": metrics, "per_question": per_question},
            f, ensure_ascii=False, indent=2,
        )
    print(f"  -> {out_path}")
    print(f"  metrics: {metrics}")
    return {"metrics": metrics, "per_question": per_question}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", default=TENANT_DEFAULT)
    ap.add_argument(
        "--config",
        default="all",
        choices=["all"] + list(ABLATION_CONFIGS.keys()),
    )
    ap.add_argument(
        "--baseline", default=None,
        help="基线评估记录路径;缺省自动取 evaluation_history/ 里上一轮",
    )
    ap.add_argument(
        "--compare-only", nargs=2, metavar=("BASELINE.json", "CURRENT.json"),
        help="干跑模式:只对比两份已有评估记录,不发起新的 API 评估",
    )
    ap.add_argument("--no-history", action="store_true", help="本轮结果不写入 evaluation_history/")
    ap.add_argument("--gate", action="store_true", help="回归结论为 FAIL 时以退出码 1 结束(CI 卡口)")
    args = ap.parse_args()

    # 干跑模式:离线对比两份记录,不加载语料/模型、不调 API
    if args.compare_only:
        baseline = load_run_record(args.compare_only[0])
        current = load_run_record(args.compare_only[1])
        report = compare_runs(baseline, current)
        print_report(report)
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[report] 回归报告: {REPORT_PATH}(compare-only 干跑,不追加失败样本库)")
        if args.gate and report["verdict"] == "FAIL":
            sys.exit(1)
        return

    dense_store, texts, metas = load_chroma_store(args.tenant)
    sparse_lookup = ensure_sparse_vectors(args.tenant, texts, metas)

    configs_to_run = (
        list(ABLATION_CONFIGS.items())
        if args.config == "all"
        else [(args.config, ABLATION_CONFIGS[args.config])]
    )

    summary_detail = {}
    for name, flags in configs_to_run:
        summary_detail[name] = run_one_config(
            name, flags, dense_store, texts, args.tenant, sparse_lookup
        )

    # 汇总文件保持旧格式({config: metrics}),不破坏既有引用
    summary = {name: d["metrics"] for name, d in summary_detail.items()}
    out_path = ROOT / "evaluation_ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n{'='*60}\n汇总报告: {out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.no_history:
        return

    # 落盘本轮记录,自动与上一轮(或指定基线)做回归对比
    record_path = save_run_record(summary_detail)
    print(f"\n[history] 本轮记录: {record_path}")
    baseline_path = Path(args.baseline) if args.baseline else latest_history_run(exclude=record_path.stem)
    if baseline_path is None:
        print("[history] 暂无历史基线,首轮已落盘;下一轮起自动回归对比。")
        return

    baseline = load_run_record(baseline_path)
    current = load_run_record(record_path)
    report = compare_runs(baseline, current)
    print_report(report)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    n = append_failure_samples(report)
    if n:
        print(f"[failures] {n} 条回退样本已追加: {FAILURES_PATH}")
    print(f"[report] 回归报告: {REPORT_PATH}")
    if args.gate and report["verdict"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
