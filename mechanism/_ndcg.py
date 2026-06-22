"""机制分析共用：qrels 读取 + NDCG@k(口径与 evaluate.py 一致,用 pytrec_eval)。"""
from __future__ import annotations
import pytrec_eval


def load_qrels(qrels_file: str) -> dict:
    """读 TREC/BEIR qrels(`qid Q0 docid rel`,空格分隔,无表头)。"""
    qrels = {}
    with open(qrels_file, "r", encoding="utf-8") as f:
        for row in f:
            parts = row.split()
            if len(parts) < 4:
                continue
            qid, docid, rel = parts[0], parts[2], parts[3]
            try:
                rel = int(rel)
            except ValueError:
                continue
            qrels.setdefault(qid, {})[docid] = rel
    return qrels


def order_to_run(qid: str, ordered_docids: list) -> dict:
    """有序 doc 列表 -> pytrec_eval run({qid:{docid:score}}),score 用倒序排名。"""
    n = len(ordered_docids)
    return {qid: {d: float(n - i) for i, d in enumerate(ordered_docids)}}


def ndcg_at_k(qrels: dict, run: dict, k: int) -> float:
    """对 run 里所有 qid 算 NDCG@k 并取均值(×100)。run 的 qid 必须在 qrels 里。"""
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut"})
    res = ev.evaluate(run)
    key = f"ndcg_cut_{k}"
    vals = [res[q][key] for q in res if q in res and key in res[q]]
    return 100.0 * sum(vals) / len(vals) if vals else 0.0


def ndcg_single(qrels: dict, qid: str, ordered_docids: list, k: int) -> float:
    """单个 query 的 NDCG@k(×100)。qid 不在 qrels 时返回 None。"""
    if qid not in qrels:
        return None
    ev = pytrec_eval.RelevanceEvaluator({qid: qrels[qid]}, {"ndcg_cut"})
    res = ev.evaluate(order_to_run(qid, ordered_docids))
    return 100.0 * res[qid][f"ndcg_cut_{k}"]
