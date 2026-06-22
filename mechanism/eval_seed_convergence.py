"""P1-1 种子收敛(评测+画图)：对每个 query 的每轮种子排名算 NDCG@10,
按轮次取均值,画"种子 NDCG@10 随轮次"曲线。预期单调上升并收敛。

- 第 0 轮 = 初始 BM25 输入序; 第 r 轮 = 第 r 轮 posterior 再播后的种子序。
- 不同 query 轮数不同: 轮数不足者用其最后一轮的种子序保持(收敛后不变)填充。
- 输出: results/analysis/seed_convergence_curve.csv + tex/figures/seed_convergence.png
- 用法: cd LLM_Rank/mechanism && <py> eval_seed_convergence.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mechanism"))
from _ndcg import load_qrels, ndcg_single

IN = ROOT / "results" / "analysis" / "seed_convergence_dl19.jsonl"
QRELS = ROOT / "data" / "TREC_data" / "dl19" / "qrels.dl19-passage.txt"
CSV_OUT = ROOT / "results" / "analysis" / "seed_convergence_curve.csv"
PNG_OUT = ROOT.parent / "tex" / "figures" / "seed_convergence.png"
K = 10


def main() -> int:
    qrels = load_qrels(str(QRELS))
    rows = [json.loads(l) for l in IN.open() if l.strip()]
    if not rows:
        print("无数据,先跑 run_seed_convergence.py"); return 1
    # 每个 query 的"逐轮种子序列": [BM25, 第1轮, 第2轮, ...]
    per_q = []
    max_r = 0
    for r in rows:
        seq = [r["initial_bm25"]] + r["round_seeds"]
        per_q.append((r["qid"], seq))
        max_r = max(max_r, len(seq))
    # 逐轮均值 NDCG@10(轮数不足者用最后一轮种子保持填充)
    curve = []
    for ri in range(max_r):
        vals = []
        for qid, seq in per_q:
            seed = seq[ri] if ri < len(seq) else seq[-1]
            v = ndcg_single(qrels, qid, seed, K)
            if v is not None:
                vals.append(v)
        if vals:
            curve.append((ri, sum(vals) / len(vals), len(vals)))
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["round", f"seed_ndcg@{K}", "n_queries"])
        for ri, nd, n in curve:
            w.writerow([ri, f"{nd:.2f}", n])
    print(f"逐轮种子 NDCG@{K}:")
    for ri, nd, n in curve:
        label = "BM25输入" if ri == 0 else f"第{ri}轮"
        print(f"  {label:8s}: {nd:.2f}  (n={n})")
    print(f"CSV -> {CSV_OUT}")
    # 画图(matplotlib 可选)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [ri for ri, _, _ in curve]; ys = [nd for _, nd, _ in curve]
        plt.figure(figsize=(5, 3.2))
        plt.plot(xs, ys, marker="o", color="#1f6feb", linewidth=2)
        plt.xlabel("Round index (0 = initial BM25 input)")
        plt.ylabel(f"Seed ranking NDCG@{K}")
        plt.title("Seed-ranking convergence (TREC DL19, Qwen3-8B)")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(PNG_OUT, dpi=200)
        print(f"PNG -> {PNG_OUT}")
    except ImportError:
        print("(matplotlib 未装,跳过画图,只出 CSV)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
