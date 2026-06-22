"""P1-2 噪声鲁棒(离线仿真)：用 qrels 真值 + 可控噪声替代 LLM 判断,
忠实复现 SwissRank / BracketRank / TourRank 的算法结构,扫描噪声看各方法
顶部 NDCG@10 掉多少。预期:SwissRank(累分+相似对战,输一场不出局)掉得最少;
BracketRank(单淘汰,早错永错)掉得最多;TourRank(固定锦标赛,累分但不再播)居中。

不调任何 LLM/API,纯离线,确定性(固定随机种子),秒级跑完。
- judge: 给一组 doc 按 (真实相关度 + 高斯噪声) 排序 —— 这是唯一被注入噪声的原语,
  三个方法共用同一个 judge,保证对比公平(差异只来自算法结构)。
- 用法: cd LLM_Rank/mechanism && <py> noise_robustness.py
"""
from __future__ import annotations
import csv, json, random, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rank_fuc"))
sys.path.insert(0, str(ROOT / "mechanism"))
from swissrank import group_documents, reorder_large_list
from _ndcg import load_qrels, ndcg_single

DATA = ROOT / "data" / "TREC_data" / "dl19" / "trec19_bm25_top100.jsonl"
QRELS = ROOT / "data" / "TREC_data" / "dl19" / "qrels.dl19-passage.txt"
CSV_OUT = ROOT / "results" / "analysis" / "noise_robustness_curve.csv"
PNG_OUT = ROOT.parent / "tex" / "figures" / "noise_robustness.png"

GROUP_SIZE = 20
K = 10
NOISE_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
N_SEEDS = 8           # 每个噪声档重复多次取均值,平滑随机性


# ---------- 被注入噪声的共用判官 ----------
def make_judge(rel: dict, sigma: float, rng: random.Random):
    """返回 judge(docs)->按(真值+N(0,sigma))降序排好的 docs。三方法共用,确保公平。"""
    def judge(docs):
        scored = [(d, rel.get(d, 0) + rng.gauss(0.0, sigma)) for d in docs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored]
    return judge


# ---------- 1) SwissRank(镜像 swissrank_choice.filter_processing) ----------
def sim_swissrank(doc_ids, judge, group_size=GROUP_SIZE, max_failure_num=-1):
    seed = list(doc_ids)
    scores = defaultdict(list); scores[0] = list(doc_ids)
    while True:
        flag = False; bucket_out = defaultdict(list); batch = []
        for score, items in scores.items():
            if len(items) < group_size or score < max_failure_num:
                bucket_out[score] = items; continue
            flag = True
            for g in group_documents(items, group_size):
                batch.append((score, g))
        if not flag:
            break
        for score, g in batch:
            n = len(g); ranked = judge(g)               # 判官给整窗排序
            bucket_out[score + 1].extend(ranked[: n // 2])  # 上半 +1
            bucket_out[score - 1].extend(ranked[n // 2:])   # 下半 -1
            seed[:] = reorder_large_list(seed, ranked)       # 后验再播
        scores = {k: sorted(v, key=lambda x: seed.index(x)) for k, v in bucket_out.items()}
    out = []
    for s in sorted(scores.keys(), reverse=True):
        out.extend(scores[s])
    return out


# ---------- 2) BracketRank(镜像 bracketrank.py 单淘汰) ----------
def _chunk(doc_ids, gs):
    return [doc_ids[i:i + gs] for i in range(0, len(doc_ids), gs)]


def _single_elim(groups, judge, gs):
    groups = [g for g in groups if g]
    advance = max(1, gs // 2)
    elim_layers = []
    while len(groups) > 1:
        nxt = []; matches = []
        i = 0
        while i < len(groups):
            if i + 1 >= len(groups):
                nxt.append(groups[i]); break
            matches.append(groups[i] + groups[i + 1]); i += 2
        round_elim = []
        for m in matches:
            ranked = judge(m)
            nxt.append(ranked[:advance])
            if ranked[advance:]:
                round_elim.append(ranked[advance:])
        if round_elim:
            elim_layers.append(round_elim)
        groups = nxt
    final = groups[0] if groups else []
    for layer in reversed(elim_layers):
        for eg in layer:
            final.extend(eg)
    return final


def sim_bracketrank(doc_ids, judge, group_size=GROUP_SIZE):
    if len(doc_ids) <= 1:
        return list(doc_ids)
    ranked_groups = [judge(g) for g in _chunk(doc_ids, group_size)]
    winners, losers = [], []
    for g in ranked_groups:
        mid = (len(g) + 1) // 2
        if g[:mid]: winners.append(g[:mid])
        if g[mid:]: losers.append(g[mid:])
    ranked = _single_elim(winners, judge, group_size) + _single_elim(losers, judge, group_size)
    seen = set(ranked)
    return ranked + [d for d in doc_ids if d not in seen]


# ---------- 3) TourRank(镜像 tourrank.py 多阶段累分) ----------
def _groups_skip(docs, n_groups, m_per):
    out = []
    for i in range(n_groups):
        cur = []
        for j in range(m_per):
            idx = j * n_groups + i
            if idx >= len(docs): break
            cur.append(docs[idx])
        if cur: out.append(cur)
    return out


def sim_tourrank(doc_ids, judge, n_tournaments=1):  # =1 让 judge 调用预算与 Swiss/Bracket 可比
    bm25_idx = {d: i for i, d in enumerate(doc_ids)}
    score = {d: 0 for d in doc_ids}
    def resort():
        return sorted(score.keys(), key=lambda d: (-score[d], bm25_idx[d]))
    for _ in range(n_tournaments):
        stages = [  # (取前多少进本阶段, 分几组, 每组多少, 选top几)
            (len(doc_ids), 5, 20, 10), (50, 5, 10, 4),
            (20, 1, 20, 10), (10, 1, 10, 5), (5, 1, 5, 2),
        ]
        ranked = resort()
        for take, ng, mp, topm in stages:
            for g in _groups_skip(ranked[:take], ng, mp):
                for d in judge(g)[:min(topm, len(g))]:
                    score[d] += 1
            ranked = resort()
    return resort()


METHODS = {"SwissRank": sim_swissrank, "BracketRank": sim_bracketrank, "TourRank": sim_tourrank}


def main() -> int:
    qrels = load_qrels(str(QRELS))
    rows = [json.loads(l) for l in DATA.open() if l.strip()]
    rows = [r for r in rows if r["qid"] in qrels]
    print(f"DL19 queries with qrels: {len(rows)}; noise levels: {NOISE_LEVELS}; seeds/level: {N_SEEDS}")
    # results[method][sigma] = mean NDCG@10
    results = {m: {} for m in METHODS}
    for sigma in NOISE_LEVELS:
        for mname, fn in METHODS.items():
            vals = []
            for si in range(N_SEEDS):
                for r in rows:
                    qid, docs = r["qid"], r["bm25_docs"]
                    rel = qrels[qid]
                    rng = random.Random(hash((qid, si, round(sigma, 3))) & 0xFFFFFFFF)
                    judge = make_judge(rel, sigma, rng)
                    ranked = fn(docs, judge)
                    v = ndcg_single(qrels, qid, ranked, K)
                    if v is not None:
                        vals.append(v)
            results[mname][sigma] = sum(vals) / len(vals)
        print(f"  sigma={sigma:>4}: " + "  ".join(f"{m}={results[m][sigma]:.2f}" for m in METHODS))
    # CSV
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["noise_sigma"] + list(METHODS))
        for sigma in NOISE_LEVELS:
            w.writerow([sigma] + [f"{results[m][sigma]:.2f}" for m in METHODS])
    print(f"CSV -> {CSV_OUT}")
    # 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 3.4))
        styles = {"SwissRank": ("#d62728", "o", "-"), "BracketRank": ("#1f77b4", "s", "--"),
                  "TourRank": ("#2ca02c", "^", "-.")}
        for m in METHODS:
            c, mk, ls = styles[m]
            plt.plot(NOISE_LEVELS, [results[m][s] for s in NOISE_LEVELS],
                     marker=mk, linestyle=ls, color=c, label=m, linewidth=2)
        plt.xlabel("Injected judgment noise $\\sigma$ (in relevance-grade units)")
        plt.ylabel(f"Top NDCG@{K}")
        plt.title("Robustness to comparison noise (TREC DL19)")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(PNG_OUT, dpi=200)
        print(f"PNG -> {PNG_OUT}")
    except ImportError:
        print("(matplotlib 未装,跳过画图,只出 CSV)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
