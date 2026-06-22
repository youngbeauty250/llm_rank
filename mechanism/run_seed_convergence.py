"""P1-1 种子收敛(数据采集)：跑真实 SwissRank,记录每一轮结束后的种子排名。

证明 posterior re-seeding 让"喂给 LLM 的输入顺序"逐轮变好。输出每个 query 的
初始 BM25 序 + 各轮种子序,供 eval_seed_convergence.py 离线算 NDCG@轮次 曲线。

- 只跑 TREC DL19(43 条),用默认 china key(config.yaml)。量小(~20min)。
- 可续跑:已写过的 qid 跳过。
- 用法: cd LLM_Rank/mechanism && <py> run_seed_convergence.py
"""
from __future__ import annotations
import copy, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rank_fuc"))
import swissrank_choice as sc
from swissrank import group_documents
from utils import get_default_client, set_seed, now_seconds

DATA = ROOT / "data" / "TREC_data" / "dl19" / "trec19_bm25_top100.jsonl"
OUT_DIR = ROOT / "results" / "analysis"
OUT = OUT_DIR / "seed_convergence_dl19.jsonl"


def done_qids(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(l)["qid"] for l in path.open() if l.strip()}


def main() -> int:
    set_seed(42)
    client = get_default_client()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = done_qids(OUT)
    smoke = os.environ.get("SMOKE_TEST")
    smoke_n = int(smoke) if smoke and smoke.isdigit() else (1 if smoke else 0)
    processed = 0
    with OUT.open("a", encoding="utf-8") as fout, DATA.open() as fin:
        for line in fin:
            if smoke_n and processed >= smoke_n:
                break
            row = json.loads(line)
            qid, query, doc_ids = row["qid"], row["query"], row["bm25_docs"]
            if qid in done:
                continue
            contents = {doc_ids[j]: row["bm25_contents"][j] for j in range(len(doc_ids))}
            client.reset_usage()
            round_seed_log = []
            params = {
                "max_failure_num": -1, "group_size": 20, "top_rerank_num": 0,
                "grouping_fn": group_documents, "seed_mode": "posterior",
                "round_seed_log": round_seed_log,
            }
            t0 = now_seconds()
            ranked = sc.filter_processing(client, query, copy.deepcopy(doc_ids), contents, params)
            rec = {
                "qid": qid, "query": query,
                "initial_bm25": doc_ids,          # 第 0 轮(输入序)
                "round_seeds": round_seed_log,    # 第 1..R 轮结束后的种子序
                "final_ranked": ranked,
                "num_rounds": len(round_seed_log),
                "elapsed_s": round(now_seconds() - t0, 1),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            processed += 1
            print(f"[seed-trace] qid={qid} rounds={len(round_seed_log)} elapsed={rec['elapsed_s']}s", flush=True)
    print("Finished seed convergence collection.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
