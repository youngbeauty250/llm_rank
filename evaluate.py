"""统一评测脚本：NDCG@1/3/5/10/20 + 效率指标（token / 调用 / 延迟 / 费用）。

相比旧版的两点增强：
1. **多模型**：`--model <tag>` 把结果路径切到 `results/.../<ds>/models/<tag>/<method>.jsonl`，
   评测 deepseek-r1 等 baseline，不动默认 qwen3-8b 主结果。
2. **内容审核丢弃**：读同名 `.skipped.jsonl` 旁路清单，报告每个 (数据集×方法) 丢弃了几条 query。
   - 默认只评测「写入结果文件的 query」，并打印丢弃数（表注用）。
   - `--fill-skipped`：把丢弃的 query 用 BM25 原序回填进 LLM 结果再算 NDCG，
     这样所有方法在「同一全量 query 集」上可比（丢弃 query 等价于回退到 BM25）。

用法示例：
    <py> evaluate.py --bench trec --datasets dl19 dl20
    <py> evaluate.py --bench trec --datasets dl19 dl20 --model deepseek-r1
    <py> evaluate.py --bench beir --datasets trec-covid scifact --fill-skipped
"""

import argparse
import json
import os

import numpy as np
import pytrec_eval

# 方法名 → 结果文件名（与 run_beir_experiments.py 的 METHODS 一致）。
METHOD_FILES = {
    "rankgpt": "rankgpt.jsonl",
    "swiss_choice": "swiss_choice.jsonl",
    "tourrank": "tourrank.jsonl",
    "blitzrank": "blitzrank.jsonl",
    "bracketrank": "bracketrank.jsonl",
    "setwise_heapsort": "setwise_heapsort.jsonl",
}

BEIR_DATASETS = [
    "trec-covid", "webis-touche2020", "dbpedia-entity", "scifact",
    "signal1m", "trec-news", "robust04", "nfcorpus",
]
TREC_DATASETS = ["dl19", "dl20"]


def read_qrels(qrels_file):
    """读 TREC/BEIR qrels（空格分隔 `qid Q0/0 docid rel`，无表头）。"""
    qrels = {}
    with open(qrels_file, "r", encoding="utf-8") as reader:
        for row in reader:
            parts = row.split()
            if len(parts) < 4:
                continue
            query_id, corpus_id, score = parts[0], parts[2], parts[3]
            try:
                score = int(score)
            except ValueError:
                continue  # 跳过可能的表头
            qrels.setdefault(query_id, {})[corpus_id] = score
    return qrels


def skipped_path(result_file):
    return result_file + ".skipped.jsonl"


def read_skipped(result_file):
    """读内容审核丢弃清单，返回 qid 列表。"""
    path = skipped_path(result_file)
    if not os.path.exists(path):
        return []
    qids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qids.append(json.loads(line)["qid"])
    return qids


def load_bm25_ranking(bm25_data_file):
    """从 *_bm25_top100.jsonl 读 qid -> bm25_docs（给丢弃 query 回填用）。"""
    ranking = {}
    if not bm25_data_file or not os.path.exists(bm25_data_file):
        return ranking
    with open(bm25_data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ranking[row["qid"]] = row["bm25_docs"]
    return ranking


def evaluate(result_file, qrels, pattern="llm_docs", fill_skipped=False, bm25_data_file=None):
    """评测单个结果文件，返回指标 dict 并打印摘要。"""
    results_bm25, results_llm = {}, {}
    tokens, prompt_tokens, completion_tokens = [], [], []
    llm_calls, api_requests = [], []
    query_latency, llm_latency, wall_clock, estimated_cost = [], [], [], []
    content_filter_failed_total = 0
    has_llm = False

    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            qid, bm25_docs = data["qid"], data["bm25_docs"]
            llm_docs = data.get(pattern, None)
            tokens.append(data.get("sum_tokens", 0))
            prompt_tokens.append(data.get("prompt_tokens", 0))
            completion_tokens.append(data.get("completion_tokens", 0))
            llm_calls.append(data.get("llm_calls", 0))
            api_requests.append(data.get("api_requests", data.get("llm_calls", 0)))
            query_latency.append(data.get("query_latency_seconds", 0))
            llm_latency.append(data.get("llm_latency_seconds", 0))
            wall_clock.append(data.get("wall_clock_seconds", data.get("query_latency_seconds", 0)))
            estimated_cost.append(data.get("estimated_cost", 0.0))
            content_filter_failed_total += data.get("content_filter_failed", 0)
            results_bm25[qid] = {doc: 1 / (i + 1) for i, doc in enumerate(bm25_docs)}
            if llm_docs:
                has_llm = True
                results_llm[qid] = {doc: 1 / (i + 1) for i, doc in enumerate(llm_docs)}

    # 内容审核丢弃的 query。
    skipped_qids = read_skipped(result_file)
    n_skipped = len(skipped_qids)
    n_evaluated = len(results_llm)

    filled = 0
    if fill_skipped and skipped_qids:
        bm25_ranking = load_bm25_ranking(bm25_data_file)
        for qid in skipped_qids:
            docs = bm25_ranking.get(qid)
            if docs:
                results_llm[qid] = {doc: 1 / (i + 1) for i, doc in enumerate(docs)}
                results_bm25.setdefault(qid, {doc: 1 / (i + 1) for i, doc in enumerate(docs)})
                filled += 1
        has_llm = has_llm or filled > 0

    bm25_ndcg = evaluate_ndcg(qrels, results_bm25) if results_bm25 else {}
    llm_ndcg = evaluate_ndcg(qrels, results_llm) if has_llm and results_llm else {}

    note = f"evaluated={n_evaluated + filled} qids, skipped={n_skipped}"
    if fill_skipped and n_skipped:
        note += f" ({filled} 条用 BM25 回填，全量可比)"
    print(f"  [{os.path.basename(result_file)}] {note}")
    if bm25_ndcg:
        print(f"    bm25 ndcg: {bm25_ndcg}")
    if llm_ndcg:
        print(f"    llm  ndcg: {llm_ndcg}")
    if tokens:
        print(
            f"    Avg Tokens={np.mean(tokens):.0f} Prompt={np.mean(prompt_tokens):.0f} "
            f"Completion={np.mean(completion_tokens):.0f} LLMCalls={np.mean(llm_calls):.1f} "
            f"APIReq={np.mean(api_requests):.1f} QueryLat={np.mean(query_latency):.1f}s "
            f"Cost={np.mean(estimated_cost):.4f}"
        )
    if content_filter_failed_total:
        # 正常不该出现：有审核的 query 已整条丢弃、不写结果。出现说明 ranker 漏判。
        print(f"    [WARN] 结果行内 content_filter_failed 合计 {content_filter_failed_total}（预期 0）")

    return {
        "result_file": result_file,
        "n_evaluated": n_evaluated,
        "n_skipped": n_skipped,
        "bm25_ndcg": bm25_ndcg,
        "llm_ndcg": llm_ndcg,
        "avg_tokens": float(np.mean(tokens)) if tokens else 0.0,
        "avg_llm_calls": float(np.mean(llm_calls)) if llm_calls else 0.0,
        "avg_query_latency": float(np.mean(query_latency)) if query_latency else 0.0,
    }


def evaluate_ndcg(qrels, results, k_values=[1, 3, 5, 10, 20]):
    ndcg = {f"NDCG@{k}": 0.0 for k in k_values}
    ndcg_string = "ndcg_cut." + ",".join(str(k) for k in k_values)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ndcg_string})
    scores = evaluator.evaluate(results)
    if not scores:
        return ndcg
    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 4) * 100
    return ndcg


def qrels_path(bench, dataset):
    if bench == "trec":
        return f"./data/TREC_data/{dataset}/qrels.{dataset}-passage.txt"
    return f"./data/BEIR_data/{dataset}/qrels.beir-v1.0.0-{dataset}.test.txt"


def bm25_data_path(bench, dataset):
    if bench == "trec":
        year = "19" if dataset == "dl19" else "20"
        return f"./data/TREC_data/{dataset}/trec{year}_bm25_top100.jsonl"
    return f"./data/BEIR_data/{dataset}/{dataset}_bm25_top100.jsonl"


def result_path(bench, dataset, method, model=None):
    root = "TREC_results" if bench == "trec" else "BEIR_results"
    model_seg = f"models/{model}/" if model else ""
    return f"./results/{root}/{dataset}/{model_seg}{METHOD_FILES[method]}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", choices=["trec", "beir"], required=True)
    parser.add_argument("--datasets", nargs="*", default=None, help="默认 trec→dl19/dl20, beir→全部")
    parser.add_argument("--methods", nargs="*", default=list(METHOD_FILES))
    parser.add_argument("--model", default=None, help="多模型 baseline 的 MODEL_TAG，如 deepseek-r1")
    parser.add_argument("--fill-skipped", action="store_true", help="丢弃 query 用 BM25 回填，保证跨方法可比")
    args = parser.parse_args()

    datasets = args.datasets or (TREC_DATASETS if args.bench == "trec" else BEIR_DATASETS)
    unknown = [m for m in args.methods if m not in METHOD_FILES]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}; valid={list(METHOD_FILES)}")

    for dataset in datasets:
        qrels_file = qrels_path(args.bench, dataset)
        if not os.path.exists(qrels_file):
            print(f"[SKIP dataset] {dataset}: 缺 qrels {qrels_file}")
            continue
        qrels = read_qrels(qrels_file)
        bm25_file = bm25_data_path(args.bench, dataset)
        header = f"===== {dataset}" + (f" / {args.model}" if args.model else "") + " ====="
        print(header)
        for method in args.methods:
            rf = result_path(args.bench, dataset, method, args.model)
            if not os.path.exists(rf):
                print(f"  [{method}] 无结果文件: {rf}")
                continue
            print(f"  {method}:")
            evaluate(rf, qrels, fill_skipped=args.fill_skipped, bm25_data_file=bm25_file)


if __name__ == "__main__":
    main()
