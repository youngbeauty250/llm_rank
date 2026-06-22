"""TourRank-N：跑 Y 次独立锦标赛，得分按轮次累加得到 tour_0/tour_1/... 渐进式排名。

相对 tourrank.py 的优化：Y 个 tournament 之间各阶段相互独立，所以同一 stage 在所有
tournament 之间也能并发——本实现把 Y×K 组 messages 一次 batch_chat 全部发出去。

修复了原版 tourrank-n.py 内自定义 get_response 不返回 usage、导致 token 完全不统计的 bug。
"""

import copy
import json
import os
import random

from tqdm import tqdm

from tourrank import (
    build_stage_messages,
    get_groups_skip,
    get_top_M,
    sort_docs_by_relevance,
)
from utils import build_result_record, get_default_client, now_seconds, set_seed


def _run_stage_multi_tour(
    client,
    query,
    all_contents,
    score_dicts,    # list[dict]，第 y 个 tournament 的得分字典
    stage_groups,   # list[list[group]]，第 y 个 tournament 的本阶段所有 group
    N,
    M,
):
    """跨 tournament 一次性 batch：所有 (y, group) 的 messages 拼一起发。"""
    batch_msgs = []
    meta = []  # (y, shuffled_group)
    for y, groups in enumerate(stage_groups):
        for g in groups:
            msgs, shuffled = build_stage_messages(query, g, all_contents, N, M)
            batch_msgs.append(msgs)
            meta.append((y, shuffled))

    if not batch_msgs:
        return
    resps = client.batch_chat(batch_msgs)
    for (y, shuffled), resp in zip(meta, resps):
        top_M_ids = get_top_M(resp.content or "", N=N, M=M, groups_docid=shuffled)
        for doc_id in top_M_ids:
            score_dicts[y][doc_id] += 1


def filter_processing_multi(client, query, docs_id, all_contents, Y=2):
    """Y 次独立锦标赛，返回 list[dict]，第 y 项为该轮次累计得分。"""
    score_dicts = [{d: 0 for d in docs_id} for _ in range(Y)]

    # Stage 1: 每个 tournament 把全 100 doc 切成 5 组，跨 Y 个 tournament 一起 batch
    stage1_groups = [
        get_groups_skip(docs_id, to_n_groups=5, m_docs_per_group=20) for _ in range(Y)
    ]
    _run_stage_multi_tour(client, query, all_contents, score_dicts, stage1_groups, N=20, M=10)

    # Stage 2: 各 tournament 用各自当前 top-50 切 5 组
    stage2_groups = []
    for y in range(Y):
        ranked = sort_docs_by_relevance(list(score_dicts[y].keys()), list(score_dicts[y].values()))
        stage2_groups.append(get_groups_skip(ranked[:50], to_n_groups=5, m_docs_per_group=10))
    _run_stage_multi_tour(client, query, all_contents, score_dicts, stage2_groups, N=10, M=4)

    # Stage 3: 各 tournament 当前 top-20 一组（Y 组并发）
    stage3_groups = []
    for y in range(Y):
        ranked = sort_docs_by_relevance(list(score_dicts[y].keys()), list(score_dicts[y].values()))
        stage3_groups.append(get_groups_skip(ranked[:20], to_n_groups=1, m_docs_per_group=20))
    _run_stage_multi_tour(client, query, all_contents, score_dicts, stage3_groups, N=20, M=10)

    # Stage 4: 各 tournament 当前 top-10 一组
    stage4_groups = []
    for y in range(Y):
        ranked = sort_docs_by_relevance(list(score_dicts[y].keys()), list(score_dicts[y].values()))
        stage4_groups.append(get_groups_skip(ranked[:10], to_n_groups=1, m_docs_per_group=10))
    _run_stage_multi_tour(client, query, all_contents, score_dicts, stage4_groups, N=10, M=5)

    # Stage 5: 各 tournament 当前 top-5 一组
    stage5_groups = []
    for y in range(Y):
        ranked = sort_docs_by_relevance(list(score_dicts[y].keys()), list(score_dicts[y].values()))
        stage5_groups.append(get_groups_skip(ranked[:5], to_n_groups=1, m_docs_per_group=5))
    _run_stage_multi_tour(client, query, all_contents, score_dicts, stage5_groups, N=5, M=2)

    return score_dicts


def main():
    set_seed(42)
    client = get_default_client()
    Y = 2
    result_file = "../results/TREC_results/dl19/tourrank_2_v2.jsonl"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    now_data = []
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                now_data.append(json.loads(line)["qid"])

    smoke = os.environ.get("SMOKE_TEST")
    smoke_n = int(smoke) if smoke and smoke.isdigit() else (2 if smoke else 0)
    processed = 0

    with open(result_file, "a+", encoding="utf-8") as f:
        with open("../data/TREC_data/dl19/trec19_bm25_top100.jsonl", "r", encoding="utf-8") as read_file:
            for line in tqdm(read_file):
                if smoke_n and processed >= smoke_n:
                    break
                client.reset_usage()
                line = json.loads(line)
                qid, query, doc_ids, contents = line["qid"], line["query"], line["bm25_docs"], line["bm25_contents"]
                if qid in now_data:
                    continue
                query_start_time = now_seconds()
                print(query)
                docs_id = copy.deepcopy(doc_ids)
                all_contents = {doc_ids[i]: contents[i] for i in range(len(doc_ids))}

                score_dicts = filter_processing_multi(client, query, docs_id, all_contents, Y=Y)

                # 渐进式累加：tour_0 用 score_dicts[0]，tour_1 用 score_dicts[0]+score_dicts[1] ...
                global_scores = {d: 0 for d in docs_id}
                result_dict = {"qid": qid, "query": query, "bm25_docs": doc_ids}
                for y in range(Y):
                    for d, s in score_dicts[y].items():
                        global_scores[d] += s
                    result_dict[f"tour_{y}"] = sort_docs_by_relevance(
                        list(global_scores.keys()), list(global_scores.values())
                    )
                result_dict = build_result_record(client, query_start_time, **result_dict)

                f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
                f.flush()
                processed += 1
        print("Finished!")


if __name__ == "__main__":
    main()
