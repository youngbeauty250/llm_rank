"""SwissRank（choice 版，论文主推）：组内只让 LLM 挑出 Top N/2，胜负各 +1/-1 分。

与 swissrank.py 的区别只在 post_prompt（让 LLM 选 Top N/2 而非整组排序）和 filter_processing
里的胜负切分点（用 N/2 而非 step_size）。其它工具函数从 swissrank 模块复用。
"""

import copy
import json
import os
import random
from collections import defaultdict

from tqdm import tqdm

from swissrank import (
    clean_response,
    get_prefix_prompt,
    group_documents,
    receive_permutation,
    remove_duplicate,
    reorder_large_list,
)
from utils import (
    build_result_record,
    get_default_client,
    get_trec_paths,
    load_done_qids,
    now_seconds,
    record_skipped_query,
    set_seed,
    truncate_doc,
)


def group_documents_chunk(docs, m):
    """连续切块分组（消融对比用）。"""
    k = len(docs)
    n = k // m
    if n == 0:
        return []
    groups = []
    for i in range(n - 1):
        groups.append(docs[i * m: (i + 1) * m])
    groups.append(docs[(n - 1) * m:])
    return groups


def group_documents_random(docs, m):
    """完全随机分组（消融对比用）。"""
    k = len(docs)
    n = k // m
    if n == 0:
        return []
    indices = list(range(k))
    random.shuffle(indices)
    group_ids = [0] * k
    for i, idx in enumerate(indices):
        group_ids[idx] = min(i // m, n - 1)
    groups = [[] for _ in range(n)]
    for idx, gid in enumerate(group_ids):
        groups[gid].append(docs[idx])
    return groups


def get_post_prompt_choice(query, M):
    return (
        f"Search Query: {query}. \nChoose and Rank the top {M} documents that are most relevant to the search query. "
        "The documents should be listed in descending order using identifiers. The most relevant documents should be "
        "listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do "
        "not say any word or explain."
    )


def build_group_messages_choice(query, group, doc_contents):
    """choice 版的 messages：post_prompt 只要 Top N/2。"""
    n = len(group)
    messages = get_prefix_prompt(query, n)
    for j, doc_id in enumerate(group):
        content = truncate_doc(doc_contents[doc_id])
        messages.append({"role": "user", "content": f"Document {j + 1}: {content}"})
        messages.append({"role": "assistant", "content": f"Received Document {j + 1}."})
    messages.append({"role": "user", "content": get_post_prompt_choice(query, n // 2)})
    return messages


def filter_processing(client, query, doc_ids, doc_contents, params):
    scores = defaultdict(list)
    scores[0] = doc_ids
    round_no = 0
    debug_failure = os.environ.get("DEBUG_SWISS_FAILURE") == "1"

    while True:
        round_no += 1
        flag = False
        score_doc_dict = defaultdict(list)
        batch_msgs = []
        batch_meta = []

        for score, doc_indexs in scores.items():
            if len(doc_indexs) < params["group_size"] or score < params["max_failure_num"]:
                score_doc_dict[score] = doc_indexs
                continue
            flag = True
            for group in params["grouping_fn"](doc_indexs, params["group_size"]):
                batch_msgs.append(build_group_messages_choice(query, group, doc_contents))
                batch_meta.append((score, group))

        if not flag:
            break

        responses = client.batch_chat(batch_msgs)
        for group_idx, ((score, group), resp) in enumerate(zip(batch_meta, responses)):
            if debug_failure and resp.content is None:
                print(
                    f"DEBUG_SWISS_FAILURE round={round_no} batch_group={group_idx} "
                    f"score={score} size={len(group)}"
                )
                print("DEBUG_SWISS_FAILURE doc_ids=" + ",".join(group))
                for j, doc_id in enumerate(group, 1):
                    snippet = truncate_doc(doc_contents[doc_id])[:300].replace("\n", " ")
                    print(f"DEBUG_SWISS_FAILURE doc{j} id={doc_id} text={snippet}")
            n = len(group)
            new_group = receive_permutation(group, resp.content or "", 0, n)
            score_doc_dict[score + 1].extend(new_group[: n // 2])
            score_doc_dict[score - 1].extend(new_group[n // 2:])
            # 后验再播（posterior correction）：把组内最新顺序写回种子排名。
            # 消融 w/o Posterior Correction（fixed/random seed）时不更新，种子保持初始序。
            if params["seed_mode"] == "posterior":
                doc_ids = reorder_large_list(doc_ids, new_group)

        scores = {
            key: sorted(value, key=lambda x: doc_ids.index(x))
            for key, value in score_doc_dict.items()
        }
        # P1-1 机制分析(opt-in)：记录本轮结束后的种子排名,用于画"种子NDCG随轮次收敛"曲线。
        # 未传 round_seed_log 时为 no-op,不影响主流程与其它实验。
        seed_log = params.get("round_seed_log")
        if seed_log is not None:
            seed_log.append(list(doc_ids))

    top_scores = sorted(scores.keys(), reverse=True)
    rerank_msgs = []
    rerank_groups = []
    for i, score in enumerate(top_scores):
        if i >= params["top_rerank_num"]:
            break
        group = scores[score]
        # 顶部精排让 LLM 输出整组完整顺序（M = n，即对整组排序）
        n = len(group)
        messages = get_prefix_prompt(query, n)
        for j, doc_id in enumerate(group):
            content = truncate_doc(doc_contents[doc_id])
            messages.append({"role": "user", "content": f"Document {j + 1}: {content}"})
            messages.append({"role": "assistant", "content": f"Received Document {j + 1}."})
        messages.append({"role": "user", "content": get_post_prompt_choice(query, n)})
        rerank_msgs.append(messages)
        rerank_groups.append((score, group))

    if rerank_msgs:
        rerank_resps = client.batch_chat(rerank_msgs)
        for (score, group), resp in zip(rerank_groups, rerank_resps):
            n = len(group)
            scores[score] = receive_permutation(group, (resp.content or "").strip(), 0, n)

    docs_ranked = []
    for score in top_scores:
        docs_ranked.extend(scores[score])
    return docs_ranked


_GROUPING_FNS = {
    "skip": group_documents,            # 默认：skip 分组（论文主推）
    "chunk": group_documents_chunk,     # 消融：连续切块
    "random": group_documents_random,   # 消融：随机分组
}


def main():
    # 消融开关（默认 = 论文主方法行为，未设环境变量时与改动前完全一致）：
    #   SWISS_GROUPING ∈ {skip, chunk, random}        —— w/o Skip Group 消融
    #   SWISS_SEED     ∈ {posterior, fixed, random}   —— w/o Posterior Correction 消融
    #   SWISS_MAX_FAILURE = 整数                        —— 剪枝阈值(min_s)扫描
    grouping_name = os.environ.get("SWISS_GROUPING", "skip").strip().lower()
    seed_mode = os.environ.get("SWISS_SEED", "posterior").strip().lower()
    if grouping_name not in _GROUPING_FNS:
        raise ValueError(f"Unsupported SWISS_GROUPING={grouping_name!r}; expected {list(_GROUPING_FNS)}")
    if seed_mode not in ("posterior", "fixed", "random"):
        raise ValueError(f"Unsupported SWISS_SEED={seed_mode!r}; expected posterior|fixed|random")
    max_failure_env = os.environ.get("SWISS_MAX_FAILURE", "").strip()
    params = {
        "max_failure_num": int(max_failure_env) if max_failure_env else -1,
        "group_size": 20,
        "top_rerank_num": 0,
        "grouping_fn": _GROUPING_FNS[grouping_name],
        "seed_mode": seed_mode,
    }
    set_seed(42)
    client = get_default_client()
    result_file, data_file = get_trec_paths("swiss_choice.jsonl")
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    now_data = load_done_qids(result_file)

    smoke = os.environ.get("SMOKE_TEST")
    smoke_n = int(smoke) if smoke and smoke.isdigit() else (2 if smoke else 0)
    processed = 0

    with open(result_file, "a+", encoding="utf-8") as f:
        with open(data_file, "r", encoding="utf-8") as read_file:
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
                doc_contents = {doc_ids[j]: contents[j] for j in range(len(doc_ids))}
                ranked_list = copy.deepcopy(doc_ids)
                # Random Seed 消融：初始种子排名随机洗牌且不做后验再播（merged in random order）。
                if params["seed_mode"] == "random":
                    random.shuffle(ranked_list)
                docs_ranked = filter_processing(client, query, ranked_list, doc_contents, params)
                cf = client.get_usage().content_filter_failed
                if cf > 0:
                    # 内容审核降级：被审核命中的分组/调用保持输入(BM25)原序，
                    # query 仍写出完整排序；content_filter_failed 计入结果行以便诚实报告。
                    print(f"[content_filter] qid={qid} degraded calls={cf}, kept ranking with BM25 fallback")
                json_line = json.dumps(
                    build_result_record(
                        client,
                        query_start_time,
                        qid=qid,
                        query=query,
                        bm25_docs=doc_ids,
                        llm_docs=docs_ranked,
                    ),
                    ensure_ascii=False,
                )
                f.write(json_line + "\n")
                f.flush()
                processed += 1
        print("Finished!")


if __name__ == "__main__":
    main()
