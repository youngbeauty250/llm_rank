"""TourRank 单次锦标赛：100→50→20→10→5→2 五阶段，每阶段被选中得 1 分，最终按累计分排名。

阶段 1/2 都是 5 组互相独立的并发 LLM 调用，用 batch_chat 一次发完；
阶段 3/4/5 每阶段只有 1 组，串行 chat。
"""

import copy
import json
import os
import random

from tqdm import tqdm

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


def get_prefix_role_prompt(query, N, M):
    return [
        {"role": "system",
         "content": "You are an intelligent assistant that can compare multiple documents based on their relevancy to the given query."},
        {"role": "user",
         "content": (
             f"I will provide you with the given query and {N} documents. \n"
             f"Consider the content of all the documents comprehensively and select the {M} documents "
             f"that are most relevant to the given query: {query}."
         )},
        {"role": "assistant", "content": "Okay, please provide the documents."},
    ]


def get_post_role_prompt(query, M):
    return (
        f"The Query is: {query}.\n"
        f"Now, you must output the top {M} documents that are most relevant to the Query using the following "
        "format strictly, and nothing else. Don't output any explanation, just the following format:\n"
        "Document 3, ..., Document 1"
    )


def sort_docs_by_relevance(doc_ids, relevance_scores):
    combined = list(zip(doc_ids, relevance_scores))
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_combined]


def get_top_M(answer, N=10, M=5, groups_docid=None):
    groups_docid = groups_docid or []
    if not answer:
        return []
    temp = answer.split("\n")
    for i in range(1, len(temp) + 1):
        if "Document" in temp[-i]:
            temp = temp[-i]
            break
    else:
        return []
    temp = temp.split(":")[-1].split(".")[0].split(",")
    top_M = []
    for doc in temp:
        try:
            if "..." in doc:
                continue
            doc_num = int(doc.split()[-1]) - 1
            top_M.append(doc_num)
        except (ValueError, IndexError):
            continue
    return [groups_docid[i] for i in top_M if 0 <= i < len(groups_docid)]


def get_groups_skip(docs_id, to_n_groups=10, m_docs_per_group=10):
    docs_groups = []
    for i in range(to_n_groups):
        cur_group = []
        for j in range(m_docs_per_group):
            idx = j * to_n_groups + i
            if idx >= len(docs_id):
                break
            cur_group.append(docs_id[idx])
        if cur_group:
            docs_groups.append(cur_group)
    return docs_groups


def build_stage_messages(query, group, all_contents, N, M):
    """构造一阶段一组的 messages；group 内顺序会被 shuffle。"""
    group = list(group)
    random.shuffle(group)
    messages = get_prefix_role_prompt(query, N, M)
    for j, doc_id in enumerate(group):
        messages.append({"role": "user", "content": f"Document {j + 1}: {truncate_doc(all_contents[doc_id])}"})
        messages.append({"role": "assistant", "content": f"Received Document {j + 1}."})
    messages.append({"role": "user", "content": get_post_role_prompt(query, M)})
    return messages, group  # group 是 shuffle 后版本，解析答案要用


def run_stage(client, query, all_contents, docs_score_dict, group_lists, N, M):
    """对若干 group 并发调用，把入选 doc 的得分 +1。"""
    batch_msgs = []
    shuffled_groups = []
    for g in group_lists:
        actual_n = len(g)
        actual_m = min(M, actual_n)
        if actual_n <= 0 or actual_m <= 0:
            continue
        msgs, shuffled = build_stage_messages(query, g, all_contents, actual_n, actual_m)
        batch_msgs.append(msgs)
        shuffled_groups.append(shuffled)
    if not batch_msgs:
        return
    resps = client.batch_chat(batch_msgs)
    for shuffled, resp in zip(shuffled_groups, resps):
        top_M_ids = get_top_M(resp.content or "", N=N, M=M, groups_docid=shuffled)
        for doc_id in top_M_ids:
            docs_score_dict[doc_id] += 1


def filter_processing(client, query, docs_id, all_contents):
    if len(docs_id) <= 1:
        return {d: 0 for d in docs_id}
    docs_score_dict = {d: 0 for d in docs_id}

    # Stage 1: 100 -> 50 (5 组 × 20 选 10)
    groups = get_groups_skip(docs_id, to_n_groups=5, m_docs_per_group=20)
    run_stage(client, query, all_contents, docs_score_dict, groups, N=20, M=10)
    ranked_list = sort_docs_by_relevance(list(docs_score_dict.keys()), list(docs_score_dict.values()))

    # Stage 2: 50 -> 20 (5 组 × 10 选 4)
    stage2_docs_id = ranked_list[:50]
    groups = get_groups_skip(stage2_docs_id, to_n_groups=5, m_docs_per_group=10)
    run_stage(client, query, all_contents, docs_score_dict, groups, N=10, M=4)
    ranked_list = sort_docs_by_relevance(list(docs_score_dict.keys()), list(docs_score_dict.values()))

    # Stage 3: 20 -> 10 (1 组 × 20 选 10)
    groups = get_groups_skip(ranked_list[:20], to_n_groups=1, m_docs_per_group=20)
    run_stage(client, query, all_contents, docs_score_dict, groups, N=20, M=10)
    ranked_list = sort_docs_by_relevance(list(docs_score_dict.keys()), list(docs_score_dict.values()))

    # Stage 4: 10 -> 5 (1 组 × 10 选 5)
    groups = get_groups_skip(ranked_list[:10], to_n_groups=1, m_docs_per_group=10)
    run_stage(client, query, all_contents, docs_score_dict, groups, N=10, M=5)
    ranked_list = sort_docs_by_relevance(list(docs_score_dict.keys()), list(docs_score_dict.values()))

    # Stage 5: 5 -> 2 (1 组 × 5 选 2)
    groups = get_groups_skip(ranked_list[:5], to_n_groups=1, m_docs_per_group=5)
    run_stage(client, query, all_contents, docs_score_dict, groups, N=5, M=2)

    return docs_score_dict


def main():
    set_seed(42)
    client = get_default_client()
    result_file, data_file = get_trec_paths("tourrank.jsonl")
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
                docs_id = copy.deepcopy(doc_ids)
                all_contents = {doc_ids[i]: contents[i] for i in range(len(doc_ids))}
                docs_score_dict = filter_processing(client, query, docs_id, all_contents)
                ranked_list = sort_docs_by_relevance(
                    list(docs_score_dict.keys()), list(docs_score_dict.values())
                )
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
                        llm_docs=ranked_list,
                    ),
                    ensure_ascii=False,
                )
                f.write(json_line + "\n")
                f.flush()
                processed += 1
        print("Finished!")


if __name__ == "__main__":
    main()
