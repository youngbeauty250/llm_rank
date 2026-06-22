"""SwissRank（permutation 版）：组内让 LLM 输出整组排序，胜者 +1、败者 -1。

每轮所有 group 互相独立，一次 batch_chat 全部并发；
轮间顺序依赖（要按上轮分数重新分桶），保留外层 while 循环。
"""

import copy
import json
import os
from collections import defaultdict

from tqdm import tqdm

from utils import build_result_record, get_default_client, now_seconds, set_seed, truncate_doc


def reorder_large_list(large_list, small_list):
    """把 small_list 的相对顺序写回 large_list 中那些位置（保持其它位置不动）。"""
    new_large_list = []
    num = 0
    for element in large_list:
        if element in small_list:
            new_large_list.append(small_list[num])
            num += 1
        else:
            new_large_list.append(element)
    return new_large_list


def group_documents(docs, m):
    """skip 分组：按 step=n 交错抽取，使每组内文档原始排名分散。"""
    k = len(docs)
    n = k // m
    if n == 0:
        return []
    groups = [[] for _ in range(n)]
    for i in range(n):
        idx = i
        while idx < k:
            groups[i].append(docs[idx])
            idx += n
    return groups


def get_prefix_prompt(query, num):
    return [
        {"role": "system",
         "content": "You are RankGPT, an intelligent assistant that can rank documents based on their relevancy to the query."},
        {"role": "user",
         "content": (
             f"I will provide you with {num} documents, each indicated by number identifier []. \n"
             f"Rank the documents based on their relevance to query: {query}."
         )},
        {"role": "assistant", "content": "Okay, please provide the documents."},
    ]


def get_post_prompt(query, num):
    return (
        f"Search Query: {query}. \nRank the {num} documents above based on their relevance to the search query. "
        "The documents should be listed in descending order using identifiers. The most relevant documents should be "
        "listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do "
        "not say any word or explain."
    )


def clean_response(response: str):
    out = ""
    for c in response:
        out += c if c.isdigit() else " "
    return out.strip()


def remove_duplicate(response):
    seen = []
    for c in response:
        if c not in seen:
            seen.append(c)
    return seen


def receive_permutation(ranking, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(ranking[rank_start:rank_end])
    original_rank = list(range(len(cut_range)))
    response = [s for s in response if s in original_rank]
    response = response + [t for t in original_rank if t not in response]
    for j, x in enumerate(response):
        ranking[j + rank_start] = cut_range[x]
    return ranking


def build_group_messages(query, group, doc_contents):
    n = len(group)
    messages = get_prefix_prompt(query, n)
    for j, doc_id in enumerate(group):
        content = truncate_doc(doc_contents[doc_id])
        messages.append({"role": "user", "content": f"Document {j + 1}: {content}"})
        messages.append({"role": "assistant", "content": f"Received Document {j + 1}."})
    messages.append({"role": "user", "content": get_post_prompt(query, n)})
    return messages


def filter_processing(client, query, doc_ids, doc_contents, params):
    scores = defaultdict(list)
    scores[0] = doc_ids

    while True:
        flag = False
        score_doc_dict = defaultdict(list)
        batch_msgs = []
        batch_meta = []  # 每项 (score, group)

        for score, doc_indexs in scores.items():
            if len(doc_indexs) < params["group_size"] or score < params["max_failure_num"]:
                score_doc_dict[score] = doc_indexs
                continue
            flag = True
            for group in group_documents(doc_indexs, params["group_size"]):
                batch_msgs.append(build_group_messages(query, group, doc_contents))
                batch_meta.append((score, group))

        if not flag:
            break

        responses = client.batch_chat(batch_msgs)
        for (score, group), resp in zip(batch_meta, responses):
            n = len(group)
            new_group = receive_permutation(group, resp.content or "", 0, n)
            score_doc_dict[score + 1].extend(new_group[: params["step_size"]])
            score_doc_dict[score - 1].extend(new_group[params["step_size"]:])
            doc_ids = reorder_large_list(doc_ids, new_group)

        scores = {
            key: sorted(value, key=lambda x: doc_ids.index(x))
            for key, value in score_doc_dict.items()
        }

    # 顶部精排：对得分最高的若干桶各做一次完整 permutation
    top_scores = sorted(scores.keys(), reverse=True)
    rerank_msgs = []
    rerank_groups = []
    for i, score in enumerate(top_scores):
        if i >= params["top_rerank_num"]:
            break
        group = scores[score]
        rerank_msgs.append(build_group_messages(query, group, doc_contents))
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


def main():
    params = {
        "max_failure_num": 0,
        "step_size": 4,
        "group_size": 20,
        "top_rerank_num": 1,
    }
    set_seed(42)
    client = get_default_client()
    result_file = "../results/TREC_results/dl20/swiss_v2.jsonl"
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
        with open("../data/TREC_data/dl20/trec20_bm25_top100.jsonl", "r", encoding="utf-8") as read_file:
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
                docs_ranked = filter_processing(client, query, ranked_list, doc_contents, params)
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
