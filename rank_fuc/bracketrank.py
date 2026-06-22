"""BracketRank：winner/loser bracket reranking.

This version keeps the paper's winner/loser bracket structure but uses a
non-reasoning listwise prompt for fair latency/cost comparison with other
LLM_Rank baselines.
"""

import copy
import json
import os

from tqdm import tqdm

from rankgpt import receive_permutation
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


def get_prefix_prompt(query, num):
    return [
        {
            "role": "system",
            "content": "You are RankGPT, an intelligent assistant that can rank documents based on their relevancy to the query.",
        },
        {
            "role": "user",
            "content": (
                f"I will provide {num} documents, each marked by a number identifier []. "
                f"Rank the documents based on their relevance to query: {query}."
            ),
        },
        {"role": "assistant", "content": "Okay, please provide the documents."},
    ]


def get_post_prompt(query, num):
    return (
        f"Search Query: {query}. \nRank the {num} documents above based on their relevance to the search query. "
        "The documents should be listed in descending order using identifiers. The most relevant documents should be "
        "listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do "
        "not say any word or explain."
    )


def build_messages(query, group, doc_contents):
    messages = get_prefix_prompt(query, len(group))
    for j, doc_id in enumerate(group):
        content = truncate_doc(doc_contents[doc_id])
        messages.append({"role": "user", "content": f"Document {j + 1}: {content}"})
        messages.append({"role": "assistant", "content": f"Received Document {j + 1}."})
    messages.append({"role": "user", "content": get_post_prompt(query, len(group))})
    return messages


def chunk_groups(doc_ids, group_size):
    return [doc_ids[i: i + group_size] for i in range(0, len(doc_ids), group_size)]


def rank_groups(client, query, groups, doc_contents, desc=None):
    messages_list = [build_messages(query, group, doc_contents) for group in groups]
    responses = client.batch_chat(messages_list, desc=desc)
    return [
        receive_permutation(group, resp.content or "", 0, len(group))
        for group, resp in zip(groups, responses)
    ]


def split_winner_loser_groups(ranked_groups):
    winner_groups, loser_groups = [], []
    for group in ranked_groups:
        midpoint = (len(group) + 1) // 2
        if group[:midpoint]:
            winner_groups.append(group[:midpoint])
        if group[midpoint:]:
            loser_groups.append(group[midpoint:])
    return winner_groups, loser_groups


def run_single_elimination(client, query, groups, doc_contents, group_size):
    groups = [group for group in groups if group]
    advance_size = max(1, group_size // 2)
    eliminated_layers = []
    while len(groups) > 1:
        next_groups = []
        match_groups = []
        for i in range(0, len(groups), 2):
            if i + 1 >= len(groups):
                next_groups.append(groups[i])
                continue
            match_groups.append(groups[i] + groups[i + 1])

        ranked_matches = rank_groups(client, query, match_groups, doc_contents)
        round_eliminated = []
        for ranked in ranked_matches:
            next_groups.append(ranked[:advance_size])
            if ranked[advance_size:]:
                round_eliminated.append(ranked[advance_size:])

        if round_eliminated:
            eliminated_layers.append(round_eliminated)
        groups = next_groups

    final_ranked = groups[0] if groups else []
    for layer in reversed(eliminated_layers):
        for eliminated_group in layer:
            final_ranked.extend(eliminated_group)
    return final_ranked


def filter_processing(client, query, doc_ids, doc_contents, params):
    if len(doc_ids) <= 1:
        return copy.deepcopy(doc_ids)
    group_size = params.get("group_size", 20)
    initial_groups = chunk_groups(doc_ids, group_size)
    ranked_groups = rank_groups(client, query, initial_groups, doc_contents, desc="BracketRank initial")
    winner_groups, loser_groups = split_winner_loser_groups(ranked_groups)

    winner_ranked = run_single_elimination(client, query, winner_groups, doc_contents, group_size)
    loser_ranked = run_single_elimination(client, query, loser_groups, doc_contents, group_size)
    ranked = winner_ranked + loser_ranked
    seen = set(ranked)
    return ranked + [doc_id for doc_id in doc_ids if doc_id not in seen]


def main():
    params = {"group_size": 20}
    set_seed(42)
    client = get_default_client()
    result_file, data_file = get_trec_paths("bracketrank.jsonl")
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
                docs_ranked = filter_processing(client, query, copy.deepcopy(doc_ids), doc_contents, params)
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
