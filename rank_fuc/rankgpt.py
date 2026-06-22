"""RankGPT：滑动窗口 listwise 重排。

窗口间顺序依赖（前一窗口的结果决定后一窗口的 doc 顺序），无法 batch；本文件仅做 client 切换。
"""

import copy
import json
import os

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


def get_prefix_prompt(query, num):
    return [
        {"role": "system",
         "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
        {"role": "user",
         "content": f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
        {"role": "assistant", "content": "Okay, please provide the passages."},
    ]


def get_post_prompt(query, num):
    return (
        f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. "
        "The passages should be listed in descending order using identifiers. The most relevant passages should be "
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


def build_window_messages(query, indexs, doc_contents):
    n = len(indexs)
    messages = get_prefix_prompt(query, n)
    for j, doc_id in enumerate(indexs):
        content = truncate_doc(doc_contents[doc_id])
        messages.append({"role": "user", "content": f"[{j + 1}] {content}"})
        messages.append({"role": "assistant", "content": f"Received passage [{j + 1}]."})
    messages.append({"role": "user", "content": get_post_prompt(query, n)})
    return messages


def filter_processing(client, query, doc_ids, doc_contents, params):
    if len(doc_ids) <= 1:
        return copy.deepcopy(doc_ids)
    ranking = copy.deepcopy(doc_ids)
    end_pos = len(ranking)
    start_pos = end_pos - params["window_size"]
    while start_pos >= 0:
        start_pos = max(start_pos, 0)
        now_indexs = ranking[start_pos:end_pos]
        messages = build_window_messages(query, now_indexs, doc_contents)
        resp = client.chat(messages)
        ranking = receive_permutation(ranking, resp.content or "", start_pos, end_pos)
        end_pos -= params["step_size"]
        start_pos -= params["step_size"]
    return ranking


def main():
    params = {"step_size": 10, "window_size": 20}
    set_seed(42)
    client = get_default_client()

    result_file, data_file = get_trec_paths("rankgpt.jsonl")
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
