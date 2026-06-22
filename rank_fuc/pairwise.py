"""Pairwise 重排：每次比较两个 doc，取 A/B + B/A 两个方向消除位置偏置。

A/B + B/A 这两次 LLM 调用没有依赖，是本文件唯一的 batch 加速点；
heapsort/bubblesort 整体仍然顺序依赖。
"""

import copy
import json
import os
import re
from collections import defaultdict
from itertools import combinations

from tqdm import tqdm

from utils import Documents, build_result_record, get_default_client, now_seconds, set_seed, truncate_doc


SYSTEM_PROMPT = (
    "You are RankGPT, an intelligent assistant specialized in selecting the most relevant "
    "passage from a pair of passages based on their relevance to the query."
)

PROMPT_TEMPLATE = """Given a query "{query}", which of the following two passages is more relevant to the query?

Passage A: "{doc1}"

Passage B: "{doc2}"

Output Passage A or Passage B:"""


def _parse_pairwise_label(output: str) -> str:
    """从 LLM 输出里抽出 'A' 或 'B'，缺省返回 'A'。"""
    if not output:
        return "A"
    matches = re.findall(r"Passage\s+([AB])", output)
    if matches:
        return matches[0]
    s = output.strip().upper()
    if s in ("A", "B"):
        return s
    print(f"Unexpected pairwise output: {output!r}")
    return "A"


class PairwiseRanker:
    def __init__(self, client, k=20, method="heapsort"):
        self.client = client
        self.k = k
        self.method = method

    def compare(self, query, doc1_text, doc2_text):
        """同时发 A/B 和 B/A 两次调用，返回 ['Passage X', 'Passage Y']。"""
        msg_ab = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    query=query,
                    doc1=truncate_doc(doc1_text),
                    doc2=truncate_doc(doc2_text),
                ),
            },
        ]
        msg_ba = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    query=query,
                    doc1=truncate_doc(doc2_text),
                    doc2=truncate_doc(doc1_text),
                ),
            },
        ]
        resps = self.client.batch_chat([msg_ab, msg_ba])
        return [
            f"Passage {_parse_pairwise_label(resps[0].content)}",
            f"Passage {_parse_pairwise_label(resps[1].content)}",
        ]

    def heapify(self, arr, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and arr[l] > arr[i]:
            largest = l
        if r < n and arr[r] > arr[largest]:
            largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.heapify(arr, n, largest)

    def heapSort(self, arr, k):
        n = len(arr)
        ranked = 0
        for i in range(n // 2, -1, -1):
            self.heapify(arr, n, i)
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            self.heapify(arr, i, 0)

    def rerank(self, query, docs):
        ranking = copy.deepcopy(docs)
        if self.method == "allpair":
            scores = defaultdict(float)
            for doc1, doc2 in combinations(ranking, 2):
                output = self.compare(query, doc1.text, doc2.text)
                if output[0] == "Passage A" and output[1] == "Passage B":
                    scores[doc1._id] += 1
                elif output[0] == "Passage B" and output[1] == "Passage A":
                    scores[doc2._id] += 1
                else:
                    scores[doc1._id] += 0.5
                    scores[doc2._id] += 0.5
            ranking = sorted(ranking, key=lambda doc: scores[doc._id], reverse=True)
        elif self.method == "heapsort":
            ranker = self

            class ComparableDoc:
                def __init__(self, _id, text):
                    self._id = _id
                    self.text = text

                def __gt__(self, other):
                    out = ranker.compare(query, self.text, other.text)
                    return out[0] == "Passage A" and out[1] == "Passage B"

            arr = [ComparableDoc(_id=doc._id, text=doc.text) for doc in ranking]
            self.heapSort(arr, self.k)
            ranking = list(reversed(arr))
        elif self.method == "bubblesort":
            k = min(self.k, len(ranking))
            last_end = len(ranking) - 1
            for i in range(k):
                current_ind = last_end
                is_change = False
                while True:
                    if current_ind <= i:
                        break
                    doc1 = ranking[current_ind]
                    doc2 = ranking[current_ind - 1]
                    output = self.compare(query, doc1.text, doc2.text)
                    if output[0] == "Passage A" and output[1] == "Passage B":
                        ranking[current_ind - 1], ranking[current_ind] = ranking[current_ind], ranking[current_ind - 1]
                        if not is_change:
                            is_change = True
                            if last_end != len(ranking) - 1:
                                last_end += 1
                    if not is_change:
                        last_end -= 1
                    current_ind -= 1
        else:
            raise NotImplementedError(f"Method {self.method} is not implemented.")

        results = []
        for doc in ranking[: self.k]:
            results.append(doc._id)
        for doc in docs:
            if doc._id not in results:
                results.append(doc._id)
        return results


def main():
    set_seed(42)
    client = get_default_client()
    ranker = PairwiseRanker(client, method="allpair")
    result_file = "../results/TREC_results/dl19/pairwise_allpair_v2.jsonl"
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
                docs = [Documents(doc_ids[j], truncate_doc(contents[j])) for j in range(len(doc_ids))]
                docs_ranked = ranker.rerank(query, docs)
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
