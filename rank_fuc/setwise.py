"""Setwise 重排：每次从 num_child+1 个候选中选 1 个最相关。

heapsort / bubblesort 都是顺序依赖，整体串行；仅替换 LLM 调用为 client.chat。
"""

import copy
import json
import os
import re

from tqdm import tqdm

from utils import (
    Documents,
    build_result_record,
    get_default_client,
    get_trec_paths,
    load_done_qids,
    now_seconds,
    record_skipped_query,
    set_seed,
    truncate_doc,
)


def _parse_setwise_label(output: str, labels: list[str]) -> str:
    if not output:
        return labels[0]
    matches = re.findall(r"Passage\s+([A-W])", output, re.IGNORECASE)
    for match in matches:
        label = match.upper()
        if label in labels:
            return label
    conclusion_matches = re.findall(
        r"(?:would be|answer is|choose|select|most relevant(?: passage)? is)\s*:?\s*([A-W])\b",
        output,
        re.IGNORECASE,
    )
    for match in conclusion_matches:
        label = match.upper()
        if label in labels:
            return label
    line_matches = re.findall(r"(?m)^\s*([A-W])\s*[\.\)]?\s*$", output)
    for match in line_matches:
        label = match.upper()
        if label in labels:
            return label
    stripped = output.strip().upper()
    if stripped in labels:
        return stripped
    print(f"Unexpected setwise output: {output}")
    return labels[0]


class SetwiseRanker:
    CHARACTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
                  "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W"]

    def __init__(self, client, num_child=3, k=20, method="heapsort"):
        self.client = client
        self.num_child = num_child
        self.k = k
        self.method = method

    def compare(self, query, docs):
        system_prompt = (
            "You are RankGPT, an intelligent assistant specialized in selecting the most relevant "
            "passage from a pool of passages based on their relevance to the query."
        )
        passages = "\n\n".join(
            [f'Passage {self.CHARACTERS[i]}: "{truncate_doc(doc.text)}"' for i, doc in enumerate(docs)]
        )
        input_text = (
            f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n'
            + passages
            + "\n\nOutput only one uppercase passage label, such as A. Do not explain."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text},
        ]
        resp = self.client.chat(messages)
        return _parse_setwise_label(resp.content or "", self.CHARACTERS[: len(docs)])

    def heapify(self, arr, n, i, query):
        if self.num_child * i + 1 < n:
            docs = [arr[i]] + arr[self.num_child * i + 1: min((self.num_child * (i + 1) + 1), n)]
            inds = [i] + list(range(self.num_child * i + 1, min((self.num_child * (i + 1) + 1), n)))
            output = self.compare(query, docs)
            try:
                best_ind = self.CHARACTERS.index(output)
            except ValueError:
                best_ind = 0
            try:
                largest = inds[best_ind]
            except IndexError:
                largest = i
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self.heapify(arr, n, largest, query)

    def heapSort(self, arr, query, k):
        n = len(arr)
        ranked = 0
        for i in range(n // self.num_child, -1, -1):
            self.heapify(arr, n, i, query)
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            self.heapify(arr, i, 0, query)

    def rerank(self, query, docs):
        ranking = copy.deepcopy(docs)
        if self.method == "heapsort":
            self.heapSort(ranking, query, self.k)
            ranking = list(reversed(ranking))
        elif self.method == "bubblesort":
            last_start = len(ranking) - (self.num_child + 1)
            for i in range(self.k):
                start_ind = last_start
                end_ind = last_start + (self.num_child + 1)
                is_change = False
                while True:
                    if start_ind < i:
                        start_ind = i
                    output = self.compare(query, ranking[start_ind:end_ind])
                    try:
                        best_ind = self.CHARACTERS.index(output)
                    except ValueError:
                        best_ind = 0
                    if best_ind != 0:
                        ranking[start_ind], ranking[start_ind + best_ind] = (
                            ranking[start_ind + best_ind],
                            ranking[start_ind],
                        )
                        if not is_change:
                            is_change = True
                            if last_start != len(ranking) - (self.num_child + 1) and best_ind == len(
                                ranking[start_ind:end_ind]
                            ) - 1:
                                last_start += len(ranking[start_ind:end_ind]) - 1

                    if start_ind == i:
                        break

                    if not is_change:
                        last_start -= self.num_child

                    start_ind -= self.num_child
                    end_ind -= self.num_child
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
    ranker = SetwiseRanker(client, method="heapsort")
    result_file, data_file = get_trec_paths("setwise_heapsort.jsonl")
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
                docs = [Documents(doc_ids[j], truncate_doc(contents[j])) for j in range(len(doc_ids))]
                docs_ranked = ranker.rerank(query, docs)
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
