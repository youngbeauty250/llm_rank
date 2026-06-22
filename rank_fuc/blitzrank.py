"""BlitzRank：tournament graph top-m reranking, adapted to the LLM_Rank runner.

This wrapper reuses the local BlitzRank tournament-graph scheduler, but sends all
LLM calls through rank_fuc.llm.LLMClient so token accounting and API config stay
consistent with the other baselines.
"""

from __future__ import annotations

import copy
import json
import os

import networkx as nx
from tqdm import tqdm

from rankgpt import receive_permutation
from swissrank import get_post_prompt, get_prefix_prompt
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


class LLMRankOracle:
    def __init__(self, client, query: str, doc_contents: dict[str, str]):
        self.client = client
        self.query = query
        self.doc_contents = doc_contents
        self.call_logs: list[dict] = []

    def _build_messages(self, doc_ids: list[str]) -> list[dict]:
        messages = get_prefix_prompt(self.query, len(doc_ids))
        for j, doc_id in enumerate(doc_ids):
            content = truncate_doc(self.doc_contents[doc_id])
            messages.append({"role": "user", "content": f"Document {j + 1}: {content}"})
            messages.append({"role": "assistant", "content": f"Received Document {j + 1}."})
        messages.append({"role": "user", "content": get_post_prompt(self.query, len(doc_ids))})
        return messages

    def compare(self, doc_ids: list[str]) -> list[tuple[str, str]]:
        before = self.client.get_usage().total_tokens
        resp = self.client.chat(self._build_messages(doc_ids))
        after = self.client.get_usage().total_tokens
        ranked_doc_ids = receive_permutation(
            doc_ids, resp.content or "", 0, len(doc_ids)
        )
        call_log = {
            "usage_total_tokens": resp.usage_total_tokens,
            "tracked_total_tokens": after - before,
        }
        self.call_logs.append(call_log)
        return [(ranked_doc_ids[i], ranked_doc_ids[i + 1]) for i in range(len(ranked_doc_ids) - 1)]


def _graph_state(graph):
    """Match BlitzRank's tournament graph state: SCCs + reachability in G."""
    condensation = nx.condensation(graph)
    scc_membership = condensation.graph["mapping"]
    scc_members = {
        scc_idx: condensation.nodes[scc_idx]["members"]
        for scc_idx in condensation.nodes()
    }
    in_reach, out_reach, known = {}, {}, {}
    for doc_id in graph.nodes():
        descendants = nx.descendants(graph, doc_id)
        ancestors = nx.ancestors(graph, doc_id)
        same_scc_others = len(scc_members[scc_membership[doc_id]]) - 1
        out_reach[doc_id] = len(descendants) - same_scc_others
        in_reach[doc_id] = len(ancestors) - same_scc_others
        known[doc_id] = len(ancestors | descendants)
    doc_ids = list(graph.nodes())
    sorted_nodes = sorted(doc_ids, key=lambda node: (in_reach[node], out_reach[node]))
    return sorted_nodes, known, scc_membership


def filter_processing(client, query, doc_ids, doc_contents, params):
    if len(doc_ids) <= 1:
        return copy.deepcopy(doc_ids)
    top_m = min(params.get("top_m", 10), len(doc_ids))
    window_size = min(params.get("window_size", 20), len(doc_ids))
    max_num_rounds = params.get("max_num_rounds", 50)

    graph = nx.DiGraph()
    graph.add_nodes_from(doc_ids)
    oracle = LLMRankOracle(client, query, doc_contents)
    nodes_for_match = doc_ids[:window_size]
    previous_schedule = None

    for _ in range(max_num_rounds):
        for winner, loser in oracle.compare(nodes_for_match):
            if winner != loser:
                graph.add_edge(winner, loser)
        sorted_nodes, known, scc_membership = _graph_state(graph)
        top_doc_ids = sorted_nodes[:top_m]
        if all(known[doc_id] >= len(doc_ids) - 1 for doc_id in top_doc_ids):
            return top_doc_ids + [doc_id for doc_id in doc_ids if doc_id not in set(top_doc_ids)]

        seen_sccs = set()
        next_match = []
        for doc_id in sorted_nodes:
            if known[doc_id] >= len(doc_ids) - 1:
                continue
            scc = scc_membership[doc_id]
            if scc in seen_sccs:
                continue
            seen_sccs.add(scc)
            next_match.append(doc_id)
            if len(next_match) >= window_size:
                break

        if len(next_match) < 2 or set(next_match) == previous_schedule:
            break
        previous_schedule = set(next_match)
        nodes_for_match = next_match

    sorted_nodes, _, _ = _graph_state(graph)
    return sorted_nodes + [doc_id for doc_id in doc_ids if doc_id not in set(sorted_nodes)]


def main():
    params = {
        "window_size": 20,
        "top_m": 10,
        "max_num_rounds": 50,
    }
    set_seed(42)
    client = get_default_client()
    result_file, data_file = get_trec_paths("blitzrank.jsonl")
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
