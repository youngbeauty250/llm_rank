"""通用工具：随机种子、Documents 容器、共享 LLMClient 单例。"""

import json
import os
import random
import time

import numpy as np

from llm import LLMClient

try:
    import torch
except ImportError:  # torch 仅用于可选的 cudnn 确定性，未安装也能跑
    torch = None

_client_singleton: LLMClient | None = None
DOC_MAX_WORDS = 300


def get_default_client() -> LLMClient:
    """进程级 LLMClient 单例，所有 ranker 共享线程池与 token 统计。"""
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = LLMClient()
    return _client_singleton


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def truncate_doc(text: str, max_words: int = DOC_MAX_WORDS) -> str:
    return " ".join(str(text).strip().split()[:max_words])


def now_seconds() -> float:
    return time.perf_counter()


def get_trec_dataset() -> str:
    dataset = os.environ.get("TREC_DATASET", "dl19").strip().lower()
    if dataset not in {"dl19", "dl20"}:
        raise ValueError(f"Unsupported TREC_DATASET={dataset!r}; expected dl19 or dl20.")
    return dataset


def _apply_model_tag(result_name: str) -> str:
    """多模型对比时把结果写到 models/<tag>/ 子目录，避免覆盖默认模型主结果。

    设了 MODEL_TAG 环境变量才生效；空则路径不变（向后兼容 qwen3-8b 主结果）。
    """
    model_tag = os.environ.get("MODEL_TAG", "").strip()
    if not model_tag:
        return result_name
    return f"models/{model_tag}/{result_name}"


def _apply_run_tag(result_name: str) -> str:
    """消融/输入顺序/剪枝等变体实验把输出写到 variants/<tag>/ 子目录，与主结果隔离。

    设了 RUN_TAG 环境变量才生效；空则路径不变（向后兼容主结果与 MODEL_TAG）。
    """
    run_tag = os.environ.get("RUN_TAG", "").strip()
    if not run_tag:
        return result_name
    return f"variants/{run_tag}/{result_name}"


def _input_order_suffix() -> str:
    """输入顺序消融：选派生数据文件后缀。bm25/空=原 BM25；inverse=逆序；random=随机洗牌。"""
    order = os.environ.get("INPUT_ORDER", "").strip().lower()
    if order in ("", "bm25"):
        return ""
    if order in ("inverse", "random"):
        return f"_{order}"
    raise ValueError(f"Unsupported INPUT_ORDER={order!r}; expected bm25|inverse|random.")


def get_trec_paths(result_name: str) -> tuple[str, str]:
    result_name = _apply_run_tag(_apply_model_tag(result_name))
    suffix = _input_order_suffix()
    beir_dataset = os.environ.get("BEIR_DATASET", "").strip()
    if beir_dataset:
        result_file = f"../results/BEIR_results/{beir_dataset}/{result_name}"
        data_file = f"../data/BEIR_data/{beir_dataset}/{beir_dataset}_bm25_top100{suffix}.jsonl"
        return result_file, data_file

    dataset = get_trec_dataset()
    year = "19" if dataset == "dl19" else "20"
    result_file = f"../results/TREC_results/{dataset}/{result_name}"
    data_file = f"../data/TREC_data/{dataset}/trec{year}_bm25_top100{suffix}.jsonl"
    return result_file, data_file


def skipped_path(result_file: str) -> str:
    """内容审核被整条丢弃的 query 记录文件（与结果文件同目录的旁路清单）。"""
    return result_file + ".skipped.jsonl"


def load_done_qids(result_file: str) -> set:
    """已处理 qid 集合：结果文件里写过的 + 跳过清单里丢弃过的，断点续跑时都不再重试。"""
    done = set()
    for path in (result_file, skipped_path(result_file)):
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                done.add(json.loads(line)["qid"])
    return done


def record_skipped_query(result_file: str, qid, query: str, reason: str = "content_filter") -> None:
    """把因内容审核丢弃的 query 追加到跳过清单，供续跑跳过与评测统计。"""
    path = skipped_path(result_file)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"qid": qid, "query": query, "reason": reason}, ensure_ascii=False) + "\n")


def build_result_record(client, query_start_time, **fields) -> dict:
    usage = client.get_usage()
    query_latency = time.perf_counter() - query_start_time
    llm_calls = usage.total_calls
    metrics = {
        "sum_tokens": usage.total_tokens,
        "prompt_tokens": usage.total_prompt_tokens,
        "completion_tokens": usage.total_completion_tokens,
        "llm_calls": llm_calls,
        "failed_llm_calls": usage.failed_calls,
        "content_filter_failed": usage.content_filter_failed,
        "api_requests": usage.total_api_requests,
        "failed_api_requests": usage.failed_api_requests,
        "llm_latency_seconds": usage.total_llm_latency_seconds,
        "avg_llm_call_latency_seconds": (
            usage.total_llm_latency_seconds / usage.total_api_requests
            if usage.total_api_requests
            else 0.0
        ),
        "query_latency_seconds": query_latency,
        "wall_clock_seconds": query_latency,
        "estimated_cost": usage.estimated_cost,
        "cost_currency": usage.cost_currency,
    }
    return {**fields, **metrics}


class Documents:
    def __init__(self, _id, text):
        self._id = _id
        self.text = text

    def __repr__(self):
        return f"Doc(id={self._id}, text={self.text!r})"
