"""Run main LLM_Rank methods on all available BEIR datasets.

This runner is intentionally resumable: each ranker already skips qids found in
its result jsonl, and this wrapper skips a method when output row count matches
the dataset row count.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RANK_FUC = ROOT / "rank_fuc"
DATA_ROOT = ROOT / "data" / "BEIR_data"
RESULT_ROOT = ROOT / "results" / "BEIR_results"
LOG_ROOT = ROOT / "results" / "logs" / "beir"

sys.path.insert(0, str(RANK_FUC))
from llm.llm_client import LLMClient  # noqa: E402

METHODS = {
    "rankgpt": ("rankgpt.py", "rankgpt.jsonl"),
    "swiss_choice": ("swissrank_choice.py", "swiss_choice.jsonl"),
    "tourrank": ("tourrank.py", "tourrank.jsonl"),
    "blitzrank": ("blitzrank.py", "blitzrank.jsonl"),
    "bracketrank": ("bracketrank.py", "bracketrank.jsonl"),
    "setwise_heapsort": ("setwise.py", "setwise_heapsort.jsonl"),
}


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def available_datasets() -> list[str]:
    datasets = []
    for path in sorted(DATA_ROOT.glob("*/*_bm25_top100.jsonl")):
        dataset = path.parent.name
        expected = DATA_ROOT / dataset / f"{dataset}_bm25_top100.jsonl"
        if path == expected:
            datasets.append(dataset)
    return datasets


def result_has_failures(path: Path) -> bool:
    # 只统计真正的瞬时失败（超时/429 重试耗尽）。内容审核失败的 query 已被 ranker
    # 整条丢弃、不写入结果文件（记到同名 .skipped.jsonl 旁路），不会进到这里，
    # 因此不会被误判为「需重跑」。
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("failed_api_requests", 0) or row.get("failed_llm_calls", 0):
                return True
    return False


def run_one(dataset: str, method: str, force: bool = False) -> int:
    script, result_name = METHODS[method]
    data_file = DATA_ROOT / dataset / f"{dataset}_bm25_top100.jsonl"
    result_file = RESULT_ROOT / dataset / result_name
    skipped_file = result_file.with_name(result_file.name + ".skipped.jsonl")
    expected_rows = count_jsonl(data_file)
    # 完成度 = 正常写入的结果行 + 因内容审核整条丢弃的 query，二者合计才覆盖全部 query。
    done_rows = count_jsonl(result_file) + count_jsonl(skipped_file)

    if not force and done_rows >= expected_rows and not result_has_failures(result_file):
        print(f"[SKIP] {dataset}/{method}: {done_rows}/{expected_rows} complete", flush=True)
        return 0

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_file = LOG_ROOT / f"{dataset}_{method}.log"
    env = os.environ.copy()
    env["BEIR_DATASET"] = dataset
    env.pop("TREC_DATASET", None)

    cmd = [sys.executable, "-u", script]
    print(f"[RUN] {dataset}/{method}: {done_rows}/{expected_rows} -> {result_file}", flush=True)
    start = time.perf_counter()
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n===== START {time.strftime('%Y-%m-%d %H:%M:%S')} {dataset}/{method} =====\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=RANK_FUC, env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start
        log.write(
            f"===== END {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"{dataset}/{method} rc={proc.returncode} elapsed={elapsed:.1f}s =====\n"
        )
    print(f"[DONE] {dataset}/{method}: rc={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
    return proc.returncode


def preflight_api() -> bool:
    client = LLMClient(max_tokens=8, max_retry=1, max_workers=1)
    resp = client.chat(
        [
            {"role": "system", "content": "You are a health check assistant."},
            {"role": "user", "content": "Reply OK."},
        ],
        max_retry=1,
    )
    usage = client.get_usage()
    if resp.content is None or usage.failed_api_requests:
        print("[PREFLIGHT_FAILED] LLM API is not usable; stop before launching BEIR runs.", flush=True)
        return False
    print("[PREFLIGHT_OK] LLM API is usable.", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=available_datasets())
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true", default=True)
    parser.add_argument("--no-preflight", action="store_true")
    args = parser.parse_args()

    unknown_methods = [m for m in args.methods if m not in METHODS]
    if unknown_methods:
        raise SystemExit(f"Unknown methods: {unknown_methods}; valid={list(METHODS)}")

    if not args.no_preflight and not preflight_api():
        return 2

    print(f"Datasets: {args.datasets}", flush=True)
    print(f"Methods: {args.methods}", flush=True)
    failures = []
    for dataset in args.datasets:
        for method in args.methods:
            rc = run_one(dataset, method, force=args.force)
            if rc != 0:
                failures.append((dataset, method, rc))
                if not args.keep_going:
                    print(f"[STOP] failure: {failures[-1]}", flush=True)
                    return rc
    if failures:
        print(f"[FINISHED_WITH_FAILURES] {failures}", flush=True)
        return 1
    print("[FINISHED] all requested BEIR runs completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
