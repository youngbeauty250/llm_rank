"""本地 vLLM 多模型实验编排器：对本机 vLLM(OpenAI 兼容)服务上的模型,
跑 generalization 实验(TREC DL19/DL20 × 6 方法),结果写 models/<tag>/。

设计与 run_model_baselines.py 同构,但不依赖 tokenkey/MODEL_PROVIDER:
- 所有模型走同一个本地端点(默认 http://localhost:8000/v1)。
- LLM_MODEL = MODEL_TAG = 模型 tag(需与 vLLM 的 --served-model-name 一致)。
- qwen3 系列自动用 LLM_THINK_STYLE=vllm(只发 chat_template_kwargs);llama 等非 qwen3 不发思考参数。
- 断点续跑、按 (结果行+跳过行)>=数据行 且无失败 整体跳过。

用法(先用 scripts/serve_vllm.sh 起好对应模型的服务):
    <py> run_local_vllm.py --models qwen3-0.6b              # 单模型, TREC dl19/dl20 × 6方法
    <py> run_local_vllm.py --models qwen3-4b --beir         # 额外跑 BEIR 7集
    <py> run_local_vllm.py --models qwen3-0.6b --datasets dl19 --methods swiss_choice rankgpt
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RANK_FUC = ROOT / "rank_fuc"
TREC_DATA = ROOT / "data" / "TREC_data"
TREC_RES = ROOT / "results" / "TREC_results"
BEIR_DATA = ROOT / "data" / "BEIR_data"
BEIR_RES = ROOT / "results" / "BEIR_results"
LOG_ROOT = ROOT / "results" / "logs" / "local"

sys.path.insert(0, str(RANK_FUC))
from llm.llm_client import LLMClient  # noqa: E402
from run_beir_experiments import METHODS, count_jsonl, result_has_failures  # noqa: E402

# 用户在新卡上准备的模型(tag 需与 vLLM --served-model-name 一致)。
DEFAULT_MODELS = ["qwen3-0.6b", "qwen3-4b", "qwen3-8b", "llama-3.1-8b", "qwen3-32b"]
PAPER_BEIR = ["trec-covid", "webis-touche2020", "dbpedia-entity", "scifact",
              "signal1m", "trec-news", "robust04"]


def trec_data_file(ds: str) -> Path:
    return TREC_DATA / ds / f"trec{'19' if ds == 'dl19' else '20'}_bm25_top100.jsonl"


def result_paths(bench: str, ds: str, model: str, method: str):
    _, rname = METHODS[method]
    if bench == "trec":
        rf = TREC_RES / ds / "models" / model / rname
        data = trec_data_file(ds)
    else:
        rf = BEIR_RES / ds / "models" / model / rname
        data = BEIR_DATA / ds / f"{ds}_bm25_top100.jsonl"
    return rf, rf.with_name(rf.name + ".skipped.jsonl"), data


def base_env(model: str, api_url: str, api_key: str) -> dict:
    env = os.environ.copy()
    env["LLM_API_URL"] = api_url
    env["LLM_API_KEY"] = api_key or "EMPTY"
    env["LLM_MODEL"] = model
    env["MODEL_TAG"] = model
    env["LLM_ENABLE_THINKING"] = "false"
    env["LLM_THINK_STYLE"] = "vllm"     # qwen3 只发 chat_template_kwargs
    return env


def preflight(model: str, api_url: str, api_key: str) -> bool:
    client = LLMClient(model_name=model, api_url=api_url, api_key=api_key or "EMPTY",
                       max_tokens=16, max_retry=1, max_workers=1)
    os.environ["LLM_THINK_STYLE"] = "vllm"
    client.chat([{"role": "system", "content": "health check"},
                 {"role": "user", "content": "Reply OK."}], max_retry=1)
    if client.get_usage().failed_api_requests:
        print(f"[PREFLIGHT_FAILED] model={model} @ {api_url} 不可用,先确认 vLLM 已起且 served-model-name={model}", flush=True)
        return False
    print(f"[PREFLIGHT_OK] model={model} @ {api_url}", flush=True)
    return True


def run_one(bench, ds, model, method, api_url, api_key, force=False) -> int:
    script, _ = METHODS[method]
    rf, sk, data = result_paths(bench, ds, model, method)
    expected = count_jsonl(data)
    done = count_jsonl(rf) + count_jsonl(sk)
    if not force and done >= expected and not result_has_failures(rf):
        print(f"[SKIP] {model}/{bench}:{ds}/{method}: {done}/{expected}", flush=True)
        return 0
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_file = LOG_ROOT / f"{model}_{ds}_{method}.log"
    env = base_env(model, api_url, api_key)
    if bench == "trec":
        env["TREC_DATASET"] = ds; env.pop("BEIR_DATASET", None)
    else:
        env["BEIR_DATASET"] = ds; env.pop("TREC_DATASET", None)
    print(f"[RUN] {model}/{bench}:{ds}/{method}: {done}/{expected} -> {rf}", flush=True)
    start = time.perf_counter()
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n===== START {time.strftime('%Y-%m-%d %H:%M:%S')} {model}/{ds}/{method} =====\n")
        log.flush()
        proc = subprocess.run([sys.executable, "-u", script], cwd=RANK_FUC, env=env,
                              stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    print(f"[DONE] {model}/{bench}:{ds}/{method}: rc={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
    return proc.returncode


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    p.add_argument("--datasets", nargs="*", default=["dl19", "dl20"])
    p.add_argument("--methods", nargs="*", default=list(METHODS))
    p.add_argument("--beir", action="store_true", help="额外跑 7 个 BEIR 数据集")
    p.add_argument("--api-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-preflight", action="store_true")
    args = p.parse_args()

    unknown = [m for m in args.methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}; valid={list(METHODS)}")

    failures = []
    for model in args.models:
        if not args.no_preflight and not preflight(model, args.api_url, args.api_key):
            failures.append((model, "*", "preflight")); continue
        for ds in args.datasets:
            for m in args.methods:
                rc = run_one("trec", ds, model, m, args.api_url, args.api_key, args.force)
                if rc != 0: failures.append((model, ds, m))
        if args.beir:
            for ds in PAPER_BEIR:
                for m in args.methods:
                    rc = run_one("beir", ds, model, m, args.api_url, args.api_key, args.force)
                    if rc != 0: failures.append((model, ds, m))
    if failures:
        print(f"[FINISHED_WITH_FAILURES] {failures}", flush=True); return 1
    print("[FINISHED] all local vLLM runs completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
