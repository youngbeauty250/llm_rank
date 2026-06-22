"""跑多模型 baseline 对比：deepseek-r1 等强模型 × TREC dl19/dl20 × 6 方法。

与 run_beir_experiments.py 同构，但多了「模型」维度，且固定在 TREC 主表数据集上跑：
- 结果写到 results/TREC_results/<ds>/models/<model>/<method>.jsonl（靠 MODEL_TAG 子目录
  隔离），不污染默认 qwen3-8b 的主结果。
- 断点续跑：结果行 + 跳过清单行 >= 数据行 即整体跳过（与 BEIR runner 一致）。
- 每个模型开跑前做一次 preflight，失败则跳过该模型、不影响其它模型。
- deepseek-r1 是推理模型，think 吃 token，单独放大 max_tokens 避免截断。

用法：
    <py> run_model_baselines.py                       # 全部 4 模型 × dl19/dl20 × 6 方法
    <py> run_model_baselines.py --models deepseek-r1  # 只跑某个模型
    <py> run_model_baselines.py --datasets dl19 --methods swiss_choice rankgpt
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
TREC_DATA_ROOT = ROOT / "data" / "TREC_data"
TREC_RESULT_ROOT = ROOT / "results" / "TREC_results"
LOG_ROOT = ROOT / "results" / "logs" / "models"

sys.path.insert(0, str(RANK_FUC))
from llm.llm_client import LLMClient  # noqa: E402
from run_beir_experiments import METHODS, count_jsonl, result_has_failures  # noqa: E402

# 泛化实验确认用 qwen3-32b(同家族大规模) + deepseek-v3(跨家族)，二者均经 DashScope
# 健康探测稳定可用；tokenkey 聚合代理实测持续返回 429 "no available accounts"，已弃用。
# qwen-max 同样可用，需要时手动 --models qwen-max 追加。
DEFAULT_MODELS = ["qwen3-32b", "deepseek-v3"]
DATASETS = ["dl19", "dl20"]
# 推理模型 think 阶段吃 token，默认 8192 易把最终排序截断，单独放大。
MODEL_MAX_TOKENS = {"deepseek-r1": "16384"}
# 某些模型跳过特别贵的方法。deepseek-r1 单 query ~170s，setwise 每 query 108 次调用，
# 全跑要数天，性价比太低 → r1 只跑轻方法。deepseek-v4-flash 快，可跑全部 6 方法。
MODEL_SKIP_METHODS = {"deepseek-r1": {"setwise_heapsort"}}

# 多 provider：tokenkey 聚合代理按模型家族用不同 sk（见 ~/.claude/CLAUDE.md）。
# 这里把模型映射到 (api_url, api_key)；未列出的模型走 config.yaml 默认(DashScope)。
TOKENKEY = "https://api.tokenkey.dev/v1"
_SK = {
    # tokenkey 各家族 key 改为从环境变量读取(勿把真实 key 提交仓库)。本地跑模型用不到,
    # 走 config.yaml 默认即可;用 tokenkey 聚合代理时设置对应环境变量。
    "deepseek": os.environ.get("TOKENKEY_DEEPSEEK_SK", ""),
    "qwen": os.environ.get("TOKENKEY_QWEN_SK", ""),
    "gpt": os.environ.get("TOKENKEY_GPT_SK", ""),
    "china": os.environ.get("TOKENKEY_CHINA_SK", ""),
}
MODEL_PROVIDER = {
    # qwen3-32b 现已可在 tokenkey 跑(用 qwen key,与 BEIR 的 china key 隔离避免抢额度)。
    "qwen3-32b": (TOKENKEY, _SK["qwen"]),
    "deepseek-v4-flash": (TOKENKEY, _SK["deepseek"]),
    "deepseek-v4-pro": (TOKENKEY, _SK["deepseek"]),
    "qwen3.7-max": (TOKENKEY, _SK["qwen"]),
    "qwen3.7-plus": (TOKENKEY, _SK["qwen"]),
    "qwen3.6-flash": (TOKENKEY, _SK["qwen"]),
    "doubao-seed-2-0-pro-260215": (TOKENKEY, _SK["china"]),
    "glm-4-7-251222": (TOKENKEY, _SK["china"]),
    "gpt-5.4-mini": (TOKENKEY, _SK["gpt"]),
    "gpt-5.4": (TOKENKEY, _SK["gpt"]),
}


def provider_for(model: str):
    """返回 (api_url, api_key)；None 表示用 config.yaml 默认(DashScope)。"""
    return MODEL_PROVIDER.get(model)


def data_file_for(dataset: str) -> Path:
    year = "19" if dataset == "dl19" else "20"
    return TREC_DATA_ROOT / dataset / f"trec{year}_bm25_top100.jsonl"


def result_paths(dataset: str, model: str, method: str) -> tuple[Path, Path]:
    _, result_name = METHODS[method]
    result_file = TREC_RESULT_ROOT / dataset / "models" / model / result_name
    skipped_file = result_file.with_name(result_file.name + ".skipped.jsonl")
    return result_file, skipped_file


def preflight(model: str) -> bool:
    # 推理模型给大一点 max_tokens，否则 think 占满后 content 为空但其实接口可用。
    max_tokens = int(MODEL_MAX_TOKENS.get(model, 64))
    kwargs = dict(model_name=model, max_tokens=max_tokens, max_retry=1, max_workers=1)
    prov = provider_for(model)
    if prov:
        kwargs["api_url"], kwargs["api_key"] = prov
    client = LLMClient(**kwargs)
    client.chat(
        [
            {"role": "system", "content": "You are a health check assistant."},
            {"role": "user", "content": "Reply OK."},
        ],
        max_retry=1,
    )
    usage = client.get_usage()
    # 只要接口调用没失败就算可用（content 是否为空对推理模型不可靠）。
    if usage.failed_api_requests:
        print(f"[PREFLIGHT_FAILED] model={model} API not usable; skip this model.", flush=True)
        return False
    print(f"[PREFLIGHT_OK] model={model} usable.", flush=True)
    return True


def run_one(dataset: str, model: str, method: str, force: bool = False) -> int:
    script, _ = METHODS[method]
    data_file = data_file_for(dataset)
    result_file, skipped_file = result_paths(dataset, model, method)
    expected_rows = count_jsonl(data_file)
    done_rows = count_jsonl(result_file) + count_jsonl(skipped_file)

    if not force and done_rows >= expected_rows and not result_has_failures(result_file):
        print(f"[SKIP] {model}/{dataset}/{method}: {done_rows}/{expected_rows} complete", flush=True)
        return 0

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_file = LOG_ROOT / f"{model}_{dataset}_{method}.log"
    env = os.environ.copy()
    env["TREC_DATASET"] = dataset
    env.pop("BEIR_DATASET", None)
    env["LLM_MODEL"] = model
    env["MODEL_TAG"] = model
    if model in MODEL_MAX_TOKENS:
        env["LLM_MAX_TOKENS"] = MODEL_MAX_TOKENS[model]
    prov = provider_for(model)
    if prov:
        env["LLM_API_URL"], env["LLM_API_KEY"] = prov

    cmd = [sys.executable, "-u", script]
    print(f"[RUN] {model}/{dataset}/{method}: {done_rows}/{expected_rows} -> {result_file}", flush=True)
    start = time.perf_counter()
    with log_file.open("a", encoding="utf-8") as log:
        log.write(
            f"\n===== START {time.strftime('%Y-%m-%d %H:%M:%S')} {model}/{dataset}/{method} =====\n"
        )
        log.flush()
        proc = subprocess.run(cmd, cwd=RANK_FUC, env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start
        log.write(
            f"===== END {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"{model}/{dataset}/{method} rc={proc.returncode} elapsed={elapsed:.1f}s =====\n"
        )
    print(f"[DONE] {model}/{dataset}/{method}: rc={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-preflight", action="store_true")
    args = parser.parse_args()

    unknown = [m for m in args.methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}; valid={list(METHODS)}")

    print(f"Models: {args.models}", flush=True)
    print(f"Datasets: {args.datasets}", flush=True)
    print(f"Methods: {args.methods}", flush=True)
    failures = []
    for model in args.models:
        if not args.no_preflight and not preflight(model):
            failures.append((model, "*", "preflight"))
            continue
        skip_methods = MODEL_SKIP_METHODS.get(model, set())
        for dataset in args.datasets:
            for method in args.methods:
                if method in skip_methods:
                    print(f"[SKIP_METHOD] {model}/{dataset}/{method}: 该模型跳过此方法", flush=True)
                    continue
                rc = run_one(dataset, model, method, force=args.force)
                if rc != 0:
                    failures.append((model, dataset, method))
    if failures:
        print(f"[FINISHED_WITH_FAILURES] {failures}", flush=True)
        return 1
    print("[FINISHED] all requested model baseline runs completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
