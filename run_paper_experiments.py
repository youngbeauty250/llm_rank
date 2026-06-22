"""论文剩余实验编排器：消融(ablation) / 输入顺序(input order) / 剪枝扫描(pruning)。

这些都用现有 ranker + 环境变量开关跑，结果靠 RUN_TAG 写到
results/TREC_results/<ds>/variants/<tag>/，与主结果、generalization 完全隔离。

设计要点：
- 顺序执行(一次一个 ranker 子进程, 子进程内部 max_workers=8)，避免和正在跑的任务抢 china key。
- --wait-for-beir：先轮询等 run_beir_experiments 退出(china key 腾出)再开跑；
  generalization 在 qwen/deepseek key 上并行不受影响。
- 断点续跑：ranker 自身按 load_done_qids 跳过已完成 qid；本编排器按
  (结果行+跳过行)>=数据行 且无失败 决定整体跳过(与 run_beir_experiments 同构)。
- 用法：
    <py> run_paper_experiments.py                       # 跑全部 job
    <py> run_paper_experiments.py --wait-for-beir       # 等 BEIR 完成后再跑
    <py> run_paper_experiments.py --only ablation       # 只跑某类
    <py> run_paper_experiments.py --dry-run             # 只打印 job 与完成度,不调用 API
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
LOG_ROOT = ROOT / "results" / "logs" / "paper"

sys.path.insert(0, str(RANK_FUC))
from run_beir_experiments import METHODS, count_jsonl, result_has_failures  # noqa: E402

ALL_METHODS = list(METHODS)
NONSETWISE_METHODS = [m for m in ALL_METHODS if m != "setwise_heapsort"]


def data_file_for(dataset: str) -> Path:
    year = "19" if dataset == "dl19" else "20"
    return TREC_DATA_ROOT / dataset / f"trec{year}_bm25_top100.jsonl"


def _job(category, tag, env, dataset, methods, desc):
    return {"category": category, "tag": tag, "env": env, "dataset": dataset,
            "methods": methods, "desc": desc}


def build_jobs(include_setwise: bool = False) -> list[dict]:
    # setwise 默认缓到最后单独跑(--include-setwise 时才纳入输入顺序图的 setwise 点)。
    io_methods = ALL_METHODS if include_setwise else NONSETWISE_METHODS
    jobs = []
    # ---------- 1) 消融表 (TREC DL19, qwen3-8b, swiss_choice 变体) ----------
    # full = skip+posterior(主方法参照); 其余四个是 w/o Skip Group / w/o Posterior Correction。
    ablation = [
        ("ablation_full",        {}, "SwissRank (full: skip + posterior)"),
        ("ablation_chunkgroup",  {"SWISS_GROUPING": "chunk"},  "w/o Skip Group: Chunk Group"),
        ("ablation_randomgroup", {"SWISS_GROUPING": "random"}, "w/o Skip Group: Random Group"),
        ("ablation_fixedseed",   {"SWISS_SEED": "fixed"},      "w/o Posterior Correction: Fixed Seed"),
        ("ablation_randomseed",  {"SWISS_SEED": "random"},     "w/o Posterior Correction: Random Seed"),
    ]
    for tag, env, desc in ablation:
        jobs.append(_job("ablation", tag, env, "dl19", ["swiss_choice"], desc))

    # ---------- 2) 输入顺序图 (TREC DL19, 全部 6 方法, 逆序/随机) ----------
    # bm25 顺序 = 主结果,直接复用,不在此重跑。
    for order in ("inverse", "random"):
        jobs.append(_job("inputorder", f"inputorder_{order}",
                         {"INPUT_ORDER": order}, "dl19", io_methods,
                         f"Input order = {order} BM25 ({len(io_methods)} methods)"))

    # ---------- 3) 剪枝阈值扫描 (TREC DL19, swiss_choice, min_s 扫描) ----------
    # max_failure_num 越负→低分组持续对战越久→token 越多。-1 ≈ 主方法默认。
    for mf in (0, -1, -2, -3):
        jobs.append(_job("pruning", f"pruning_mf{mf}",
                         {"SWISS_MAX_FAILURE": str(mf)}, "dl19", ["swiss_choice"],
                         f"Pruning sweep: SWISS_MAX_FAILURE={mf}"))
    return jobs


def result_file_for(job, method) -> Path:
    _, result_name = METHODS[method]
    return TREC_RESULT_ROOT / job["dataset"] / "variants" / job["tag"] / result_name


def run_method(job, method, force=False) -> tuple[str, int]:
    script, _ = METHODS[method]
    dataset = job["dataset"]
    result_file = result_file_for(job, method)
    skipped_file = result_file.with_name(result_file.name + ".skipped.jsonl")
    expected = count_jsonl(data_file_for(dataset))
    done = count_jsonl(result_file) + count_jsonl(skipped_file)

    if not force and done >= expected and not result_has_failures(result_file):
        print(f"[SKIP] {job['tag']}/{dataset}/{method}: {done}/{expected} complete", flush=True)
        return ("skip", 0)

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_file = LOG_ROOT / f"{job['tag']}_{dataset}_{method}.log"
    env = os.environ.copy()
    env["TREC_DATASET"] = dataset
    env.pop("BEIR_DATASET", None)
    env.pop("MODEL_TAG", None)            # 确保不串到 models/ 路径
    env["RUN_TAG"] = job["tag"]
    # 先清掉可能残留的变体开关,再按本 job 设定,避免跨 job 串味。
    for k in ("SWISS_GROUPING", "SWISS_SEED", "SWISS_MAX_FAILURE", "INPUT_ORDER"):
        env.pop(k, None)
    env.update(job["env"])

    cmd = [sys.executable, "-u", script]
    print(f"[RUN] {job['tag']}/{dataset}/{method}: {done}/{expected} -> {result_file}", flush=True)
    start = time.perf_counter()
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n===== START {time.strftime('%Y-%m-%d %H:%M:%S')} {job['tag']}/{dataset}/{method} "
                  f"env={job['env']} =====\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=RANK_FUC, env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start
        log.write(f"===== END rc={proc.returncode} elapsed={elapsed:.1f}s =====\n")
    print(f"[DONE] {job['tag']}/{dataset}/{method}: rc={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
    return ("run", proc.returncode)


def beir_running() -> bool:
    try:
        out = subprocess.run(["pgrep", "-f", "run_beir_experiments"],
                             capture_output=True, text=True)
        return out.returncode == 0 and out.stdout.strip() != ""
    except Exception:
        return False


def wait_for_beir(poll_seconds: int = 120) -> None:
    print(f"[WAIT] 等待 run_beir_experiments 退出后再开跑(每{poll_seconds}s 轮询)...", flush=True)
    while beir_running():
        time.sleep(poll_seconds)
    print("[WAIT] BEIR 已退出, china key 腾出, 开始论文剩余实验。", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None,
                        help="只跑这些类别: ablation / inputorder / pruning")
    parser.add_argument("--wait-for-beir", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-setwise", action="store_true",
                        help="把输入顺序图的 setwise 点也纳入(默认缓到最后)")
    args = parser.parse_args()

    jobs = build_jobs(include_setwise=args.include_setwise)
    if args.only:
        jobs = [j for j in jobs if j["category"] in args.only]

    print(f"共 {len(jobs)} 个 job, 类别分布: "
          f"{ {c: sum(1 for j in jobs if j['category']==c) for c in sorted({j['category'] for j in jobs})} }",
          flush=True)
    if args.dry_run:
        for j in jobs:
            for m in j["methods"]:
                rf = result_file_for(j, m)
                exp = count_jsonl(data_file_for(j["dataset"]))
                done = count_jsonl(rf) + count_jsonl(rf.with_name(rf.name + ".skipped.jsonl"))
                fail = "FAIL" if result_has_failures(rf) else ""
                print(f"  [{j['category']}] {j['tag']}/{j['dataset']}/{m}: {done}/{exp} {fail}  ({j['desc']})")
        return 0

    if args.wait_for_beir:
        wait_for_beir()

    failures = []
    for j in jobs:
        print(f"\n##### JOB {j['tag']} ({j['desc']}) #####", flush=True)
        for m in j["methods"]:
            kind, rc = run_method(j, m, force=args.force)
            if rc != 0:
                failures.append((j["tag"], j["dataset"], m, rc))
    if failures:
        print(f"[FINISHED_WITH_FAILURES] {failures}", flush=True)
        return 1
    print("[FINISHED] all paper variant experiments completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
