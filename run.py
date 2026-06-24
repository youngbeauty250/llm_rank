#!/usr/bin/env python
"""LLM_Rank 统一入口：用一条命令在 TREC / BEIR 数据集上跑任意一个排序方法。

每个方法本身是 rank_fuc/ 下的一个独立脚本，通过环境变量约定输入/输出路径：
  - TREC_DATASET = dl19 | dl20      （跑 TREC DL 时）
  - BEIR_DATASET = <数据集名>         （跑 BEIR 时，设了它就走 BEIR 分支）
本入口只是把命令行参数翻译成这些环境变量，再调用对应方法脚本，省去手工 export。

用法示例：
    # TREC DL19 上跑 SwissRank
    python run.py --method swiss_choice --bench trec --dataset dl19

    # BEIR scifact 上跑 RankGPT
    python run.py --method rankgpt --bench beir --dataset scifact

    # 冒烟测试：只跑前 2 条 query（验证环境/接口是否通）
    SMOKE_TEST=1 python run.py --method swiss_choice --bench trec --dataset dl19

结果写到 results/{TREC,BEIR}_results/<dataset>/<method>.jsonl（断点续跑，自动跳过已完成 qid）。
跑完用 evaluate.py 算 NDCG：
    python evaluate.py --bench trec --datasets dl19 dl20
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RANK_FUC = ROOT / "rank_fuc"

# 方法名 -> rank_fuc/ 下的脚本文件名
METHODS = {
    "rankgpt": "rankgpt.py",
    "swiss_choice": "swissrank_choice.py",   # 本论文方法 SwissRank
    "tourrank": "tourrank.py",
    "blitzrank": "blitzrank.py",
    "bracketrank": "bracketrank.py",
    "setwise_heapsort": "setwise.py",
    "pairwise": "pairwise.py",
    "tourrank_n": "tourrank-n.py",
}

TREC_DATASETS = ["dl19", "dl20"]
BEIR_DATASETS = [
    "trec-covid", "webis-touche2020", "dbpedia-entity", "scifact",
    "signal1m", "trec-news", "robust04", "nfcorpus",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--method", required=True, choices=list(METHODS),
                        help="要运行的排序方法")
    parser.add_argument("--bench", required=True, choices=["trec", "beir"],
                        help="基准：trec(TREC DL) 或 beir(BEIR)")
    parser.add_argument("--dataset", required=True,
                        help="数据集：trec 用 dl19/dl20；beir 用 scifact/trec-covid 等")
    args = parser.parse_args()

    if args.bench == "trec" and args.dataset not in TREC_DATASETS:
        raise SystemExit(f"trec 基准的 --dataset 应为 {TREC_DATASETS}，收到 {args.dataset!r}")
    if args.bench == "beir" and args.dataset not in BEIR_DATASETS:
        raise SystemExit(f"beir 基准的 --dataset 应为 {BEIR_DATASETS}，收到 {args.dataset!r}")

    # 通过环境变量告诉方法脚本该读哪个数据集（与 rank_fuc/utils.py 的约定一致）。
    if args.bench == "trec":
        os.environ["TREC_DATASET"] = args.dataset
        os.environ.pop("BEIR_DATASET", None)
    else:
        os.environ["BEIR_DATASET"] = args.dataset
        os.environ.pop("TREC_DATASET", None)

    # 方法脚本之间用相对 import（如 swissrank_choice 依赖 swissrank、utils），
    # 因此切到 rank_fuc/ 目录并把它加入 sys.path，再以 __main__ 方式执行目标脚本。
    script = RANK_FUC / METHODS[args.method]
    os.chdir(RANK_FUC)
    sys.path.insert(0, str(RANK_FUC))
    print(f"[run] method={args.method} bench={args.bench} dataset={args.dataset} -> {script.name}",
          flush=True)
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
