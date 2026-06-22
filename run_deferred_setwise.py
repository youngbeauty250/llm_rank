"""收尾阶段：统一补跑之前缓下来的所有 setwise 实验。

按用户要求,setwise(最慢)被推迟到所有其他实验跑完后再补。本脚本一次性跑完三处 setwise:
  1. BEIR setwise        —— run_beir_experiments.py --methods setwise_heapsort
  2. generalization setwise —— run_model_baselines.py --methods setwise_heapsort
                               (qwen3-32b 的 setwise 已完成,会自动跳过;只补 deepseek-v4-flash)
  3. 输入顺序图的 setwise   —— run_paper_experiments.py --only inputorder --include-setwise

全部可断点续跑;失败行需先各自清理(沿用既有套路)。串行执行,避免抢 key。
用法: cd LLM_Rank && <py> run_deferred_setwise.py
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

PHASES = [
    ("BEIR setwise",
     [PY, "-u", "run_beir_experiments.py", "--methods", "setwise_heapsort", "--no-preflight"]),
    ("generalization setwise",
     [PY, "-u", "run_model_baselines.py", "--models", "qwen3-32b", "deepseek-v4-flash",
      "--methods", "setwise_heapsort", "--no-preflight"]),
    ("input-order setwise",
     [PY, "-u", "run_paper_experiments.py", "--only", "inputorder", "--include-setwise"]),
]


def main() -> int:
    rc_all = 0
    for name, cmd in PHASES:
        print(f"\n########## 收尾 setwise 阶段: {name} ##########", flush=True)
        print("  " + " ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        print(f"  [{name}] rc={rc}", flush=True)
        rc_all = rc_all or rc
    print(f"\n[FINISHED] 所有缓跑 setwise 完成 (rc={rc_all})", flush=True)
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
