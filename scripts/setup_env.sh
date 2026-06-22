#!/usr/bin/env bash
# 安装 vLLM 0.10 + LLM_Rank 实验依赖。建议 Python 3.10/3.11, CUDA 12.x。
# 用法: bash scripts/setup_env.sh
set -e
pip install -U pip
pip install "vllm==0.10.*"
pip install "openai>=1.0" pyyaml tqdm numpy matplotlib
# 评测用的 pytrec_eval(由 pytrec-eval-terrier 提供 pytrec_eval 模块)
pip install pytrec-eval-terrier
echo "环境就绪。验证: python -c 'import vllm, openai, pytrec_eval; print(\"ok\")'"
