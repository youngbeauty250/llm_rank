#!/usr/bin/env bash
# 启动 vLLM(OpenAI 兼容)服务,端口 8000。served-model-name = tag(代码用它当 LLM_MODEL)。
# 用法: bash scripts/serve_vllm.sh <tag> [tp_size]
#   tag ∈ {qwen3-0.6b, qwen3-4b, qwen3-8b, llama-3.1-8b, qwen3-32b}
set -e
TAG=${1:?用法: serve_vllm.sh <tag> [tp_size]}
case "$TAG" in
  qwen3-0.6b)   HF=Qwen/Qwen3-0.6B;                 DTP=1 ;;
  qwen3-4b)     HF=Qwen/Qwen3-4B;                   DTP=1 ;;
  qwen3-8b)     HF=Qwen/Qwen3-8B;                   DTP=1 ;;
  llama-3.1-8b) HF=meta-llama/Llama-3.1-8B-Instruct; DTP=1 ;;
  qwen3-32b)    HF=Qwen/Qwen3-32B;                  DTP=2 ;;
  *) echo "未知 tag: $TAG (支持: qwen3-0.6b qwen3-4b qwen3-8b llama-3.1-8b qwen3-32b)"; exit 1 ;;
esac
TP=${2:-$DTP}
echo "启动 vLLM: $HF  (served-model-name=$TAG, tensor-parallel=$TP, port 8000)"
# 排序任务关闭思考(请求侧由 chat_template_kwargs 控制),max-model-len 给足长 prompt(setwise/大窗口)。
exec vllm serve "$HF" \
  --served-model-name "$TAG" \
  --tensor-parallel-size "$TP" \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90
