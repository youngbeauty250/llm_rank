#!/usr/bin/env bash
# 全自动跑完所有模型的 generalization 实验(TREC dl19/dl20 × 6 方法)。
# 逐个模型: 起 vLLM → 等 /health 就绪 → 跑实验 → 评测 → 关服务 → 下一个。
# 用法: bash scripts/run_all.sh [tag1 tag2 ...]   # 不带参数=全部5个模型
#   可选: BEIR=1 也跑 7 个 BEIR 数据集; PY=python3 指定解释器
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
PY=${PY:-python}
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(qwen3-0.6b qwen3-4b qwen3-8b llama-3.1-8b qwen3-32b)
BEIR_FLAG=""; [ "${BEIR:-0}" = "1" ] && BEIR_FLAG="--beir"

for TAG in "${MODELS[@]}"; do
  echo "================ $TAG ================"
  bash scripts/serve_vllm.sh "$TAG" > "logs_serve_${TAG}.log" 2>&1 &
  VLLM_PID=$!
  echo "等待 vLLM 就绪(最多 ~10min)..."
  ready=0
  for i in $(seq 1 120); do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then ready=1; echo "vLLM 就绪"; break; fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vLLM 启动失败,见 logs_serve_${TAG}.log"; exit 1; fi
    sleep 5
  done
  [ "$ready" = 1 ] || { echo "vLLM 就绪超时"; kill "$VLLM_PID" 2>/dev/null; exit 1; }

  $PY run_local_vllm.py --models "$TAG" --datasets dl19 dl20 $BEIR_FLAG --no-preflight
  echo "----- $TAG NDCG -----"
  $PY evaluate.py --bench trec --datasets dl19 dl20 --model "$TAG" || true

  kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null || true
  sleep 3
done
echo "全部完成。结果: results/TREC_results/<ds>/models/<tag>/  | 汇总: bash scripts/collect_results.sh"
