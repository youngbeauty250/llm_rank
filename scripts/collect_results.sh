#!/usr/bin/env bash
# 汇总各本地模型的 NDCG@10(填 generalization 表用)。
# 用法: bash scripts/collect_results.sh [tag1 tag2 ...]   # 不带参数=全部5个
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
PY=${PY:-python}
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(qwen3-0.6b qwen3-4b qwen3-8b llama-3.1-8b qwen3-32b)
for TAG in "${MODELS[@]}"; do
  echo "==================== $TAG ===================="
  $PY evaluate.py --bench trec --datasets dl19 dl20 --model "$TAG"
done
