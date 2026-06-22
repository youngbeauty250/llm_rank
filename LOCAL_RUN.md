# 本地 vLLM 跑实验指南（小模型泛化）

用你新卡上的 vLLM 0.10 跑 **跨规模 + 跨家族泛化实验**，填充论文 generalization 表。

## 你有的模型 → 这条泛化曲线
| tag | HF 模型 | 角色 |
|---|---|---|
| `qwen3-0.6b` | Qwen/Qwen3-0.6B | 同家族最小 |
| `qwen3-4b` | Qwen/Qwen3-4B | 同家族小 |
| `qwen3-8b` | Qwen/Qwen3-8B | 主模型规模 |
| `qwen3-32b` | Qwen/Qwen3-32B | 同家族大 |
| `llama-3.1-8b` | meta-llama/Llama-3.1-8B-Instruct | **跨家族**(同 8B) |

→ 同家族 **0.6B→4B→8B→32B** 四档规模 + 8B 上 **Qwen vs Llama** 跨家族，比现在 API 版(qwen3-32b + deepseek-v4-flash)的泛化论证强得多。

## 该跑哪些实验
1. **【主要】generalization**：5 个模型 × TREC DL19/DL20 × 6 方法（rankgpt/swiss_choice/tourrank/blitzrank/bracketrank/setwise_heapsort）。→ 一条命令 `bash scripts/run_all.sh`。
2. **【可选】BEIR 泛化**：同样 5 模型 × 7 BEIR 数据集（量大、慢）。→ `BEIR=1 bash scripts/run_all.sh`。
3. **【可选】主结果本地复现**：qwen3-8b 的 TREC 主表已有 API 结果；如需全本地一致可重跑（`run_local_vllm.py --models qwen3-8b`）。

> 消融 / 输入顺序 / 剪枝 / 机制图 这些仍用主模型(qwen3-8b)，不需要在这 5 个模型上重复。

## 三步跑起来
```bash
# 0) 装环境
bash scripts/setup_env.sh

# 1) 拉数据(从 HF)
bash download_datasets.sh

# 2) 全自动跑(逐模型: 起vLLM→跑→评测→关→下一个)
bash scripts/run_all.sh                 # 全部5个模型, TREC
#   或单个:  bash scripts/run_all.sh qwen3-0.6b
#   或加BEIR: BEIR=1 bash scripts/run_all.sh
```

## 手动控制（按需）
```bash
# 单独起某个模型的服务(占住一个终端)
bash scripts/serve_vllm.sh qwen3-4b          # 32b 默认 TP=2: bash scripts/serve_vllm.sh qwen3-32b 4
# 另开终端跑实验
python run_local_vllm.py --models qwen3-4b --datasets dl19 dl20
# 评测出 NDCG
python evaluate.py --bench trec --datasets dl19 dl20 --model qwen3-4b
# 汇总所有模型
bash scripts/collect_results.sh
```

## 关键说明
- **思考模式已关**：qwen3 系列排序时 `enable_thinking=false`（脚本设 `LLM_THINK_STYLE=vllm`，只发 `chat_template_kwargs`，避免 vLLM 顶层参数 400）。llama 无思考模式，自动不发。
- **结果隔离**：写到 `results/TREC_results/<ds>/models/<tag>/`，靠 `MODEL_TAG` 与默认结果隔离，互不覆盖。
- **断点续跑**：中断后重跑自动跳过已完成 qid。
- **GPU**：≤8B 单卡(TP=1)即可；qwen3-32b 默认 TP=2（按显存调，如 4×24G 用 `serve_vllm.sh qwen3-32b 4`）。
- **接口零改动**：代码走 OpenAI 兼容接口，`config.yaml` 默认已指 `http://localhost:8000/v1`，脚本会自动用本地端点。

## 跑完后填论文
`bash scripts/collect_results.sh` 打印各模型 NDCG@10，按规模/家族填进 `tab:generalization`。表头可扩成：Qwen3-0.6B / 4B / 8B / 32B / Llama-3.1-8B。
