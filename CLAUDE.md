# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

`LLM_Rank/` 是一个**基于 LLM 的零样本文档重排（reranking）代码库**：所有方法的输入统一是 BM25 的 Top-100 候选，输出是 LLM 重排后的有序文档列表。本目录是父级 LaTeX 论文（SwissRank，SIGIR-AP 2025，位于 `../`）的实验实现，与论文目录是两个独立项目——**不要把这里的改动与 `../CLAUDE.md` 描述的 LaTeX 论文混淆**。

逐方法的算法细节（SwissRank / RankGPT / Setwise / Pairwise / TourRank / BlitzRank / BracketRank）已在 `README.md` 中详细讲解，本文件只补充跨文件才能看懂的大图景架构与运行方式。需要某个方法的具体机制时，去看 `README.md`。

## 常用命令

Python 解释器固定用 `/Users/yangmeili/Downloads/Code/.venv/bin/python`（记为 `<py>`）。

```bash
# 跑单个方法：每个 ranker 是独立脚本，配置写死在各自的 main() 里，
# 且用相对路径 ../data ../results —— 必须在 rank_fuc/ 目录下运行
cd rank_fuc && <py> swissrank_choice.py

# 数据集切换靠环境变量（见 utils.get_trec_paths）：
TREC_DATASET=dl19 <py> swissrank_choice.py      # 或 dl20
BEIR_DATASET=trec-covid <py> swissrank_choice.py # 设置后覆盖为 BEIR 路径

# Smoke test：只跑前 N 个 qid 验证流水线
SMOKE_TEST=2 <py> rank_fuc/swissrank_choice.py

# 批量跑「全部方法 × 全部 BEIR 数据集」，可断点续跑、带 API 预检
<py> run_beir_experiments.py [--datasets ...] [--methods ...] [--force] [--no-preflight]

# 多模型 baseline：deepseek-r1/v3/qwen3-32b/qwen-max × TREC dl19/dl20（结果写 models/<model>/）
<py> run_model_baselines.py [--models ...] [--datasets ...] [--methods ...] [--force]

# 评测（NDCG@1/3/5/10/20 + 平均 token/调用/延迟/费用），已参数化为 CLI
<py> evaluate.py --bench trec --datasets dl19 dl20                  # qwen3-8b 主结果
<py> evaluate.py --bench trec --datasets dl19 --model deepseek-r1   # 某个 baseline 模型
<py> evaluate.py --bench beir --datasets trec-news --fill-skipped   # 丢弃 query 用 BM25 回填
```

- `evaluate.py` 已从写死路径改为 argparse CLI（`--bench trec|beir`、`--datasets`、`--methods`、`--model`、`--fill-skipped`）。它会读同名 `.skipped.jsonl` 报告每个 (数据集×方法) 丢弃了几条 query；默认只评测写入的 query，`--fill-skipped` 把丢弃 query 用 BM25 原序回填以保证跨方法全量可比。底层 `evaluate(result_file, qrels, pattern='llm_docs', ...)` 仍可直接调用；TourRank 多锦标赛结果用 `pattern='tour_0'/'tour_1'`。
- 若 `pytrec_eval` 缺失，**不要** `pip install pytrec_eval`，按 `README.md` 第 4 节的源码方式安装。

## 大图景架构

- **统一 I/O 契约**：所有 ranker 读 `../data/{TREC,BEIR}_data/<ds>/*_bm25_top100.jsonl`（字段 `qid/query/bm25_docs/bm25_contents`），以 **append** 方式写 `../results/.../<method>.jsonl`（字段 `qid/query/bm25_docs/llm_docs/sum_tokens` 加一组效率指标）。新增/修改方法要遵守这个契约，否则 `evaluate.py` / `run_beir_experiments.py` 无法处理。

- **断点续跑约定（两层）**：写入是 append；各 ranker 启动时用 `utils.load_done_qids()` 跳过「结果文件 + `.skipped.jsonl` 跳过清单」里已处理的 qid；`run_beir_experiments.py` 再在外层按「(结果行数 + 跳过行数) >= 数据行数 且无失败」决定是否整体跳过一个方法（见 `run_one` / `result_has_failures`）。

- **内容审核 → 整条 query 丢弃**：DashScope 对部分数据集正文（如 trec-news 的犯罪/暴力新闻）会返回 `data_inspection_failed`(400)。`llm_client.chat()` 识别该错误后**立即放弃不重试**（计入独立的 `content_filter_failed`，不污染 `failed_*`）；ranker 在 query 结束时若发现 `client.get_usage().content_filter_failed > 0`，就**不写结果行**、把 qid 记入 `<method>.jsonl.skipped.jsonl`（`utils.record_skipped_query`）。评测时这些 qid 缺失，需单独报告每个数据集丢弃了几条。

- **多模型对比（MODEL_TAG）**：设 `MODEL_TAG` 环境变量后，`utils.get_trec_paths()` 把结果改写到 `.../<ds>/models/<tag>/<method>.jsonl`，与默认 qwen3-8b 主结果隔离。`run_model_baselines.py` 用它跑 `deepseek-r1 / deepseek-v3 / qwen3-32b / qwen-max` × dl19/dl20 × 6 方法（每模型独立 preflight，deepseek-r1 自动放大 `LLM_MAX_TOKENS`）。`_build_extra_body()` 只对 `qwen3` 下发 `enable_thinking`，其余模型（deepseek/qwen-max/qwen2.5）传了会 400。

- **共享 LLM 层**：`rank_fuc/llm/llm_client.py` 暴露 `LLMClient`（`chat` / `batch_chat`）。所有 ranker 通过 `utils.get_default_client()` 拿**进程级单例**，共享线程池与线程安全的 token 统计。`batch_chat` 用 `ThreadPoolExecutor` 并发、retry + 指数退避——这是各方法在「天然 batch 边界」（同分桶组间、锦标赛同阶段等）并行的基础。

- **配置优先级**：`config.yaml` 的 `llm:` 段是默认值；同名环境变量优先（`LLM_MODEL` / `LLM_API_KEY` / `LLM_API_URL` / `LLM_MAX_WORKERS` / `LLM_MAX_TOKENS` / `LLM_TIMEOUT` / `LLM_ENABLE_THINKING`，解析逻辑见 `_get_config_value`）。排序实验默认保持 `enable_thinking=false`（输出更短更稳）。退避/限流旋钮：`rate_limit_backoff_base/max`（429 专用长退避）、`backoff_jitter`（多线程抖动）、`requests_per_minute`（0=禁用主动限流，持续 429 时设为账号档位 QPM）。`chat()` 按错误类型差异化重试：审核不重试、限流长退避、超时常规退避。

- **效率指标采集**：`utils.build_result_record()` 在每个 query 结束时调用 `client.get_usage()`，把 token、LLM 调用数、API 请求数、延迟、估算费用打进结果行。费用按 `config.yaml` 的 `llm.pricing`（单价 / 1K token）计算。

- **公共工具**：`utils.py` 提供 `set_seed`、`truncate_doc`（`DOC_MAX_WORDS=300`，每篇文档进 prompt 前截断到前 300 词）、`Documents` 容器、数据集路径解析。

- **ranker 家族与映射**：`swissrank.py`（permutation 版）/ `swissrank_choice.py`（choice 版，论文主推）是本工作；`rankgpt / setwise / pairwise / tourrank / tourrank-n / blitzrank / bracketrank` 是 baseline。`run_beir_experiments.py` 顶部的 `METHODS` 字典是「方法名 → (脚本, 结果文件名)」的权威映射，新增方法需在此登记。

## 编辑 / 运行注意事项

- 每个 ranker 的关键超参写在各自 `main()` 的 `params` 里（如 SwissRank 的 `group_size` / `max_failure_num` / `top_rerank_num`）；改实验配置就改那里。其中 `max_failure_num` 是 SwissRank 控制 token 预算的关键开关（值越小，低分组持续对战，效果略好但 token 涨）。
- 输入顺序消融（BM25 / inverse BM25 / random）在各 `main()` 里以注释切换。
- `config.yaml` 内现有一个明文 API key；运行真实实验时建议用 `LLM_API_KEY` 环境变量覆盖，不要把 key 提交进仓库。
