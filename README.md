# LLM_Rank：基于 LLM 的零样本文档排序方法库

本目录汇总了 **SwissRank**（本论文方法）以及若干主流 baseline 在 TREC DL / BEIR 上做零样本文档重排（reranking）的实现代码。所有方法的输入都是 BM25 的 Top-100 候选文档，输出是 LLM 重排后的有序文档列表。

---

## 1. 目录结构

```
LLM_Rank/
├── evaluate.py                # 统一评测脚本（计算 NDCG@1/3/5/10/20、平均 token 数）
├── rank_fuc/                  # 各种排序方法的实现
│   ├── utils.py               # 通用工具：LLM 调用、随机种子、Documents 类
│   ├── swissrank.py           # SwissRank（生成式 permutation 版）
│   ├── swissrank_choice.py    # SwissRank（Top-K choice 版，论文主推）
│   ├── rankgpt.py             # RankGPT 滑动窗口
│   ├── setwise.py             # Setwise（heapsort / bubblesort）
│   ├── pairwise.py            # Pairwise（heapsort / bubblesort）
│   ├── tourrank.py            # TourRank（单次锦标赛）
│   ├── tourrank-n.py          # TourRank（多锦标赛累加投票）
│   ├── blitzrank.py           # BlitzRank（tournament graph）
│   └── bracketrank.py         # BracketRank（winner/loser bracket）
└── results/                   # 输出结果
    ├── TREC_results/          # TREC DL19 / DL20
    └── BEIR_results/          # BEIR 各子集
```

---

## 2. 通用约定

所有方法统一使用 `rank_fuc/utils.py::DOC_MAX_WORDS = 300`，即每篇候选文档进入 LLM prompt 前最多保留前 300 个词。

### 2.1 输入数据格式
所有方法都从 `../data/{TREC,BEIR}_data/<dataset>/*.jsonl` 读取每行一个查询：

```json
{
  "qid": "...",
  "query": "...",
  "bm25_docs": ["docid1", "docid2", ...],     // BM25 给出的 top-100 文档ID
  "bm25_contents": ["text1", "text2", ...]    // 与 bm25_docs 一一对应的文本
}
```

### 2.2 输出数据格式
每个方法都将重排结果以 jsonl 追加写入 `results/.../<method>.jsonl`：

```json
{
  "qid": "...",
  "query": "...",
  "bm25_docs": [...],                 // 原始 BM25 顺序
  "llm_docs": [...],                  // LLM 重排后的顺序（TourRank 用 tour_0, tour_1, ...）
  "sum_tokens": 12345                 // 本次查询的总 token 消耗
}
```

### 2.3 LLM 调用
`rank_fuc/llm/llm_client.py` 统一封装 OpenAI 兼容接口，配置从 `config.yaml` 读取，环境变量优先级更高。默认配置位于：

```bash
LLM_Rank/config.yaml
```

真实运行前至少需要设置 API key，推荐用环境变量，避免把密钥写进仓库：

```bash
export LLM_API_KEY="your-api-key"
```

常用覆盖项包括 `LLM_MODEL`、`LLM_API_URL`、`LLM_MAX_WORKERS`、`LLM_MAX_TOKENS`、`LLM_MAX_RETRY`、`LLM_TIMEOUT`、`LLM_ENABLE_THINKING`。

`LLM_ENABLE_THINKING=true` 时，客户端会自动使用 streaming 请求，并在返回的 `LLMResponse` 中分开保存：
- `content`：最终回答，baseline 解析仍使用这个字段
- `reasoning_content`：模型 thinking 过程

排序实验默认建议保持 `LLM_ENABLE_THINKING=false`，这样输出更短、更稳定，延迟和 token 成本也更低。

每条结果 JSONL 会记录效率指标：
- `sum_tokens`、`prompt_tokens`、`completion_tokens`
- `llm_calls`、`failed_llm_calls`
- `api_requests`、`failed_api_requests`
- `llm_latency_seconds`：所有 API 请求耗时之和
- `query_latency_seconds` / `wall_clock_seconds`：该 query 的端到端墙钟时间
- `estimated_cost`、`cost_currency`

费用估算使用 `config.yaml` 中的 `llm.pricing`。单价单位是每 1K token，未填写时默认为 0。

### 2.4 输入顺序消融
所有方法的主函数中都预留了三种输入顺序，注释切换即可：
- BM25 顺序（默认）
- `list(reversed(...))` —— inverse BM25
- `random.sample(...)` —— random BM25

---

## 3. 方法说明

### 3.1 SwissRank（本论文，`swissrank_choice.py` / `swissrank.py`）

**核心思想**：受瑞士制锦标赛启发，迭代进行「分组 → 组内比较 → 按得分重新分配」直到无法继续分组，然后对最优分数组做最终精排。

**两种变体**：
| 文件 | 组内 LLM 任务 | 得分更新 | 论文中名称 |
|---|---|---|---|
| `swissrank.py` | 让 LLM 输出整组的完整 permutation | 前 `step_size` 个 +1 分，后面 -1 分 | SwissRank（permutation） |
| `swissrank_choice.py` | 让 LLM 只挑选 Top-`N/2` | 前 `N/2` 个 +1 分，后 `N/2` 个 -1 分 | **SwissRank（choice，主推）** |

**关键参数**（位于 `main()` 内的 `params`）：
- `group_size`：每组文档数，默认 20
- `step_size`（仅 permutation 版）：每组上升的文档数，默认 4
- `max_failure_num`：允许的最低得分，得分低于此值的组不再参与对局；越小表示对低分文档越宽松，影响最终覆盖范围
- `top_rerank_num`：最终对得分最高的前几个分数组做一次额外精排，默认 1

**算法流程**（见 `filter_processing`）：
1. 初始所有文档得分为 0；
2. 按得分分桶；对每个桶（大小 ≥ `group_size` 且分数 ≥ `max_failure_num`）做 **skip 分组**（`group_documents`：交错抽取，使每组内文档原始排名分散）；
3. 多进程并发让 LLM 对每组排序 / 挑 Top-N/2；
4. 按结果更新得分桶：胜者得分 +1、败者 -1；
5. 重复直到所有桶都不可分；
6. 取最高分桶做一次完整重排作为 Top-K。

**输入顺序鲁棒性**：`swissrank_choice.py` 中提供 `group_documents_chunk`（顺序切块）、`group_documents_random`（完全随机分组）作为分组策略消融对比。

---

### 3.2 RankGPT（`rankgpt.py`）

**滑动窗口式 listwise 排序**：自后向前用大小为 `window_size` 的窗口滑动，每个窗口让 LLM 输出整窗 permutation，按 `step_size` 步长前移。

- 默认参数：`window_size=20`、`step_size=10`
- 优点：实现简单、调用次数少
- 缺点：窗口内对比有限，初始 BM25 顺序对结果影响大

---

### 3.3 Setwise（`setwise.py`）

**核心**：每次比较从 `num_child+1` 个候选中选 **1 个最相关**（输出大写字母 A/B/C…）。两种排序算法：

| `method` | 思路 | 适用 |
|---|---|---|
| `heapsort` | 把候选建成 N 叉堆，反复弹出堆顶得到 Top-K | 效果好、调用次数稳定 |
| `bubblesort` | 从底向上做改良冒泡，每次确认一个 Top 位置 | 调用次数多、对深位敏感 |

- 默认 `num_child=3`、`k=20`
- 提示词只问「哪一段最相关」，输出鲁棒性高

---

### 3.4 Pairwise（`pairwise.py`）

**两两比较**：每次给 LLM 两个文档（同时给 A/B、B/A 两个方向消除位置偏置）。
- 排序算法：`allpair`、`heapsort` 或 `bubblesort`
- 默认 `k=20`
- 调用次数最多，token 成本高，是其它方法的对照下界

---

### 3.5 TourRank（`tourrank.py` 单次、`tourrank-n.py` 多次）

**多阶段锦标赛 + 投票**：

阶段流程（每阶段并发）：
```
100 → 50 (5 组 × 20 选 10)
 50 → 20 (5 组 × 10 选 4)
 20 → 10 (1 组 × 20 选 10)
 10 →  5 (1 组 × 10 选 5)
  5 →  2 (1 组 ×  5 选 2)
```
每被某阶段选中得 1 分，最终按累计分排序。

- `tourrank.py`：跑 **1 次**锦标赛
- `tourrank-n.py`：跑 **Y 次**锦标赛（默认 Y=2），结果累加得到 `tour_0`、`tour_1`、… 渐进式排名（论文中的 TourRank-1 / TourRank-2）。重构后跨 tournament 同一阶段的 group 也走 batch_chat 一起并发。
- `get_groups_skip`：与 SwissRank 类似的交错分组，降低输入顺序偏置

---

### 3.6 BlitzRank（`blitzrank.py`）

**Tournament graph top-m**：每次让 LLM 对最多 `window_size` 个文档做 listwise 排序，排序结果转成 preference graph 中的边；随后用传递闭包计算每个文档的 `in_reach / out_reach / known_relationships`，优先调度当前 top 区域里尚未 resolved 的 SCC representative，直到 top-`m` 已经 resolved 或达到最大轮数。

- 默认参数：`window_size=20`、`top_m=10`、`max_num_rounds=50`
- 本实现按 `LLM_Rank` 形式重写了图调度和 LLM 调用层，不依赖本地 `BlitzRank` 包，避免额外安装 `networkx/litellm` 等依赖
- 输出 `llm_docs`：先放 BlitzRank 确定的 top-`m`，后面按当前图排序/BM25 兜底补齐剩余文档，方便直接用 `evaluate.py` 评测

---

### 3.7 BracketRank（`bracketrank.py`）

**Reasoning-enhanced competitive elimination**：先按 `group_size` 连续切分 BM25 top-100，对每组并发做带 reasoning 指令的 listwise 排序；每组 top half 进入 winner bracket，bottom half 进入 loser bracket；两个 bracket 各自做 single-elimination 合并重排，最终 `winner_ranked + loser_ranked` 拼接成全局排序。

- 默认参数：`group_size=20`
- 本地 `BracketRank` 仓库为空，因此这里根据论文详解实现 single-elimination 主版本
- prompt 会要求模型先做简短相关性推理，并在最后输出 `Final Ranking: [1] > [2] ...`；解析仍复用现有 permutation parser，失败时保留原输入顺序兜底

---

## 4. 评测：`evaluate.py`

```bash
cd LLM_Rank
python evaluate.py
```

`evaluate(result_file, qrels, pattern='llm_docs')` 会同时打印：
- BM25 baseline 的 NDCG@1/3/5/10/20
- LLM 重排结果的 NDCG@1/3/5/10/20
- 该结果文件的平均 token 数、LLM 调用数、API 请求数、query latency、wall-clock time、估算费用

对 TourRank 多锦标赛结果（pattern 用 `tour_0` / `tour_1`）即可输出 TourRank-1 / TourRank-2 的指标。

`read_qrels` 解析 TREC 格式（空格分隔，`qid Q0 docid rel`）。

如果当前环境没有 `pytrec_eval`，不要直接 `pip install pytrec_eval`。按如下源码方式安装：

```bash
wget https://files.pythonhosted.org/packages/2e/03/e6e84df6a7c1265579ab26bbe30ff7f8c22745aa77e0799bba471c0a3a19/pytrec_eval-0.5.tar.gz
tar -zxvf pytrec_eval-0.5.tar.gz
wget https://github.com/usnistgov/trec_eval/archive/refs/tags/v9.0.8.tar.gz
tar -zxvf trec_eval-9.0.8.tar.gz
mv trec_eval-9.0.8 pytrec_eval-0.5/trec_eval
cd pytrec_eval-0.5
python setup.py install
```

---

## 5. 跑通一个方法的最小步骤

以 SwissRank（choice 版）+ TREC DL19 为例：

1. 准备数据：`data/TREC_data/dl19/trec19_bm25_top100.jsonl` 与 `qrels.dl19-passage.txt`。
2. 在 `swissrank_choice.py::main()` 内配置：
   - `result_file = '../results/TREC_results/dl19/swiss_choice.jsonl'`
   - 数据集路径与 params。
3. 运行：
   ```bash
   cd LLM_Rank/rank_fuc
   python swissrank_choice.py
   ```
   结果以 **追加** 模式写入，已处理过的 `qid` 会自动跳过（断点续跑）。
4. 评测：
   ```bash
   cd ..
   python evaluate.py
   ```

---

## 6. 方法对比一览

| 方法 | 粒度 | 每次比较的文档数 | 排序结构 | 并行 | 论文中地位 |
|---|---|---|---|---|---|
| RankGPT | Listwise | 20（滑动窗口） | 线性扫描 | 否 | 经典 baseline |
| Setwise | Setwise | 10（选 1） | heap / bubble | 否 | 强 baseline |
| Pairwise | Pairwise | 2 | heap / bubble | 否 | 成本下界 |
| TourRank | Listwise | 5–20 | 多阶段淘汰 + 投票 | 阶段内并行 | 近期 SOTA |
| BlitzRank | Listwise | 20 | tournament graph + SCC | 否 | 图证据累积 baseline |
| BracketRank | Listwise | 20 | winner/loser bracket | 组内并行 | reasoning tournament baseline |
| **SwissRank** | Listwise | 10（permutation / choose top-N/2） | 瑞士制循环对局 + 种子排序 | 同分桶组间并行 | **本工作** |

各方法的实测对比见 `utils.py` 末尾的可视化代码（NDCG@10 vs Avg. Tokens 散点图），SwissRank 在 NDCG@10 和 token 成本之间取得最优折中。

---

## 7. 代码改进 / 注意事项

1. **LLM 调用层（已重构）**：`rank_fuc/llm/llm_client.py` 暴露轻量 `LLMClient`（`chat` / `batch_chat`），所有 ranker 通过 `utils.get_default_client()` 共用进程级单例。`batch_chat` 用 `ThreadPoolExecutor`（默认 16 并发，可 `LLM_MAX_WORKERS` env 覆盖），retry 5 次、指数退避，token 统计线程安全。
2. **配置读取**：模型 / API key / API URL 走环境变量 `LLM_MODEL` / `LLM_API_KEY` / `LLM_API_URL`，未设置时回退到用户 `CLAUDE.md` 中的 dashscope qwen3-8b 兼容接口。**没有硬编码 key**。
3. **天然 batch 边界**（重构后已生效）：
   - SwissRank：每轮 N 个 group 一次 batch_chat 全部并发（替代旧 `multiprocessing.Process + Manager`）。
   - TourRank：阶段 1/2 各 5 组一次 batch；阶段 3/4/5 单组 chat。
   - TourRank-N：跨 Y 个 tournament 的同 stage（共 Y×K 组）一次 batch；并自动修复了原版 token 不统计的 bug。
   - BracketRank：初始分组和每轮 bracket pair 的 group 调用走 batch_chat。
   - BlitzRank：轮间图状态有依赖，保持串行 LLM 调用。
   - Pairwise：A/B + B/A 两次方向调用合并成一次 batch_chat。
   - RankGPT / Setwise / Pairwise.heapsort/bubblesort：因前后顺序依赖保持串行。
4. **结果文件**：重构后写入 `*_v2.jsonl`（如 `swiss_choice_v2.jsonl`），与旧结果分开便于对比；`evaluate.py` 中相应路径加 `_v2` 后缀即可评测。
5. **`max_failure_num`** 是 SwissRank 控制 token 预算的关键开关：值越小（如 `-4`），低分组持续对战，效果略好但 token 涨；评测脚本中的 `swiss_choice_max_fail_-1/-2/-3/-4` 即对应这组消融。
6. **smoke test**：每个 ranker `main()` 检测 `SMOKE_TEST` 环境变量；`SMOKE_TEST=2 /Users/yangmeili/Downloads/Code/.venv/bin/python rank_fuc/swissrank_choice.py` 只跑前 2 个 qid，验证 pipeline。
