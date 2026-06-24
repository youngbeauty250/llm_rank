# LLM_Rank：基于 LLM 的零样本文档排序方法库

本仓库实现了 **SwissRank**（受瑞士制锦标赛启发的零样本 LLM 文档排序方法）以及若干主流 baseline，在 TREC DL / BEIR 上做零样本文档重排（reranking）。所有方法的输入统一是 BM25 的 Top-100 候选文档，输出是 LLM 重排后的有序文档列表。

---

## 1. 目录结构

```
LLM_Rank/
├── run.py                     # 统一入口：一条命令在 TREC/BEIR 上跑任意方法
├── evaluate.py                # 评测脚本（NDCG@1/3/5/10/20 + 效率指标）
├── config.yaml                # LLM 接口配置（key 留空，用环境变量注入）
├── requirements.txt
└── rank_fuc/                  # 各排序方法的实现
    ├── llm/llm_client.py      # OpenAI 兼容接口封装（并发、重试、退避、用量统计）
    ├── utils.py               # 通用工具：路径约定、随机种子、结果记录、文档截断
    ├── swissrank.py           # SwissRank（生成式 permutation 版 + 共享 prompt 工具）
    ├── swissrank_choice.py    # SwissRank（Top-N/2 choice 版，论文主推）
    ├── rankgpt.py             # RankGPT（滑动窗口 listwise）
    ├── setwise.py             # Setwise（heapsort）
    ├── pairwise.py            # Pairwise（heapsort / bubblesort）
    ├── tourrank.py            # TourRank（单次锦标赛）
    ├── tourrank-n.py          # TourRank（多锦标赛累加投票）
    ├── blitzrank.py           # BlitzRank（tournament graph）
    └── bracketrank.py         # BracketRank（winner/loser bracket）
```

> 数据（`data/`）与结果（`results/`）不入库，需自行准备，见第 4 节。

---

## 2. 安装与配置

```bash
pip install -r requirements.txt
```

`pytrec_eval` 若 pip 装不上，按源码方式安装：

```bash
wget https://files.pythonhosted.org/packages/2e/03/e6e84df6a7c1265579ab26bbe30ff7f8c22745aa77e0799bba471c0a3a19/pytrec_eval-0.5.tar.gz
tar -zxvf pytrec_eval-0.5.tar.gz
wget https://github.com/usnistgov/trec_eval/archive/refs/tags/v9.0.8.tar.gz
tar -zxvf v9.0.8.tar.gz
mv trec_eval-9.0.8 pytrec_eval-0.5/trec_eval
cd pytrec_eval-0.5 && python setup.py install
```

**LLM 接口**走 OpenAI 兼容协议。推荐用环境变量注入 key（不要把 key 写进仓库）：

```bash
export LLM_API_KEY="your-api-key"
export LLM_API_URL="https://your-endpoint/v1"   # 可选，默认读 config.yaml
export LLM_MODEL="qwen3-8b"                      # 可选
```

环境变量优先级高于 `config.yaml`。其它可覆盖项：`LLM_MAX_WORKERS` / `LLM_MAX_TOKENS` / `LLM_MAX_RETRY` / `LLM_TIMEOUT` / `LLM_ENABLE_THINKING`。排序实验默认 `enable_thinking=false`（输出更短更稳、成本更低）。

---

## 3. 快速开始

统一入口是 `run.py`：

```bash
# TREC DL19 上跑 SwissRank（论文主推 choice 版）
python run.py --method swiss_choice --bench trec --dataset dl19

# BEIR scifact 上跑 RankGPT
python run.py --method rankgpt --bench beir --dataset scifact

# 冒烟测试：只跑前 2 条 query，验证接口是否打通
SMOKE_TEST=1 python run.py --method swiss_choice --bench trec --dataset dl19
```

可选 `--method`：`swiss_choice`、`rankgpt`、`setwise_heapsort`、`tourrank`、`blitzrank`、`bracketrank`、`pairwise`、`tourrank_n`。

`--dataset`：trec 用 `dl19` / `dl20`；beir 用 `trec-covid` / `webis-touche2020` / `dbpedia-entity` / `scifact` / `signal1m` / `trec-news` / `robust04` / `nfcorpus`。

结果以 **追加** 模式写入 `results/{TREC,BEIR}_results/<dataset>/<method>.jsonl`，已处理过的 `qid` 自动跳过（**断点续跑**）。

跑完用 `evaluate.py` 算指标：

```bash
python evaluate.py --bench trec --datasets dl19 dl20
python evaluate.py --bench beir --datasets scifact trec-covid
```

---

## 4. 数据格式

数据放在 `data/{TREC,BEIR}_data/<dataset>/`，每个数据集需要两个文件：

**① BM25 Top-100 候选**（`*_bm25_top100.jsonl`，每行一个查询）：

```json
{
  "qid": "...",
  "query": "...",
  "bm25_docs": ["docid1", "docid2", "..."],
  "bm25_contents": ["text1", "text2", "..."]
}
```

**② qrels 相关性标注**（TREC 格式，空格分隔 `qid Q0 docid rel`）：
- TREC：`data/TREC_data/<ds>/qrels.<ds>-passage.txt`
- BEIR：`data/BEIR_data/<ds>/qrels.beir-v1.0.0-<ds>.test.txt`

BM25 候选可用 [pyserini](https://github.com/castorini/pyserini) 以默认设置检索得到。所有方法进入 prompt 前对文档截断到前 300 个词（`utils.py::DOC_MAX_WORDS`）。

每条结果 JSONL 还会记录效率指标：`sum_tokens` / `prompt_tokens` / `completion_tokens`、`llm_calls` / `api_requests`（及对应 `failed_*`）、`llm_latency_seconds`、`query_latency_seconds`、`estimated_cost`。

---

## 5. 方法说明

### SwissRank（本论文，`swissrank_choice.py`）
受瑞士制锦标赛启发，迭代「分组 → 组内比较 → 按累计得分重新分桶」，直到无法继续分组，再对最高分桶做最终精排。两个关键设计：
- **后验重排种子（posterior re-seeding）**：每轮用组内比较结果回写全局种子排序，使下一轮喂给 LLM 的输入顺序逐轮变好，从源头降低单次调用误差。
- **相似对战 + 累计得分**：只把得分相近的文档分到一组，胜 +1 / 负 -1，单次不利比较不会让文档直接出局，从而抑制误差传播。
- 组内任务用 choice 版（让 LLM 挑出 Top-N/2），同分桶的各组互相独立、并发调用。
- `swissrank.py` 是 permutation 变体，同时为其它方法提供共享 prompt 工具函数。

### RankGPT（`rankgpt.py`）
滑动窗口 listwise：自后向前用大小 `window_size=20` 的窗口、`step_size=10` 步长滑动，每窗输出完整 permutation。实现简单、调用少，但受初始顺序影响大。

### Setwise（`setwise.py`）
每次从若干候选里选 **1 个最相关**（输出 A/B/C…），用 heapsort 反复弹出堆顶得到 Top-K。提示词简单、解析鲁棒，但串行调用次数多。

### Pairwise（`pairwise.py`）
两两比较（同时给 A/B、B/A 两个方向消除位置偏置），heapsort / bubblesort 排序。调用次数最多，作为成本下界对照。

### TourRank（`tourrank.py` / `tourrank-n.py`）
多阶段锦标赛 + 投票：`100→50→20→10→5→2` 逐级淘汰，每被选中得 1 分，按累计分排序。阶段内并发。`tourrank-n.py` 跑多次锦标赛累加，得到渐进式排名（TourRank-1 / TourRank-2）。

### BlitzRank（`blitzrank.py`）
Tournament graph：每次 listwise 排序结果转成 preference graph 的边，用传递闭包计算可达关系，优先调度 top 区域里尚未 resolved 的 SCC，直到 top-`m` 确定。

### BracketRank（`bracketrank.py`）
按 `group_size` 连续切分后并发做带 reasoning 的组内排序；每组上半进 winner bracket、下半进 loser bracket，各自 single-elimination 合并，最终 `winner + loser` 拼接。

---

## 6. 方法对比一览

| 方法 | 粒度 | 单次比较文档数 | 排序结构 | 并行 |
|---|---|---|---|---|
| RankGPT | Listwise | 20（滑窗） | 线性扫描 | 否 |
| Setwise | Setwise | 选 1 | heapsort | 否 |
| Pairwise | Pairwise | 2 | heap / bubble | 否 |
| TourRank | Listwise | 5–20 | 多阶段淘汰 + 投票 | 阶段内 |
| BlitzRank | Listwise | 20 | tournament graph + SCC | 否 |
| BracketRank | Listwise | 20 | winner/loser bracket | 组内 |
| **SwissRank** | Listwise | 10（choice Top-N/2） | 瑞士制循环对局 + 种子排序 | 同分桶组间 |
