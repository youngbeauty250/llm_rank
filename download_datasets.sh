#!/usr/bin/env bash
# 下载 LLM_Rank 实验所需数据：BM25 Top-100 候选(qid/query/bm25_docs/bm25_contents) + qrels。
# 覆盖 TREC DL19/DL20 与 7 个 BEIR 数据集(trec-covid/webis-touche2020/dbpedia-entity/
# scifact/signal1m/trec-news/robust04)。
#
# 依赖: pip install -U "huggingface_hub[cli]"
# 运行后目录结构: ./data/TREC_data/<ds>/...  ./data/BEIR_data/<ds>/...
set -e
HF_REPO="YoungBeauty25000/BEIR_TOP100"   # HuggingFace dataset repo
hf download "$HF_REPO" --repo-type dataset --local-dir ./data
echo "数据已下载到 ./data"
