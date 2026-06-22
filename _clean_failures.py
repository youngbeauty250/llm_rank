"""删除带瞬时失败的结果行(failed_api_requests>0 或 failed_llm_calls>0),
使 ranker 续跑时重新处理这些 qid。content_filter 丢弃的 query 在 .skipped.jsonl 里,不动。
原文件备份到 <file>.prefail.bak。"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET_DIRS = [ROOT / "results" / "BEIR_results", ROOT / "results" / "TREC_results"]

def has_fail(row):
    return row.get("failed_api_requests", 0) or row.get("failed_llm_calls", 0)

cleaned = []
for base in TARGET_DIRS:
    for f in base.rglob("*.jsonl"):
        if f.name.endswith(".skipped.jsonl") or ".prefail.bak" in f.name:
            continue
        lines = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        keep, dropped_qids = [], []
        for l in lines:
            r = json.loads(l)
            if has_fail(r):
                dropped_qids.append(r.get("qid"))
            else:
                keep.append(l)
        if dropped_qids:
            shutil.copy2(f, f.with_suffix(f.suffix + ".prefail.bak"))
            f.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
            cleaned.append((str(f.relative_to(ROOT)), len(lines), len(keep), len(dropped_qids)))

if not cleaned:
    print("没有需要清理的失败行。")
else:
    print(f"{'文件':60s} {'原行':>5s} {'保留':>5s} {'删除(待重跑)':>10s}")
    tot = 0
    for path, before, after, dropped in cleaned:
        print(f"{path:60s} {before:5d} {after:5d} {dropped:10d}")
        tot += dropped
    print(f"\n共清理 {len(cleaned)} 个文件,删除 {tot} 行失败 qid 待重跑。")
