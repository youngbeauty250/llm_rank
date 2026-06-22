import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run_beir_experiments import METHODS, count_jsonl, result_has_failures

TREC_DATA = ROOT / "data" / "TREC_data"
TREC_RES = ROOT / "results" / "TREC_results"
BEIR_DATA = ROOT / "data" / "BEIR_data"
BEIR_RES = ROOT / "results" / "BEIR_results"

def status(result_file, expected):
    done = count_jsonl(result_file) + count_jsonl(result_file.with_name(result_file.name + ".skipped.jsonl"))
    if done == 0:
        return f"MISSING (0/{expected})"
    if done < expected:
        return f"PARTIAL ({done}/{expected})"
    if result_has_failures(result_file):
        return f"HAS_FAILURES ({done}/{expected})"
    return f"OK ({done}/{expected})"

def trec_data_rows(ds):
    yr = "19" if ds == "dl19" else "20"
    return count_jsonl(TREC_DATA / ds / f"trec{yr}_bm25_top100.jsonl")

print("########## TREC 主结果 (qwen3-8b) ##########")
for ds in ["dl19", "dl20"]:
    exp = trec_data_rows(ds)
    print(f"\n[{ds}] expected={exp}")
    for m, (_, rname) in METHODS.items():
        print(f"  {m:18s} {status(TREC_RES / ds / rname, exp)}")

print("\n########## BEIR 结果 (qwen3-8b) ##########")
for ds in sorted(p.name for p in BEIR_DATA.iterdir() if p.is_dir()):
    exp = count_jsonl(BEIR_DATA / ds / f"{ds}_bm25_top100.jsonl")
    print(f"\n[{ds}] expected={exp}")
    for m, (_, rname) in METHODS.items():
        print(f"  {m:18s} {status(BEIR_RES / ds / rname, exp)}")

print("\n########## 多模型 baseline (TREC) ##########")
model_dirs = set()
for ds in ["dl19", "dl20"]:
    md = TREC_RES / ds / "models"
    if md.exists():
        model_dirs |= {p.name for p in md.iterdir() if p.is_dir()}
for model in sorted(model_dirs):
    print(f"\n=== model={model} ===")
    for ds in ["dl19", "dl20"]:
        exp = trec_data_rows(ds)
        print(f"  [{ds}] expected={exp}")
        for m, (_, rname) in METHODS.items():
            rf = TREC_RES / ds / "models" / model / rname
            print(f"    {m:18s} {status(rf, exp)}")
