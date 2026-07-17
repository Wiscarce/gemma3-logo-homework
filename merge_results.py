"""
合并 baseline + generated 结果为单一 results.json（作业要求）
在 Colab 执行:
    python gemma3-logo-homework/merge_results.py
"""
import json, sys

BASELINE = "/content/results_baseline_v2.json"
GENERATED = "/content/results_generated_v2.json"
OUTPUT = "/content/results.json"

baseline = json.load(open(BASELINE, encoding="utf-8"))
generated = json.load(open(GENERATED, encoding="utf-8"))

merged = {
    "description": "SVG Logo reward comparison: Sonnet (ground-truth) vs Gemma 3 270M LoRA",
    "reward_version": "v2",
    "n_samples": baseline["n_samples"],
    "baseline": {
        "model": "Claude Sonnet",
        "avg_score": baseline["avg_score"],
        "min_score": baseline["min_score"],
        "max_score": baseline["max_score"],
    },
    "generated": {
        "model": "Gemma 3 270M + LoRA (rank=16, checkpoint-100)",
        "avg_score": generated["avg_score"],
        "min_score": generated["min_score"],
        "max_score": generated["max_score"],
    },
    "gap": baseline["avg_score"] - generated["avg_score"],
    # 逐条对比
    "per_sample": [],
}

if baseline["mode"] == "baseline" and generated["mode"] in ("generate", "score"):
    for i in range(len(baseline.get("samples", []))):
        b = baseline["samples"][i]
        g = generated["samples"][i]
        merged["per_sample"].append({
            "index": i,
            "score_baseline": b["score"],
            "score_generated": g["score"],
            "diff": b["score"] - g["score"],
        })

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print(f"Merged → {OUTPUT}")
print(f"Baseline avg: {merged['baseline']['avg_score']:.4f}")
print(f"Generated avg: {merged['generated']['avg_score']:.4f}")
print(f"Gap: {merged['gap']:.4f}")
print(f"Per-sample entries: {len(merged['per_sample'])}")
