"""
自评估脚本: 使用 reward.py 对模型生成的 SVG 进行打分

用法:
    # 模式1: 对 valid.jsonl 中的 ground-truth (Sonnet) SVGs 打基准分
    python eval_self.py --data logo-detailed-prompt/valid.jsonl --mode baseline

    # 模式2: 用 LoRA 模型生成 SVG 并打分
    python eval_self.py --data logo-detailed-prompt/valid.jsonl --mode generate \
        --model ./output/gemma3-270m-svg-lora

    # 模式3: 对比模式 (生成 vs ground-truth)
    python eval_self.py --data logo-detailed-prompt/valid.jsonl --mode compare \
        --model ./output/gemma3-270m-svg-lora

    # 模式4: 对已有推理结果的 JSONL 打分
    python eval_self.py --data results_inference.jsonl --mode score \
        --svg-field svg_output --prompt-field prompt

输出:
    results.json — 包含每个样本的 score 和 detail
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# 确保 student_kit 在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward import score, score_batch


# ============================================================================
# 数据加载
# ============================================================================

def load_chat_jsonl(filepath: str) -> List[Dict[str, str]]:
    """加载 chat-format JSONL, 提取 prompt 和 ground-truth SVG."""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record["messages"]
            # system = messages[0], user = messages[1], assistant = messages[2]
            samples.append({
                "prompt": messages[1]["content"],
                "svg_gt": messages[2]["content"],   # ground truth (Sonnet)
            })
    return samples


def load_flat_jsonl(filepath: str, svg_field: str = "svg_output",
                    prompt_field: str = "prompt") -> List[Dict[str, str]]:
    """加载扁平 JSONL (每行有 prompt 和 svg_output 字段)."""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            samples.append({
                "prompt": record.get(prompt_field, ""),
                "svg_output": record.get(svg_field, ""),
            })
    return samples


# ============================================================================
# 模型推理
# ============================================================================

def load_lora_model(model_path: str, base_model: str = "google/gemma-3-270m-it"):
    """加载 LoRA 微调后的模型和 tokenizer."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    return model, tokenizer


def generate_svg(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str = None,
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """用模型生成单个 SVG."""
    if system_prompt is None:
        system_prompt = (
            "You are an expert logo designer working in clean, scalable vector graphics. "
            "Given a description of a logo's visual elements, output ONE complete SVG document "
            'for the logo.\n\nRules:\n- Output ONLY the SVG: a single <svg ...>...</svg> '
            'element with an xmlns and viewBox="0 0 256 256". No prose, no markdown, '
            "no code fences.\n- Compose centered, content roughly within 16..240. "
            "Use a small cohesive palette.\n- Put gradients/filters in <defs>; use vector "
            "primitives only (<path>, <circle>, <ellipse>, <rect>, <polygon>, <line>, <g>). "
            "No <image>, external refs, or scripts.\n- Draw exactly what the description specifies."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # 清理: 提取 <svg>...</svg>
    import re
    match = re.search(r"<svg[\s\S]*?</svg>", generated, re.IGNORECASE)
    if match:
        return match.group(0)
    elif "<svg" in generated:
        idx = generated.index("<svg")
        return generated[idx:] + "</svg>"
    return generated.strip()


def batch_generate(
    model,
    tokenizer,
    samples: List[Dict[str, str]],
    verbose: bool = True,
) -> List[Dict[str, str]]:
    """批量生成 SVG."""
    results = []
    total = len(samples)
    for i, sample in enumerate(samples):
        if verbose:
            print(f"  [{i+1}/{total}] Generating...", end=" ", flush=True)
        t0 = time.time()
        svg_output = generate_svg(model, tokenizer, sample["prompt"])
        elapsed = time.time() - t0
        if verbose:
            print(f"({elapsed:.1f}s, {len(svg_output)} chars)")
        results.append({
            "prompt": sample["prompt"],
            "svg_output": svg_output,
            "svg_gt": sample.get("svg_gt", ""),
        })
    return results


# ============================================================================
# 主流程
# ============================================================================

def run_baseline(samples: List[Dict[str, str]]) -> Dict[str, Any]:
    """对 ground-truth (Sonnet) SVGs 打分."""
    print(f"Scoring {len(samples)} ground-truth SVGs...")
    batch = [{"prompt": s["prompt"], "svg_output": s["svg_gt"]} for s in samples]
    results = score_batch(batch)

    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"Baseline average: {avg:.4f}  (min={min(scores):.4f}, max={max(scores):.4f})")

    return {
        "mode": "baseline",
        "n_samples": len(samples),
        "avg_score": avg,
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "samples": results,
    }


def run_score(samples: List[Dict[str, str]]) -> Dict[str, Any]:
    """对已有推理结果的 JSONL 打分."""
    print(f"Scoring {len(samples)} generated SVGs...")
    results = score_batch(samples)

    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"Average: {avg:.4f}  (min={min(scores):.4f}, max={max(scores):.4f})")

    return {
        "mode": "score",
        "n_samples": len(samples),
        "avg_score": avg,
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "samples": results,
    }


def run_generate(samples: List[Dict[str, str]], model_path: str,
                 base_model: str = "google/gemma-3-270m-it") -> Dict[str, Any]:
    """用 LoRA 模型生成 SVG 并打分."""
    print(f"Loading model from: {model_path}")
    model, tokenizer = load_lora_model(model_path, base_model)

    print(f"Generating SVGs for {len(samples)} prompts...")
    generated = batch_generate(model, tokenizer, samples)

    print(f"Scoring generated SVGs...")
    results = score_batch(generated)

    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"Generated average: {avg:.4f}  (min={min(scores):.4f}, max={max(scores):.4f})")

    return {
        "mode": "generate",
        "model_path": model_path,
        "n_samples": len(samples),
        "avg_score": avg,
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "samples": results,
    }


def run_compare(samples: List[Dict[str, str]], model_path: str,
                base_model: str = "google/gemma-3-270m-it") -> Dict[str, Any]:
    """对比模式: 生成 vs ground-truth."""
    print(f"Loading model from: {model_path}")
    model, tokenizer = load_lora_model(model_path, base_model)

    print(f"Generating SVGs for {len(samples)} prompts...")
    generated = batch_generate(model, tokenizer, samples)

    print(f"Scoring both generated and ground-truth SVGs...")
    gen_results = score_batch(
        [{"prompt": g["prompt"], "svg_output": g["svg_output"]} for g in generated]
    )
    gt_results = score_batch(
        [{"prompt": s["prompt"], "svg_output": s["svg_gt"]} for s in samples]
    )

    gen_scores = [r["score"] for r in gen_results]
    gt_scores = [r["score"] for r in gt_results]

    gen_avg = sum(gen_scores) / len(gen_scores) if gen_scores else 0.0
    gt_avg = sum(gt_scores) / len(gt_scores) if gt_scores else 0.0

    print(f"\n{'='*50}")
    print(f"Ground Truth (Sonnet)  avg: {gt_avg:.4f}")
    print(f"LoRA Model            avg: {gen_avg:.4f}")
    print(f"Gap (GT - Model):          {gt_avg - gen_avg:+.4f}")
    print(f"{'='*50}")

    # 合并每条样本的对比
    comparison = []
    for i, (gen_r, gt_r) in enumerate(zip(gen_results, gt_results)):
        comparison.append({
            "index": i,
            "prompt": gen_r["prompt"],
            "svg_generated": gen_r["svg_output"],
            "svg_ground_truth": samples[i]["svg_gt"],
            "score_generated": gen_r["score"],
            "score_ground_truth": gt_r["score"],
            "diff": gen_r["score"] - gt_r["score"],
            "detail_generated": gen_r["detail"],
            "detail_ground_truth": gt_r["detail"],
        })

    return {
        "mode": "compare",
        "model_path": model_path,
        "n_samples": len(samples),
        "avg_score_ground_truth": gt_avg,
        "avg_score_generated": gen_avg,
        "gap": gt_avg - gen_avg,
        "comparison": comparison,
    }


# ============================================================================
# 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SVG Logo 自评估 — 使用 reward.py 打分",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python eval_self.py -d valid.jsonl -m baseline
  python eval_self.py -d valid.jsonl -m generate --model ./output/gemma3-270m-svg-lora
  python eval_self.py -d valid.jsonl -m compare  --model ./output/gemma3-270m-svg-lora
  python eval_self.py -d results.jsonl -m score --svg-field svg_output
        """,
    )
    parser.add_argument("-d", "--data", required=True,
                        help="输入数据: chat-format JSONL 或 flat JSONL")
    parser.add_argument("-m", "--mode", required=True,
                        choices=["baseline", "generate", "compare", "score"],
                        help="baseline: GT打分 | generate: 模型生成+打分 | "
                             "compare: 生成vsGT对比 | score: 对已有结果打分")
    parser.add_argument("--model", default="./output/gemma3-270m-svg-lora",
                        help="LoRA 模型路径 (mode=generate/compare 时需要)")
    parser.add_argument("--base-model", default="google/gemma-3-270m-it",
                        help="基础模型名称或路径")
    parser.add_argument("-o", "--output", default="results.json",
                        help="输出 JSON 路径 (默认: results.json)")
    parser.add_argument("--svg-field", default="svg_output",
                        help="flat JSONL 中 SVG 字段名 (mode=score 时)")
    parser.add_argument("--prompt-field", default="prompt",
                        help="flat JSONL 中 prompt 字段名 (mode=score 时)")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制评估样本数 (0=全部)")
    parser.add_argument("--no-verbose", action="store_true",
                        help="关闭逐条进度输出")

    args = parser.parse_args()

    # --- 加载数据 ---
    print(f"Loading data: {args.data}")
    if args.mode == "score":
        samples = load_flat_jsonl(args.data, args.svg_field, args.prompt_field)
    else:
        samples = load_chat_jsonl(args.data)

    if args.limit > 0:
        samples = samples[:args.limit]
    print(f"  Loaded {len(samples)} samples")

    # --- 执行 ---
    if args.mode == "baseline":
        result = run_baseline(samples)
    elif args.mode == "generate":
        result = run_generate(samples, args.model, args.base_model)
    elif args.mode == "compare":
        result = run_compare(samples, args.model, args.base_model)
    elif args.mode == "score":
        result = run_score(samples)

    # --- 保存 ---
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
