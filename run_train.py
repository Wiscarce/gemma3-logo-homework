"""
ms-swift 4.4.1 LoRA training launcher — Gemma 3 270M → SVG Logo
Bypasses --config YAML parsing, passes args directly to SwiftSft

Usage:
    python gemma3-logo-homework/run_train.py

Env: Tesla T4 15GB / Ubuntu 22.04 / CUDA 13.0 / fp16
"""

import os
import sys

from swift.pipelines.train.sft import SwiftSft
from swift.arguments import SftArguments


def build_args() -> SftArguments:
    """Build ms-swift 4.4.1 training args (T4 GPU tuned)."""
    return SftArguments(
        # ---- Model & Data ----
        model="/content/gemma3-270m",
        tuner_type="lora",
        dataset="/content/logo-detailed-prompt/train_flat.jsonl",
        val_dataset="/content/logo-detailed-prompt/valid_flat.jsonl",
        max_length=4096,
        truncation_strategy="right",

        # ---- LoRA ----
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",

        # ---- Training (T4 15GB safe) ----
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,       # effective batch = 1 x 8 = 8

        # ---- Optimizer ----
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,

        # ---- Precision: T4 only supports fp16 ----
        torch_dtype="float16",

        # ---- Save & Log ----
        output_dir="/content/output",
        logging_steps=10,
        save_steps=50,
        eval_steps=200,                      # NOT synced with save_steps to avoid T4 OOM
        save_total_limit=3,

        # ---- Eval generation (lighter to fit T4) ----
        predict_with_generate=True,
        max_new_tokens=2048,                 # reduced from 4096 to save VRAM during eval
        temperature=0.7,
        top_p=0.9,

        # ---- Misc ----
        seed=42,
        dataloader_num_workers=2,
    )


def main():
    args = build_args()
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Data: {args.dataset} / {args.val_dataset}")
    print(f"[INFO] Dtype: {args.torch_dtype}")
    print(f"[INFO] Batch: {args.per_device_train_batch_size} x {args.gradient_accumulation_steps} = "
          f"{args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"[INFO] LoRA rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"[INFO] max_length={args.max_length}, epochs={args.num_train_epochs}")

    trainer = SwiftSft(args)
    trainer.main()


if __name__ == "__main__":
    main()
