"""
ms-swift 4.4.1 LoRA 训练启动脚本 — Gemini 3 270M → SVG Logo
绕过 --config YAML 解析问题，直接以 Python 方式传入参数

用法:
    python gemma3-logo-homework/run_train.py

环境: Tesla T4 15GB / Ubuntu 22.04 / CUDA 13.0 / fp16
"""

import os
import sys

# 确保 ms-swift 可导入
from swift.pipelines.train.sft import SwiftSft
from swift.arguments import SftArguments


def build_args() -> SftArguments:
    """构建 ms-swift 4.4.1 训练参数 (T4 GPU 适配)."""
    return SftArguments(
        # ---- 模型与数据 ----
        model="/content/gemma3-270m",
        tuner_type="lora",
        dataset="/content/logo-detailed-prompt/train.jsonl",
        val_dataset="/content/logo-detailed-prompt/valid.jsonl",
        max_length=4096,
        truncation_strategy="right",

        # ---- LoRA ----
        lora_rank=16,
        lora_alpha=32,
        lora_dropout_p=0.05,
        target_modules="all-linear",

        # ---- 训练超参 (T4 15GB 适配) ----
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=5,
        per_device_train_batch_size=1,      # T4 15GB 保守值
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,       # 有效 batch = 1 × 8 = 8

        # ---- 优化器 ----
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,

        # ---- 精度: T4 只支持 fp16, 不支持 bf16 ----
        torch_dtype="float16",

        # ---- 保存 & 日志 ----
        output_dir="/content/output",
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        save_total_limit=3,

        # ---- 评估时生成 ----
        predict_with_generate=True,
        max_new_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,

        # ---- 其他 ----
        seed=42,
        dataloader_num_workers=2,
    )


def main():
    args = build_args()
    print(f"[INFO] 模型: {args.model}")
    print(f"[INFO] 数据: {args.dataset} / {args.val_dataset}")
    print(f"[INFO] 精度: {args.torch_dtype}")
    print(f"[INFO] batch: {args.per_device_train_batch_size} × {args.gradient_accumulation_steps} = "
          f"{args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"[INFO] LoRA rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"[INFO] max_length={args.max_length}, epochs={args.num_train_epochs}")

    trainer = SwiftSft(args)
    trainer.main()


if __name__ == "__main__":
    main()
