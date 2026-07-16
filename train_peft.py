"""
LoRA 微调脚本: Gemma 3 270M → SVG Logo 生成
使用 transformers + PEFT (不依赖 ms-swift)

用法:
    python train_peft.py

环境:
    pip install transformers peft accelerate datasets pandas

模型下载 (二选一):
    # HuggingFace
    export HF_TOKEN=your_token

    # ModelScope
    pip install modelscope
    python -c "from modelscope import snapshot_download; snapshot_download('LLM-Research/gemma-3-270m-it', local_dir='./gemma3-270m')"

参考: GitHub roboticcam/logo-detailed-prompt
"""

import os
import sys
import json
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset

# ============================================================================
# 超参数 (可直接修改, 也可通过命令行覆盖)
# ============================================================================

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="google/gemma-3-270m-it",
        metadata={"help": "HuggingFace 模型名称或本地路径"}
    )
    lora_rank: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "LoRA 目标模块, 逗号分隔"}
    )

@dataclass
class DataArguments:
    train_file: str = field(default="logo-detailed-prompt/train.jsonl")
    val_file: str = field(default="logo-detailed-prompt/valid.jsonl")
    max_length: int = field(default=4096)
    mask_prompt: bool = field(default=True)  # 只对 assistant token 计算 loss

@dataclass
class MyTrainingArguments(TrainingArguments):
    output_dir: str = field(default="./output/gemma3-270m-svg-lora")
    num_train_epochs: int = field(default=5)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-4)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.05)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=10)
    save_steps: int = field(default=50)
    eval_steps: int = field(default=50)
    save_total_limit: int = field(default=3)
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    max_grad_norm: float = field(default=1.0)
    report_to: str = field(default="tensorboard")
    remove_unused_columns: bool = field(default=False)
    seed: int = field(default=42)


# ============================================================================
# 数据处理
# ============================================================================

def load_jsonl_to_dataset(filepath: str) -> Dataset:
    """将 chat-format JSONL 转为 HuggingFace Dataset."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return Dataset.from_list(records)


def format_and_tokenize(
    examples: Dict[str, Sequence],
    tokenizer: AutoTokenizer,
    max_length: int,
    mask_prompt: bool,
) -> Dict[str, torch.Tensor]:
    """
    将 messages 列表转为 token ids + labels。
    - system/user token 的 label 设为 -100 (不计算 loss)
    - assistant token 保留原 token id 作为 label
    """
    all_input_ids = []
    all_labels = []

    for messages_list in examples["messages"]:
        # 用 tokenizer 的 chat template 格式化
        text = tokenizer.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=False,
        )

        # 分别 tokenize 各部分以生成 mask
        system_text = tokenizer.apply_chat_template(
            [messages_list[0]], tokenize=False, add_generation_prompt=False
        )
        user_text = tokenizer.apply_chat_template(
            messages_list[:2], tokenize=False, add_generation_prompt=False
        )

        sys_ids = tokenizer(system_text, add_special_tokens=False)["input_ids"]
        user_ids = tokenizer(user_text, add_special_tokens=False)["input_ids"]

        full_ids = tokenizer(text, truncation=True, max_length=max_length)["input_ids"]

        if mask_prompt:
            # 找到 assistant 部分开始的位置
            prompt_len = len(user_ids)
            labels = [-100] * min(prompt_len, len(full_ids)) + full_ids[prompt_len:]
            # 补齐
            if len(labels) < len(full_ids):
                labels += full_ids[len(labels):]
            labels = labels[:len(full_ids)]
        else:
            labels = full_ids.copy()

        all_input_ids.append(full_ids)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
    }


# ============================================================================
# 主训练逻辑
# ============================================================================

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- 1. 加载 tokenizer ---
    print(f"Loading tokenizer: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 2. 加载数据 ---
    print(f"Loading data: {data_args.train_file}")
    train_ds = load_jsonl_to_dataset(data_args.train_file)
    val_ds = load_jsonl_to_dataset(data_args.val_file)

    print(f"  Train: {len(train_ds)} samples, Valid: {len(val_ds)} samples")

    # --- 3. Tokenize ---
    print("Tokenizing...")
    train_ds = train_ds.map(
        lambda x: format_and_tokenize(
            x, tokenizer, data_args.max_length, data_args.mask_prompt
        ),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        lambda x: format_and_tokenize(
            x, tokenizer, data_args.max_length, data_args.mask_prompt
        ),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    # --- 4. 加载模型 ---
    print(f"Loading model: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # gradient checkpointing 兼容

    # --- 5. 配置 LoRA ---
    target_modules = [m.strip() for m in model_args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 6. Data Collator ---
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # --- 7. 训练 ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    # --- 8. 保存 ---
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"LoRA adapter saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()
