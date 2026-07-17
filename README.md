# SVG Logo 生成 — LoRA 微调 Gemma 3 270M

## 项目简介 / Overview

基于 **Gemma 3 270M** 模型，使用 219 条详细视觉描述 → SVG Logo 的配对数据进行 LoRA 监督微调，使小模型学会根据自然语言描述生成完整的矢量图形 Logo。

本项目为研究生课程《大语言模型数学》Part B 作业。训练数据来源于 [roboticcam/logo-detailed-prompt](https://github.com/roboticcam/logo-detailed-prompt)，由 Claude Sonnet 生成的高质量 SVG Logo 与其对应的详细视觉描述组成。

---

A LoRA fine-tuning project that teaches **Gemma 3 270M** to generate SVG logos from detailed visual descriptions. Trained on 219 text→SVG pairs produced by Claude Sonnet.

This is a graduate course assignment (Part B) on LLM mathematics and fine-tuning.

---

## 文件结构 / Repository Structure

```
├── adapter/                          # LoRA adapter 权重
│   ├── adapter_config.json           # LoRA 配置文件
│   └── adapter_model.safetensors    # LoRA 权重（Git LFS）
├── reward.py                         # SVG Logo 三维度奖励函数
├── train_config.yaml                 # ms-swift LoRA 训练超参数
├── train_config.json                 # 同上（JSON 格式，更兼容）
├── run_train.py                      # 训练启动脚本（Python API）
├── convert_data.py                   # 数据格式转换脚本
├── eval_self.py                      # 自评估脚本（推理 + 评分）
├── run.ipynb                         # Colab 训练 Notebook
├── results_baseline.json             # Sonnet ground-truth 评分
├── results_generated.json            # LoRA 模型生成评分
├── report.md                         # 实验报告（中文）
└── README.md                         # 本文件
```

---

## 环境要求 / Requirements

| Component | Version | Notes |
|---|---|---|
| Python | 3.12 | |
| ms-swift | 4.4.1 | `pip install ms-swift` |
| transformers | 5.12.1 | |
| peft | 0.19.1 | |
| datasets | 3.0.1 | ⚠️ 4.x 不兼容, 3.2+ 需测试 |
| torch | 2.11.0+cu128 | |
| GPU | Tesla T4 15GB | 仅支持 fp16（无 bf16） |

```bash
pip install ms-swift==4.4.1 transformers peft datasets==3.0.1 accelerate torchao>=0.16.0
```

---

## 快速开始 / Quick Start

### 1. 下载模型和数据

```bash
# 下载 Gemma 3 270M（HuggingFace，需授权）
huggingface-cli download google/gemma-3-270m-it --local-dir /content/gemma3-270m

# 或从 ModelScope 下载（国内更快）
pip install modelscope
modelscope download --model LLM-Research/gemma-3-270m-it --local_dir /content/gemma3-270m

# 下载训练数据
git clone https://github.com/roboticcam/logo-detailed-prompt /content/logo-detailed-prompt
```

### 2. 转换数据格式

ms-swift 4.4.1 在某些环境中处理 chat-format JSONL 存在兼容性问题，建议转为扁平格式：

```bash
python convert_data.py
```

### 3. 启动训练

```bash
# 方式 A：Python 脚本（推荐，绕过 --config YAML 解析问题）
python run_train.py

# 方式 B：ms-swift CLI（JSON config）
swift sft --config train_config.json
```

### 4. 评估模型

```bash
# 基线分（Sonnet ground-truth）
python eval_self.py -d /content/logo-detailed-prompt/valid.jsonl -m baseline -o results_baseline.json

# LoRA 模型生成 + 评分
python eval_self.py -d /content/logo-detailed-prompt/valid.jsonl -m generate \
    --model /content/output/.../checkpoint-100 --base-model /content/gemma3-270m \
    -o results_generated.json
```

---

## 实验结果 / Results

| 模型 Model | avg | min | max |
|---|---|---|---|
| **Sonnet (ground-truth)** | 0.9857 | 0.9065 | 1.0000 |
| **Gemma 3 270M + LoRA** | 0.4056 | 0.3135 | 0.7323 |
| **差距 Gap** | **-0.5801** | — | — |

评分使用自建的 `reward.py`，从 **结构有效性**（XML 合法性）、**设计规则合规性**（禁止元素检查）、**Prompt 对齐度**（颜色/形状匹配）三个维度综合评估（详见 [`report.md`](report.md)）。

Scores are computed by `reward.py` across three dimensions: structural validity (XML), design-rule compliance, and prompt–SVG alignment. See [`report.md`](report.md) for details.

---

## 已知问题 / Known Issues

| 问题 | 原因 | 解决方案 |
|---|---|---|
| `--config xxx.yaml` 不生效 | ms-swift 4.4.1 + HfArgumentParser YAML 解析兼容性 | 使用 `train_config.json` 或 `run_train.py` |
| `datasets.features` ImportError | datasets 4.x 移除 `Json`/`List` 类型 | 降级至 `datasets==3.0.1` |
| T4 训练中 OOM | 保存 + 评估 + 生成同时触发 | `eval_steps` 设为 > 训练总步数 |
| 生成 SVG 长度不稳定 | 270M 模型对 SVG 结构化输出控制力弱 | 增加数据量、使用更大模型、添加 stop_words |

---

## 关键参数 / Key Hyperparameters

| Parameter | Value | Note |
|---|---|---|
| Model | Gemma 3 270M it | |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 | `target_modules=all-linear` |
| Learning rate | 2e-4 | cosine schedule |
| Epochs | 5 | |
| Batch size | 1 × 8 grad_accum = 8 | T4 15GB 安全值 |
| Max length | 4,096 tokens | |
| Precision | fp16 | T4 不支持 bf16 |
| Training time | ~8.5 min | T4, 140 steps |

---

## 许可证 / License

训练数据来自 [roboticcam/logo-detailed-prompt](https://github.com/roboticcam/logo-detailed-prompt)，本仓库仅包含 LoRA adapter 权重和训练代码。
