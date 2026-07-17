# SVG Logo 生成 — LoRA 微调 Gemma 3 270M

## 项目简介 / Overview

基于 **Gemma 3 270M**，使用 219 条详细视觉描述 → SVG Logo 配对数据进行 LoRA 监督微调，使小模型学会根据自然语言描述生成完整矢量图形 Logo。研究生课程《大语言模型数学》Part B 作业。

训练数据来自 [roboticcam/logo-detailed-prompt](https://github.com/roboticcam/logo-detailed-prompt)，由 Claude Sonnet 生成。

---

LoRA fine-tuning **Gemma 3 270M** to generate SVG logos from detailed visual descriptions. Trained on 219 text→SVG pairs produced by Claude Sonnet. Graduate course assignment (LLM Mathematics, Part B).

---

## 文件结构 / Repository Structure

```
├── adapter/                          # LoRA adapter 权重（Git LFS）
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── reward.py                         # SVG Logo 四维度奖励函数（v2）
├── train_config.yaml                 # ms-swift LoRA 训练超参数
├── train_config.json                 # JSON 格式备选
├── run_train.py                      # 训练启动脚本
├── convert_data.py                   # 数据格式转换（chat → flat）
├── eval_self.py                      # 评估脚本（推理 + 评分）
├── export_svg.py                     # 导出 SVG / HTML 对比图
├── run.ipynb                         # Colab 训练 Notebook
├── results.json                      # Sonnet GT vs LoRA 评分对比
├── report.md                         # 实验报告（中文）
└── README.md                         # 本文件
```

---

## 环境 / Requirements

| Component | Version | Notes |
|---|---|---|
| Python | 3.12 | |
| ms-swift | 4.4.1 | |
| transformers | 5.12.1 | |
| peft | 0.19.1 | |
| datasets | 3.0.1 | ⚠️ 4.x 不兼容 |
| torch | 2.11.0+cu128 | |
| GPU | Tesla T4 15GB | fp16 only |

```bash
pip install ms-swift==4.4.1 transformers peft datasets==3.0.1 accelerate "torchao>=0.16.0"
```

---

## 快速开始 / Quick Start

### 1. 准备模型和数据

```bash
# HuggingFace（需授权）
huggingface-cli download google/gemma-3-270m-it --local-dir /content/gemma3-270m

# ModelScope（国内）
pip install modelscope
modelscope download --model LLM-Research/gemma-3-270m-it --local_dir /content/gemma3-270m

# 数据
git clone https://github.com/roboticcam/logo-detailed-prompt /content/logo-detailed-prompt
```

### 2. 转换数据格式

```bash
python convert_data.py
```

### 3. 训练

```bash
python run_train.py
# 或: swift sft --config train_config.json
```

### 4. 评估

```bash
# 基线分
python eval_self.py -d valid.jsonl -m baseline -o results_baseline.json

# 模型生成 + 评分
python eval_self.py -d valid.jsonl -m generate \
    --model /content/output/.../checkpoint-100 \
    --base-model /content/gemma3-270m -o results_generated.json

# 导出 SVG 对比图
python export_svg.py
```

---

## 实验结果 / Results（reward.py v2）

| Model | avg | min | max |
|---|---|---|---|
| **Sonnet (ground-truth)** | 0.9426 | 0.9058 | 0.9859 |
| **Gemma 3 270M + LoRA** | 0.3645 | 0.3085 | 0.6898 |
| **Gap** | **0.5781** | — | — |

评分维度：结构有效性(0.25) + 设计合规性(0.20) + 视觉质量(0.15) + Prompt对齐度(0.40)。详见 [`report.md`](report.md)。

---

## 已知问题 / Known Issues

| Problem | Cause | Fix |
|---|---|---|
| `--config xxx.yaml` not working | ms-swift 4.4.1 YAML parsing bug | Use `train_config.json` or `run_train.py` |
| `datasets.features` ImportError | datasets 4.x removed `Json`/`List` | `pip install datasets==3.0.1` |
| T4 OOM at eval | save + eval + generate at same step | Set `eval_steps` > total steps |
| SVG length unstable | 270M poor structured output control | Larger model, `stop_words=["</svg>"]` |

---

## 关键参数 / Key Hyperparameters

| Parameter | Value |
|---|---|
| Model | Gemma 3 270M it |
| LoRA (rank/alpha/dropout) | 16 / 32 / 0.05 |
| Learning rate | 2e-4, cosine |
| Epochs | 5 |
| Batch size | 1 × 8 = 8 |
| Precision | fp16 |
| Training time | ~8.5 min (T4) |

---

## 许可证 / License

训练数据：[roboticcam/logo-detailed-prompt](https://github.com/roboticcam/logo-detailed-prompt)。本仓库仅含 LoRA adapter 权重与训练/评估代码。
