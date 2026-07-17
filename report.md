# 实验报告：LoRA 微调 Gemma 3 270M 生成 SVG Logo

## 一、任务概述

使用 219 条 text→SVG 配对数据对 **Gemma 3 270M** 进行 LoRA 监督微调（SFT），使其根据详细视觉描述生成完整的矢量 Logo。训练数据来自 [roboticcam/logo-detailed-prompt](https://github.com/roboticcam/logo-detailed-prompt)，由 Claude Sonnet 生成。训练完成后使用自建 `reward.py`（四维度评分函数）评估生成质量，并与 Sonnet ground-truth 对比分析。

---

## 二、数据

| 数据集 | 样本数 | 格式 |
|---|---|---|
| `train.jsonl` | 219 | chat-format: system / user / assistant |
| `valid.jsonl` | 17 | 同上（Sonnet 生成的高质量 SVG） |

- SVG 长度：419 ~ 5,348 字符，平均 ~1,800 字符
- 所有 SVG 均以 `<svg xmlns="..." viewBox="0 0 256 256">` 开头
- 仅使用矢量图元（`<path>`, `<circle>`, `<rect>`, `<polygon>` 等），无外部引用

---

## 三、方法

### 3.1 模型与微调

| 配置 | 值 |
|---|---|
| 基座模型 | `google/gemma-3-270m-it` |
| 微调方法 | LoRA (rank=16, alpha=32, dropout=0.05) |
| 目标模块 | `all-linear` |
| 框架 | ms-swift 4.4.1 |
| 环境 | Tesla T4 15GB / CUDA 13.0 / fp16 |

### 3.2 超参数

| 参数 | 值 |
|---|---|
| Learning rate | 2e-4, cosine schedule |
| Epochs | 5 |
| Effective batch size | 1 × 8 gradient accumulation = 8 |
| Max sequence length | 4,096 tokens |
| Optimizer | AdamW (weight_decay=0.01) |
| Warmup ratio | 0.05 |

### 3.3 Loss 掩码

训练时仅对 assistant（SVG）部分的 token 计算 loss，system 和 user token 忽略。使用扁平格式（system/query/response）由 ms-swift 自动处理。

### 3.4 训练耗时

140 步，约 8.5 分钟（Tesla T4）。

---

## 四、Reward 函数设计（v2 增强版）

自建 `reward.py` 从 **四个维度** 评估 SVG 质量（总分 0~1），v2 版核心改进包括 CIE76 感知色差、ElementTree 树遍历形状检测、语义关键词分组、阶梯评分防聚集。

| 维度 | 权重 | 评估内容 |
|---|---|---|
| **结构有效性** | 0.25 | XML 解析、xmlns/viewBox、截断检测、标签平衡 |
| **设计合规性** | 0.20 | 禁止元素、标签白名单、颜色使用合理性 |
| **视觉质量** | 0.15 | 形状多样性、坐标越界、defs/gradient 使用 |
| **Prompt 对齐度** | 0.40 | CIE76 色差匹配、形状检测、语义组覆盖、复杂度对齐 |

对齐度采用阶梯式评分（不足 30%→0.2，30-60%→0.5，60-80%→0.7，80%+→0.9+），消除 GT 分数在 0.99 的顶部聚集效应。每次评分返回 `failure_reasons` 列表，标明具体扣分原因。

---

## 五、实验结果

### 5.1 评分对比（reward.py v2）

| 模型 | avg | min | max |
|---|---|---|---|
| **Sonnet (ground-truth)** | 0.9426 | 0.9058 | 0.9859 |
| **Gemma 3 270M + LoRA** | 0.3645 | 0.3085 | 0.6898 |
| **差距** | **0.5781** | — | — |

GT 评分不再触碰天花板（max < 0.99），LoRA 模型得分底线更低（< 0.31），reward 区分度显著提升。

### 5.2 生成质量分析

对 17 条验证样本逐一分析：

| 类型 | 样本数 | 特征 | reward |
|---|---|---|---|
| 严重截断 (< 200 字符) | 8 条 | 仅含 1~2 个元素，无闭合标签或过早 EOS | 0.31~0.35 |
| 部分完整 (400~1,400 字符) | 5 条 | 结构较完整，有多元素和颜色，但细节不精确 | 0.42~0.69 |
| 生成失控 (> 8,000 字符) | 4 条 | 大量冗余重复元素，远超训练数据平均长度 | 0.33~0.40 |

**核心问题：**

1. **生成截断**（~50% 样本）：270M 模型对 SVG 语法的学习不充分，EOS token 过早出现
2. **长度失控**（~25% 样本）：模型未能学会控制 SVG 的合理长度
3. **对齐不精确**：即使较好的样本也倾向于生成泛化的 SVG，而非严格遵循 prompt 中指定的具体颜色、形状和布局

---

## 六、Sonnet 表现更优的原因分析

1. **模型规模差距**：Sonnet 参数量远超 270M，SVG 生成需同时理解自然语言、空间布局和 XML 语法，对小模型是三重挑战
2. **训练数据有限**：仅 219 条样本，且各 logo 风格差异大，模型难以学到泛化的 SVG 生成能力
3. **预训练差异**：Gemma 3 270M 预训练语料中 SVG 代码占比极低；Sonnet 经过 RLHF 对齐，结构化输出能力强得多
4. **指令遵循**：Sonnet 能严格遵循 system prompt 的格式约束；270M 模型指令遵循能力天然较弱

---

## 七、改进方向

1. **增大模型**：使用 Gemma 3 1B/4B 或 Qwen2.5-1.5B
2. **数据增强**：利用 Sonnet API 扩增训练数据，加入负面样本
3. **训练策略**：使用 DPO/RLHF 让 reward 直接参与优化；增加 `response_prefix="<svg"`；提升 LoRA rank 至 64+
4. **推理控制**：使用 `stop_words=["</svg>"]` 精确截断；beam search 替代随机采样；添加 SVG 后处理修复

---

## 八、交付物清单

| 文件 | 说明 |
|---|---|
| `adapter/adapter_config.json` + `adapter_model.safetensors` | LoRA 权重（checkpoint-100），Git LFS |
| `reward.py` | SVG Logo 四维度奖励函数（v2 增强版） |
| `train_config.yaml` | ms-swift 4.4.1 LoRA 训练超参数 |
| `results.json` | Sonnet GT vs LoRA 评分对比 |
| `report.md` | 本报告 |
