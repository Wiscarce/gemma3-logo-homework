# 实验报告：LoRA 微调 Gemma 3 270M 生成 SVG Logo

## 一、任务概述

使用 `train.jsonl`（219 条）和 `valid.jsonl`（17 条）训练数据，对 **Gemma 3 270M** 进行 LoRA 监督微调（SFT），使其能够根据详细的视觉描述（text prompt）生成完整的 SVG Logo。训练数据中的每条样本包含 system prompt（SVG 设计规范）、user prompt（详细视觉描述）和 assistant response（Claude Sonnet 生成的 SVG 文档，`viewBox="0 0 256 256"`）。

训练完成后，使用自建的 `reward.py` 评分函数评估生成质量，并与 Sonnet ground-truth 进行对比。

---

## 二、数据

| 数据集 | 样本数 | 格式 |
|---|---|---|
| `train.jsonl` | 219 | chat-format: system / user / assistant |
| `valid.jsonl` | 17 | 同上（来源一致，由 Sonnet 生成的高质量 SVG） |

- SVG 长度范围：419 ~ 5,348 字符，平均 ~1,800 字符
- 所有 SVG 均以 `<svg xmlns="..." viewBox="0 0 256 256">` 开头
- 仅使用矢量图元（`<path>`, `<circle>`, `<rect>`, `<polygon>` 等），无外部引用

---

## 三、方法

### 3.1 模型与微调

| 配置项 | 值 |
|---|---|
| 基座模型 | `google/gemma-3-270m-it` |
| 微调方法 | LoRA (rank=16, alpha=32, dropout=0.05, target_modules=all-linear) |
| 框架 | ms-swift 4.4.1 |
| 训练环境 | Tesla T4 15GB, CUDA 13.0, fp16 |
| 精度 | torch.float16（T4 不支持 bf16） |

### 3.2 超参数

| 参数 | 值 |
|---|---|
| Learning rate | 2e-4, cosine schedule |
| Epochs | 5 |
| Batch size | 1 × 8 gradient accumulation = 8 |
| Max sequence length | 4,096 tokens |
| Optimizer | AdamW (weight_decay=0.01) |
| Warmup ratio | 0.05 |

### 3.3 Loss 掩码

训练时仅对 assistant（SVG）部分的 token 计算 loss，system 和 user 部分的 token label 设为 -100（忽略）。ms-swift 对扁平格式（system/query/response）的数据自动处理此逻辑。

### 3.4 训练耗时

- 总步数：140 步（219 样本 × 5 epoch / batch 8 ≈ 137 步，实际 140）
- 训练时间：**约 8.5 分钟**（T4）

---

## 四、Reward 函数设计

自建的 `reward.py` 从三个维度评估 SVG 质量（总分 0~1）：

| 维度 | 权重 | 评估内容 |
|---|---|---|
| **结构有效性** | 0.30 | XML 可解析性、xmlns/viewBox 是否正确 |
| **设计规则合规性** | 0.25 | 禁止元素检测（`<image>`, `<script>` 等）、标签白名单、颜色使用合理性 |
| **Prompt 对齐度** | 0.45 | 颜色匹配（hex + 颜色名映射）、形状/视觉元素检测、SVG 复杂度 vs 描述复杂度 |

对齐度评分采用启发式方法：从 prompt 中提取颜色名称（如 "coral", "teal"），映射到 hex 值，检查 SVG 中是否存在近似颜色；从 SVG 标签结构中检测常见形状模式（circle, hexagon, sun, wave 等），与 prompt 中提及的概念进行交集匹配；同时考虑了 SVG 元素复杂度是否与 prompt 的长度匹配，以及非标签文本的比例。

Sonnet ground-truth SVG 在验证集上平均得分为 **0.9857**，验证了 reward 函数对高质量 SVG 的判断与人工预期一致。

---

## 五、实验结果

### 5.1 评分对比

| 模型 | avg | min | max |
|---|---|---|---|
| **Sonnet (ground-truth)** | 0.9857 | 0.9065 | 1.0000 |
| **Gemma 3 270M + LoRA** | 0.4056 | 0.3135 | 0.7323 |
| **差距** | **-0.5801** | — | — |

### 5.2 生成质量分析

对 17 条验证样本逐一检查生成结果，发现以下模式：

| 生成长度 | 样本数 | 典型特征 | reward 范围 |
|---|---|---|---|
| < 200 字符 | 8 条 | SVG 极短（92~183 字符），通常仅含 1~2 个元素，未完成即截断 | 0.31~0.38 |
| 400~1,400 字符 | 5 条 | 结构较完整，有多个元素和合理颜色 | 0.42~0.73 |
| > 8,000 字符 | 4 条 | 生成过长，包含大量冗余或重复元素 | 0.33~0.45 |

**核心问题：**

1. **生成截断**：约半数输出在 200 字符以内中止，SVG 结构不完整（缺少闭合标签、有效内容极少）。可能原因：270M 模型对 SVG 语法的学习不充分，或 `max_new_tokens=2048` 的配置下模型过早生成了 EOS token。

2. **长度失控**：约 1/4 样本生成超过 8,000 字符，远长于训练数据中平均 1,800 字符的 SVG。模型未能学会"生成适中、完整的 SVG"的模式。

3. **颜色/形状对齐弱**：即使是较好的样本，也缺乏对 prompt 中详细视觉描述的精确重现——模型倾向于生成"泛化的 SVG"，而非严格遵循 prompt 中指定的具体元素。

---

## 六、Sonnet 表现更优的原因分析

1. **模型规模**：Sonnet（参数量远超 270M）具有更强的文本理解和结构化输出能力。SVG 生成需要同时理解自然语言描述、空间布局逻辑和严格的 XML 语法——这对小模型是三重挑战。

2. **训练数据规模不足**：仅 219 条训练样本，且每条样本的 SVG 结构差异大（不同 logo 的元素组合、颜色方案、布局差异显著），模型难以在 5 个 epoch 内学到泛化的"SVG 生成能力"。

3. **预训练差异**：Gemma 3 270M 的预训练语料中，SVG 代码占比极低。而 Sonnet 经过 RLHF 和其他对齐训练，生成结构化输出的能力更强。

4. **生成控制**：Sonnet 通过 system prompt 中的显式指令（"Output ONLY the SVG"）能够严格控制输出格式；而小模型在指令遵循（instruction following）方面天然较弱。

---

## 七、改进方向

1. **增大模型规模**：使用 Gemma 3 1B/4B 或 Qwen2.5-1.5B 等更大模型，提升指令遵循和结构化生成能力。

2. **数据增强**：
   - 利用 Sonnet API 生成更多训练数据（当前只在 219 条上训练）
   - 添加"负面样本"（破损的 SVG、不完整的 SVG）作为对比，帮助模型学习边界
   - 加入多样化 prompt（简化版、详细版、多语言版）

3. **改进训练策略**：
   - 使用 DPO/RLHF 替代纯 SFT，让 reward 直接参与优化
   - 增加 response_prefix="<svg" 强制从正确位置开始生成
   - 调高 max_length 至 8192 以包含完整的 SVG
   - 使用更大的 LoRA rank（如 64 或 128）

4. **推理优化**：
   - 使用 beam search 替代随机采样，生成更稳定的结果
   - 添加后处理：自动修复不平衡的标签、截断多余的重复内容
   - 使用 stop_words=["</svg>"] 让模型在 SVG 闭合时立即停止

---

## 八、交付物清单

| 文件 | 说明 |
|---|---|
| `adapter_config.json` + `adapter_model.safetensors` | LoRA 权重（checkpoint-100） |
| `reward.py` | SVG Logo 三维度奖励函数 |
| `train_config.yaml` | ms-swift LoRA 训练超参数配置 |
| `results_baseline.json` | Sonnet ground-truth 评分结果（avg=0.9857） |
| `results_generated.json` | LoRA 模型生成评分结果（avg=0.4056） |
| `report.md` | 本报告 |
