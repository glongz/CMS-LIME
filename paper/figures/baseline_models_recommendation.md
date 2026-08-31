# 推荐对比基线与大模型（多架构充分对比）

用于与 CMS-LIME 后处理管道对比的**基模型**建议如下，兼顾经典基线、不同架构与“大模型/重模型”，便于在论文中做充分对比。

---

## 一、建议纳入的对比模型

### 1. 经典 EEG 专用 CNN（必选）

| 模型 | 说明 | 理由 |
|------|------|------|
| **EEGNet** | 轻量 CNN，深度可分离卷积 + 时间卷积 | 当前主基线；BCI/癫痫领域最常用，便于复现与引用。 |
| **ShallowConvNet** | 浅层卷积（Schirrmeister et al.） | BNCI/EEG 标准基线，与 DeepConvNet 成对对比。 |
| **DeepConvNet** | 深层卷积，更多滤波器 | 代表“更深”的 CNN，与 Shallow 对比可看深度对敏感性/可解释性的影响。 |

### 2. 现代 CNN 变体（强烈推荐）

| 模型 | 说明 | 理由 |
|------|------|------|
| **EEG-Inception** | 多尺度 Inception 模块 + 可选 SE | 论文已提及；多尺度与 CMS-LIME 的多尺度 primitives 可对照。 |
| **TCN (Temporal Convolutional Network)** | 因果膨胀卷积，长程依赖 | 时序建模强，与 LSTM/Transformer 对比“纯卷积”的边界。 |

### 3. Transformer / 大模型类（推荐至少 1 个）

| 模型 | 说明 | 理由 |
|------|------|------|
| **EEG Transformer** | Patch 化 + 自注意力（可参考 EEG ViT / PatchTST 思路） | 代表“注意力/长程”架构；可验证后处理对非 CNN 是否同样有效。 |
| **SeizureFormer** | CNN patch 嵌入 + 多头自注意力 + SE | 近年发作预测专用 Transformer，若有公开实现可作强对比。 |
| **EEG-U-Transformer** | U 形 + 卷积与自注意力，时间步级输出 | 2025 检测竞赛表现突出，适合作为“大/重”模型代表。 |

### 4. 可选：更大/更重模型

| 模型 | 说明 | 理由 |
|------|------|------|
| **ResNet / 1D-ResNet** | 残差块堆叠 | 参数量与容量明显大于 EEGNet，可看“模型越大，后处理增益是否仍存在”。 |
| **预训练 + 微调** | 若使用 EEG 预训练（如 self-supervised）再微调 | 可讨论“预训练表示 + CMS-LIME 后处理”的协同效应。 |

---

## 二、实施建议

- **最少对比集**（可接受）：EEGNet + ShallowConvNet + DeepConvNet + **1 个 Transformer/大模型**（如 EEG Transformer 或 SeizureFormer/EEG-U-Transformer 若可得）。
- **推荐对比集**（充分）：上述 3 个经典 CNN + EEG-Inception + TCN + **1 个 Transformer**（共 6 个基模型），表格与图中统一报告 Baseline / +Biomarker / +Full。
- **统一设定**：所有基模型使用相同的输入（5 s 窗、相同预处理）、相同的 K-of-N/不应期与事件匹配规则，仅替换“窗级预测”的来源，以保证增益归因于后处理而非数据或规则差异。

---

## 三、与 Fig. 2–4 的对应

- **Fig. 2**：横轴为基模型（如 6 个），每组 3 根柱（Baseline / +Biomarker / +Full），分 Patient-specific 与 Cross-patient 两子图；可同时展示 Sensitivity 与 FDR。
- **Fig. 3**：按基模型分面（small multiples），每个子图仍为“基线敏感性 vs 完整后处理敏感性”散点，便于逐架构看低敏感性患者获益。
- **Fig. 4**：Ablation（Baseline / +Unfiltered / +Filtered / +Full）可按模型分面或分组柱状图，展示各架构上“生物标志物过滤 + 动态阈值”的边际贡献。

以上模型列表可直接用于扩展 experiments.tex 中的“Compared methods”与表格/图注。
