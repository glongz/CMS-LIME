# 两阶段基元选拔框架

## 概述

本框架实现了一个两阶段的基元选拔流程，用于从EEG数据中提取和评估生物标志物候选：

1. **第一场选拔（基元提取）**：基于先验信息从原始数据中提取候选基元
2. **第二场选拔（因果扰动解释）**：通过类LIME方法评估基元在/不在对模型预测的影响

## 文件结构

```
├── prior_knowledge_config.py          # 先验信息配置
├── primitive_selection_stage1.py      # 第一场选拔：基元提取
├── primitive_selection_stage2.py      # 第二场选拔：因果扰动解释
├── two_stage_primitive_selection.py   # 主框架整合
├── run_two_stage_selection.py         # 主程序（集成CHB-MIT数据集）
└── utils/
    └── visualization.py                # 可视化工具
```

## 使用方法

### 基本使用

```python
from prior_knowledge_config import create_default_prior_config
from two_stage_primitive_selection import TwoStagePrimitiveSelection
import numpy as np

# 1. 创建配置
prior_config = create_default_prior_config()

# 2. 创建框架（需要提供模型）
framework = TwoStagePrimitiveSelection(
    prior_config=prior_config,
    model=your_model.predict_proba,  # 模型需要提供predict_proba方法
    n_perturbations=10,
    importance_threshold=0.1,
    top_k_biomarkers=20
)

# 3. 第一场选拔：提取候选基元
X_train = ...  # 训练数据 (n_samples, n_channels, n_timepoints)
y_train = ...  # 训练标签
framework.fit(X_train, y_train)

# 4. 第二场选拔：评估基元重要性
X_sample = ...  # 样本数据 (n_channels, n_timepoints)
y_sample = ...  # 样本标签
biomarkers = framework.explain(X_sample, y_sample)

# 5. 保存结果
framework.save_results("output_folder")
```

### 使用CHB-MIT数据集

直接运行主程序：

```bash
python run_two_stage_selection.py
```

程序会自动：
1. 加载CHB-MIT数据集
2. 加载EEG模型
3. 执行两阶段选拔
4. 保存结果和可视化

## 先验信息配置

先验信息配置包含三种基元类型的参数：

### Shapelet先验信息
- `min_length`: 最小长度 (默认: 10)
- `max_length`: 最大长度 (默认: 100)
- `information_gain_threshold`: 信息增益阈值 (默认: 0.0)
- `n_shapelets_selected`: 选择的shapelet数量 (默认: 50)

### 微状态先验信息
- `n_microstates`: 微状态数量 (默认: 4)
- `gfp_threshold_percentile`: GFP阈值百分位数 (默认: 90.0)
- `min_duration`: 最小持续时间 (默认: 3)
- `min_explained_variance`: 最小解释方差 (默认: 0.3)

### 时频先验信息
- `freq_bands`: 频带定义
  - Delta: (0.5, 4) Hz
  - Theta: (4, 8) Hz
  - Alpha: (8, 13) Hz
  - Beta: (13, 30) Hz
  - Gamma: (30, 100) Hz
- `time_window_size`: 时间窗口大小 (默认: 50)
- `overlap_ratio`: 重叠比例 (默认: 0.5)

## 输出结果

运行完成后，会在结果文件夹中生成：

1. **prior_knowledge_config.json**: 先验信息配置
2. **prior_knowledge_record.json**: 先验信息使用记录
3. **candidate_primitives.json**: 第一场选拔的候选基元
4. **biomarkers.json**: 第二场选拔的生物标志物
5. **selection_report.txt**: 综合报告
6. **stage1_visualization.png**: 第一场选拔可视化
7. **stage2_visualization.png**: 第二场选拔可视化
8. **final_summary.json**: 最终摘要

## 模型要求

模型需要提供 `predict_proba` 方法，接受输入：
- 形状: `(batch_size, n_channels, n_timepoints)` 或 `(n_channels, n_timepoints)`
- 返回: `(batch_size, n_classes)` 的概率分布

示例：

```python
class YourModel:
    def predict_proba(self, X):
        # X: (batch, channels, timepoints) 或 (channels, timepoints)
        # 返回: (batch, n_classes) 的概率分布
        return probabilities
```

## 注意事项

1. 数据格式：框架期望数据格式为 `(n_samples, n_channels, n_timepoints)`
2. CHB-MIT数据集：默认使用22通道，1280时间点（5秒@256Hz）
3. 计算效率：第一场选拔可能较慢，建议限制训练样本数量
4. 内存使用：大量基元可能占用较多内存

## 扩展

### 自定义先验信息

```python
from prior_knowledge_config import PriorKnowledgeConfig, ShapeletPriors, MicrostatePriors, TimeFreqPriors

# 创建自定义配置
shapelet_priors = ShapeletPriors(
    min_length=20,
    max_length=150,
    n_shapelets_selected=100
)

prior_config = PriorKnowledgeConfig(
    shapelet_priors=shapelet_priors,
    # ... 其他配置
)
```

### 自定义可视化

参考 `utils/visualization.py` 中的函数，可以创建自定义可视化。

## 作者

CMS-LIME Framework
Date: 2025
