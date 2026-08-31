# 统一基元选拔框架

## 概述

本框架实现了**合并的基元提取和重要性评估流程**，解决了原有两阶段框架中基元匹配的问题。

### 核心优势

1. **解决匹配问题**：基元从当前样本提取，确保存在性
2. **简化流程**：一次遍历完成提取和评估
3. **提高效率**：避免重复匹配和验证
4. **更准确**：每个基元的重要性基于其实际存在的位置计算

### 工作流程

```
样本 → 提取基元 → 立即评估重要性 → 筛选高重要性基元 → 保存为生物标志物
```

## 文件结构

```
unified_primitive_selection/
├── __init__.py                          # 包初始化
├── config.py                            # 配置管理
├── primitive_importance_evaluator.py   # 重要性评估器
├── unified_primitive_extractor.py      # 统一提取器（提取+评估）
├── unified_selection_framework.py      # 主框架类
├── run_unified_selection.py            # 主程序
├── README.md                            # 本文件
└── utils/
    ├── __init__.py
    └── visualization.py                # 可视化工具
```

## 使用方法

### 基本使用

```python
from unified_primitive_selection import (
    UnifiedPrimitiveSelection,
    create_default_config
)

# 1. 创建配置
config = create_default_config()

# 2. 创建框架（需要提供模型）
framework = UnifiedPrimitiveSelection(
    config=config,
    model=your_model.predict_proba
)

# 3. 处理样本
X_samples = ...  # (n_samples, n_channels, n_timepoints)
y_samples = ...  # 标签

sample_biomarkers = framework.process_samples(X_samples, y_samples)

# 4. 获取top K生物标志物
top_biomarkers = framework.get_top_biomarkers(top_k=20)

# 5. 保存结果
framework.save_results("output_folder")
```

### 运行主程序

```bash
# 从项目根目录运行
python unified_primitive_selection/run_unified_selection.py
```

## 配置说明

### 先验配置（复用）

- Shapelet分析参数
- 微状态分析参数
- 时频分析参数
- 基元数量配置（数量模式）

### 评估配置（新增）

- `n_perturbations`: 每个基元的扰动次数（默认10）
- `stability_iterations`: 稳定性迭代次数（默认3）
- `importance_threshold`: 重要性阈值（默认0.01）
- `top_k_biomarkers`: Top K生物标志物数量（默认20）
- `min_importance_score`: 最小重要性分数（默认0.001）

## 输出结果

### 保存的文件

1. **config.json**: 配置信息
2. **all_biomarkers.json**: 所有提取的生物标志物
3. **top_biomarkers.json**: Top K生物标志物
4. **sample_biomarkers.json**: 每个样本的生物标志物
5. **statistics.json**: 统计信息

### 生物标志物格式

每个生物标志物包含：
- `primitive_id`: 基元ID
- `primitive_type`: 基元类型（shapelet/microstate/timefreq）
- `channels`: 通道列表
- `time_range`: 时间范围
- `importance_score`: **重要性分数**（关键指标）
- `quality_score`: 质量分数
- `metadata`: 元数据

## 关键改进

### 与原两阶段框架的对比

| 特性 | 两阶段框架 | 统一框架 |
|------|-----------|---------|
| 基元提取 | 从训练数据批量提取 | 从当前样本提取 |
| 重要性评估 | 后续单独评估 | 提取后立即评估 |
| 基元匹配 | 需要匹配到测试样本 | 无需匹配（已存在） |
| 准确性 | 可能存在匹配错误 | 更准确（基于实际位置） |

### 解决的问题

1. **基元匹配问题**：原框架中，基元从训练数据提取，在测试时需要匹配到测试样本，可能存在匹配错误。新框架直接从测试样本提取，确保基元存在。

2. **效率问题**：原框架需要两遍遍历（提取+评估），新框架一次遍历完成。

3. **准确性问题**：新框架中，每个基元的重要性基于其实际存在的位置计算，更准确。

## 注意事项

1. **计算成本**：对每个样本都要进行基元提取和评估，计算成本较高。建议：
   - 限制处理的样本数量
   - 减少每类基元的提取数量
   - 使用GPU加速模型预测

2. **内存使用**：每个样本的基元都会保存在内存中，处理大量样本时注意内存使用。

3. **模型要求**：需要提供可调用的模型函数，接受 `(n_channels, n_timepoints)` 输入，返回概率分布。

## 依赖

- numpy
- torch
- matplotlib
- seaborn
- tqdm
- shapelet_analysis.py
- microstate_analysis.py
- timefreq_analysis.py
- cms_lime_chb_analysis_final.py

## 示例输出

```
统一基元选拔：提取并评估生物标志物
================================================================================
处理 10 个样本...

处理样本: 100%|████████████| 10/10 [02:30<00:00, 15.0s/个]

处理完成！
  - 处理的样本数: 10
  - 提取的生物标志物总数: 1500
  - 聚合后的生物标志物数: 60

Top 20 个生物标志物:
  1. shapelet - 重要性: 0.123456, 质量: 0.8500
  2. microstate - 重要性: 0.098765, 质量: 0.7200
  ...
```
