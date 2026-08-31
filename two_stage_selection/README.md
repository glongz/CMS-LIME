# 两阶段基元选拔框架

## 概述

本框架实现了一个两阶段的基元选拔流程，用于从EEG数据中提取和评估生物标志物候选：

1. **第一场选拔（基元提取）**：基于先验信息从原始数据中提取候选基元
2. **第二场选拔（因果扰动解释）**：通过类LIME方法评估基元在/不在对模型预测的影响

## 文件结构

```
two_stage_selection/
├── __init__.py                          # 包初始化文件
├── prior_knowledge_config.py            # 先验信息配置
├── primitive_selection_stage1.py        # 第一场选拔：基元提取
├── primitive_selection_stage2.py        # 第二场选拔：因果扰动解释
├── two_stage_primitive_selection.py     # 主框架整合
├── run_two_stage_selection.py           # 主程序（集成CHB-MIT数据集）
├── test_imports.py                      # 导入测试脚本
├── README.md                            # 本文件
└── utils/
    ├── __init__.py                      # 工具包初始化
    └── visualization.py                 # 可视化工具
```

## 使用方法

### 方法1：从项目根目录运行

```bash
# 在项目根目录下运行
python two_stage_selection/run_two_stage_selection.py
```

### 方法2：从two_stage_selection目录运行

```bash
# 进入two_stage_selection目录
cd two_stage_selection

# 运行主程序（需要确保在项目根目录的上下文中）
python run_two_stage_selection.py
```

### 方法3：作为Python包使用

```python
from two_stage_selection import (
    TwoStagePrimitiveSelection,
    create_default_prior_config
)
import numpy as np

# 创建配置
prior_config = create_default_prior_config()

# 创建框架
framework = TwoStagePrimitiveSelection(
    prior_config=prior_config,
    model=your_model.predict_proba,
    n_perturbations=10,
    importance_threshold=0.1,
    top_k_biomarkers=20
)

# 第一场选拔
X_train = ...  # (n_samples, n_channels, n_timepoints)
y_train = ...  # 标签
framework.fit(X_train, y_train)

# 第二场选拔
X_sample = ...  # (n_channels, n_timepoints)
y_sample = ...  # 标签
biomarkers = framework.explain(X_sample, y_sample)

# 保存结果
framework.save_results("output_folder")
```

## 测试导入

运行测试脚本验证所有模块可以正确导入：

```bash
python two_stage_selection/test_imports.py
```

## 依赖模块

本框架依赖以下模块（位于项目根目录）：

- `shapelet_analysis.py` - Shapelet分析
- `microstate_analysis.py` - 微状态分析
- `timefreq_analysis.py` - 时频分析
- `cms_lime_chb_analysis_final.py` - CHB-MIT数据加载（用于主程序）

## 注意事项

1. **路径设置**：代码会自动添加父目录到 `sys.path`，以便导入依赖模块
2. **相对导入**：包内模块使用相对导入（如 `from .module import ...`）
3. **绝对导入**：从父目录导入依赖模块使用绝对导入
4. **运行位置**：建议从项目根目录运行，确保所有依赖模块可以正确导入

## 输出结果

运行完成后，会在结果文件夹中生成：

1. `prior_knowledge_config.json` - 先验信息配置
2. `prior_knowledge_record.json` - 先验信息使用记录
3. `candidate_primitives.json` - 第一场选拔的候选基元
4. `biomarkers.json` - 第二场选拔的生物标志物
5. `selection_report.txt` - 综合报告
6. `stage1_visualization.png` - 第一场选拔可视化
7. `stage2_visualization.png` - 第二场选拔可视化
8. `final_summary.json` - 最终摘要

## 作者

CMS-LIME Framework
Date: 2025
