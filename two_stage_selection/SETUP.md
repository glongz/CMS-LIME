# 两阶段基元选拔框架 - 设置说明

## 文件已成功移动到 two_stage_selection 文件夹

所有相关代码文件已成功移动到 `two_stage_selection/` 文件夹下，并已更新导入路径，确保代码可以正常运行。

## 文件结构

```
two_stage_selection/
├── __init__.py                          # 包初始化文件
├── prior_knowledge_config.py            # 先验信息配置
├── primitive_selection_stage1.py        # 第一场选拔：基元提取
├── primitive_selection_stage2.py        # 第二场选拔：因果扰动解释
├── two_stage_primitive_selection.py     # 主框架整合
├── run_two_stage_selection.py          # 主程序（集成CHB-MIT数据集）
├── test_imports.py                      # 导入测试脚本
├── README.md                            # 使用说明
├── SETUP.md                             # 本文件
└── utils/
    ├── __init__.py                      # 工具包初始化
    └── visualization.py                 # 可视化工具
```

## 导入路径更新说明

所有文件的导入路径已更新，支持从新文件夹运行：

1. **包内模块**：使用相对导入（`from .module import ...`）
2. **父目录依赖**：自动添加父目录到 `sys.path`，使用绝对导入
3. **包导入**：可以从项目根目录使用 `from two_stage_selection import ...`

## 运行方式

### 方式1：从项目根目录运行（推荐）

```bash
# 在项目根目录下
python two_stage_selection/run_two_stage_selection.py
```

### 方式2：作为Python包使用

```python
from two_stage_selection import TwoStagePrimitiveSelection, create_default_prior_config

# 使用框架...
```

### 方式3：测试导入

```bash
# 从项目根目录运行
python two_stage_selection/test_imports.py
```

## 验证

所有模块已通过导入测试：
- ✓ 先验信息配置
- ✓ 第一场选拔模块
- ✓ 第二场选拔模块
- ✓ 主框架
- ✓ 可视化工具
- ✓ 依赖模块（shapelet_analysis, microstate_analysis, timefreq_analysis）

## 注意事项

1. **运行位置**：建议从项目根目录运行，确保所有依赖模块可以正确导入
2. **路径设置**：代码会自动处理路径，无需手动设置
3. **依赖模块**：确保项目根目录下有 `shapelet_analysis.py`、`microstate_analysis.py`、`timefreq_analysis.py` 等依赖文件

## 完成状态

✅ 所有文件已移动到 `two_stage_selection/` 文件夹
✅ 导入路径已更新
✅ 创建了 `__init__.py` 文件使成为Python包
✅ 测试脚本验证所有模块可以正确导入
✅ 代码可以正常运行
