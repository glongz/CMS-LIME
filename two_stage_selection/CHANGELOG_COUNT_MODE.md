# 数量模式更新说明

## 更新内容

已将第一场选拔从**质量阈值筛选模式**改为**数量模式**，确保每类基元都能提取到目标数量。

## 主要修改

### 1. 先验信息配置 (`prior_knowledge_config.py`)

新增配置项：
- `target_shapelet_count`: Shapelet基元目标数量（默认：100）
- `target_microstate_count`: 微状态基元目标数量（默认：100）
- `target_timefreq_count`: 时频基元目标数量（默认：100）
- `use_count_mode`: 是否使用数量模式（默认：True）

### 2. 第一场选拔逻辑 (`primitive_selection_stage1.py`)

#### Shapelet提取优化：
- **数量模式**：只应用最低质量要求（quality >= 0.001），按质量排序后取前N个
- **质量模式**：应用严格的质量阈值筛选
- 自动增加候选数量（目标数量的3倍）以确保有足够候选

#### 微状态提取优化：
- **数量模式**：
  - 增加样本数（从10增加到50）
  - 提取多个时间段（每个微状态最多3个连续段）
  - 放宽质量要求（explained_variance >= 0.1）
  - 按质量排序后取前N个
- **质量模式**：保持原有严格筛选逻辑

#### 时频提取优化：
- **数量模式**：
  - 使用多个样本（从1个增加到10个）
  - 放宽质量要求（quality >= 0.01）
  - 按质量排序后取前N个
- **质量模式**：保持原有严格筛选逻辑

### 3. 最终筛选逻辑

- **数量模式**：只应用最低质量要求（quality >= 0.001）作为最终安全检查
- **质量模式**：应用全局质量阈值（global_quality_threshold）

## 使用方式

### 默认使用数量模式

```python
from two_stage_selection import create_default_prior_config

config = create_default_prior_config()
# 默认已启用数量模式
# config.use_count_mode = True
# config.target_shapelet_count = 100
# config.target_microstate_count = 100
# config.target_timefreq_count = 100
```

### 自定义目标数量

```python
config = create_default_prior_config()
config.target_shapelet_count = 150
config.target_microstate_count = 80
config.target_timefreq_count = 120
```

### 切换回质量模式

```python
config = create_default_prior_config()
config.use_count_mode = False
# 将使用质量阈值筛选
```

## 解决的问题

1. ✅ **Shapelet基元为0的问题**：现在会按质量排序后取前100个（或目标数量）
2. ✅ **微状态基元为0的问题**：增加样本数和时间段提取，确保有足够候选
3. ✅ **时频基元数量不足的问题**：使用多个样本，按质量排序后取前100个

## 输出示例

使用数量模式后，输出类似：

```
1. 提取Shapelet基元...
   按数量模式选择: 从 90 个shapelet候选中选择前 90 个（目标: 100，可用: 90）
   提取了 90 个Shapelet基元

2. 提取微状态基元...
   按数量模式选择: 从所有候选中选择前 100 个（目标: 100，可用: 120）
   提取了 100 个微状态基元

3. 提取时频基元...
   按数量模式选择: 从所有候选中选择前 100 个（目标: 100，可用: 150）
   提取了 100 个时频基元
```

## 注意事项

1. **候选数量不足**：如果候选数量少于目标数量，会提取所有可用的基元
2. **质量排序**：在数量模式下，仍然按质量分数排序，优先选择高质量基元
3. **最低质量要求**：仍然应用最低质量要求（如 quality >= 0.001），过滤掉明显无效的基元
