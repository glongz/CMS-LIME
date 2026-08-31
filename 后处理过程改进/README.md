# 后处理过程改进方案

## 概述

本文件夹包含利用生物标志物文件改进癫痫预测任务敏感性和特异性的后处理分析方法。特别针对指标特别低的患者设计了专门的应对策略。

## 核心策略

### 1. 生物标志物增强预测
- 利用已发现的生物标志物作为辅助特征
- 结合模型预测概率和生物标志物置信度
- 动态调整预测阈值

### 2. 低性能患者识别与处理
- 识别敏感性和特异性特别低的患者
- 针对低性能患者采用特殊策略
- 多模型集成和置信度校准

### 3. 动态阈值调整
- 基于患者历史表现调整阈值
- 考虑生物标志物出现频率和强度
- 平衡敏感性和特异性

## 文件说明

- `biomarker_enhanced_prediction.py`: 生物标志物增强预测主程序
- `biomarker_fusion.py`: 与预测器一致的概率融合（无 PyTorch 依赖，供评估与可视化复用）
- `biomarker_evaluation_viz.py`: 逐样本日志、权重/置信度扫描与论文级图表（ROC/PR、Sens–Spec 位移等）
- `low_performance_patient_handler.py`: 低性能患者处理模块
- `dynamic_threshold_adjustment.py`: 动态阈值调整模块
- `post_processing_pipeline.py`: 完整后处理流程
- `evaluation_metrics.py`: 评估指标计算

## 使用方法

详见各文件的文档说明。

### 论文插图建议（图序分工）

1. **标志物库与质量**（与后处理权重无关）：使用仓库内 [`对生物标志物进行分析/biomarker_analysis.py`](../对生物标志物进行分析/biomarker_analysis.py) 生成的总览、类型/频率与波形图，作为「发现的可解释模式」铺垫。
2. **加权机制与超参**：运行 `biomarker_evaluation_viz.run_all_plots_from_log`，使用 `collect_per_sample_log` 或 `log_row_from_pipeline_info` 汇总得到的 CSV，得到 `weight_sweep_curves.png` 与 `weight_minconf_f1_heatmap.png`，说明 `biomarker_weight` 与 `min_biomarker_confidence` 的合理区间。
3. **样本级可解释主图**：`sample_waterfall_<patient>_<sample>.png`，展示 `base_p_pre -> delta_pre -> enh_p_pre` 的单样本贡献链路；默认自动选择“高置信+高增益”代表样本，也可通过 `waterfall_patient_id`/`waterfall_sample_index` 或 `waterfall_row_index` 指定。
4. **样本级机制证据**：`confidence_delta_scatter.png`，展示 `max_confidence` 与 `ΔP_pre` 的关系；`patient_level_explainability.png` 与 `patient_level_explainability.csv` 给出每患者平均增益、boost 触发率和平均置信度（含中位数与四分位区间）。
5. **患者级临床指标位移**：`sens_spec_shift.png`（固定阈值 0.5 下，基线概率 vs 生物标志物增强概率的敏感性–特异性）。
6. **概率校准与分类能力**：`prob_before_after_delta.png`、`roc_pr_overlay.png`。
7. **决策翻转补充**：`decision_flips.png` / `decision_flips.json`（FN→TP、FP→TN 等），与 `evaluation_metrics.plot_metrics_comparison` 柱状图互补。

快速自检（合成数据演示）：在后处理目录执行 `python biomarker_evaluation_viz.py`，输出位于 `_biomarker_viz_demo_out/`。
