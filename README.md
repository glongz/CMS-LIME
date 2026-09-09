# CMS-LIME

**Causal Multi-Scale Local Interpretable Model-agnostic Explanations**

面向 EEG 癫痫预测的可解释 AI 框架：用微状态、shapelet 与时频基元构造多尺度解释单元，在因果图约束下生成生理上合理的扰动样本，再通过局部稀疏回归与多样性选择给出神经语义一致的归因。

论文标题（拟定）：*CMS-LIME: Causal Multi-Scale Interpretable Explanations for EEG-Based Seizure Prediction*

## 方法要点

1. **多尺度基元**：微状态（拓扑）、shapelet（时域判别子序列）、时频单元（CWT/STFT 频带）。
2. **因果一致性扰动**：由 Granger 等得到通道有向图，扰动时按因果闭包与时滞同步改写父/子节点，避免不合理伪样本。
3. **局部稀疏代理**：多维核加权 + Ridge/Lasso，将预测变化归因到基元。
4. **多样性选择**：DPP / 子模块优化，压缩为紧凑、少冗余的解释集。
5. **生物标志物后处理**：将高重要性基元筛选为跨段可复用标志物，用于增强预测敏感性/特异性。

本仓库只收录**实现与实验脚本**，不含 CHB-MIT / Siena 原始 EEG、训练权重、基元库 `.pkl` 以及大规模实验输出。

## 仓库结构

```
CMS-LIME/
├── cms_lime.py                      # 框架定义与配置
├── cms_lime_explainer.py            # 主解释器
├── cms_lime_enhanced_explainer.py   # 带基元库的增强解释器
├── microstate_analysis.py
├── shapelet_analysis.py
├── timefreq_analysis.py
├── causal_perturbation.py
├── local_regression.py
├── primitive_library.py
├── cms_lime_example.py              # 合成数据示例
├── cms_lime_chb_analysis_final.py   # CHB-MIT 分析入口
├── siena/                           # Siena 外部验证预处理（对齐 CHB 30-1-240）
├── unified_primitive_selection/     # 统一基元提取 + 重要性评估
├── two_stage_selection/             # 两阶段基元选拔
├── 可解释归因示例/                 # Faithfulness / 归因可视化
├── 后处理过程改进/                 # 标志物融合与动态阈值
├── 对生物标志物进行分析/
└── 跨患者分析/
```

## 快速开始

```bash
pip install -r requirements.txt
python cms_lime_example.py
```

最小调用：

```python
from cms_lime_explainer import CMSLimeExplainer, CMSLimeConfig

config = CMSLimeConfig(
    n_microstates=6,
    n_shapelets=50,
    causal_method="granger",
    n_perturbations=200,
    n_features=20,
)
explainer = CMSLimeExplainer(config)
explainer.fit(X_train, y_train, model)          # X: (n, channels, time)
explanation = explainer.explain_instance(
    test_sample,                                 # (channels, time)
    unit_types=["microstate", "shapelet", "timefreq"],
)
print(explainer.get_explanation_summary(explanation))
```

请在仓库根目录运行脚本，或将根目录加入 `PYTHONPATH`。

## CHB-MIT 实验

数据需自行从 [PhysioNet CHB-MIT](https://physionet.org/content/chbmit/) 获取。

**与论文主表一致的协议（务必对齐）：**

- 分段目录标签：`30-1-240` → **SOP = 30 min，SPH = 1 min**（preictal = `[onset−31 min, onset−1 min)`）
- 评价患者：`chb01`–`chb11`、`chb13`–`chb23`（共 22 人；排除 `chb12`、`chb24`）
- 通道：18 公共双极导联；另有 11 个 montage 不兼容 EDF 在文件级丢弃（见公开预处理仓 `config.py` / ignore-list）
- 无额外 band-pass / notch（原生 256 Hz）
- 预处理公开仓：https://github.com/DongDongBan/chbmit-seizure-prediction/tree/master/dataset_specific/chbmit （路径中的 `dataset_specific` 含下划线）

默认路径可用环境变量覆盖（见 `可解释归因示例/chb_paths.py`）：

- `CHB_DATA_CLEAN_ROOT`
- `CHB_SEGMENT_INFO_ROOT`（论文主表对应 `.../segment_clean_18channels/30-1-240` 或等价 `segment_clean/30-1-240`）

留一发作交叉验证权重路径见 `可解释归因示例/chb_loocv_weights.py`。

Faithfulness 协议示例：

```bash
python 可解释归因示例/faithfulness_protocol_experiment.py --fast
```

## Siena 外部验证预处理

用于跨数据集复现（与 CHB 主表同一 **30-1-240** 协议）。原始数据请自行从
[PhysioNet Siena Scalp EEG](https://physionet.org/content/siena-scalp-eeg/1.0.0/) 获取；本仓库只提供脚本。

要点：

- 重建与 CHB 相同的 18 双极导联（T3/T4/T5/T6 → T7/T8/P7/P8；通道顺序对齐 CHB maj，便于迁移）
- 抗混叠重采样至 256 Hz；**仅 Siena** 加 50 Hz notch
- 输出布局镜像 CHB：`1_data_clean_18channels/PNXX/*.npy` + `segment_clean_18channels/30-1-240/PNXX/segment_info.json`
- Lead seizure 需文件内前置 ≥31 min 才建 `Pre*`；短片段（如部分 PN00）仅记 Onset 属预期

路径环境变量（见 `siena/paths.py`）：

- `SIENA_RAW_ROOT`
- `SIENA_OUT_ROOT` / `SIENA_DATA_CLEAN_ROOT` / `SIENA_SEGMENT_INFO_ROOT`
- `SIENA_REPORTS_DIR`（inventory / stats CSV，默认 `siena/reports/`）

```bash
# 在仓库根目录
python -m siena.inventory
python -m siena.preprocess --dry-run
python -m siena.preprocess
python -m siena.make_patient_stats
```

更细说明见 [`siena/README.md`](siena/README.md)。

## 依赖

核心：`numpy`、`scipy`、`scikit-learn`、`torch`、`matplotlib`、`PyWavelets`、`networkx`、`lime`。

Siena 预处理另需 `mne`。复现 CHB 实验中的部分对比模型时需要 `braindecode`。

## 未纳入本仓库的内容

- `.venv`、IDE 配置、调试一次性脚本
- 患者 EEG、`.pkl` 基元库、`.pth` 权重
- Faithfulness 等实验的完整输出目录
- 论文稿件与作图脚本、实验用 backbone 源码（本地保留）

## 许可

MIT License.
