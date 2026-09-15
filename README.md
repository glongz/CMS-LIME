# CMS-LIME

**Causal Multi-Scale Local Interpretable Model-agnostic Explanations**

An explainable-AI framework for EEG-based seizure prediction. CMS-LIME builds multi-scale explanation units from microstates, shapelets, and time–frequency primitives, generates physiologically plausible perturbations under a causal graph, and attributes the change in model output with sparse local regression plus diversity selection.

Paper (working title): *CMS-LIME: Causal Multi-Scale Interpretable Explanations for EEG-Based Seizure Prediction*

## Method

1. **Multi-scale primitives.** Microstates (topography), shapelets (discriminative time-domain subsequences), and time–frequency tiles (CWT/STFT bands).
2. **Causally consistent perturbation.** A directed channel graph (e.g. Granger) is used so that edits to a node are applied together with lagged parents/children, avoiding implausible counterfactuals.
3. **Sparse local surrogate.** Kernel-weighted Ridge/Lasso maps prediction change to primitives.
4. **Diversity selection.** DPP / submodular selection compresses the explanation into a compact, low-redundancy set.
5. **Statistically screened candidate markers.** High-importance primitives are admitted only after a permutation / BH-FDR match-rate filter; the resulting *candidate markers* can refine \(P(\text{preictal})\). This is **not** a clinical biomarker claim.

This repository ships **implementation, experiment scripts, and CHB-MIT / Siena preprocess code**. It does not include raw EEG, trained weights, primitive-library `.pkl` files, or large experiment dumps.

**Setting B (paper-aligned).** Cross-patient fusion uses a shared group-recurring candidate-marker pool built from source patients (family-specific template matching; motifs that recur in \(\ge 2\) sources). The held-out patient does **not** re-fit the dictionary \(\mathcal{U}\) or causal graph \(\mathcal{G}\); fusion always uses fixed \(\gamma=0.3\). The adaptive-\(\gamma\) rule is Setting A only.

## Layout

```
CMS-LIME/
├── cms_lime.py                      # framework config
├── cms_lime_explainer.py            # main explainer
├── cms_lime_enhanced_explainer.py   # explainer with a primitive library
├── microstate_analysis.py
├── shapelet_analysis.py
├── timefreq_analysis.py
├── causal_perturbation.py
├── local_regression.py
├── primitive_library.py
├── cms_lime_example.py              # synthetic-data demo
├── cms_lime_chb_analysis_final.py   # CHB-MIT analysis entry
├── CHBMIT/                          # CHB-MIT preprocess (montage, segments, ignore-list)
├── siena/                           # Siena preprocess (CHB-matched 30-1-240)
├── unified_primitive_selection/     # joint primitive extraction + importance
├── two_stage_selection/             # two-stage primitive selection
└── utils/
```

## Quick start

```bash
pip install -r requirements.txt
python cms_lime_example.py
```

Minimal API:

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

Run scripts from the repository root, or add the root to `PYTHONPATH`.

## CHB-MIT experiments

Download recordings from [PhysioNet CHB-MIT](https://physionet.org/content/chbmit/).

**Protocol (must match the paper tables):**

- Segment tag `30-1-240`: **SOP = 30 min**, **SPH = 1 min** (preictal = `[onset-31 min, onset-1 min)`)
- Evaluation cohort: `chb01`–`chb11`, `chb13`–`chb23` (22 subjects; exclude `chb12` and `chb24`)
- Montage: 18 shared bipolar channels; montage-incompatible EDFs are dropped at file level (ignore-list in `CHBMIT/chbmit/config.py`)
- No extra band-pass / notch (native 256 Hz)
- Preprocess scripts in this repo: [`CHBMIT/`](CHBMIT/) (see [`CHBMIT/README.md`](CHBMIT/README.md))

Override local paths with environment variables:

- `CHB_DATA_CLEAN_ROOT`
- `CHB_SEGMENT_INFO_ROOT` (paper tables: `.../segment_clean_18channels/30-1-240` or equivalent `segment_clean/30-1-240`)

## Siena external-validation preprocess

Same **30-1-240** protocol as CHB. Obtain raw files from
[PhysioNet Siena Scalp EEG](https://physionet.org/content/siena-scalp-eeg/1.0.0/); this repo provides scripts only.

- Rebuild the same 18 bipolar channels (T3/T4/T5/T6 → T7/T8/P7/P8; channel order matches CHB maj for transfer)
- Anti-alias and resample to 256 Hz; **Siena only** also applies a 50 Hz notch
- Output layout mirrors CHB: `1_data_clean_18channels/PNXX/*.npy` and `segment_clean_18channels/30-1-240/PNXX/segment_info.json`
- A lead seizure needs ≥31 min in-file preamble to create `Pre*`; short clips (e.g. some PN00 files) record Onset only — expected

Environment variables (see `siena/paths.py`):

- `SIENA_RAW_ROOT`
- `SIENA_OUT_ROOT` / `SIENA_DATA_CLEAN_ROOT` / `SIENA_SEGMENT_INFO_ROOT`
- `SIENA_REPORTS_DIR` (inventory / stats CSV; default `siena/reports/`)

```bash
python -m siena.inventory
python -m siena.preprocess --dry-run
python -m siena.preprocess
python -m siena.make_patient_stats
```

Details: [`siena/README.md`](siena/README.md).

## Dependencies

Core: `numpy`, `scipy`, `scikit-learn`, `torch`, `matplotlib`, `PyWavelets`, `networkx`, `lime`.

Siena preprocess also needs `mne`. Some CHB comparison models need `braindecode`.

## Not in this repository

- Virtualenv / IDE config / one-off debug scripts
- Patient EEG, `.pkl` primitive libraries, `.pth` weights
- Full faithfulness experiment output trees
- Manuscript sources, figure-drawing scripts, and backbone training code (kept locally)

## License

MIT License.
