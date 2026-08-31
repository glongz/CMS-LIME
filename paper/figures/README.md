# Figure scripts for CMS-LIME paper (Experiments)

Scripts generate PDFs into the project `pictures/` directory for use in the LaTeX experiments section.

## Requirements

- Python 3
- `matplotlib` (and `numpy` for Fig. 2–4)

From project root:

```bash
pip install matplotlib numpy
```

## How to run

From the **project root** (`paper-main/`):

```bash
python paper/figures/plot_fig1_experiment_pipeline.py
python paper/figures/plot_fig2_main_results.py
python paper/figures/plot_fig3_per_patient.py
python paper/figures/plot_fig4_ablation.py
python paper/figures/plot_fig5_primitive_library_interpretability.py
```

Or from `paper/figures/`:

```bash
python plot_fig1_experiment_pipeline.py
python plot_fig2_main_results.py
python plot_fig3_per_patient.py
python plot_fig4_ablation.py
python plot_fig5_primitive_library_interpretability.py
```

(Scripts resolve the project root and write to `pictures/`.)

## Output files

| Script | Output |
|--------|--------|
| `plot_fig1_experiment_pipeline.py` | `pictures/fig1_experiment_pipeline.pdf`, `.png` |
| `plot_fig2_main_results.py` | `pictures/fig2_sensitivity_fdr_comparison.pdf` |
| `plot_fig3_per_patient.py` | `pictures/fig3_per_patient_sensitivity.pdf` |
| `plot_fig4_ablation.py` | `pictures/fig4_ablation.pdf` |
| `plot_fig5_primitive_library_interpretability.py` | `pictures/fig5_primitive_library_interpretability.pdf`, `.png` |

## Design (multi-base-model)

Fig. 2–4 support **multiple base models** (e.g. EEGNet, ShallowConvNet, DeepConvNet, EEG-Inception, Transformer). The list is in each script as `MODELS`; adjust or load from config/CSV. See **`baseline_models_recommendation.md`** for suggested baselines and “large” architectures to compare.

## Data

- **Fig. 1**: No data; pipeline layout is fixed.
- **Fig. 2**: Per-model, per-setting: `PATIENT_SPECIFIC` and `CROSS_PATIENT` dicts keyed by model; values are `(sensitivity[3], sens_std[3], fdr[3], fdr_std[3])` for Baseline / +Biomarker / +Full. Replace with real results or CSV.
- **Fig. 3**: `PER_PATIENT_BY_MODEL`: dict model → list of `(patient_id, baseline_sens%, full_sens%)`. Replace with per-patient, per-model results.
- **Fig. 4**: `ABLATION_BY_MODEL`: dict model → `(sens[4], sens_std[4], fdr[4], fdr_std[4])` for the four conditions. Replace with ablation results.
- **Fig. 5**: Auto-discovers the latest `biomarker_statistics_report.txt` in the project (by file modification time), parses primitive counts for `microstate/shapelet/timefreq`, and renders both absolute counts and percentages in the left quantity panel. If no report is found or parsing fails, fallback counts are used and a warning is printed.

After running your experiments, replace placeholders with real values and re-run the scripts to update the figures.
