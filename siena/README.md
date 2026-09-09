# Siena Scalp EEG → CHB-matched protocol (30-1-240)

Preprocessing for the Neurocomputing revision **R3.7** external validation on the
[Siena Scalp EEG Database](https://physionet.org/content/siena-scalp-eeg/1.0.0/).

Part of [CMS-LIME](https://github.com/glongz/CMS-LIME).

## Layout

| Module | Role |
|--------|------|
| `paths.py` | Default roots + env overrides (`SIENA_RAW_ROOT`, …) |
| `common.py` | C₁₈ montage, electrode aliases, clock / EDF-name helpers |
| `inventory.py` | Step A — C18 feasibility + seizure-list inventory |
| `preprocess.py` | Step B — EDF → `(18,T)` npy + `segment_info.json` |
| `make_patient_stats.py` | Hours / event counts after preprocess |

Processed data (not committed) mirrors CHB:

```
$SIENA_OUT_ROOT/
  1_data_clean_18channels/PNXX/*.npy
  segment_clean_18channels/30-1-240/PNXX/segment_info.json
```

Small CSVs / summaries go to `siena/reports/` (gitignored by default).

## Environment

| Variable | Default |
|----------|---------|
| `SIENA_RAW_ROOT` | `D:\public_data\siena-scalp-eeg-database-1.0.0` |
| `SIENA_OUT_ROOT` | `D:\public_data\SIENA` |
| `SIENA_DATA_CLEAN_ROOT` | `$SIENA_OUT_ROOT/1_data_clean_18channels` |
| `SIENA_SEGMENT_INFO_ROOT` | `$SIENA_OUT_ROOT/segment_clean_18channels/30-1-240` |
| `SIENA_REPORTS_DIR` | `siena/reports` |

Dependency: `mne`, `numpy` (see repo `requirements.txt`).

## Run (from `paper-main` root)

```powershell
cd D:\2025_important_projects\paper-main
python -m siena.inventory
python -m siena.preprocess --dry-run
python -m siena.preprocess
python -m siena.make_patient_stats
```

## Protocol notes

- SOP=30 min, SPH=1 min; lead seizure needs ≥31 min in-file preamble for Pre*
- 50 Hz notch (Siena only); resample 512→256 Hz
- Channel order = CHB maj (`chb01/channel_info.json`) for cross-dataset transfer
- PN00 short clips often fail the 31 min preamble (Onset only) — expected
