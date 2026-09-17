# Siena Scalp EEG preprocessing under the proposed retained-block rule

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
| `make_patient_stats.py` | Candidate per-patient counts from a generated index |
| `segment_rules.py` | Preictal and interictal intervals in sample units |
| `intervals.py` | Interval union and subtraction |

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

## Run (from the CMS-LIME repository root)

```bash
python -m siena.inventory
python -m siena.preprocess --dry-run
python -m siena.preprocess --index-only --annotation-overrides PATH_TO_VERIFIED_OVERRIDES.json
python -m siena.preprocess
python -m siena.make_patient_stats
```

## Protocol notes

- SOP=30 min, SPH=1 min; retain an onset when its valid nominal preictal
  interval contains at least one complete, non-overlapping 60 s block.
- Exclude 60 min before every onset and 60 min after every offset from
  interictal labeling; retain other eligible EEG, including seizure-free files.
- A clustered onset inside a prior seizure guard receives no second preictal
  label. EDF boundaries and declared acquisition gaps truncate intervals.
- 50 Hz notch (Siena only); resample 512→256 Hz
- Channel order = CHB maj (`chb01/channel_info.json`) for montage consistency.
- PN10 contains alternative or malformed annotation clocks. The parser now
  stops rather than silently dropping them; use `--annotation-overrides` only
  with the time choices documented by the actual experiment.
- EDF dates in the public release do not establish cross-file chronology.
  Cross-file guard decisions must be checked against the experiment timeline.
- `--index-only` does not read or write EEG arrays; use it to inspect the
  generated labels before committing the processed data to a training run.
