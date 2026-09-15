# CHB-MIT preprocessing

CHB-MIT channel-harmonisation and segmentation scripts used in the CMS-LIME paper.

Run notebooks from `CHBMIT/chbmit/` with the **repository root** on `PYTHONPATH` (so that `from global_config import *` and `from CHBMIT.config import *` resolve).

Typical order:

1. `chbmit/chbmit_channels_meta.ipynb` â€?common 18-channel montage; file-level ignore list in `chbmit/config.py`
2. `chbmit/chbmit_clean_retrieval.ipynb` â€?extract mapped channels
3. `chbmit/generate_timeline_json.ipynb` â€?seizure clocks
4. `create_label_tgt_classification.py` â€?SOP/SPH segment labels (set SOP/SPH to **30 / 1** to match the paper tables)

Raw recordings are **not** included; download [PhysioNet CHB-MIT](https://physionet.org/content/chbmit/) and set paths in `chbmit/config.py`.

## Source

Vendored from [DongDongBan/chbmit-seizure-prediction](https://github.com/DongDongBan/chbmit-seizure-prediction), with attribution. This folder was previously named `CHBMIT`. Siena preprocessing lives in `siena/` in this same repository.
