# -*- coding: utf-8 -*-
"""
CHB-MIT 默认数据与 segment_info 路径（可用环境变量覆盖）。

与训练脚本中约定一致，例如::

    inter_folder_path = r'D:\\public_data\\CHBMIT\\1_data_clean\\chb%02d' % patient
    inter_segment_clean = r'D:\\public_data\\CHBMIT\\segment_clean\\30-1-240\\chb%02d\\segment_info.json' % patient
"""
from __future__ import annotations

import os
from pathlib import Path

# 前期信号段根目录: .../1_data_clean/chb01, chb02, ...
CHB_DATA_CLEAN_ROOT: Path = Path(
    os.environ.get("CHB_DATA_CLEAN_ROOT", r"D:\public_data\CHBMIT\1_data_clean")
)

# segment_clean 下子版本目录（你当前使用的 30-1-240）
CHB_SEGMENT_CLEAN_ROOT: Path = Path(
    os.environ.get("CHB_SEGMENT_INFO_ROOT", r"D:\public_data\CHBMIT\segment_clean\30-1-240")
)


def chb_patient_data_dir(patient: int) -> Path:
    """``patient=1`` -> ``.../chb01``."""
    return CHB_DATA_CLEAN_ROOT / f"chb{int(patient):02d}"


def chb_patient_segment_info(patient: int) -> Path:
    """``patient=1`` -> ``.../30-1-240/chb01/segment_info.json``."""
    return CHB_SEGMENT_CLEAN_ROOT / f"chb{int(patient):02d}" / "segment_info.json"


# 留一法 .pth 解析见 ``chb_loocv_weights``（``apply_loocv_model_cli`` + ``resolve_loocv_model_path``）
