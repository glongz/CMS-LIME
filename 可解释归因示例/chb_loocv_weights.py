# -*- coding: utf-8 -*-
"""
留一法（组合留一法）权重路径解析，与训练/评估脚本约定一致。

例如（见 ``biomarker_enhanced_prediction_v2.extract_model_scores``）::

    weight_model_path = r'D:\\public_data\\CHBMIT\\weight\\组合留一法\\%s' % model_name
    # weight_model_path = r'E:\\...\\LOOCV\\%s' % model_name  # 备用盘

子目录按 ``natural_keys`` 排序后，再按受试者编号映射到某一子文件夹；该文件夹内
``.pth`` 按 ``sort_key``（文件名中 ``FDR`` 后 ``数字-数字``）排序，取第
``select_loop_n_w - 1`` 个文件。
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional, Tuple

# 与默认训练脚本一致；可用环境变量覆盖（例如映射到 E:\\...\\LOOCV）
CHB_LOOCV_WEIGHT_ROOT: Path = Path(
    os.environ.get("CHB_LOOCV_WEIGHT_ROOT", r"D:\public_data\CHBMIT\weight\组合留一法")
)


def natural_keys(text: str):
    def atoi(t):
        return int(t) if t.isdigit() else t

    return [atoi(c) for c in re.split(r"(\d+)", str(text))]


def sort_key_pth(filename: str):
    match = re.search(r"(\d+)-(\d+)", filename)
    if match:
        return (int(match.group(1)), int(match.group(2)), str(filename))
    return (10**9, 10**9, str(filename))


def patient_remap_for_loocv(patient_w: int) -> Tuple[int, int]:
    """
    与 ``extract_model_scores`` 一致：返回
    (用于 ``chb%02d`` 数据目录的 patient 编号, 在排序后 ``weight_sub_dirs`` 中的 0-based 下标)。

    特别地，原 CHB 受试者 12 在留一法目录中跳过一位，并改用与 patient 13 相同的数据/目录映射逻辑。
    """
    patient = int(patient_w)
    patient_folder = int(patient_w)
    if patient > 12:
        patient_folder = patient - 1
    elif patient == 12:
        patient = 13
        patient_folder = patient - 1
    return patient, patient_folder - 1


def loocv_data_patient_id(patient_w: int) -> int:
    """与留一法 checkpoint 配套时，应使用的 ``chb%02d`` 编号（如 12→13）。"""
    p, _ = patient_remap_for_loocv(patient_w)
    return p


def resolve_loocv_model_path(
    model_name: str,
    patient_w: int,
    select_loop_n_w: int,
    weight_root: Path | None = None,
) -> Path:
    """
    :param model_name: 如 ``eeginception``，对应 ``组合留一法/eeginception/`` 下子目录。
    :param patient_w: 临床受试者编号 1..24（与脚本中 ``patient_w`` 一致）。
    :param select_loop_n_w: 与训练里 ``select_loop_n_w`` 一致；取第 ``select_loop_n_w - 1`` 个 pth（0-based）。
    """
    root = Path(weight_root) if weight_root is not None else CHB_LOOCV_WEIGHT_ROOT
    model_root = root / str(model_name)
    if not model_root.is_dir():
        raise FileNotFoundError(f"LOOCV model class dir not found: {model_root}")

    subdirs = sorted([p.name for p in model_root.iterdir() if p.is_dir()], key=natural_keys)
    if not subdirs:
        raise FileNotFoundError(f"No subject subdirectories under {model_root}")

    _, idx = patient_remap_for_loocv(patient_w)
    if not (0 <= idx < len(subdirs)):
        raise IndexError(
            f"patient_w={patient_w} maps to subdir index {idx}, but only {len(subdirs)} entries exist"
        )
    sub = model_root / subdirs[idx]

    pth_names = sorted(
        [p.name for p in sub.iterdir() if p.suffix.lower() == ".pth"],
        key=sort_key_pth,
    )
    n_seizure = int(select_loop_n_w) - 1
    if not (0 <= n_seizure < len(pth_names)):
        raise IndexError(
            f"select_loop_n_w={select_loop_n_w} -> pth index {n_seizure}, "
            f"but {sub} has {len(pth_names)} .pth files: {pth_names[:5]!r}..."
        )
    return (sub / pth_names[n_seizure]).resolve()


def apply_loocv_model_cli(args: Any) -> bool:
    """
    若 ``args.loocv_model_name`` 非空：解析并写入 ``args.model_path``，并可按
    ``args.loocv_sync_data`` 重写 ``args.data_dir``（需已设置 ``args.chb_patient``）。

    可选 ``args.loocv_weight_root``：留一法根目录（默认同环境 / ``CHB_LOOCV_WEIGHT_ROOT``）。

    :return: 是否应用了留一法解析
    """
    name = (getattr(args, "loocv_model_name", None) or "").strip()
    if not name:
        return False
    cp = int(getattr(args, "chb_patient", 0) or 0)
    if cp <= 0:
        raise SystemExit("--loocv_model_name requires --chb_patient (1..99)")
    wr = (getattr(args, "loocv_weight_root", None) or "").strip()
    root: Optional[Path] = Path(wr) if wr else None
    sel = int(getattr(args, "loocv_select_loop", 1) or 1)
    args.model_path = str(
        resolve_loocv_model_path(name, cp, sel, weight_root=root)
    )
    if bool(getattr(args, "loocv_sync_data", False)):
        from chb_paths import chb_patient_data_dir

        args.data_dir = str(chb_patient_data_dir(loocv_data_patient_id(cp)))
    return True
