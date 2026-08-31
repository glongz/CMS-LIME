#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生物标志物去重筛选

从生物标志物目录中筛选出不同的标志物，同一标志物（相同 primitive_type + primitive_id）只保留一个。

用法:
    python filter_unique_biomarkers.py --data_dir <生物标志物目录> [--output_dir <输出目录>] [--by id|type_id]
"""

import os
import glob
import json
import shutil
import argparse
from typing import List, Dict, Tuple, Any
from collections import OrderedDict

# 复用 biomarker_analysis 中的类
from biomarker_analysis import BiomarkerData, load_biomarker_data


def get_biomarker_key(biomarker: BiomarkerData, by: str = "id") -> Tuple:
    """
    生成标志物的唯一键，用于去重。
    
    Parameters
    ----------
    biomarker : BiomarkerData
        生物标志物对象
    by : str
        "id"   : 仅按 primitive_id 去重（同一 id 只留一个）
        "type_id" : 按 (primitive_type, primitive_id) 去重
    
    Returns
    -------
    key : tuple
        用于去重的键
    """
    if by == "type_id":
        return (biomarker.primitive_type or "unknown", biomarker.primitive_id or "unknown")
    else:
        return (biomarker.primitive_id or "unknown",)


def filter_unique_biomarkers(
    biomarkers: List[BiomarkerData],
    by: str = "id",
    keep: str = "first",
) -> List[BiomarkerData]:
    """
    筛选出不同的标志物，同一标志物只保留一个。
    
    Parameters
    ----------
    biomarkers : List[BiomarkerData]
        全部生物标志物列表
    by : str
        "id"      : 按 primitive_id 去重
        "type_id" : 按 (primitive_type, primitive_id) 去重
    keep : str
        "first" : 保留每组中第一个遇到的
        "last"  : 保留每组中最后一个遇到的
    
    Returns
    -------
    unique_biomarkers : List[BiomarkerData]
        去重后的生物标志物列表（每个标志物只保留一个）
    """
    seen = OrderedDict()
    
    for b in biomarkers:
        key = get_biomarker_key(b, by=by)
        if key not in seen:
            seen[key] = b
        else:
            if keep == "last":
                seen[key] = b
    
    return list(seen.values())


def copy_unique_biomarkers_to_dir(
    unique_biomarkers: List[BiomarkerData],
    output_dir: str,
    by: str = "id",
) -> List[str]:
    """
    将去重后的生物标志物文件复制到目标目录，便于后续使用。
    
    Parameters
    ----------
    unique_biomarkers : List[BiomarkerData]
        去重后的生物标志物列表
    output_dir : str
        输出目录路径
    by : str
        用于生成输出文件名的键类型
    
    Returns
    -------
    copied_paths : List[str]
        复制后的文件路径列表
    """
    output_dir = os.path.normpath(os.path.abspath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    copied = []
    
    for i, b in enumerate(unique_biomarkers):
        key = get_biomarker_key(b, by=by)
        if by == "type_id":
            safe_name = f"{key[0]}_{key[1]}"
        else:
            safe_name = str(key[0])
        # 避免文件名非法字符
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_name)
        ext = os.path.splitext(b.filename)[1] or ".npz"
        out_name = f"unique_{i:04d}_{safe_name}{ext}"
        out_path = os.path.join(output_dir, out_name)
        shutil.copy2(b.filepath, out_path)
        copied.append(out_path)
    
    return copied


def main():
    parser = argparse.ArgumentParser(
        description="筛选不同的生物标志物，同一标志物只保留一个"
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=None,
        help="生物标志物 npz 文件所在目录",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default=None,
        help="去重后文件保存目录，默认: D:\\2025_important_projects\\data",
    )
    parser.add_argument(
        "--by",
        choices=["id", "type_id"],
        default="id",
        help="去重依据: id=按 primitive_id, type_id=按 (primitive_type, primitive_id)",
    )
    parser.add_argument(
        "--keep",
        choices=["first", "last"],
        default="first",
        help="同一标志物保留哪一个: first 或 last",
    )
    parser.add_argument(
        "--list_only",
        action="store_true",
        help="只打印去重后的列表，不复制文件",
    )
    # 去重后的标志物原数据默认保存到此目录（注意路径中 data 与 chbmit 之间要有反斜杠）
    DEFAULT_OUTPUT_DIR = r"D:\2025_important_projects\data\chbmit_biomarkers\%02d\preprocess_eeg"%patient
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    if not data_dir or not os.path.isdir(data_dir):
        # 默认示例路径，便于在项目内直接运行
        data_dir = r"D:\2025_important_projects\data\chbmit_biomarkers\%02d\raw_eeg"%patient
        if not os.path.isdir(data_dir):
            print("请指定有效的生物标志物目录: python filter_unique_biomarkers.py <data_dir>")
            return
    
    output_dir = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR
    output_dir = os.path.normpath(os.path.abspath(output_dir))
    
    biomarkers = load_biomarker_data(data_dir)
    if not biomarkers:
        print("未加载到任何生物标志物，请检查目录路径。")
        return
    
    unique = filter_unique_biomarkers(biomarkers, by=args.by, keep=args.keep)
    
    print(f"原始数量: {len(biomarkers)}")
    print(f"去重后数量: {len(unique)} (by={args.by}, keep={args.keep})")
    
    print("\n去重后的标志物 (primitive_type, primitive_id, filepath):")
    for b in unique:
        print(f"  {b.primitive_type}, {b.primitive_id}, {os.path.basename(b.filepath)}")
    
    if not args.list_only:
        copied = copy_unique_biomarkers_to_dir(unique, output_dir, by=args.by)
        # 校验：确认文件确实存在
        existing = [p for p in copied if os.path.isfile(p)]
        if len(existing) != len(copied):
            print(f"\n警告: 仅 {len(existing)}/{len(copied)} 个文件在目标位置可访问")
        print(f"\n已复制 {len(copied)} 个唯一标志物文件到:")
        print(f"  {output_dir}")
        if copied:
            print(f"  示例文件: {os.path.basename(copied[0])}")
    else:
        print("\n(未复制文件，因使用了 --list_only)")


if __name__ == "__main__":
    # 若直接运行且未传参，可在此修改默认数据目录
    # import sys
    # if len(sys.argv) == 1:
    for patient in range(22,23):
        default_data = r"D:\2025_important_projects\data\chbmit_biomarkers\%02d\raw_eeg"%patient
            # if os.path.isdir(default_data):
            #     sys.argv.append(default_data)
        main()
