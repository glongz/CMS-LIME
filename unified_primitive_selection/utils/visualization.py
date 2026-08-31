#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一基元选拔可视化工具

Author: CMS-LIME Framework
Date: 2025
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import os
import sys

# 支持相对导入和绝对导入
try:
    from ..unified_primitive_extractor import BiomarkerUnit
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from unified_primitive_extractor import BiomarkerUnit


def visualize_biomarkers(biomarkers: List[BiomarkerUnit], save_path: str):
    """
    可视化生物标志物
    
    Parameters:
    -----------
    biomarkers : list
        生物标志物列表
    save_path : str
        保存路径
    """
    if len(biomarkers) == 0:
        print("没有生物标志物可可视化")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 重要性分数分布
    importance_scores = [abs(b.importance_score) for b in biomarkers]
    axes[0, 0].hist(importance_scores, bins=30, edgecolor='black')
    axes[0, 0].set_xlabel('重要性分数 (绝对值)')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('重要性分数分布')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 类型分布
    type_counts = {}
    for b in biomarkers:
        type_counts[b.primitive_type] = type_counts.get(b.primitive_type, 0) + 1
    
    axes[0, 1].bar(type_counts.keys(), type_counts.values())
    axes[0, 1].set_xlabel('基元类型')
    axes[0, 1].set_ylabel('数量')
    axes[0, 1].set_title('生物标志物类型分布')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Top K重要性分数
    top_k = min(20, len(biomarkers))
    top_biomarkers = sorted(biomarkers, key=lambda x: abs(x.importance_score), reverse=True)[:top_k]
    top_importances = [abs(b.importance_score) for b in top_biomarkers]
    top_labels = [f"{b.primitive_type}\n{i+1}" for i, b in enumerate(top_biomarkers)]
    
    axes[1, 0].barh(range(len(top_importances)), top_importances)
    axes[1, 0].set_yticks(range(len(top_importances)))
    axes[1, 0].set_yticklabels(top_labels, fontsize=8)
    axes[1, 0].set_xlabel('重要性分数 (绝对值)')
    axes[1, 0].set_title(f'Top {top_k} 生物标志物')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 4. 重要性 vs 质量分数
    importance_scores = [abs(b.importance_score) for b in biomarkers]
    quality_scores = [b.quality_score for b in biomarkers]
    
    axes[1, 1].scatter(quality_scores, importance_scores, alpha=0.5)
    axes[1, 1].set_xlabel('质量分数')
    axes[1, 1].set_ylabel('重要性分数 (绝对值)')
    axes[1, 1].set_title('重要性 vs 质量分数')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化已保存: {save_path}")


def visualize_importance_distribution(biomarkers: List[BiomarkerUnit], save_path: str):
    """可视化重要性分布"""
    if len(biomarkers) == 0:
        return
    
    importance_scores = [abs(b.importance_score) for b in biomarkers]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 直方图
    axes[0].hist(importance_scores, bins=30, edgecolor='black')
    axes[0].set_xlabel('重要性分数 (绝对值)')
    axes[0].set_ylabel('频数')
    axes[0].set_title('重要性分数分布')
    axes[0].grid(True, alpha=0.3)
    
    # 箱线图
    axes[1].boxplot(importance_scores, vert=True)
    axes[1].set_ylabel('重要性分数 (绝对值)')
    axes[1].set_title('重要性分数箱线图')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_type_distribution(biomarkers: List[BiomarkerUnit], save_path: str):
    """可视化类型分布"""
    if len(biomarkers) == 0:
        return
    
    type_counts = {}
    type_importances = {}
    
    for b in biomarkers:
        ptype = b.primitive_type
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
        if ptype not in type_importances:
            type_importances[ptype] = []
        type_importances[ptype].append(abs(b.importance_score))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 数量分布
    axes[0].bar(type_counts.keys(), type_counts.values())
    axes[0].set_xlabel('基元类型')
    axes[0].set_ylabel('数量')
    axes[0].set_title('生物标志物类型分布（数量）')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 平均重要性
    type_avg_importance = {
        ptype: np.mean(scores) for ptype, scores in type_importances.items()
    }
    axes[1].bar(type_avg_importance.keys(), type_avg_importance.values())
    axes[1].set_xlabel('基元类型')
    axes[1].set_ylabel('平均重要性分数')
    axes[1].set_title('生物标志物类型分布（平均重要性）')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
