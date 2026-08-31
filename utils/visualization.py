#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两阶段基元选拔可视化工具

Author: CMS-LIME Framework
Date: 2025
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import os
import gc

from primitive_selection_stage1 import PrimitiveUnit
from primitive_selection_stage2 import BiomarkerCandidate


def visualize_stage1_results(primitives: List[PrimitiveUnit], save_path: str):
    """可视化第一场选拔结果"""
    try:
        plt.close('all')
        gc.collect()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('第一场选拔：基元提取结果', fontsize=16, fontweight='bold')
        
        # 1. 按类型统计
        ax1 = axes[0, 0]
        type_counts = {}
        for p in primitives:
            type_counts[p.primitive_type] = type_counts.get(p.primitive_type, 0) + 1
        
        if type_counts:
            types = list(type_counts.keys())
            counts = list(type_counts.values())
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            ax1.bar(types, counts, color=colors[:len(types)], alpha=0.8, edgecolor='black')
            ax1.set_ylabel('数量', fontsize=12)
            ax1.set_title('基元类型分布', fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # 2. 质量分数分布
        ax2 = axes[0, 1]
        quality_scores = [p.quality_score for p in primitives]
        if quality_scores:
            ax2.hist(quality_scores, bins=20, color='#4ECDC4', alpha=0.7, edgecolor='black')
            ax2.axvline(np.mean(quality_scores), color='red', linestyle='--', 
                       linewidth=2, label=f'均值: {np.mean(quality_scores):.3f}')
            ax2.set_xlabel('质量分数', fontsize=12)
            ax2.set_ylabel('频数', fontsize=12)
            ax2.set_title('质量分数分布', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 时间范围分布
        ax3 = axes[1, 0]
        time_spans = [p.time_range[1] - p.time_range[0] for p in primitives]
        if time_spans:
            ax3.hist(time_spans, bins=20, color='#45B7D1', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('时间跨度 (样本点)', fontsize=12)
            ax3.set_ylabel('频数', fontsize=12)
            ax3.set_title('时间跨度分布', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # 4. 通道使用统计
        ax4 = axes[1, 1]
        channel_usage = {}
        for p in primitives:
            for ch in p.channels:
                channel_usage[ch] = channel_usage.get(ch, 0) + 1
        
        if channel_usage:
            channels = sorted(channel_usage.keys())
            usage_counts = [channel_usage[ch] for ch in channels]
            ax4.bar(channels, usage_counts, color='#FF6B6B', alpha=0.8, edgecolor='black')
            ax4.set_xlabel('通道索引', fontsize=12)
            ax4.set_ylabel('使用次数', fontsize=12)
            ax4.set_title('通道使用统计', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        gc.collect()
        
        print(f"第一场选拔可视化已保存: {save_path}")
        
    except Exception as e:
        print(f"可视化第一场选拔结果失败: {e}")
        plt.close('all')
        gc.collect()


def visualize_stage2_results(biomarkers: List[BiomarkerCandidate], save_path: str):
    """可视化第二场选拔结果"""
    try:
        plt.close('all')
        gc.collect()
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.5], hspace=0.3, wspace=0.3)
        fig.suptitle('第二场选拔：生物标志物评估结果', fontsize=16, fontweight='bold')
        
        if not biomarkers:
            ax = fig.add_subplot(gs[0, :])
            ax.text(0.5, 0.5, '没有生物标志物候选', ha='center', va='center', 
                   fontsize=14, fontweight='bold')
            ax.axis('off')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            return
        
        # 1. 重要性分数分布（顶部，跨3列）
        ax_importance = fig.add_subplot(gs[0, :])
        
        importances = [b.importance_score for b in biomarkers]
        primitive_ids = [f"{b.primitive_id[:20]}..." if len(b.primitive_id) > 20 else b.primitive_id 
                        for b in biomarkers]
        
        # 按类型着色
        colors = []
        for b in biomarkers:
            if b.primitive_type == 'microstate':
                colors.append('#FF6B6B')
            elif b.primitive_type == 'shapelet':
                colors.append('#4ECDC4')
            elif b.primitive_type == 'timefreq':
                colors.append('#45B7D1')
            else:
                colors.append('#96CEB4')
        
        bars = ax_importance.bar(range(len(importances)), importances, color=colors, 
                                alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax_importance.set_xlabel('生物标志物索引', fontsize=12, fontweight='bold')
        ax_importance.set_ylabel('重要性分数', fontsize=12, fontweight='bold')
        ax_importance.set_title('生物标志物重要性分布', fontweight='bold', fontsize=14)
        ax_importance.grid(True, alpha=0.3, linestyle='--')
        
        # 添加阈值线
        threshold = 0.1
        ax_importance.axhline(y=threshold, color='red', linestyle='--', 
                             linewidth=2, alpha=0.7, label=f'阈值 ({threshold})')
        ax_importance.axhline(y=-threshold, color='red', linestyle='--', 
                             linewidth=2, alpha=0.7)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='Microstate'),
            Patch(facecolor='#4ECDC4', label='Shapelet'),
            Patch(facecolor='#45B7D1', label='Timefreq'),
            plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, 
                      label=f'阈值 (±{threshold})')
        ]
        ax_importance.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # 添加统计信息
        stats_text = f'数量: {len(importances)} | 均值: {np.mean(importances):.4f} | 最大: {max(importances):.4f} | 最小: {min(importances):.4f}'
        ax_importance.text(0.02, 0.98, stats_text, transform=ax_importance.transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. 按类型统计（左下）
        ax2 = fig.add_subplot(gs[1, 0])
        type_counts = {}
        for b in biomarkers:
            type_counts[b.primitive_type] = type_counts.get(b.primitive_type, 0) + 1
        
        if type_counts:
            types = list(type_counts.keys())
            counts = list(type_counts.values())
            colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            ax2.pie(counts, labels=types, autopct='%1.1f%%', colors=colors_pie[:len(types)],
                   startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
            ax2.set_title('类型分布', fontweight='bold')
        
        # 3. Top 10 生物标志物表格（中下）
        ax3 = fig.add_subplot(gs[1, 1:])
        ax3.axis('off')
        
        top_10 = biomarkers[:10]
        table_data = []
        for i, b in enumerate(top_10, 1):
            table_data.append([
                str(i),
                b.primitive_id[:30] + '...' if len(b.primitive_id) > 30 else b.primitive_id,
                b.primitive_type.capitalize(),
                f'{b.importance_score:.4f}',
                f'{b.time_range[0]}-{b.time_range[1]}',
                f'{len(b.channels)}'
            ])
        
        table = ax3.table(cellText=table_data,
                         colLabels=['排名', '基元ID', '类型', '重要性', '时间范围', '通道数'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(0.9, 1.8)
        
        # 样式化表格
        for i in range(len(table_data) + 1):
            for j in range(6):
                cell = table[(i, j)]
                if i == 0:  # 表头
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    # 重要性列着色
                    if j == 3 and i > 0:
                        imp_val = float(table_data[i-1][3])
                        if abs(imp_val) >= threshold:
                            cell.set_facecolor('#E8F5E8')
                        else:
                            cell.set_facecolor('#FFF3E0')
                    else:
                        cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax3.set_title('Top 10 生物标志物', fontweight='bold', fontsize=14, pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        gc.collect()
        
        print(f"第二场选拔可视化已保存: {save_path}")
        
    except Exception as e:
        print(f"可视化第二场选拔结果失败: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        gc.collect()


def visualize_biomarker_comparison(biomarkers_list: List[List[BiomarkerCandidate]], 
                                  sample_labels: List[str],
                                  save_path: str):
    """可视化多个样本的生物标志物比较"""
    try:
        plt.close('all')
        gc.collect()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('多样本生物标志物比较', fontsize=16, fontweight='bold')
        
        # 1. 重要性分数比较
        ax1 = axes[0, 0]
        for i, (biomarkers, label) in enumerate(zip(biomarkers_list, sample_labels)):
            if biomarkers:
                importances = [abs(b.importance_score) for b in biomarkers]
                ax1.plot(range(len(importances)), importances, marker='o', 
                        label=f'{label} (n={len(biomarkers)})', alpha=0.7)
        ax1.set_xlabel('生物标志物排名', fontsize=12)
        ax1.set_ylabel('重要性分数 (绝对值)', fontsize=12)
        ax1.set_title('重要性分数比较', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 类型分布比较
        ax2 = axes[0, 1]
        all_types = set()
        for biomarkers in biomarkers_list:
            for b in biomarkers:
                all_types.add(b.primitive_type)
        
        type_data = {t: [] for t in all_types}
        for biomarkers in biomarkers_list:
            type_counts = {t: 0 for t in all_types}
            for b in biomarkers:
                type_counts[b.primitive_type] += 1
            for t in all_types:
                type_data[t].append(type_counts[t])
        
        x = np.arange(len(sample_labels))
        width = 0.8 / len(all_types)
        colors = {'microstate': '#FF6B6B', 'shapelet': '#4ECDC4', 'timefreq': '#45B7D1'}
        
        for i, t in enumerate(all_types):
            offset = (i - len(all_types)/2 + 0.5) * width
            ax2.bar(x + offset, type_data[t], width, label=t.capitalize(), 
                   color=colors.get(t, '#96CEB4'), alpha=0.8)
        
        ax2.set_xlabel('样本', fontsize=12)
        ax2.set_ylabel('数量', fontsize=12)
        ax2.set_title('类型分布比较', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(sample_labels)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 平均重要性比较
        ax3 = axes[1, 0]
        mean_importances = []
        for biomarkers in biomarkers_list:
            if biomarkers:
                mean_imp = np.mean([abs(b.importance_score) for b in biomarkers])
                mean_importances.append(mean_imp)
            else:
                mean_importances.append(0.0)
        
        bars = ax3.bar(sample_labels, mean_importances, color='#4ECDC4', alpha=0.8, edgecolor='black')
        ax3.set_ylabel('平均重要性分数', fontsize=12)
        ax3.set_title('平均重要性比较', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars, mean_importances):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 生物标志物数量比较
        ax4 = axes[1, 1]
        n_biomarkers = [len(biomarkers) for biomarkers in biomarkers_list]
        bars = ax4.bar(sample_labels, n_biomarkers, color='#45B7D1', alpha=0.8, edgecolor='black')
        ax4.set_ylabel('生物标志物数量', fontsize=12)
        ax4.set_title('生物标志物数量比较', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars, n_biomarkers):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        gc.collect()
        
        print(f"生物标志物比较可视化已保存: {save_path}")
        
    except Exception as e:
        print(f"可视化生物标志物比较失败: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        gc.collect()
