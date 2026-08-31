#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标计算模块

计算和比较后处理前后的性能指标，包括敏感性、特异性、准确率等
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def calculate_metrics(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     y_probs: Optional[np.ndarray] = None) -> Dict:
    """
    计算完整的评估指标
    
    Parameters:
    -----------
    y_true : np.ndarray
        真实标签
    y_pred : np.ndarray
        预测标签
    y_probs : np.ndarray, optional
        预测概率，形状为 (n_samples, 2) 或 (n_samples,)
    
    Returns:
    --------
    metrics : Dict
        包含所有指标的字典
    """
    # 基本指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
    else:
        # 处理只有一个类别的情况
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 0:
                TN, FP, FN, TP = len(y_true), 0, 0, 0
            else:
                TN, FP, FN, TP = 0, 0, 0, len(y_true)
        else:
            TN, FP, FN, TP = 0, 0, 0, 0
    
    # 敏感性和特异性
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    
    # 假阳性率（FDR）
    fdr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,  # 与recall相同
        'specificity': specificity,
        'f1_score': f1,
        'fdr': fdr,
        'confusion_matrix': cm.tolist(),
        'TP': int(TP),
        'TN': int(TN),
        'FP': int(FP),
        'FN': int(FN)
    }
    
    # 如果有概率，计算ROC和PR曲线相关指标
    if y_probs is not None:
        if y_probs.ndim == 1:
            # 如果只有一维，假设是Preictal的概率
            y_probs_2d = np.column_stack([1 - y_probs, y_probs])
        else:
            y_probs_2d = y_probs
        
        try:
            # ROC AUC
            if len(np.unique(y_true)) > 1:
                roc_auc = roc_auc_score(y_true, y_probs_2d[:, 1])
                metrics['roc_auc'] = roc_auc
                
                # PR AUC
                precision_curve, recall_curve, _ = precision_recall_curve(
                    y_true, y_probs_2d[:, 1]
                )
                pr_auc = auc(recall_curve, precision_curve)
                metrics['pr_auc'] = pr_auc
        except Exception as e:
            print(f"计算AUC时出错: {e}")
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
    
    return metrics


def compare_metrics(before_metrics: Dict,
                   after_metrics: Dict,
                   metric_names: Optional[List[str]] = None) -> Dict:
    """
    比较后处理前后的指标
    
    Parameters:
    -----------
    before_metrics : Dict
        后处理前的指标
    after_metrics : Dict
        后处理后的指标
    metric_names : List[str], optional
        要比较的指标名称列表，如果为None则比较所有指标
    
    Returns:
    --------
    comparison : Dict
        比较结果，包含改进幅度
    """
    if metric_names is None:
        metric_names = ['accuracy', 'precision', 'recall', 'sensitivity', 
                       'specificity', 'f1_score', 'roc_auc', 'pr_auc']
    
    comparison = {}
    
    for metric_name in metric_names:
        if metric_name in before_metrics and metric_name in after_metrics:
            before_value = before_metrics[metric_name]
            after_value = after_metrics[metric_name]
            improvement = after_value - before_value
            improvement_percent = (improvement / before_value * 100) if before_value > 0 else 0.0
            
            comparison[metric_name] = {
                'before': before_value,
                'after': after_value,
                'improvement': improvement,
                'improvement_percent': improvement_percent
            }
    
    return comparison


def print_metrics_comparison(comparison: Dict):
    """打印指标比较结果"""
    print("\n" + "=" * 80)
    print("性能指标比较（后处理 vs 原始）")
    print("=" * 80)
    print(f"{'指标':<20} {'原始':<12} {'后处理':<12} {'改进':<12} {'改进率':<12}")
    print("-" * 80)
    
    for metric_name, values in comparison.items():
        before = values['before']
        after = values['after']
        improvement = values['improvement']
        improvement_percent = values['improvement_percent']
        
        print(f"{metric_name:<20} {before:<12.4f} {after:<12.4f} "
              f"{improvement:+.4f} ({improvement_percent:+.2f}%)")
    
    print("=" * 80)


def plot_metrics_comparison(comparison: Dict, save_path: Optional[str] = None):
    """
    绘制指标比较图
    
    Parameters:
    -----------
    comparison : Dict
        比较结果
    save_path : str, optional
        保存路径
    """
    metrics = list(comparison.keys())
    before_values = [comparison[m]['before'] for m in metrics]
    after_values = [comparison[m]['after'] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, before_values, width, label='原始', alpha=0.8)
    bars2 = ax.bar(x + width/2, after_values, width, label='后处理', alpha=0.8)
    
    ax.set_xlabel('指标')
    ax.set_ylabel('分数')
    ax.set_title('后处理前后性能指标比较')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrices(before_cm: np.ndarray,
                           after_cm: np.ndarray,
                           save_path: Optional[str] = None):
    """
    绘制混淆矩阵比较图
    
    Parameters:
    -----------
    before_cm : np.ndarray
        后处理前的混淆矩阵
    after_cm : np.ndarray
        后处理后的混淆矩阵
    save_path : str, optional
        保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (cm, title) in enumerate([(before_cm, '原始'), (after_cm, '后处理')]):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Interictal', 'Preictal'],
                   yticklabels=['Interictal', 'Preictal'])
        ax.set_title(f'{title}混淆矩阵')
        ax.set_ylabel('真实标签')
        ax.set_xlabel('预测标签')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_patient_wise_performance(y_true_dict: Dict[int, np.ndarray],
                                     y_pred_dict: Dict[int, np.ndarray],
                                     y_probs_dict: Optional[Dict[int, np.ndarray]] = None) -> Dict:
    """
    按患者评估性能
    
    Parameters:
    -----------
    y_true_dict : Dict[int, np.ndarray]
        每个患者的真实标签
    y_pred_dict : Dict[int, np.ndarray]
        每个患者的预测标签
    y_probs_dict : Dict[int, np.ndarray], optional
        每个患者的预测概率
    
    Returns:
    --------
    results : Dict
        每个患者的性能指标
    """
    results = {}
    
    for patient_id in y_true_dict.keys():
        y_true = y_true_dict[patient_id]
        y_pred = y_pred_dict[patient_id]
        y_probs = y_probs_dict.get(patient_id) if y_probs_dict else None
        
        metrics = calculate_metrics(y_true, y_pred, y_probs)
        results[patient_id] = metrics
    
    return results


def identify_low_performance_patients(patient_metrics: Dict,
                                    sensitivity_threshold: float = 0.5,
                                    specificity_threshold: float = 0.5) -> List[int]:
    """
    识别低性能患者
    
    Parameters:
    -----------
    patient_metrics : Dict
        患者性能指标字典
    sensitivity_threshold : float
        敏感性阈值
    specificity_threshold : float
        特异性阈值
    
    Returns:
    --------
    low_perf_patients : List[int]
        低性能患者ID列表
    """
    low_perf_patients = []
    
    for patient_id, metrics in patient_metrics.items():
        sensitivity = metrics.get('sensitivity', 0.0)
        specificity = metrics.get('specificity', 0.0)
        
        if sensitivity < sensitivity_threshold or specificity < specificity_threshold:
            low_perf_patients.append(patient_id)
    
    return low_perf_patients


if __name__ == "__main__":
    print("评估指标计算模块")
    print("=" * 80)
    
    # 示例用法
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
    y_pred_before = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
    y_pred_after = y_true.copy()  # 假设后处理完美预测
    
    metrics_before = calculate_metrics(y_true, y_pred_before)
    metrics_after = calculate_metrics(y_true, y_pred_after)
    
    comparison = compare_metrics(metrics_before, metrics_after)
    print_metrics_comparison(comparison)
