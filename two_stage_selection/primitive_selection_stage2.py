#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第二场选拔：因果扰动解释模块

基于LIME方法评估基元在/不在对模型预测的影响，筛选生物标志物候选

Author: CMS-LIME Framework
Date: 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import json
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    # 如果没有tqdm，使用简单的进度显示
    def tqdm(iterable, desc=None, total=None, **kwargs):
        if desc:
            print(f"{desc}...")
        return iterable

# 支持相对导入和绝对导入
try:
    from .primitive_selection_stage1 import PrimitiveUnit
except ImportError:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from primitive_selection_stage1 import PrimitiveUnit


@dataclass
class BiomarkerCandidate:
    """生物标志物候选"""
    primitive_id: str
    primitive_type: str
    importance_score: float
    channels: List[int]
    time_range: Tuple[int, int]
    
    # 统计信息
    mean_prediction_change: float = 0.0
    std_prediction_change: float = 0.0
    n_perturbations: int = 0
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'primitive_id': self.primitive_id,
            'primitive_type': self.primitive_type,
            'importance_score': float(self.importance_score),
            'channels': self.channels,
            'time_range': list(self.time_range),
            'mean_prediction_change': float(self.mean_prediction_change),
            'std_prediction_change': float(self.std_prediction_change),
            'n_perturbations': int(self.n_perturbations),
            'metadata': self.metadata
        }


class CausalPerturbationExplainer:
    """因果扰动解释器"""
    
    def __init__(self, model: Callable, n_perturbations: int = 10, 
                 stability_iterations: int = 3, importance_threshold: float = 0.1):
        """
        初始化因果扰动解释器
        
        Parameters:
        -----------
        model : Callable
            预测模型，接受 (n_channels, n_timepoints) 输入，返回概率分布
        n_perturbations : int
            每个基元的扰动次数
        stability_iterations : int
            稳定性迭代次数（多次扰动取平均）
        importance_threshold : float
            重要性阈值（用于筛选生物标志物）
        """
        self.model = model
        self.n_perturbations = n_perturbations
        self.stability_iterations = stability_iterations
        self.importance_threshold = importance_threshold
    
    def explain(self, X_sample: np.ndarray, 
                candidate_primitives: List[PrimitiveUnit],
                target_class: int) -> List[BiomarkerCandidate]:
        """
        解释样本，评估基元重要性
        
        Parameters:
        -----------
        X_sample : np.ndarray
            样本数据，形状为 (n_channels, n_timepoints)
        candidate_primitives : list
            候选基元列表（来自第一场选拔）
        target_class : int
            目标类别
        
        Returns:
        --------
        biomarkers : list
            生物标志物候选列表
        """
        print("=" * 80)
        print("第二场选拔：因果扰动解释")
        print("=" * 80)
        
        # 确保输入格式正确
        if X_sample.ndim == 3:
            X_sample = X_sample[0]  # 取第一个样本
        
        # 获取原始预测
        original_pred = self._predict_single(X_sample)
        
        # 确保original_pred是一维数组
        original_pred = np.asarray(original_pred).flatten()
        
        # 确保target_class是有效的整数索引
        try:
            target_class_idx = int(target_class) if isinstance(target_class, (str, np.str_)) else int(target_class)
        except (ValueError, TypeError):
            target_class_idx = int(np.argmax(original_pred))
        
        if target_class_idx >= len(original_pred):
            target_class_idx = int(np.argmax(original_pred))
        
        # 确保提取的是标量值
        original_score = float(np.asarray(original_pred[target_class_idx]).item())
        
        print(f"\n原始预测 (类别 {target_class_idx}): {original_score:.4f}")
        print(f"评估 {len(candidate_primitives)} 个候选基元...")
        
        # 计算每个基元的重要性
        biomarker_candidates = []
        all_importances = []  # 记录所有重要性分数用于分析
        
        # 使用进度条显示处理进度
        for primitive in tqdm(candidate_primitives, desc="处理基元", unit="个", ncols=80):
            importance = self._compute_primitive_importance(
                X_sample, primitive, target_class_idx, original_score
            )
            
            all_importances.append(importance)
            
            # 创建生物标志物候选
            if abs(importance) >= self.importance_threshold:
                biomarker = BiomarkerCandidate(
                    primitive_id=primitive.primitive_id,
                    primitive_type=primitive.primitive_type,
                    importance_score=importance,
                    channels=primitive.channels,
                    time_range=primitive.time_range,
                    metadata={
                        'quality_score': float(primitive.quality_score),
                        **primitive.metadata
                    }
                )
                biomarker_candidates.append(biomarker)
        
        # 输出重要性统计信息
        if all_importances:
            all_importances = np.array(all_importances)
            abs_importances = np.abs(all_importances)
            print(f"\n重要性统计:")
            print(f"  - 总数: {len(all_importances)}")
            print(f"  - 平均绝对值: {np.mean(abs_importances):.6f}")
            print(f"  - 最大绝对值: {np.max(abs_importances):.6f}")
            print(f"  - 最小绝对值: {np.min(abs_importances):.6f}")
            print(f"  - 中位数绝对值: {np.median(abs_importances):.6f}")
            print(f"  - 阈值: {self.importance_threshold}")
            print(f"  - 超过阈值的数量: {np.sum(abs_importances >= self.importance_threshold)}")
            
            # 如果所有重要性都低于阈值，使用自适应阈值
            if len(biomarker_candidates) == 0 and len(all_importances) > 0:
                # 使用top 20%或中位数的较大值作为阈值
                median_abs = np.median(abs_importances)
                percentile_80 = np.percentile(abs_importances, 80)
                adaptive_threshold = max(median_abs, percentile_80, 0.001)  # 至少0.001
                
                print(f"\n警告: 没有基元超过阈值 {self.importance_threshold}")
                print(f"使用自适应阈值: {adaptive_threshold:.6f} (中位数: {median_abs:.6f}, 80%分位数: {percentile_80:.6f})")
                
                # 使用自适应阈值重新筛选
                for i, (primitive, importance) in enumerate(zip(candidate_primitives, all_importances)):
                    if abs(importance) >= adaptive_threshold:
                        biomarker = BiomarkerCandidate(
                            primitive_id=primitive.primitive_id,
                            primitive_type=primitive.primitive_type,
                            importance_score=float(importance),
                            channels=primitive.channels,
                            time_range=primitive.time_range,
                            metadata={
                                'quality_score': float(primitive.quality_score),
                                **primitive.metadata,
                                'adaptive_threshold_used': True,
                                'adaptive_threshold_value': float(adaptive_threshold)
                            }
                        )
                        biomarker_candidates.append(biomarker)
                
                print(f"使用自适应阈值后，筛选出 {len(biomarker_candidates)} 个生物标志物候选")
        
        # 按重要性分数排序
        biomarker_candidates.sort(key=lambda x: abs(x.importance_score), reverse=True)
        
        print(f"\n筛选出 {len(biomarker_candidates)} 个生物标志物候选")
        print("=" * 80)
        
        return biomarker_candidates
    
    def _compute_primitive_importance(self, X_sample: np.ndarray,
                                     primitive: PrimitiveUnit,
                                     target_class_idx: int,
                                     original_score: float) -> float:
        """
        计算基元重要性
        
        Parameters:
        -----------
        X_sample : np.ndarray
            样本数据
        primitive : PrimitiveUnit
            基元单位
        target_class_idx : int
            目标类别索引
        original_score : float
            原始预测分数
        
        Returns:
        --------
        importance : float
            重要性分数
        """
        importance_scores = []
        prediction_changes = []
        
        for _ in range(self.stability_iterations):
            # 生成扰动样本（基元不存在）
            perturbed_samples = self._generate_perturbations(X_sample, primitive)
            
            # 计算扰动后的预测
            perturbed_predictions = []
            for perturbed_x in perturbed_samples:
                pred = self._predict_single(perturbed_x)
                # 确保pred是一维数组，然后提取标量值
                pred = np.asarray(pred).flatten()
                pred_score = float(np.asarray(pred[target_class_idx]).item())
                perturbed_predictions.append(pred_score)
            
            # 计算平均预测变化
            avg_perturbed_score = float(np.mean(perturbed_predictions))
            prediction_change = original_score - avg_perturbed_score
            
            importance_scores.append(prediction_change)
            prediction_changes.append(prediction_change)
        
        # 返回平均重要性
        if len(importance_scores) > 0:
            mean_importance = float(np.mean(importance_scores))
        else:
            mean_importance = 0.0
        
        std_importance = np.std(importance_scores) if len(importance_scores) > 1 else 0.0
        
        # 更新基元的统计信息（如果可能）
        if hasattr(primitive, 'mean_prediction_change'):
            primitive.mean_prediction_change = np.mean(prediction_changes) if prediction_changes else 0.0
            primitive.std_prediction_change = std_importance
            primitive.n_perturbations = len(perturbed_samples) * self.stability_iterations
        
        return mean_importance
    
    def _generate_perturbations(self, X_sample: np.ndarray,
                               primitive: PrimitiveUnit) -> List[np.ndarray]:
        """
        生成扰动样本（基元不存在）
        
        Parameters:
        -----------
        X_sample : np.ndarray
            原始样本
        primitive : PrimitiveUnit
            要扰动的基元
        
        Returns:
        --------
        perturbed_samples : list
            扰动样本列表
        """
        perturbed_samples = []
        
        for _ in range(self.n_perturbations):
            perturbed_x = X_sample.copy()
            
            # 根据基元类型进行不同的扰动
            if primitive.primitive_type == 'shapelet':
                # 对shapelet区域进行掩码
                for ch in primitive.channels:
                    start, end = primitive.time_range
                    if 0 <= start < end <= perturbed_x.shape[1]:
                        # 用零掩码
                        perturbed_x[ch, start:end] = 0
                        # 或者用随机噪声替换
                        # noise = np.random.normal(0, perturbed_x[ch].std(), end - start)
                        # perturbed_x[ch, start:end] = noise
            
            elif primitive.primitive_type == 'microstate':
                # 对微状态区域进行扰动
                for ch in primitive.channels:
                    start, end = primitive.time_range
                    if 0 <= start < end <= perturbed_x.shape[1]:
                        # 用均值替换（移除微状态特征）
                        segment_mean = perturbed_x[ch, start:end].mean()
                        perturbed_x[ch, start:end] = segment_mean
            
            elif primitive.primitive_type == 'timefreq':
                # 对时频区域进行频域滤波
                for ch in primitive.channels:
                    start, end = primitive.time_range
                    if 0 <= start < end <= perturbed_x.shape[1]:
                        # 移除特定频带的能量
                        segment = perturbed_x[ch, start:end].copy()
                        # 简单的频域掩码：移除高频成分
                        fft_segment = np.fft.fft(segment)
                        # 移除中间频率成分（模拟移除特定频带）
                        n = len(fft_segment)
                        fft_segment[n//4:3*n//4] = 0
                        filtered_segment = np.real(np.fft.ifft(fft_segment))
                        perturbed_x[ch, start:end] = filtered_segment
            
            perturbed_samples.append(perturbed_x)
        
        return perturbed_samples
    
    def _predict_single(self, X: np.ndarray) -> np.ndarray:
        """
        对单个样本进行预测
        
        Parameters:
        -----------
        X : np.ndarray
            样本数据，形状为 (n_channels, n_timepoints)
        
        Returns:
        --------
        predictions : np.ndarray
            预测概率分布
        """
        # 确保输入格式正确
        if X.ndim == 2:
            # (n_channels, n_timepoints) -> (1, n_channels, n_timepoints)
            X = X[np.newaxis, :, :]
        elif X.ndim == 3:
            # 已经是 (batch, channels, timepoints)
            pass
        else:
            raise ValueError(f"Unexpected input shape: {X.shape}")
        
        # 调用模型预测
        try:
            predictions = self.model.predict_proba(X)
            if predictions.ndim == 2 and predictions.shape[0] == 1:
                predictions = predictions[0]
            # 确保返回numpy数组，且值为float类型
            predictions = np.asarray(predictions, dtype=np.float64)
            return predictions
        except Exception as e:
            # 如果模型没有predict_proba方法，尝试其他方式
            try:
                if hasattr(self.model, '__call__'):
                    result = self.model(X)
                    if isinstance(result, np.ndarray):
                        return np.asarray(result, dtype=np.float64)
                # 返回默认值
                return np.array([0.5, 0.5], dtype=np.float64)
            except:
                return np.array([0.5, 0.5], dtype=np.float64)


class BiomarkerEvaluator:
    """生物标志物评估器"""
    
    def __init__(self, top_k: int = 20, importance_threshold: float = 0.1):
        """
        初始化评估器
        
        Parameters:
        -----------
        top_k : int
            选择top K个生物标志物
        importance_threshold : float
            重要性阈值
        """
        self.top_k = top_k
        self.importance_threshold = importance_threshold
    
    def select_biomarkers(self, candidates: List[BiomarkerCandidate]) -> List[BiomarkerCandidate]:
        """
        选择生物标志物
        
        Parameters:
        -----------
        candidates : list
            生物标志物候选列表
        
        Returns:
        --------
        biomarkers : list
            选定的生物标志物列表
        """
        # 按重要性分数排序（已经排序过了）
        # 选择top K个
        selected = candidates[:self.top_k]
        
        return selected
    
    def generate_summary(self, biomarkers: List[BiomarkerCandidate]) -> Dict[str, Any]:
        """
        生成生物标志物摘要
        
        Parameters:
        -----------
        biomarkers : list
            生物标志物列表
        
        Returns:
        --------
        summary : dict
            摘要信息
        """
        if not biomarkers:
            return {
                'n_biomarkers': 0,
                'mean_importance': 0.0,
                'type_distribution': {},
                'top_biomarkers': []
            }
        
        # 统计信息
        importances = [abs(b.importance_score) for b in biomarkers]
        mean_importance = np.mean(importances)
        std_importance = np.std(importances)
        
        # 类型分布
        type_distribution = {}
        for b in biomarkers:
            type_distribution[b.primitive_type] = type_distribution.get(b.primitive_type, 0) + 1
        
        # Top生物标志物
        top_biomarkers = [
            {
                'primitive_id': b.primitive_id,
                'type': b.primitive_type,
                'importance': float(b.importance_score)
            }
            for b in biomarkers[:10]
        ]
        
        summary = {
            'n_biomarkers': len(biomarkers),
            'mean_importance': float(mean_importance),
            'std_importance': float(std_importance),
            'type_distribution': type_distribution,
            'top_biomarkers': top_biomarkers,
            'importance_threshold': self.importance_threshold
        }
        
        return summary
    
    def save_biomarkers(self, biomarkers: List[BiomarkerCandidate], filepath: str):
        """保存生物标志物到JSON文件"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'n_biomarkers': len(biomarkers),
            'biomarkers': [b.to_dict() for b in biomarkers],
            'summary': self.generate_summary(biomarkers)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def get_summary_text(self, biomarkers: List[BiomarkerCandidate]) -> str:
        """获取文本摘要"""
        summary = self.generate_summary(biomarkers)
        
        text = "=" * 80 + "\n"
        text += "第二场选拔：生物标志物评估摘要\n"
        text += "=" * 80 + "\n\n"
        
        text += f"生物标志物数量: {summary['n_biomarkers']}\n"
        text += f"平均重要性: {summary['mean_importance']:.4f} ± {summary['std_importance']:.4f}\n"
        text += f"重要性阈值: {summary['importance_threshold']}\n\n"
        
        text += "类型分布:\n"
        for ptype, count in summary['type_distribution'].items():
            text += f"  - {ptype}: {count}\n"
        
        text += "\nTop 10 生物标志物:\n"
        for i, bm in enumerate(summary['top_biomarkers'], 1):
            text += f"  {i}. {bm['primitive_id']} ({bm['type']}): {bm['importance']:.4f}\n"
        
        text += "=" * 80 + "\n"
        
        return text
