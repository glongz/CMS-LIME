#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基元重要性评估器

对提取的基元进行扰动分析，计算重要性分数

Author: CMS-LIME Framework
Date: 2025
"""

import numpy as np
from typing import List, Callable, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class PrimitiveUnit:
    """基元单位（简化版，用于重要性评估）"""
    primitive_id: str
    primitive_type: str  # 'shapelet', 'microstate', 'timefreq'
    channels: List[int]
    time_range: Tuple[int, int]
    quality_score: float = 0.0
    shapelet_data: np.ndarray = None
    microstate_id: int = None
    freq_band: Tuple[float, float] = None
    freq_band_name: str = None
    metadata: dict = None


class PrimitiveImportanceEvaluator:
    """基元重要性评估器"""
    
    def __init__(self, 
                 model: Callable,
                 n_perturbations: int = 10,
                 stability_iterations: int = 3,
                 use_causal_perturbation: bool = False,
                 causal_graph: Optional[Any] = None):
        """
        初始化重要性评估器
        
        Parameters:
        -----------
        model : Callable
            预测模型，接受 (n_channels, n_timepoints) 输入，返回概率分布
        n_perturbations : int
            每个基元的扰动次数
        stability_iterations : int
            稳定性迭代次数（多次扰动取平均）
        use_causal_perturbation : bool
            是否使用因果扰动
        causal_graph : Optional[Any]
            因果图对象，需要有 get_causal_closure 方法
        """
        self.model = model
        self.n_perturbations = n_perturbations
        self.stability_iterations = stability_iterations
        self.use_causal_perturbation = use_causal_perturbation
        self.causal_graph = causal_graph
    
    def evaluate_importance(self,
                            X_sample: np.ndarray,
                            primitive: PrimitiveUnit,
                            target_class: int) -> float:
        """
        评估基元重要性
        
        Parameters:
        -----------
        X_sample : np.ndarray
            样本数据，形状为 (n_channels, n_timepoints)
        primitive : PrimitiveUnit
            要评估的基元
        target_class : int
            目标类别索引
        
        Returns:
        --------
        importance : float
            重要性分数（原始预测 - 扰动后预测）
        """
        # 获取原始预测
        original_pred = self._predict_single(X_sample)
        original_pred = np.asarray(original_pred).flatten()
        
        # 确保target_class是有效的
        if target_class >= len(original_pred):
            target_class = int(np.argmax(original_pred))
        
        original_score = float(np.asarray(original_pred[target_class]).item())
        
        # 多次迭代计算平均重要性（提高稳定性）
        importance_scores = []
        
        for _ in range(self.stability_iterations):
            # 生成扰动样本
            perturbed_samples = self._generate_perturbations(X_sample, primitive)
            
            # 计算扰动后的预测
            perturbed_predictions = []
            for perturbed_x in perturbed_samples:
                pred = self._predict_single(perturbed_x)
                pred = np.asarray(pred).flatten()
                pred_score = float(np.asarray(pred[target_class]).item())
                perturbed_predictions.append(pred_score)
            
            # 计算平均预测变化
            avg_perturbed_score = float(np.mean(perturbed_predictions))
            prediction_change = original_score - avg_perturbed_score
            
            importance_scores.append(prediction_change)
        
        # 返回平均重要性
        mean_importance = float(np.mean(importance_scores)) if importance_scores else 0.0
        
        return mean_importance
    
    def _generate_perturbations(self, 
                                X_sample: np.ndarray,
                                primitive: PrimitiveUnit) -> List[np.ndarray]:
        """
        生成扰动样本（基元不存在）
        根据配置选择使用普通扰动或因果扰动
        
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
        if self.use_causal_perturbation and self.causal_graph is not None:
            return self._generate_causal_perturbations(X_sample, primitive)
        else:
            return self._generate_normal_perturbations(X_sample, primitive)
    
    def _generate_normal_perturbations(self, 
                                       X_sample: np.ndarray,
                                       primitive: PrimitiveUnit) -> List[np.ndarray]:
        """
        生成普通扰动样本（基元不存在）
        
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
                        # 简单的频域掩码：移除中间频率成分
                        fft_segment = np.fft.fft(segment)
                        n = len(fft_segment)
                        fft_segment[n//4:3*n//4] = 0
                        filtered_segment = np.real(np.fft.ifft(fft_segment))
                        perturbed_x[ch, start:end] = filtered_segment
            
            perturbed_samples.append(perturbed_x)
        
        return perturbed_samples
    
    def _generate_causal_perturbations(self, 
                                       X_sample: np.ndarray,
                                       primitive: PrimitiveUnit) -> List[np.ndarray]:
        """
        生成因果一致性扰动样本（基元不存在）
        基于 cms_lime.py 中的 causal_consistent_perturbation 实现
        
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
            
            # 获取因果闭包
            # 支持两种格式的因果图：
            # 1. cms_lime.CausalGraph: get_causal_closure(channels, time_range) -> List[Tuple[int, Tuple[int, int]]]
            # 2. causal_perturbation.CausalGraph: get_causal_closure(nodes, max_lag) -> Set[Tuple[int, int]]
            closure = []
            start, end = primitive.time_range
            
            try:
                # 尝试使用 cms_lime 格式（推荐）
                # 方法签名: get_causal_closure(channels, time_range) -> List[Tuple[int, Tuple[int, int]]]
                closure_raw = self.causal_graph.get_causal_closure(
                    primitive.channels, 
                    primitive.time_range
                )
                # 检查返回格式
                if closure_raw and len(closure_raw) > 0:
                    first_item = closure_raw[0]
                    if isinstance(first_item, tuple) and len(first_item) == 2:
                        if isinstance(first_item[1], tuple):
                            # 格式: [(channel, (start, end)), ...]
                            closure = closure_raw
                        else:
                            # 可能是其他格式，尝试转换
                            closure = [(ch, (start, end)) for ch, _ in closure_raw]
                    else:
                        closure = []
                        
            except (TypeError, AttributeError):
                # 如果失败，尝试使用 causal_perturbation 格式（需要适配）
                try:
                    # 方法签名: get_causal_closure(nodes, max_lag) -> Set[Tuple[int, int]]
                    max_lag = getattr(self.causal_graph, 'max_lag', 5)
                    closure_raw = self.causal_graph.get_causal_closure(
                        primitive.channels,
                        max_lag=max_lag
                    )
                    
                    # 适配格式：将 (node, lag_offset) 转换为 (channel, (start, end))
                    # lag_offset 为负表示过去，为正表示未来
                    for item in closure_raw:
                        if isinstance(item, tuple) and len(item) == 2:
                            node, lag_offset = item
                            adjusted_start = max(0, start + lag_offset)
                            adjusted_end = end + lag_offset
                            if adjusted_end > adjusted_start:
                                closure.append((node, (adjusted_start, adjusted_end)))
                                
                except Exception as e:
                    import warnings
                    warnings.warn(
                        f"无法获取因果闭包: {e}。将只扰动原始基元。",
                        UserWarning
                    )
                    closure = []
            
            # 首先扰动原始基元
            if primitive.primitive_type == 'shapelet':
                for ch in primitive.channels:
                    start, end = primitive.time_range
                    if 0 <= start < end <= perturbed_x.shape[1]:
                        perturbed_x[ch, start:end] = 0
            elif primitive.primitive_type == 'microstate':
                for ch in primitive.channels:
                    start, end = primitive.time_range
                    if 0 <= start < end <= perturbed_x.shape[1]:
                        segment_mean = perturbed_x[ch, start:end].mean()
                        perturbed_x[ch, start:end] = segment_mean
            elif primitive.primitive_type == 'timefreq':
                for ch in primitive.channels:
                    start, end = primitive.time_range
                    if 0 <= start < end <= perturbed_x.shape[1]:
                        segment = perturbed_x[ch, start:end].copy()
                        fft_segment = np.fft.fft(segment)
                        n = len(fft_segment)
                        fft_segment[n//4:3*n//4] = 0
                        filtered_segment = np.real(np.fft.ifft(fft_segment))
                        perturbed_x[ch, start:end] = filtered_segment
            
            # 扰动因果闭包中的相关单元
            for channel, time_range in closure:
                if time_range[1] <= perturbed_x.shape[1] and time_range[0] >= 0:
                    # 对因果闭包中的区域进行掩码
                    start, end = time_range
                    if 0 <= start < end <= perturbed_x.shape[1]:
                        perturbed_x[channel, start:end] = 0
            
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
        if X.ndim == 3:
            X = X[0]
        
        # 调用模型
        predictions = self.model(X)
        
        # 确保返回的是numpy数组
        if not isinstance(predictions, np.ndarray):
            predictions = np.asarray(predictions)
        
        # 确保是一维数组
        predictions = predictions.flatten()
        
        # 转换为float64
        predictions = predictions.astype(np.float64)
        
        return predictions
