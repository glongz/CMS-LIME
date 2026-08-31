#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第一场选拔：基元提取模块

基于先验信息从原始数据中提取候选基元，记录所有先验标准

Author: CMS-LIME Framework
Date: 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from datetime import datetime
import sys
import os

# 添加父目录到路径，以便导入依赖模块
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 支持相对导入和绝对导入
try:
    from .prior_knowledge_config import PriorKnowledgeConfig
except ImportError:
    from prior_knowledge_config import PriorKnowledgeConfig

from shapelet_analysis import ShapeletAnalyzer, ShapeletCandidate
from microstate_analysis import MicrostateAnalyzer
from timefreq_analysis import TimeFreqAnalyzer, TimeFreqUnit


@dataclass
class PrimitiveUnit:
    """统一的基元单位类"""
    primitive_id: str
    primitive_type: str  # 'shapelet', 'microstate', 'timefreq'
    channels: List[int]
    time_range: Tuple[int, int]
    quality_score: float = 0.0
    
    # 类型特定信息
    shapelet_data: Optional[np.ndarray] = None
    microstate_id: Optional[int] = None
    freq_band: Optional[Tuple[float, float]] = None
    freq_band_name: Optional[str] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        result = {
            'primitive_id': self.primitive_id,
            'primitive_type': self.primitive_type,
            'channels': self.channels,
            'time_range': list(self.time_range),
            'quality_score': float(self.quality_score),
            'metadata': self.metadata
        }
        
        if self.microstate_id is not None:
            result['microstate_id'] = int(self.microstate_id)
        
        if self.freq_band is not None:
            result['freq_band'] = list(self.freq_band)
            result['freq_band_name'] = self.freq_band_name
        
        return result


class QualityEvaluator:
    """质量评估器"""
    
    def __init__(self, config: PriorKnowledgeConfig):
        self.config = config
    
    def evaluate_shapelet_quality(self, shapelet: ShapeletCandidate) -> float:
        """评估shapelet质量"""
        # 基于信息增益
        quality = shapelet.information_gain
        
        # 如果使用数量模式，保留原始信息增益值（可能为负数）
        # 否则归一化到0-1范围
        if not self.config.use_count_mode:
            quality = max(0.0, min(1.0, quality))
        
        return quality
    
    def evaluate_microstate_quality(self, microstate_stats: Dict, 
                                   microstate_id: int) -> float:
        """评估微状态质量"""
        # 基于解释方差和覆盖率
        explained_var = microstate_stats.get('explained_variance', 0.0)
        coverage = microstate_stats.get('coverage', [0.0] * self.config.microstate_priors.n_microstates)
        
        if microstate_id < len(coverage):
            quality = explained_var * coverage[microstate_id]
        else:
            quality = explained_var * 0.25  # 默认值
        
        return quality
    
    def evaluate_timefreq_quality(self, unit: TimeFreqUnit, 
                                 sample_data: np.ndarray) -> float:
        """评估时频单元质量"""
        try:
            features = unit.extract_features(sample_data)
            
            # 提取相对功率特征（通常在特征向量的后半部分）
            if len(features) > len(unit.channels):
                # 假设相对功率在中间位置
                relative_power_idx = len(unit.channels)
                if relative_power_idx < len(features):
                    relative_power = features[relative_power_idx]
                    quality = float(relative_power)
                else:
                    quality = 0.0
            else:
                quality = 0.0
            
            # 归一化
            quality = max(0.0, min(1.0, quality))
            
        except Exception as e:
            quality = 0.0
        
        return quality


class PrimitiveExtractor:
    """基元提取器 - 整合三种基元提取方法"""
    
    def __init__(self, prior_config: PriorKnowledgeConfig):
        self.prior_config = prior_config
        self.quality_evaluator = QualityEvaluator(prior_config)
        
        # 初始化分析器
        # 如果使用数量模式，增加候选数量以确保有足够的候选
        if prior_config.use_count_mode:
            n_shapelets_to_find = max(
                prior_config.target_shapelet_count * 2,  # 至少是目标数量的2倍
                prior_config.shapelet_priors.n_shapelets_selected
            )
        else:
            n_shapelets_to_find = prior_config.shapelet_priors.n_shapelets_selected
        
        self.shapelet_analyzer = ShapeletAnalyzer(
            method=prior_config.shapelet_priors.method,
            min_length=prior_config.shapelet_priors.min_length,
            max_length=prior_config.shapelet_priors.max_length,
            n_shapelets=n_shapelets_to_find
        )
        
        # 如果使用数量模式，增加max_candidates以确保有足够的候选
        if prior_config.use_count_mode and hasattr(self.shapelet_analyzer, 'finder'):
            if hasattr(self.shapelet_analyzer.finder, 'max_candidates'):
                # 增加候选数量，至少是目标数量的3倍
                self.shapelet_analyzer.finder.max_candidates = max(
                    prior_config.target_shapelet_count * 3,
                    prior_config.shapelet_priors.max_candidates
                )
        
        self.microstate_analyzer = MicrostateAnalyzer(
            n_microstates=prior_config.microstate_priors.n_microstates,
            method=prior_config.microstate_priors.method,
            min_duration=prior_config.microstate_priors.min_duration
        )
        
        self.timefreq_analyzer = TimeFreqAnalyzer(
            method=prior_config.timefreq_priors.method,
            sampling_rate=prior_config.timefreq_priors.sampling_rate,
            freq_bands=prior_config.timefreq_priors.freq_bands,
            time_window_size=prior_config.timefreq_priors.time_window_size,
            overlap_ratio=prior_config.timefreq_priors.overlap_ratio
        )
        
        # 存储提取的基元
        self.extracted_primitives: List[PrimitiveUnit] = []
        self.prior_knowledge_record: Dict = {}
    
    def extract(self, X: np.ndarray, y: np.ndarray) -> List[PrimitiveUnit]:
        """
        提取候选基元
        
        Parameters:
        -----------
        X : np.ndarray
            训练数据，形状为 (n_samples, n_channels, n_timepoints)
        y : np.ndarray
            标签
        
        Returns:
        --------
        primitives : list
            候选基元列表
        """
        print("=" * 80)
        print("第一场选拔：基元提取")
        print("=" * 80)
        
        all_primitives = []
        
        # 1. 提取Shapelet基元
        print("\n1. 提取Shapelet基元...")
        shapelet_primitives = self._extract_shapelets(X, y)
        all_primitives.extend(shapelet_primitives)
        print(f"   提取了 {len(shapelet_primitives)} 个Shapelet基元")
        
        # 2. 提取微状态基元
        print("\n2. 提取微状态基元...")
        microstate_primitives = self._extract_microstates(X, y)
        all_primitives.extend(microstate_primitives)
        print(f"   提取了 {len(microstate_primitives)} 个微状态基元")
        
        # 3. 提取时频基元
        print("\n3. 提取时频基元...")
        timefreq_primitives = self._extract_timefreq(X)
        all_primitives.extend(timefreq_primitives)
        print(f"   提取了 {len(timefreq_primitives)} 个时频基元")
        
        # 4. 应用最终筛选
        if self.prior_config.use_count_mode:
            print("\n4. 应用最终质量筛选（数量模式）...")
        else:
            print("\n4. 应用质量阈值筛选...")
        filtered_primitives = self._apply_quality_threshold(all_primitives)
        print(f"   筛选后剩余 {len(filtered_primitives)} 个基元")
        
        # 5. 记录先验信息
        self._record_prior_knowledge(len(shapelet_primitives), 
                                   len(microstate_primitives),
                                   len(timefreq_primitives),
                                   len(filtered_primitives))
        
        self.extracted_primitives = filtered_primitives
        
        print("\n" + "=" * 80)
        print(f"第一场选拔完成：共提取 {len(filtered_primitives)} 个候选基元")
        print("=" * 80)
        
        return filtered_primitives
    
    def _extract_shapelets(self, X: np.ndarray, y: np.ndarray) -> List[PrimitiveUnit]:
        """提取Shapelet基元"""
        # 确保数据格式正确
        if len(X.shape) == 4:
            X = X.squeeze(1)
        
        # 拟合shapelet分析器
        self.shapelet_analyzer.fit(X, y)
        
        # 获取发现的shapelets
        shapelets = self.shapelet_analyzer.get_shapelets()
        
        primitives = []
        for i, shapelet in enumerate(shapelets):
            # 评估质量
            quality = self.quality_evaluator.evaluate_shapelet_quality(shapelet)
            
            # 如果使用数量模式，只应用最低质量要求；否则应用严格的质量阈值
            if self.prior_config.use_count_mode:
                # 数量模式：只过滤掉明显无效的
                # 允许信息增益为0或负数的shapelet（它们可能仍然有用）
                # 只过滤掉质量异常的（NaN或极端负值）
                if np.isnan(quality) or np.isinf(quality) or quality < -10.0:  # 只过滤明显异常的
                    continue
                # 不应用信息增益阈值，因为即使是负信息增益的shapelet也可能有用
                # 在数量模式下，我们接受所有合理的shapelet，然后按质量排序选择最好的
            else:
                # 质量模式：应用严格的质量阈值
                if quality < self.prior_config.shapelet_priors.min_quality_score:
                    continue
                if shapelet.information_gain < self.prior_config.shapelet_priors.information_gain_threshold:
                    continue
            
            # 创建基元单位
            # 在数量模式下，quality_score使用原始信息增益；否则使用归一化的quality
            if self.prior_config.use_count_mode:
                quality_score = float(shapelet.information_gain)
            else:
                quality_score = quality
            
            primitive = PrimitiveUnit(
                primitive_id=f"shapelet_{i}",
                primitive_type='shapelet',
                channels=[shapelet.channel],
                time_range=(shapelet.start_pos, shapelet.start_pos + shapelet.length),
                quality_score=quality_score,
                shapelet_data=shapelet.data,
                metadata={
                    'information_gain': float(shapelet.information_gain),
                    'length': int(shapelet.length),
                    'series_id': int(shapelet.series_id),
                    'class_label': int(shapelet.class_label) if shapelet.class_label is not None else None
                }
            )
            primitives.append(primitive)
        
        # 如果使用数量模式，按质量排序并取前N个
        if self.prior_config.use_count_mode:
            # 按信息增益排序（使用quality_score，因为在数量模式下它已经是原始信息增益）
            primitives.sort(key=lambda x: x.quality_score, reverse=True)
            target_count = self.prior_config.target_shapelet_count
            # 如果候选数量不足目标数量，提取所有可用的
            actual_count = min(len(primitives), target_count)
            primitives = primitives[:actual_count]
            if len(primitives) > 0:
                print(f"   按数量模式选择: 从 {len(shapelets)} 个shapelet候选中选择前 {actual_count} 个（目标: {target_count}，可用: {len(primitives)}）")
                print(f"   信息增益范围: {primitives[0].metadata.get('information_gain', 0):.4f} ~ {primitives[-1].metadata.get('information_gain', 0):.4f}")
            else:
                print(f"   警告: 所有 {len(shapelets)} 个shapelet候选都被过滤掉了！")
                print(f"   尝试放宽过滤条件...")
                # 如果所有都被过滤，尝试接受所有shapelet（除了NaN）
                primitives = []
                for i, shapelet in enumerate(shapelets):
                    quality = self.quality_evaluator.evaluate_shapelet_quality(shapelet)
                    if not (np.isnan(quality) or np.isinf(quality)):
                        primitive = PrimitiveUnit(
                            primitive_id=f"shapelet_{i}",
                            primitive_type='shapelet',
                            channels=[shapelet.channel],
                            time_range=(shapelet.start_pos, shapelet.start_pos + shapelet.length),
                            quality_score=quality,
                            shapelet_data=shapelet.data,
                            metadata={
                                'information_gain': float(shapelet.information_gain),
                                'length': int(shapelet.length),
                                'series_id': int(shapelet.series_id),
                                'class_label': int(shapelet.class_label) if shapelet.class_label is not None else None
                            }
                        )
                        primitives.append(primitive)
                
                if primitives:
                    primitives.sort(key=lambda x: x.metadata.get('information_gain', x.quality_score), reverse=True)
                    actual_count = min(len(primitives), target_count)
                    primitives = primitives[:actual_count]
                    print(f"   放宽条件后: 选择了 {actual_count} 个shapelet基元")
        
        return primitives
    
    def _extract_microstates(self, X: np.ndarray, y: np.ndarray) -> List[PrimitiveUnit]:
        """提取微状态基元"""
        primitives = []
        
        # 如果使用数量模式，增加样本数以获得更多候选基元
        if self.prior_config.use_count_mode:
            n_samples = min(50, X.shape[0])  # 增加样本数以获得更多候选
        else:
            n_samples = min(10, X.shape[0])  # 限制样本数以提高效率
        
        for sample_idx in range(n_samples):
            sample_data = X[sample_idx]  # (n_channels, n_timepoints)
            
            # 拟合微状态分析器（使用第一个样本）
            if sample_idx == 0:
                self.microstate_analyzer.fit(sample_data, 
                                            sampling_rate=self.prior_config.sampling_rate)
            
            # 转换数据
            sequence, correlations, stats = self.microstate_analyzer.transform(sample_data)
            
            # 如果使用数量模式，放宽质量要求；否则应用严格阈值
            if self.prior_config.use_count_mode:
                # 数量模式：只过滤掉明显无效的
                explained_var = stats.get('explained_variance', 0.0)
                if explained_var < 0.1:  # 非常低的阈值
                    continue
            else:
                # 质量模式：应用严格阈值
                explained_var = stats.get('explained_variance', 0.0)
                if explained_var < self.prior_config.microstate_priors.min_explained_variance:
                    continue
            
            # 为每个微状态创建基元
            for ms_id in range(self.prior_config.microstate_priors.n_microstates):
                # 找到该微状态出现的时间段
                ms_mask = sequence == ms_id
                if not np.any(ms_mask):
                    continue
                
                # 找到所有连续的时间段（不只是第一个）
                ms_indices = np.where(ms_mask)[0]
                if len(ms_indices) == 0:
                    continue
                
                # 如果使用数量模式，提取多个时间段；否则只取第一个
                if self.prior_config.use_count_mode:
                    # 找到所有连续段
                    segments = []
                    start = ms_indices[0]
                    for i in range(1, len(ms_indices)):
                        if ms_indices[i] != ms_indices[i-1] + 1:
                            segments.append((start, ms_indices[i-1] + 1))
                            start = ms_indices[i]
                    segments.append((start, ms_indices[-1] + 1))
                    
                    # 为每个连续段创建基元（最多每个微状态3个段）
                    for seg_idx, (start_time, end_time) in enumerate(segments[:3]):
                        if end_time - start_time < 10:  # 跳过太短的段
                            continue
                        
                        # 评估质量
                        quality = self.quality_evaluator.evaluate_microstate_quality(stats, ms_id)
                        
                        # 数量模式：只过滤掉质量极低的
                        if quality < 0.01:
                            continue
                        
                        # 创建基元单位
                        primitive = PrimitiveUnit(
                            primitive_id=f"microstate_{sample_idx}_{ms_id}_{seg_idx}",
                            primitive_type='microstate',
                            channels=list(range(sample_data.shape[0])),
                            time_range=(start_time, end_time),
                            quality_score=quality,
                            microstate_id=ms_id,
                            metadata={
                                'sample_idx': int(sample_idx),
                                'explained_variance': float(explained_var),
                                'coverage': float(stats['coverage'][ms_id]),
                                'mean_duration': float(stats['mean_duration'][ms_id]),
                                'segment_idx': seg_idx
                            }
                        )
                        primitives.append(primitive)
                else:
                    # 质量模式：只取第一个连续段
                    start_time = ms_indices[0]
                    end_time = ms_indices[-1] + 1
                    
                    # 评估质量
                    quality = self.quality_evaluator.evaluate_microstate_quality(stats, ms_id)
                    
                    if quality < self.prior_config.microstate_priors.min_quality_score:
                        continue
                    
                    # 创建基元单位
                    primitive = PrimitiveUnit(
                        primitive_id=f"microstate_{sample_idx}_{ms_id}",
                        primitive_type='microstate',
                        channels=list(range(sample_data.shape[0])),
                        time_range=(start_time, end_time),
                        quality_score=quality,
                        microstate_id=ms_id,
                        metadata={
                            'sample_idx': int(sample_idx),
                            'explained_variance': float(explained_var),
                            'coverage': float(stats['coverage'][ms_id]),
                            'mean_duration': float(stats['mean_duration'][ms_id])
                        }
                    )
                    primitives.append(primitive)
        
        # 如果使用数量模式，按质量排序并取前N个
        if self.prior_config.use_count_mode:
            primitives.sort(key=lambda x: x.quality_score, reverse=True)
            target_count = self.prior_config.target_microstate_count
            # 如果候选数量不足目标数量，提取所有可用的
            actual_count = min(len(primitives), target_count)
            primitives = primitives[:actual_count]
            print(f"   按数量模式选择: 从所有候选中选择前 {actual_count} 个（目标: {target_count}，可用: {len(primitives)}）")
        
        return primitives
    
    def _extract_timefreq(self, X: np.ndarray) -> List[PrimitiveUnit]:
        """提取时频基元"""
        # 确保数据格式正确
        if len(X.shape) == 4:
            X = X.squeeze(1)
        
        # 如果使用数量模式，使用多个样本以获得更多候选基元
        if self.prior_config.use_count_mode:
            n_samples_to_use = min(10, X.shape[0])
        else:
            n_samples_to_use = 1
        
        all_primitives = []
        
        for sample_idx in range(n_samples_to_use):
            sample_data = X[sample_idx]  # (n_channels, n_timepoints)
            
            # 使用每个样本拟合（或只拟合第一个）
            if sample_idx == 0:
                self.timefreq_analyzer.fit(sample_data)
            
            # 获取时频单元
            timefreq_units = self.timefreq_analyzer.get_units()
            
            for i, unit in enumerate(timefreq_units):
                # 评估质量
                quality = self.quality_evaluator.evaluate_timefreq_quality(unit, sample_data)
                
                # 如果使用数量模式，只应用最低质量要求；否则应用严格阈值
                if self.prior_config.use_count_mode:
                    # 数量模式：只过滤掉质量极低的
                    if quality < 0.01:  # 非常低的质量阈值
                        continue
                else:
                    # 质量模式：应用严格阈值
                    if quality < self.prior_config.timefreq_priors.min_quality_score:
                        continue
                    if quality < self.prior_config.timefreq_priors.min_relative_power:
                        continue
                
                # 创建基元单位
                primitive = PrimitiveUnit(
                    primitive_id=f"timefreq_{sample_idx}_{i}",
                    primitive_type='timefreq',
                    channels=unit.channels,
                    time_range=unit.time_range,
                    quality_score=quality,
                    freq_band=unit.freq_band,
                    freq_band_name=unit.freq_band_name,
                    metadata={
                        'unit_id': unit.unit_id,
                        'window_idx': unit.metadata.get('window_idx', -1),
                        'channel_idx': unit.metadata.get('channel_idx', -1),
                        'sample_idx': sample_idx
                    }
                )
                all_primitives.append(primitive)
        
        # 如果使用数量模式，按质量排序并取前N个
        if self.prior_config.use_count_mode:
            all_primitives.sort(key=lambda x: x.quality_score, reverse=True)
            target_count = self.prior_config.target_timefreq_count
            # 如果候选数量不足目标数量，提取所有可用的
            actual_count = min(len(all_primitives), target_count)
            all_primitives = all_primitives[:actual_count]
            print(f"   按数量模式选择: 从所有候选中选择前 {actual_count} 个（目标: {target_count}，可用: {len(all_primitives)}）")
        
        return all_primitives
    
    def _apply_quality_threshold(self, primitives: List[PrimitiveUnit]) -> List[PrimitiveUnit]:
        """应用质量阈值筛选"""
        if self.prior_config.use_count_mode:
            # 数量模式：只过滤掉明显无效的基元（NaN、Inf等）
            # 不应用质量阈值，因为已经按数量提取了
            filtered = []
            for primitive in primitives:
                # 只过滤掉质量异常的（NaN、Inf或极端值）
                quality = primitive.quality_score
                if np.isnan(quality) or np.isinf(quality):
                    continue
                # 对于不同基元类型，使用不同的检查
                if primitive.primitive_type == 'shapelet':
                    # Shapelet的quality_score是原始信息增益，可能为负数，只过滤极端负值
                    if quality < -10.0:
                        continue
                else:
                    # 微状态和时频的quality_score是归一化的，只过滤极端负值
                    if quality < -1.0:
                        continue
                filtered.append(primitive)
            
            # 按质量分数排序（保持原有顺序，因为已经按质量排序过了）
            # 不重新排序，保持每类基元内部的排序
            return filtered
        else:
            # 质量模式：应用全局质量阈值
            filtered = []
            for primitive in primitives:
                if primitive.quality_score >= self.prior_config.global_quality_threshold:
                    filtered.append(primitive)
            
            # 按质量分数排序
            filtered.sort(key=lambda x: x.quality_score, reverse=True)
            return filtered
    
    def _record_prior_knowledge(self, n_shapelets: int, n_microstates: int, 
                               n_timefreq: int, n_filtered: int):
        """记录先验信息"""
        self.prior_knowledge_record = {
            'timestamp': datetime.now().isoformat(),
            'selection_mode': 'count_mode' if self.prior_config.use_count_mode else 'quality_mode',
            'target_counts': {
                'target_shapelet_count': self.prior_config.target_shapelet_count,
                'target_microstate_count': self.prior_config.target_microstate_count,
                'target_timefreq_count': self.prior_config.target_timefreq_count
            },
            'shapelet_priors': {
                'min_length': self.prior_config.shapelet_priors.min_length,
                'max_length': self.prior_config.shapelet_priors.max_length,
                'length_step': self.prior_config.shapelet_priors.length_step,
                'max_candidates': self.prior_config.shapelet_priors.max_candidates,
                'information_gain_threshold': self.prior_config.shapelet_priors.information_gain_threshold,
                'n_shapelets_selected': self.prior_config.shapelet_priors.n_shapelets_selected,
                'normalize_shapelet': self.prior_config.shapelet_priors.normalize_shapelet,
                'method': self.prior_config.shapelet_priors.method,
                'min_quality_score': self.prior_config.shapelet_priors.min_quality_score,
                'extracted_count': n_shapelets
            },
            'microstate_priors': {
                'n_microstates': self.prior_config.microstate_priors.n_microstates,
                'gfp_threshold_percentile': self.prior_config.microstate_priors.gfp_threshold_percentile,
                'min_duration': self.prior_config.microstate_priors.min_duration,
                'method': self.prior_config.microstate_priors.method,
                'zscore_normalize': self.prior_config.microstate_priors.zscore_normalize,
                'remove_mean': self.prior_config.microstate_priors.remove_mean,
                'use_absolute_correlation': self.prior_config.microstate_priors.use_absolute_correlation,
                'min_explained_variance': self.prior_config.microstate_priors.min_explained_variance,
                'min_quality_score': self.prior_config.microstate_priors.min_quality_score,
                'extracted_count': n_microstates
            },
            'timefreq_priors': {
                'freq_bands': {k: list(v) for k, v in self.prior_config.timefreq_priors.freq_bands.items()},
                'time_window_size': self.prior_config.timefreq_priors.time_window_size,
                'overlap_ratio': self.prior_config.timefreq_priors.overlap_ratio,
                'method': self.prior_config.timefreq_priors.method,
                'sampling_rate': self.prior_config.timefreq_priors.sampling_rate,
                'min_relative_power': self.prior_config.timefreq_priors.min_relative_power,
                'min_quality_score': self.prior_config.timefreq_priors.min_quality_score,
                'extracted_count': n_timefreq
            },
            'quality_thresholds': {
                'global_quality_threshold': self.prior_config.global_quality_threshold,
                'min_shapelet_quality': self.prior_config.shapelet_priors.min_quality_score,
                'min_microstate_quality': self.prior_config.microstate_priors.min_quality_score,
                'min_timefreq_quality': self.prior_config.timefreq_priors.min_quality_score
            },
            'extraction_summary': {
                'total_shapelets': n_shapelets,
                'total_microstates': n_microstates,
                'total_timefreq': n_timefreq,
                'total_before_filtering': n_shapelets + n_microstates + n_timefreq,
                'total_after_filtering': n_filtered
            }
        }
    
    def save_primitives(self, filepath: str):
        """保存提取的基元到JSON文件"""
        data = {
            'prior_knowledge_record': self.prior_knowledge_record,
            'primitives': [p.to_dict() for p in self.extracted_primitives]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def save_prior_knowledge(self, filepath: str):
        """保存先验信息记录"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.prior_knowledge_record, f, indent=2, ensure_ascii=False)
    
    def get_summary(self) -> str:
        """获取提取摘要"""
        summary = "=" * 80 + "\n"
        summary += "第一场选拔：基元提取摘要\n"
        summary += "=" * 80 + "\n\n"
        
        if self.prior_knowledge_record:
            ext_summary = self.prior_knowledge_record.get('extraction_summary', {})
            summary += f"提取统计:\n"
            summary += f"  - Shapelet基元: {ext_summary.get('total_shapelets', 0)}\n"
            summary += f"  - 微状态基元: {ext_summary.get('total_microstates', 0)}\n"
            summary += f"  - 时频基元: {ext_summary.get('total_timefreq', 0)}\n"
            summary += f"  - 筛选前总数: {ext_summary.get('total_before_filtering', 0)}\n"
            summary += f"  - 筛选后总数: {ext_summary.get('total_after_filtering', 0)}\n\n"
        
        summary += f"候选基元总数: {len(self.extracted_primitives)}\n"
        
        # 按类型统计
        type_counts = {}
        for p in self.extracted_primitives:
            type_counts[p.primitive_type] = type_counts.get(p.primitive_type, 0) + 1
        
        summary += "\n按类型分布:\n"
        for ptype, count in type_counts.items():
            summary += f"  - {ptype}: {count}\n"
        
        summary += "=" * 80 + "\n"
        
        return summary
