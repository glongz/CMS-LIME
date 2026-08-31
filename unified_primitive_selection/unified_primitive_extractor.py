#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一基元提取器

从单个样本提取基元，并立即进行重要性评估

Author: CMS-LIME Framework
Date: 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import sys
import os

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from shapelet_analysis import ShapeletAnalyzer, ShapeletCandidate
from microstate_analysis import MicrostateAnalyzer
from timefreq_analysis import TimeFreqAnalyzer, TimeFreqUnit

# 支持相对导入和绝对导入
try:
    from .config import UnifiedSelectionConfig
    from .primitive_importance_evaluator import PrimitiveImportanceEvaluator, PrimitiveUnit
except ImportError:
    from config import UnifiedSelectionConfig
    from primitive_importance_evaluator import PrimitiveImportanceEvaluator, PrimitiveUnit

# 不在这里使用tqdm，避免在处理单个样本时显示进度条
# try:
#     from tqdm import tqdm
# except ImportError:
#     def tqdm(iterable, desc=None, **kwargs):
#         return iterable


@dataclass
class BiomarkerUnit:
    """生物标志物单位（带重要性分数）"""
    primitive_id: str
    primitive_type: str
    channels: List[int]
    time_range: Tuple[int, int]
    importance_score: float
    quality_score: float = 0.0
    shapelet_data: np.ndarray = None
    microstate_id: int = None
    freq_band: Tuple[float, float] = None
    freq_band_name: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        result = {
            'primitive_id': self.primitive_id,
            'primitive_type': self.primitive_type,
            'channels': self.channels,
            'time_range': list(self.time_range),
            'importance_score': float(self.importance_score),
            'quality_score': float(self.quality_score),
            'metadata': self.metadata
        }
        
        if self.microstate_id is not None:
            result['microstate_id'] = int(self.microstate_id)
        
        if self.freq_band is not None:
            result['freq_band'] = list(self.freq_band)
            result['freq_band_name'] = self.freq_band_name
        
        return result


class UnifiedPrimitiveExtractor:
    """统一基元提取器：提取+评估"""
    
    def __init__(self, 
                 config: UnifiedSelectionConfig,
                 importance_evaluator: PrimitiveImportanceEvaluator):
        """
        初始化统一提取器
        
        Parameters:
        -----------
        config : UnifiedSelectionConfig
            配置对象
        importance_evaluator : PrimitiveImportanceEvaluator
            重要性评估器
        """
        self.config = config
        self.prior_config = config.prior_config
        self.eval_config = config.evaluation_config
        self.importance_evaluator = importance_evaluator
        
        # 初始化分析器
        self._init_analyzers()
    
    def _init_analyzers(self):
        """初始化各种分析器"""
        # Shapelet分析器
        if self.prior_config.use_count_mode:
            n_shapelets = max(
                self.prior_config.target_shapelet_count * 2,
                self.prior_config.shapelet_priors.n_shapelets_selected
            )
        else:
            n_shapelets = self.prior_config.shapelet_priors.n_shapelets_selected
        
        self.shapelet_analyzer = ShapeletAnalyzer(
            method=self.prior_config.shapelet_priors.method,
            min_length=self.prior_config.shapelet_priors.min_length,
            max_length=self.prior_config.shapelet_priors.max_length,
            n_shapelets=n_shapelets
        )
        
        # 微状态分析器
        self.microstate_analyzer = MicrostateAnalyzer(
            n_microstates=self.prior_config.microstate_priors.n_microstates,
            method=self.prior_config.microstate_priors.method,
            min_duration=self.prior_config.microstate_priors.min_duration
        )
        
        # 时频分析器
        self.timefreq_analyzer = TimeFreqAnalyzer(
            method=self.prior_config.timefreq_priors.method,
            sampling_rate=self.prior_config.timefreq_priors.sampling_rate,
            freq_bands=self.prior_config.timefreq_priors.freq_bands,
            time_window_size=self.prior_config.timefreq_priors.time_window_size,
            overlap_ratio=self.prior_config.timefreq_priors.overlap_ratio
        )
    
    def extract_and_evaluate(self,
                            X_sample: np.ndarray,
                            y_sample: int) -> List[BiomarkerUnit]:
        """
        从单个样本提取基元并立即评估重要性
        
        Parameters:
        -----------
        X_sample : np.ndarray
            样本数据，形状为 (n_channels, n_timepoints)
        y_sample : int
            样本标签
        
        Returns:
        --------
        biomarkers : list
            带重要性分数的生物标志物列表
        """
        # 确保输入格式正确
        if X_sample.ndim == 3:
            X_sample = X_sample[0]
        
        # 准备批量数据（分析器需要批量输入）
        X_batch = X_sample[np.newaxis, :, :]  # (1, n_channels, n_timepoints)
        y_batch = np.array([y_sample])
        
        all_biomarkers = []
        
        # 1. 提取并评估Shapelet基元
        shapelet_biomarkers = self._extract_and_evaluate_shapelets(
            X_sample, X_batch, y_batch, y_sample
        )
        all_biomarkers.extend(shapelet_biomarkers)
        
        # 2. 提取并评估微状态基元
        microstate_biomarkers = self._extract_and_evaluate_microstates(
            X_sample, X_batch, y_sample
        )
        all_biomarkers.extend(microstate_biomarkers)
        
        # 3. 提取并评估时频基元
        timefreq_biomarkers = self._extract_and_evaluate_timefreq(
            X_sample, X_batch, y_sample
        )
        all_biomarkers.extend(timefreq_biomarkers)
        
        return all_biomarkers
    
    def _extract_and_evaluate_shapelets(self,
                                       X_sample: np.ndarray,
                                       X_batch: np.ndarray,
                                       y_batch: np.ndarray,
                                       y_sample: int) -> List[BiomarkerUnit]:
        """提取并评估Shapelet基元"""
        biomarkers = []
        
        try:
            # 拟合并提取shapelets
            self.shapelet_analyzer.fit(X_batch, y_batch)
            shapelets = self.shapelet_analyzer.get_shapelets()
            
            if len(shapelets) == 0:
                return biomarkers
            
            # 先按信息增益排序，只评估前N个（提高效率）
            shapelets_sorted = sorted(shapelets, key=lambda x: x.information_gain, reverse=True)
            # 限制评估数量，避免计算时间过长
            max_to_evaluate = min(len(shapelets_sorted), self.prior_config.target_shapelet_count * 2)
            shapelets_to_evaluate = shapelets_sorted[:max_to_evaluate]
            
            # print(f"   评估 {len(shapelets_to_evaluate)} 个shapelet的重要性...")  # 注释掉
            
            # 对每个shapelet进行评估
            for i, shapelet in enumerate(shapelets_to_evaluate):
                # 创建基元单位
                primitive = PrimitiveUnit(
                    primitive_id=f"shapelet_{i}",
                    primitive_type='shapelet',
                    channels=[shapelet.channel],
                    time_range=(shapelet.start_pos, shapelet.start_pos + shapelet.length),
                    quality_score=float(shapelet.information_gain),
                    shapelet_data=shapelet.data,
                    metadata={
                        'information_gain': float(shapelet.information_gain),
                        'length': int(shapelet.length),
                        'series_id': int(shapelet.series_id),
                        'class_label': int(shapelet.class_label) if shapelet.class_label is not None else None
                    }
                )
                
                # 立即评估重要性
                importance = self.importance_evaluator.evaluate_importance(
                    X_sample, primitive, y_sample
                )
                
                # 创建生物标志物单位
                biomarker = BiomarkerUnit(
                    primitive_id=primitive.primitive_id,
                    primitive_type=primitive.primitive_type,
                    channels=primitive.channels,
                    time_range=primitive.time_range,
                    importance_score=importance,
                    quality_score=primitive.quality_score,
                    shapelet_data=primitive.shapelet_data,
                    metadata=primitive.metadata
                )
                
                biomarkers.append(biomarker)
            
            # 按重要性排序并选择top K
            biomarkers.sort(key=lambda x: abs(x.importance_score), reverse=True)
            target_count = self.prior_config.target_shapelet_count
            biomarkers = biomarkers[:min(len(biomarkers), target_count)]
            
        except Exception as e:
            print(f"   提取Shapelet基元时出错: {e}")
        
        return biomarkers
    
    def _extract_and_evaluate_microstates(self,
                                         X_sample: np.ndarray,
                                         X_batch: np.ndarray,
                                         y_sample: int) -> List[BiomarkerUnit]:
        """提取并评估微状态基元"""
        biomarkers = []
        
        try:
            # 拟合微状态分析器
            self.microstate_analyzer.fit(
                X_sample,
                sampling_rate=self.prior_config.sampling_rate
            )
            
            # 使用transform方法获取微状态序列和统计信息
            sequence, correlations, stats = self.microstate_analyzer.transform(X_sample)
            
            if sequence is None or len(sequence) == 0:
                return biomarkers
            
            # 提取微状态段
            n_samples = self.prior_config.target_microstate_count if self.prior_config.use_count_mode else 50
            segments = []
            
            # 为每个微状态找到所有连续的时间段
            for ms_id in range(self.prior_config.microstate_priors.n_microstates):
                ms_mask = sequence == ms_id
                if not np.any(ms_mask):
                    continue
                
                ms_indices = np.where(ms_mask)[0]
                if len(ms_indices) == 0:
                    continue
                
                # 找到所有连续段
                start = ms_indices[0]
                for i in range(1, len(ms_indices)):
                    if ms_indices[i] != ms_indices[i-1] + 1:
                        if ms_indices[i-1] + 1 - start >= self.prior_config.microstate_priors.min_duration:
                            segments.append((ms_id, start, ms_indices[i-1] + 1))
                        start = ms_indices[i]
                # 处理最后一个段
                if ms_indices[-1] + 1 - start >= self.prior_config.microstate_priors.min_duration:
                    segments.append((ms_id, start, ms_indices[-1] + 1))
            
            # 限制段数量
            if len(segments) > n_samples:
                segments = segments[:n_samples]
            
            # print(f"   评估 {len(segments)} 个微状态段的重要性...")  # 注释掉
            
            # 对每个微状态段进行评估
            for seg_idx, (ms_id, start_time, end_time) in enumerate(segments):
                explained_var = stats.get('explained_variance', 0.0)
                
                # 评估质量
                coverage = stats.get('coverage', [0.0] * self.prior_config.microstate_priors.n_microstates)
                if ms_id < len(coverage):
                    quality = explained_var * coverage[ms_id]
                else:
                    quality = explained_var * 0.25
                
                # 创建基元单位
                primitive = PrimitiveUnit(
                    primitive_id=f"microstate_{ms_id}_{seg_idx}",
                    primitive_type='microstate',
                    channels=list(range(X_sample.shape[0])),
                    time_range=(start_time, end_time),
                    quality_score=float(quality),
                    microstate_id=int(ms_id),
                    metadata={
                        'explained_variance': float(explained_var),
                        'coverage': float(coverage[ms_id] if ms_id < len(coverage) else 0.0),
                        'mean_duration': float(stats.get('mean_duration', [0.0] * self.prior_config.microstate_priors.n_microstates)[ms_id] if ms_id < len(stats.get('mean_duration', [])) else 0.0),
                        'segment_idx': seg_idx
                    }
                )
                
                # 立即评估重要性
                importance = self.importance_evaluator.evaluate_importance(
                    X_sample, primitive, y_sample
                )
                
                # 创建生物标志物单位
                biomarker = BiomarkerUnit(
                    primitive_id=primitive.primitive_id,
                    primitive_type=primitive.primitive_type,
                    channels=primitive.channels,
                    time_range=primitive.time_range,
                    importance_score=importance,
                    quality_score=primitive.quality_score,
                    microstate_id=primitive.microstate_id,
                    metadata=primitive.metadata
                )
                
                biomarkers.append(biomarker)
            
            # 按重要性排序并选择top K
            biomarkers.sort(key=lambda x: abs(x.importance_score), reverse=True)
            target_count = self.prior_config.target_microstate_count
            biomarkers = biomarkers[:min(len(biomarkers), target_count)]
            
        except Exception as e:
            print(f"   提取微状态基元时出错: {e}")
            import traceback
            traceback.print_exc()
        
        return biomarkers
    
    def _extract_and_evaluate_timefreq(self,
                                      X_sample: np.ndarray,
                                      X_batch: np.ndarray,
                                      y_sample: int) -> List[BiomarkerUnit]:
        """提取并评估时频基元"""
        biomarkers = []
        
        try:
            # 拟合时频分析器
            self.timefreq_analyzer.fit(X_sample)
            
            # 获取时频单元（使用get_units方法）
            timefreq_units = self.timefreq_analyzer.get_units()
            
            if len(timefreq_units) == 0:
                return biomarkers
            
            # 限制评估数量
            max_to_evaluate = min(len(timefreq_units), self.prior_config.target_timefreq_count * 2)
            timefreq_units_to_evaluate = timefreq_units[:max_to_evaluate]
            
            # print(f"   评估 {len(timefreq_units_to_evaluate)} 个时频单元的重要性...")  # 注释掉
            
            # 对每个时频单元进行评估
            for unit in timefreq_units_to_evaluate:
                # 计算质量分数（相对功率）
                quality = 0.0
                if hasattr(unit, 'relative_power'):
                    quality = float(unit.relative_power)
                elif hasattr(unit, 'features') and len(unit.features) > len(unit.channels):
                    quality = float(unit.features[len(unit.channels)])
                
                # 创建基元单位
                primitive = PrimitiveUnit(
                    primitive_id=unit.unit_id,
                    primitive_type='timefreq',
                    channels=unit.channels,
                    time_range=unit.time_range,
                    quality_score=quality,
                    freq_band=unit.freq_band,
                    freq_band_name=unit.freq_band_name,
                    metadata={
                        'unit_id': unit.unit_id,
                        'window_idx': unit.metadata.get('window_idx', -1) if hasattr(unit, 'metadata') else -1,
                        'channel_idx': unit.metadata.get('channel_idx', -1) if hasattr(unit, 'metadata') else -1
                    }
                )
                
                # 立即评估重要性
                importance = self.importance_evaluator.evaluate_importance(
                    X_sample, primitive, y_sample
                )
                
                # 创建生物标志物单位
                biomarker = BiomarkerUnit(
                    primitive_id=primitive.primitive_id,
                    primitive_type=primitive.primitive_type,
                    channels=primitive.channels,
                    time_range=primitive.time_range,
                    importance_score=importance,
                    quality_score=primitive.quality_score,
                    freq_band=primitive.freq_band,
                    freq_band_name=primitive.freq_band_name,
                    metadata=primitive.metadata
                )
                
                biomarkers.append(biomarker)
            
            # 按重要性排序并选择top K
            biomarkers.sort(key=lambda x: abs(x.importance_score), reverse=True)
            target_count = self.prior_config.target_timefreq_count
            biomarkers = biomarkers[:min(len(biomarkers), target_count)]
            
        except Exception as e:
            print(f"   提取时频基元时出错: {e}")
        
        return biomarkers
