#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
先验信息配置模块

从shapelet_analysis.py、microstate_analysis.py、timefreq_analysis.py中
提取的所有先验信息参数，用于第一场选拔（基元提取）

Author: CMS-LIME Framework
Date: 2025
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import json


@dataclass
class ShapeletPriors:
    """Shapelet分析先验信息"""
    # 长度范围
    min_length: int = 10
    max_length: int = 100
    length_step: int = 5
    
    # 候选数量限制
    max_candidates: int = 100
    
    # 信息增益阈值（用于筛选判别性shapelet）
    information_gain_threshold: float = 0.0
    
    # 选择数量
    n_shapelets_selected: int = 50
    
    # 标准化要求
    normalize_shapelet: bool = True
    
    # 方法选择
    method: str = 'information_gain'  # 'information_gain' or 'learnable'
    
    # 质量阈值
    min_quality_score: float = 0.01


@dataclass
class MicrostatePriors:
    """微状态分析先验信息"""
    # 微状态数量
    n_microstates: int = 4
    
    # GFP阈值（使用分位数）
    gfp_threshold_percentile: float = 90.0
    
    # 最小持续时间（样本点数）
    min_duration: int = 3
    
    # 方法选择
    method: str = 'kmeans'  # 'kmeans' or 'atomize_agglomerate'
    
    # 预处理要求
    zscore_normalize: bool = True
    remove_mean: bool = True  # 重新参考
    
    # 相关性阈值（使用绝对值）
    use_absolute_correlation: bool = True
    
    # 质量阈值
    min_explained_variance: float = 0.3
    min_quality_score: float = 0.1


@dataclass
class TimeFreqPriors:
    """时频分析先验信息"""
    # 频带定义
    freq_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    })
    
    # 时间窗口
    time_window_size: int = 50  # 样本点数
    overlap_ratio: float = 0.5
    
    # 方法选择
    method: str = 'stft'  # 'stft' or 'cwt'
    
    # CWT参数（如果使用CWT）
    cwt_wavelet: str = 'cmor1.5-1.0'
    cwt_scales: Optional[List[float]] = None
    
    # STFT参数（如果使用STFT）
    stft_window: str = 'hann'
    stft_nperseg: int = 256
    stft_noverlap: Optional[int] = None
    
    # 采样率
    sampling_rate: float = 250.0
    
    # 功率特征阈值
    min_relative_power: float = 0.1
    min_quality_score: float = 0.1


@dataclass
class PriorKnowledgeConfig:
    """先验信息配置主类"""
    # 三种基元类型的先验信息
    shapelet_priors: ShapeletPriors = field(default_factory=ShapeletPriors)
    microstate_priors: MicrostatePriors = field(default_factory=MicrostatePriors)
    timefreq_priors: TimeFreqPriors = field(default_factory=TimeFreqPriors)
    
    # 全局质量阈值（用于最终筛选，如果使用数量模式则作为最低质量要求）
    global_quality_threshold: float = 0.1
    
    # 基元数量配置（第一场选拔的目标数量）
    target_shapelet_count: int = 100  # Shapelet基元目标数量
    target_microstate_count: int = 100  # 微状态基元目标数量
    target_timefreq_count: int = 100  # 时频基元目标数量
    
    # 是否使用数量模式（True：按数量提取，False：按质量阈值筛选）
    use_count_mode: bool = True
    
    # 数据格式配置
    n_channels: int = 22
    n_timepoints: int = 1280
    sampling_rate: float = 256.0
    
    def to_dict(self) -> Dict:
        """转换为字典格式（用于JSON序列化）"""
        return {
            'shapelet_priors': {
                'min_length': self.shapelet_priors.min_length,
                'max_length': self.shapelet_priors.max_length,
                'length_step': self.shapelet_priors.length_step,
                'max_candidates': self.shapelet_priors.max_candidates,
                'information_gain_threshold': self.shapelet_priors.information_gain_threshold,
                'n_shapelets_selected': self.shapelet_priors.n_shapelets_selected,
                'normalize_shapelet': self.shapelet_priors.normalize_shapelet,
                'method': self.shapelet_priors.method,
                'min_quality_score': self.shapelet_priors.min_quality_score
            },
            'microstate_priors': {
                'n_microstates': self.microstate_priors.n_microstates,
                'gfp_threshold_percentile': self.microstate_priors.gfp_threshold_percentile,
                'min_duration': self.microstate_priors.min_duration,
                'method': self.microstate_priors.method,
                'zscore_normalize': self.microstate_priors.zscore_normalize,
                'remove_mean': self.microstate_priors.remove_mean,
                'use_absolute_correlation': self.microstate_priors.use_absolute_correlation,
                'min_explained_variance': self.microstate_priors.min_explained_variance,
                'min_quality_score': self.microstate_priors.min_quality_score
            },
            'timefreq_priors': {
                'freq_bands': {k: list(v) for k, v in self.timefreq_priors.freq_bands.items()},
                'time_window_size': self.timefreq_priors.time_window_size,
                'overlap_ratio': self.timefreq_priors.overlap_ratio,
                'method': self.timefreq_priors.method,
                'cwt_wavelet': self.timefreq_priors.cwt_wavelet,
                'stft_window': self.timefreq_priors.stft_window,
                'stft_nperseg': self.timefreq_priors.stft_nperseg,
                'sampling_rate': self.timefreq_priors.sampling_rate,
                'min_relative_power': self.timefreq_priors.min_relative_power,
                'min_quality_score': self.timefreq_priors.min_quality_score
            },
            'global_quality_threshold': self.global_quality_threshold,
            'target_counts': {
                'target_shapelet_count': self.target_shapelet_count,
                'target_microstate_count': self.target_microstate_count,
                'target_timefreq_count': self.target_timefreq_count
            },
            'use_count_mode': self.use_count_mode,
            'data_format': {
                'n_channels': self.n_channels,
                'n_timepoints': self.n_timepoints,
                'sampling_rate': self.sampling_rate
            }
        }
    
    def save_to_json(self, filepath: str):
        """保存配置到JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'PriorKnowledgeConfig':
        """从JSON文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 重建配置对象
        shapelet_priors = ShapeletPriors(**data['shapelet_priors'])
        microstate_priors = MicrostatePriors(**data['microstate_priors'])
        timefreq_priors = TimeFreqPriors(
            freq_bands={k: tuple(v) for k, v in data['timefreq_priors']['freq_bands'].items()},
            time_window_size=data['timefreq_priors']['time_window_size'],
            overlap_ratio=data['timefreq_priors']['overlap_ratio'],
            method=data['timefreq_priors']['method'],
            cwt_wavelet=data['timefreq_priors']['cwt_wavelet'],
            stft_window=data['timefreq_priors']['stft_window'],
            stft_nperseg=data['timefreq_priors']['stft_nperseg'],
            sampling_rate=data['timefreq_priors']['sampling_rate'],
            min_relative_power=data['timefreq_priors']['min_relative_power'],
            min_quality_score=data['timefreq_priors']['min_quality_score']
        )
        
        config = cls(
            shapelet_priors=shapelet_priors,
            microstate_priors=microstate_priors,
            timefreq_priors=timefreq_priors,
            global_quality_threshold=data.get('global_quality_threshold', 0.1),
            target_shapelet_count=data.get('target_counts', {}).get('target_shapelet_count', 100),
            target_microstate_count=data.get('target_counts', {}).get('target_microstate_count', 100),
            target_timefreq_count=data.get('target_counts', {}).get('target_timefreq_count', 100),
            use_count_mode=data.get('use_count_mode', True),
            n_channels=data['data_format']['n_channels'],
            n_timepoints=data['data_format']['n_timepoints'],
            sampling_rate=data['data_format']['sampling_rate']
        )
        
        return config
    
    def get_summary(self) -> str:
        """获取配置摘要"""
        summary = "=" * 80 + "\n"
        summary += "先验信息配置摘要\n"
        summary += "=" * 80 + "\n\n"
        
        summary += "1. Shapelet分析先验信息:\n"
        summary += f"   - 长度范围: {self.shapelet_priors.min_length}-{self.shapelet_priors.max_length} (步长: {self.shapelet_priors.length_step})\n"
        summary += f"   - 最大候选数: {self.shapelet_priors.max_candidates}\n"
        summary += f"   - 信息增益阈值: {self.shapelet_priors.information_gain_threshold}\n"
        summary += f"   - 选择数量: {self.shapelet_priors.n_shapelets_selected}\n"
        summary += f"   - 方法: {self.shapelet_priors.method}\n"
        summary += f"   - 最小质量分数: {self.shapelet_priors.min_quality_score}\n\n"
        
        summary += "2. 微状态分析先验信息:\n"
        summary += f"   - 微状态数量: {self.microstate_priors.n_microstates}\n"
        summary += f"   - GFP阈值百分位数: {self.microstate_priors.gfp_threshold_percentile}\n"
        summary += f"   - 最小持续时间: {self.microstate_priors.min_duration} 样本点\n"
        summary += f"   - 方法: {self.microstate_priors.method}\n"
        summary += f"   - 最小解释方差: {self.microstate_priors.min_explained_variance}\n"
        summary += f"   - 最小质量分数: {self.microstate_priors.min_quality_score}\n\n"
        
        summary += "3. 时频分析先验信息:\n"
        summary += f"   - 频带定义: {list(self.timefreq_priors.freq_bands.keys())}\n"
        for band_name, (low, high) in self.timefreq_priors.freq_bands.items():
            summary += f"     * {band_name}: {low}-{high} Hz\n"
        summary += f"   - 时间窗口大小: {self.timefreq_priors.time_window_size} 样本点\n"
        summary += f"   - 重叠比例: {self.timefreq_priors.overlap_ratio}\n"
        summary += f"   - 方法: {self.timefreq_priors.method}\n"
        summary += f"   - 最小相对功率: {self.timefreq_priors.min_relative_power}\n"
        summary += f"   - 最小质量分数: {self.timefreq_priors.min_quality_score}\n\n"
        
        summary += "4. 全局配置:\n"
        summary += f"   - 全局质量阈值: {self.global_quality_threshold}\n"
        summary += f"   - 使用数量模式: {self.use_count_mode}\n"
        if self.use_count_mode:
            summary += f"   - Shapelet目标数量: {self.target_shapelet_count}\n"
            summary += f"   - 微状态目标数量: {self.target_microstate_count}\n"
            summary += f"   - 时频目标数量: {self.target_timefreq_count}\n"
        summary += f"   - 数据格式: {self.n_channels}通道 × {self.n_timepoints}时间点\n"
        summary += f"   - 采样率: {self.sampling_rate} Hz\n"
        summary += "=" * 80 + "\n"
        
        return summary


def create_default_prior_config() -> PriorKnowledgeConfig:
    """创建默认的先验信息配置"""
    return PriorKnowledgeConfig()


if __name__ == "__main__":
    # 测试配置
    config = create_default_prior_config()
    print(config.get_summary())
    
    # 保存配置
    config.save_to_json("default_prior_config.json")
    print("\n配置已保存到 default_prior_config.json")
    
    # 测试加载
    loaded_config = PriorKnowledgeConfig.load_from_json("default_prior_config.json")
    print("\n配置加载成功！")
