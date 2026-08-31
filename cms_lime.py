#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal-Microstate Shapelet LIME (CMS-LIME) Framework

面向EEG预测任务的解释框架，在时空-时频-因果维度上生成神经语义一致、
跨个体可复用、稳健的解释。

核心特性：
- 以微状态/shapelet为扰动基本单元
- 施加因果一致性约束
- 在时频域与时空图结构中进行扰动与回归
- 保持模型无关性

Author: CMS-LIME Framework
Date: 2025
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import pairwise_distances
from scipy import signal
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CMSLimeConfig:
    """CMS-LIME配置类"""
    # 微状态配置
    n_microstates: int = 8
    microstate_method: str = 'kmeans'  # 'kmeans', 'atomize_agglomerate'
    
    # Shapelet配置
    n_shapelets: int = 50
    shapelet_length_range: Tuple[int, int] = (10, 100)
    shapelet_method: str = 'information_gain'  # 'information_gain', 'learnable'
    
    # 时频配置
    freq_bands: Dict[str, Tuple[float, float]] = None
    time_window_size: int = 50
    overlap_ratio: float = 0.5
    
    # 因果配置
    causal_method: str = 'granger'  # 'granger', 'pcmci', 'notears'
    max_lag: int = 5
    causal_threshold: float = 0.05
    
    # 扰动配置
    perturbation_method: str = 'mask'  # 'mask', 'replace', 'noise'
    n_perturbations: int = 1000
    
    # 回归配置
    regression_method: str = 'ridge'  # 'ridge', 'lasso', 'sparse_multitask'
    alpha: float = 1.0
    n_features_select: int = 10
    
    def __post_init__(self):
        if self.freq_bands is None:
            self.freq_bands = {
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 100)
            }


class BaseUnit(ABC):
    """基元单位抽象基类"""
    
    def __init__(self, unit_id: str, channels: List[int], 
                 time_range: Tuple[int, int], metadata: Dict = None):
        self.unit_id = unit_id
        self.channels = channels
        self.time_range = time_range
        self.metadata = metadata or {}
    
    @abstractmethod
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """提取基元特征"""
        pass
    
    @abstractmethod
    def compute_distance(self, other: 'BaseUnit', data: np.ndarray) -> float:
        """计算与其他基元的距离"""
        pass


class MicrostateUnit(BaseUnit):
    """微状态基元单位"""
    
    def __init__(self, unit_id: str, channels: List[int], 
                 time_range: Tuple[int, int], microstate_label: int,
                 topography: np.ndarray, metadata: Dict = None):
        super().__init__(unit_id, channels, time_range, metadata)
        self.microstate_label = microstate_label
        self.topography = topography
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """提取微状态特征"""
        segment = data[self.channels, self.time_range[0]:self.time_range[1]]
        # 计算全局场功率(GFP)
        gfp = np.std(segment, axis=0)
        # 计算拓扑相关性
        topo_corr = np.corrcoef(segment.mean(axis=1), self.topography)[0, 1]
        return np.array([gfp.mean(), gfp.std(), topo_corr])
    
    def compute_distance(self, other: 'MicrostateUnit', data: np.ndarray) -> float:
        """计算微状态编辑距离"""
        if self.microstate_label == other.microstate_label:
            return 0.0
        # 基于拓扑相似性的距离
        topo_dist = 1 - np.corrcoef(self.topography, other.topography)[0, 1]
        return topo_dist


class ShapeletUnit(BaseUnit):
    """Shapelet基元单位"""
    
    def __init__(self, unit_id: str, channels: List[int], 
                 time_range: Tuple[int, int], shapelet_pattern: np.ndarray,
                 information_gain: float, metadata: Dict = None):
        super().__init__(unit_id, channels, time_range, metadata)
        self.shapelet_pattern = shapelet_pattern
        self.information_gain = information_gain
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """提取shapelet特征"""
        segment = data[self.channels, self.time_range[0]:self.time_range[1]]
        # 计算与shapelet模式的相似性
        similarity = self._compute_similarity(segment)
        return np.array([similarity, self.information_gain])
    
    def _compute_similarity(self, segment: np.ndarray) -> float:
        """计算与shapelet模式的相似性"""
        if segment.shape != self.shapelet_pattern.shape:
            return 0.0
        return np.corrcoef(segment.flatten(), self.shapelet_pattern.flatten())[0, 1]
    
    def compute_distance(self, other: 'ShapeletUnit', data: np.ndarray) -> float:
        """计算shapelet距离"""
        # 基于模式相似性的距离
        pattern_dist = 1 - np.corrcoef(
            self.shapelet_pattern.flatten(), 
            other.shapelet_pattern.flatten()
        )[0, 1]
        return pattern_dist


class TimeFreqUnit(BaseUnit):
    """时频基元单位"""
    
    def __init__(self, unit_id: str, channels: List[int], 
                 time_range: Tuple[int, int], freq_band: Tuple[float, float],
                 freq_band_name: str, metadata: Dict = None):
        super().__init__(unit_id, channels, time_range, metadata)
        self.freq_band = freq_band
        self.freq_band_name = freq_band_name
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """提取时频特征"""
        segment = data[self.channels, self.time_range[0]:self.time_range[1]]
        # 计算功率谱密度
        freqs, psd = signal.welch(segment, axis=1)
        # 提取目标频带功率
        freq_mask = (freqs >= self.freq_band[0]) & (freqs <= self.freq_band[1])
        band_power = psd[:, freq_mask].mean(axis=1)
        return band_power
    
    def compute_distance(self, other: 'TimeFreqUnit', data: np.ndarray) -> float:
        """计算时频距离"""
        # 时间距离
        time_dist = abs(self.time_range[0] - other.time_range[0])
        # 频率距离
        freq_dist = abs(self.freq_band[0] - other.freq_band[0])
        # 通道距离
        channel_dist = len(set(self.channels).symmetric_difference(set(other.channels)))
        return time_dist + freq_dist + channel_dist


class CausalGraph:
    """因果图类"""
    
    def __init__(self, n_channels: int, max_lag: int = 5):
        self.n_channels = n_channels
        self.max_lag = max_lag
        self.adjacency_matrix = np.zeros((n_channels, n_channels, max_lag + 1))
        self.causal_strengths = np.zeros((n_channels, n_channels, max_lag + 1))
    
    def add_edge(self, from_channel: int, to_channel: int, lag: int, strength: float):
        """添加因果边"""
        if 0 <= lag <= self.max_lag:
            self.adjacency_matrix[from_channel, to_channel, lag] = 1
            self.causal_strengths[from_channel, to_channel, lag] = strength
    
    def get_causal_closure(self, channels: List[int], time_range: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取因果闭包"""
        closure = []
        for channel in channels:
            for lag in range(self.max_lag + 1):
                # 父节点
                parents = np.where(self.adjacency_matrix[:, channel, lag] > 0)[0]
                for parent in parents:
                    parent_time = (time_range[0] - lag, time_range[1] - lag)
                    if parent_time[0] >= 0:
                        closure.append((parent, parent_time))
                
                # 子节点
                children = np.where(self.adjacency_matrix[channel, :, lag] > 0)[0]
                for child in children:
                    child_time = (time_range[0] + lag, time_range[1] + lag)
                    closure.append((child, child_time))
        
        return list(set(closure))


class PerturbationEngine:
    """扰动引擎"""
    
    def __init__(self, config: CMSLimeConfig, causal_graph: Optional[CausalGraph] = None):
        self.config = config
        self.causal_graph = causal_graph
    
    def perturb_unit(self, data: np.ndarray, unit: BaseUnit, 
                     method: str = 'mask') -> np.ndarray:
        """扰动单个基元"""
        perturbed_data = data.copy()
        
        if method == 'mask':
            perturbed_data[unit.channels, unit.time_range[0]:unit.time_range[1]] = 0
        elif method == 'replace':
            # 用随机噪声替换
            noise = np.random.normal(0, data.std(), 
                                   (len(unit.channels), 
                                    unit.time_range[1] - unit.time_range[0]))
            perturbed_data[unit.channels, unit.time_range[0]:unit.time_range[1]] = noise
        elif method == 'noise':
            # 添加噪声
            noise = np.random.normal(0, data.std() * 0.1, 
                                   (len(unit.channels), 
                                    unit.time_range[1] - unit.time_range[0]))
            perturbed_data[unit.channels, unit.time_range[0]:unit.time_range[1]] += noise
        
        return perturbed_data
    
    def causal_consistent_perturbation(self, data: np.ndarray, unit: BaseUnit) -> np.ndarray:
        """因果一致性扰动"""
        if self.causal_graph is None:
            return self.perturb_unit(data, unit)
        
        # 获取因果闭包
        closure = self.causal_graph.get_causal_closure(unit.channels, unit.time_range)
        
        perturbed_data = data.copy()
        
        # 扰动原始单元
        perturbed_data = self.perturb_unit(perturbed_data, unit)
        
        # 扰动因果闭包中的相关单元
        for channel, time_range in closure:
            if time_range[1] <= data.shape[1] and time_range[0] >= 0:
                closure_unit = BaseUnit(f"closure_{channel}", [channel], time_range)
                perturbed_data = self.perturb_unit(perturbed_data, closure_unit)
        
        return perturbed_data


class WeightKernel:
    """权重核函数"""
    
    def __init__(self, config: CMSLimeConfig):
        self.config = config
    
    def compute_weights(self, original_data: np.ndarray, 
                      perturbed_samples: List[np.ndarray],
                      units: List[BaseUnit]) -> np.ndarray:
        """计算权重"""
        n_samples = len(perturbed_samples)
        weights = np.zeros(n_samples)
        
        for i, perturbed_data in enumerate(perturbed_samples):
            # 时频距离
            time_freq_dist = self._compute_time_freq_distance(original_data, perturbed_data)
            
            # 微状态编辑距离（如果适用）
            microstate_dist = 0
            if isinstance(units[i], MicrostateUnit):
                microstate_dist = self._compute_microstate_distance(original_data, perturbed_data)
            
            # 因果一致性评分
            causal_score = self._compute_causal_consistency_score(original_data, perturbed_data)
            
            # 综合权重
            total_dist = time_freq_dist + microstate_dist + (1 - causal_score)
            weights[i] = np.exp(-total_dist)
        
        # 安全的权重归一化
        weights_sum = weights.sum()
        if weights_sum > 0:
            return weights / weights_sum
        else:
            # 如果所有权重都为0，返回均匀权重
            return np.ones(len(weights)) / len(weights)
    
    def _compute_time_freq_distance(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """计算时频距离"""
        # 简化的时频距离计算
        return np.linalg.norm(data1 - data2)
    
    def _compute_microstate_distance(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """计算微状态编辑距离"""
        # 简化的微状态距离计算
        return np.corrcoef(data1.flatten(), data2.flatten())[0, 1]
    
    def _compute_causal_consistency_score(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """计算因果一致性评分"""
        # 简化的因果一致性评分
        return 1.0  # 占位符实现


class LocalRegressor:
    """局部回归器"""
    
    def __init__(self, config: CMSLimeConfig):
        self.config = config
        self.regressor = None
        self._initialize_regressor()
    
    def _initialize_regressor(self):
        """初始化回归器"""
        if self.config.regression_method == 'ridge':
            self.regressor = Ridge(alpha=self.config.alpha)
        elif self.config.regression_method == 'lasso':
            self.regressor = Lasso(alpha=self.config.alpha)
        else:
            self.regressor = Ridge(alpha=self.config.alpha)
    
    def fit_and_select(self, X: np.ndarray, y: np.ndarray, 
                      weights: np.ndarray, units: List[BaseUnit]) -> Tuple[np.ndarray, List[int]]:
        """拟合回归器并选择重要特征"""
        # 加权回归
        self.regressor.fit(X, y, sample_weight=weights)
        
        # 获取特征重要性
        if hasattr(self.regressor, 'coef_'):
            feature_importance = np.abs(self.regressor.coef_)
        else:
            feature_importance = np.ones(X.shape[1])
        
        # 选择top-k特征
        selected_indices = np.argsort(feature_importance)[-self.config.n_features_select:]
        
        return feature_importance, selected_indices.tolist()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        return self.regressor.predict(X)


class CMSLimeExplainer:
    """CMS-LIME解释器主类"""
    
    def __init__(self, config: CMSLimeConfig = None):
        self.config = config or CMSLimeConfig()
        self.causal_graph = None
        self.perturbation_engine = None
        self.weight_kernel = WeightKernel(self.config)
        self.local_regressor = LocalRegressor(self.config)
        self.units = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, model: Callable):
        """拟合解释器"""
        # 构建因果图
        self.causal_graph = self._build_causal_graph(X)
        
        # 初始化扰动引擎
        self.perturbation_engine = PerturbationEngine(self.config, self.causal_graph)
        
        # 构造基元单位
        self.units = self._construct_units(X, y)
        
        print(f"CMS-LIME解释器已拟合，构造了{len(self.units)}个基元单位")
    
    def explain_instance(self, instance: np.ndarray, model: Callable, 
                        target_class: Optional[int] = None) -> Dict:
        """解释单个实例"""
        if not self.units:
            raise ValueError("解释器尚未拟合，请先调用fit方法")
        
        # 生成扰动样本
        perturbed_samples, perturbation_matrix = self._generate_perturbations(instance)
        
        # 获取模型预测
        predictions = np.array([model(sample) for sample in perturbed_samples])
        
        # 如果是分类任务且指定了目标类别
        if target_class is not None and len(predictions.shape) > 1:
            predictions = predictions[:, target_class]
        
        # 计算权重
        weights = self.weight_kernel.compute_weights(instance, perturbed_samples, self.units)
        
        # 局部回归
        feature_importance, selected_indices = self.local_regressor.fit_and_select(
            perturbation_matrix, predictions, weights, self.units
        )
        
        # 构建解释结果
        explanation = {
            'feature_importance': feature_importance,
            'selected_units': [self.units[i] for i in selected_indices],
            'selected_indices': selected_indices,
            'weights': weights,
            'local_fidelity': self._compute_local_fidelity(perturbation_matrix, predictions, weights)
        }
        
        return explanation
    
    def _build_causal_graph(self, X: np.ndarray) -> CausalGraph:
        """构建因果图"""
        n_channels = X.shape[1] if len(X.shape) == 3 else X.shape[0]
        causal_graph = CausalGraph(n_channels, self.config.max_lag)
        
        # 简化的Granger因果分析
        if self.config.causal_method == 'granger':
            for i in range(n_channels):
                for j in range(n_channels):
                    if i != j:
                        # 简化的因果强度计算
                        strength = np.random.random()  # 占位符
                        if strength > self.config.causal_threshold:
                            causal_graph.add_edge(i, j, 1, strength)
        
        return causal_graph
    
    def _construct_units(self, X: np.ndarray, y: np.ndarray) -> List[BaseUnit]:
        """构造基元单位"""
        units = []
        
        # 构造微状态单元
        microstate_units = self._construct_microstate_units(X)
        units.extend(microstate_units)
        
        # 构造shapelet单元
        shapelet_units = self._construct_shapelet_units(X, y)
        units.extend(shapelet_units)
        
        # 构造时频单元
        timefreq_units = self._construct_timefreq_units(X)
        units.extend(timefreq_units)
        
        return units
    
    def _construct_microstate_units(self, X: np.ndarray) -> List[MicrostateUnit]:
        """构造微状态单元"""
        units = []
        
        # 简化的微状态分析
        if len(X.shape) == 3:  # (n_samples, n_channels, n_timepoints)
            n_samples, n_channels, n_timepoints = X.shape
            
            # 对每个样本进行微状态分割
            for sample_idx in range(min(10, n_samples)):  # 限制样本数量
                sample_data = X[sample_idx]
                
                # K-means聚类
                kmeans = KMeans(n_clusters=self.config.n_microstates, random_state=42)
                microstate_labels = kmeans.fit_predict(sample_data.T)
                
                # 创建微状态单元
                for ms_label in range(self.config.n_microstates):
                    ms_indices = np.where(microstate_labels == ms_label)[0]
                    if len(ms_indices) > 0:
                        time_range = (ms_indices[0], ms_indices[-1] + 1)
                        topography = kmeans.cluster_centers_[ms_label]
                        
                        unit = MicrostateUnit(
                            unit_id=f"ms_{sample_idx}_{ms_label}",
                            channels=list(range(n_channels)),
                            time_range=time_range,
                            microstate_label=ms_label,
                            topography=topography
                        )
                        units.append(unit)
        
        return units
    
    def _construct_shapelet_units(self, X: np.ndarray, y: np.ndarray) -> List[ShapeletUnit]:
        """构造shapelet单元"""
        units = []
        
        if len(X.shape) == 3:
            n_samples, n_channels, n_timepoints = X.shape
            
            # 简化的shapelet发现
            for _ in range(self.config.n_shapelets):
                # 随机选择shapelet
                sample_idx = np.random.randint(0, n_samples)
                channel_idx = np.random.randint(0, n_channels)
                length = np.random.randint(*self.config.shapelet_length_range)
                start_time = np.random.randint(0, max(1, n_timepoints - length))
                
                shapelet_pattern = X[sample_idx, channel_idx, start_time:start_time+length]
                
                # 计算信息增益（简化）
                information_gain = np.random.random()  # 占位符
                
                unit = ShapeletUnit(
                    unit_id=f"shapelet_{len(units)}",
                    channels=[channel_idx],
                    time_range=(start_time, start_time + length),
                    shapelet_pattern=shapelet_pattern.reshape(1, -1),
                    information_gain=information_gain
                )
                units.append(unit)
        
        return units
    
    def _construct_timefreq_units(self, X: np.ndarray) -> List[TimeFreqUnit]:
        """构造时频单元"""
        units = []
        
        if len(X.shape) == 3:
            n_samples, n_channels, n_timepoints = X.shape
            
            # 为每个频带和时间窗口创建单元
            for freq_name, freq_band in self.config.freq_bands.items():
                n_windows = max(1, n_timepoints // self.config.time_window_size)
                
                for window_idx in range(n_windows):
                    start_time = window_idx * self.config.time_window_size
                    end_time = min(start_time + self.config.time_window_size, n_timepoints)
                    
                    for channel_idx in range(n_channels):
                        unit = TimeFreqUnit(
                            unit_id=f"tf_{freq_name}_{window_idx}_{channel_idx}",
                            channels=[channel_idx],
                            time_range=(start_time, end_time),
                            freq_band=freq_band,
                            freq_band_name=freq_name
                        )
                        units.append(unit)
        
        return units
    
    def _generate_perturbations(self, instance: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """生成扰动样本"""
        perturbed_samples = []
        perturbation_matrix = np.zeros((self.config.n_perturbations, len(self.units)))
        
        for i in range(self.config.n_perturbations):
            # 随机选择要扰动的单元
            n_units_to_perturb = np.random.randint(1, min(10, len(self.units)))
            units_to_perturb = np.random.choice(len(self.units), n_units_to_perturb, replace=False)
            
            perturbed_data = instance.copy()
            
            for unit_idx in units_to_perturb:
                unit = self.units[unit_idx]
                perturbed_data = self.perturbation_engine.causal_consistent_perturbation(
                    perturbed_data, unit
                )
                perturbation_matrix[i, unit_idx] = 1
            
            perturbed_samples.append(perturbed_data)
        
        return perturbed_samples, perturbation_matrix
    
    def _compute_local_fidelity(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """计算局部保真度"""
        y_pred = self.local_regressor.predict(X)
        weighted_mse = np.average((y - y_pred) ** 2, weights=weights)
        return 1 / (1 + weighted_mse)


def create_cms_lime_explainer(config: Dict = None) -> CMSLimeExplainer:
    """创建CMS-LIME解释器的便捷函数"""
    if config:
        cms_config = CMSLimeConfig(**config)
    else:
        cms_config = CMSLimeConfig()
    
    return CMSLimeExplainer(cms_config)


if __name__ == "__main__":
    # 示例使用
    print("CMS-LIME框架已加载")
    print("使用示例：")
    print("explainer = create_cms_lime_explainer()")
    print("explainer.fit(X_train, y_train, model)")
    print("explanation = explainer.explain_instance(instance, model)")