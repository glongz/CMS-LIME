#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMS-LIME解释器主类
整合微状态、shapelet、时频、因果扰动和局部回归模块
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score

# 导入各个模块
from microstate_analysis import MicrostateAnalyzer
from shapelet_analysis import ShapeletAnalyzer
from timefreq_analysis import TimeFreqAnalyzer
from causal_perturbation import CausalPerturbationEngine
from local_regression import LocalRegressionSelector

@dataclass
class CMSLimeConfig:
    """CMS-LIME配置类"""
    # 基元构造配置
    microstate_method: str = 'kmeans'  # 'kmeans' or 'atomize_agglomerate'
    n_microstates: int = 4
    shapelet_method: str = 'information_gain'  # 'information_gain' or 'learnable'
    n_shapelets: int = 100
    timefreq_method: str = 'cwt'  # 'cwt' or 'stft'
    freq_bands: Dict[str, Tuple[float, float]] = None
    
    # 因果分析配置
    causal_method: str = 'granger'  # 'granger', 'pcmci', 'notears'
    max_lag: int = 5
    significance_level: float = 0.05
    
    # 扰动配置
    perturbation_method: str = 'mask'  # 'mask', 'replace', 'diffusion'
    n_perturbations: int = 1000
    preserve_statistics: bool = True
    
    # 局部回归配置
    kernel_type: str = 'multidim'  # 'multidim', 'rbf', 'linear'
    regression_method: str = 'ridge'  # 'ridge', 'lasso', 'sparse_multitask'
    selection_method: str = 'dpp'  # 'dpp', 'submodular'
    n_features: int = 20
    
    # 其他配置
    random_state: int = 42
    n_jobs: int = -1
    verbose: bool = True
    
    def __post_init__(self):
        if self.freq_bands is None:
            self.freq_bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 100)
            }

class CMSLimeExplainer:
    """CMS-LIME解释器主类"""
    
    def __init__(self, config: CMSLimeConfig = None):
        self.config = config or CMSLimeConfig()
        
        # 初始化各个模块
        self.microstate_analyzer = MicrostateAnalyzer(
            method=self.config.microstate_method,
            n_microstates=self.config.n_microstates
        )
        
        self.shapelet_analyzer = ShapeletAnalyzer(
            method=self.config.shapelet_method,
            n_shapelets=self.config.n_shapelets
        )
        
        self.timefreq_analyzer = TimeFreqAnalyzer(
            method=self.config.timefreq_method,
            freq_bands=self.config.freq_bands
        )
        
        # 因果引擎将在fit时初始化
        self.causal_engine = None
        self.causal_method = self.config.causal_method
        self.max_lag = self.config.max_lag
        self.significance_level = self.config.significance_level
        
        # 创建权重核函数
        from local_regression import WeightKernel, SparseRegressor
        weight_kernel = WeightKernel(
            kernel_type=self.config.kernel_type
        )
        
        # 创建稀疏回归器
        sparse_regressor = SparseRegressor(
            method=self.config.regression_method
        )
        
        self.local_regressor = LocalRegressionSelector(
            weight_kernel=weight_kernel,
            regressor=sparse_regressor,
            selector_type=self.config.selection_method,
            max_features=self.config.n_features
        )
        
        # 存储训练数据和模型
        self.model = None
        self.training_data = None
        self.is_fitted = False
        
        # 存储解释结果
        self.explanations = {}
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, model: Any):
        """拟合解释器
        
        Args:
            X_train: 训练数据 (n_samples, n_channels, n_timepoints)
            y_train: 训练标签 (n_samples,)
            model: 预训练模型
        """
        if self.config.verbose:
            print("开始拟合CMS-LIME解释器...")
            
        self.model = model
        self.training_data = (X_train, y_train)
        
        # 1. 拟合微状态分析器
        if self.config.verbose:
            print("拟合微状态分析器...")
        self.microstate_analyzer.fit(X_train)
        
        # 2. 拟合shapelet分析器
        if self.config.verbose:
            print("拟合shapelet分析器...")
        self.shapelet_analyzer.fit(X_train, y_train)
        
        # 3. 拟合时频分析器
        if self.config.verbose:
            print("拟合时频分析器...")
        self.timefreq_analyzer.fit(X_train)
        
        # 4. 构建因果图和初始化因果引擎
        if self.config.verbose:
            print("构建因果图...")
        
        # 首先创建因果分析器来构建因果图
        from causal_perturbation import GrangerCausalityAnalyzer, CausalGraph, CausalPerturbationEngine
        
        # 根据方法选择分析器
        if self.causal_method == 'granger':
            causal_analyzer = GrangerCausalityAnalyzer(
                max_lag=self.max_lag,
                significance_level=self.significance_level
            )
        else:
            # 默认使用Granger
            causal_analyzer = GrangerCausalityAnalyzer(
                max_lag=self.max_lag,
                significance_level=self.significance_level
            )
        
        # 拟合因果分析器
        causal_analyzer.fit(X_train)
        
        # 构建因果图
        n_channels = X_train.shape[1] if X_train.ndim == 3 else X_train.shape[0]
        causal_graph = CausalGraph(n_channels)
        
        # 添加因果边
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j and causal_analyzer.causal_matrix[i, j] > 0.5:
                    causal_graph.add_edge(i, j, causal_analyzer.causal_matrix[i, j])
        
        # 初始化因果扰动引擎
        self.causal_engine = CausalPerturbationEngine(
            causal_graph=causal_graph,
            perturbation_strength=0.1,
            preserve_statistics=True
        )
        
        self.is_fitted = True
        
        if self.config.verbose:
            print("CMS-LIME解释器拟合完成!")
    
    def explain_instance(self, 
                        x: np.ndarray, 
                        target_class: Optional[int] = None,
                        unit_types: List[str] = ['microstate', 'shapelet', 'timefreq']) -> Dict[str, Any]:
        """解释单个实例
        
        Args:
            x: 输入样本 (n_channels, n_timepoints)
            target_class: 目标类别（如果为None则使用预测类别）
            unit_types: 使用的基元类型列表
            
        Returns:
            解释结果字典
        """
        if not self.is_fitted:
            raise ValueError("解释器尚未拟合，请先调用fit()方法")
            
        if self.config.verbose:
            print(f"开始解释实例，使用基元类型: {unit_types}")
            
        # 获取原始预测
        original_pred = self._predict_single(x)
        if target_class is None:
            target_class = np.argmax(original_pred)
            
        explanation = {
            'original_prediction': original_pred,
            'target_class': target_class,
            'unit_explanations': {},
            'combined_explanation': None
        }
        
        all_units = []
        all_importances = []
        
        # 1. 微状态基元解释
        if 'microstate' in unit_types:
            if self.config.verbose:
                print("生成微状态基元解释...")
            microstate_exp = self._explain_microstate_units(x, target_class)
            explanation['unit_explanations']['microstate'] = microstate_exp
            all_units.extend(microstate_exp['units'])
            all_importances.extend(microstate_exp['importances'])
            
        # 2. Shapelet基元解释
        if 'shapelet' in unit_types:
            if self.config.verbose:
                print("生成shapelet基元解释...")
            shapelet_exp = self._explain_shapelet_units(x, target_class)
            explanation['unit_explanations']['shapelet'] = shapelet_exp
            all_units.extend(shapelet_exp['units'])
            all_importances.extend(shapelet_exp['importances'])
            
        # 3. 时频基元解释
        if 'timefreq' in unit_types:
            if self.config.verbose:
                print("生成时频基元解释...")
            timefreq_exp = self._explain_timefreq_units(x, target_class)
            explanation['unit_explanations']['timefreq'] = timefreq_exp
            all_units.extend(timefreq_exp['units'])
            all_importances.extend(timefreq_exp['importances'])
            
        # 4. 综合解释选择
        if len(all_units) > 0:
            if self.config.verbose:
                print("进行综合解释选择...")
            combined_exp = self._combine_explanations(all_units, all_importances, x)
            explanation['combined_explanation'] = combined_exp
            
            # 将综合解释的结果提升到顶层
            explanation.update(combined_exp)
            
        return explanation
    
    def _explain_microstate_units(self, x: np.ndarray, target_class: int) -> Dict[str, Any]:
        """微状态基元解释"""
        # 获取微状态序列
        microstate_seq = self.microstate_analyzer.transform(x.reshape(1, *x.shape))[0]
        
        # 生成微状态单元
        units = self._generate_microstate_units(microstate_seq)
        
        # 因果一致性扰动
        perturbations = []
        predictions = []
        
        # 确保target_class是有效的整数索引
        original_pred_full = self._predict_single(x)
        try:
            target_class_idx = int(target_class) if isinstance(target_class, (str, np.str_)) else target_class
        except (ValueError, TypeError):
            target_class_idx = np.argmax(original_pred_full)
        
        if target_class_idx >= len(original_pred_full):
            target_class_idx = np.argmax(original_pred_full)
        
        for unit in units:
            # 生成扰动样本
            perturbed_samples = self._generate_causal_perturbations(x, unit, 'microstate')
            
            # 预测扰动样本
            unit_predictions = []
            for perturbed_x in perturbed_samples:
                pred = self._predict_single(perturbed_x)
                unit_predictions.append(pred[target_class_idx])
                
            perturbations.append(perturbed_samples)
            predictions.append(np.mean(unit_predictions))
            
        # 计算重要性
        original_score = self._predict_single(x)[target_class_idx]
        importances = [original_score - pred for pred in predictions]
        
        return {
            'units': units,
            'importances': importances,
            'perturbations': perturbations,
            'predictions': predictions
        }
    
    def _explain_shapelet_units(self, x: np.ndarray, target_class: int) -> Dict[str, Any]:
        """Shapelet基元解释"""
        # 获取shapelet匹配
        shapelet_matches = self.shapelet_analyzer.transform(x.reshape(1, *x.shape))[0]
        
        # 生成shapelet单元
        units = self._generate_shapelet_units(shapelet_matches)
        
        # 因果一致性扰动
        perturbations = []
        predictions = []
        
        # 确保target_class是有效的整数索引
        original_pred_full = self._predict_single(x)
        try:
            target_class_idx = int(target_class) if isinstance(target_class, (str, np.str_)) else target_class
        except (ValueError, TypeError):
            target_class_idx = np.argmax(original_pred_full)
        
        if target_class_idx >= len(original_pred_full):
            target_class_idx = np.argmax(original_pred_full)
        
        for unit in units:
            # 生成扰动样本
            perturbed_samples = self._generate_causal_perturbations(x, unit, 'shapelet')
            
            # 预测扰动样本
            unit_predictions = []
            for perturbed_x in perturbed_samples:
                pred = self._predict_single(perturbed_x)
                unit_predictions.append(pred[target_class_idx])
                
            perturbations.append(perturbed_samples)
            predictions.append(np.mean(unit_predictions))
            
        # 计算重要性
        # 确保target_class是有效的整数索引
        original_pred_full = self._predict_single(x)
        try:
            target_class_idx = int(target_class) if isinstance(target_class, (str, np.str_)) else target_class
        except (ValueError, TypeError):
            target_class_idx = np.argmax(original_pred_full)
        
        if target_class_idx >= len(original_pred_full):
            target_class_idx = np.argmax(original_pred_full)
            
        original_score = original_pred_full[target_class_idx]
        importances = [original_score - pred for pred in predictions]
        
        return {
            'units': units,
            'importances': importances,
            'perturbations': perturbations,
            'predictions': predictions
        }
    
    def _explain_timefreq_units(self, x: np.ndarray, target_class: int) -> Dict[str, Any]:
        """时频基元解释"""
        # 获取时频表示
        timefreq_repr = self.timefreq_analyzer.transform(x.reshape(1, *x.shape))[0]
        
        # 生成时频单元
        units = self._generate_timefreq_units(timefreq_repr)
        
        # 因果一致性扰动
        perturbations = []
        predictions = []

        # 确保target_class是有效的整数索引
        original_pred_full = self._predict_single(x)
        try:
            target_class_idx = int(target_class) if isinstance(target_class, (str, np.str_)) else target_class
        except (ValueError, TypeError):
            target_class_idx = np.argmax(original_pred_full)

        if target_class_idx >= len(original_pred_full):
            target_class_idx = np.argmax(original_pred_full)
        
        for unit in units:
            # 生成扰动样本
            perturbed_samples = self._generate_causal_perturbations(x, unit, 'timefreq')
            
            # 预测扰动样本
            unit_predictions = []
            for perturbed_x in perturbed_samples:
                pred = self._predict_single(perturbed_x)
                unit_predictions.append(pred[target_class_idx])
                
            perturbations.append(perturbed_samples)
            predictions.append(np.mean(unit_predictions))
            
        # 计算重要性
        original_score = original_pred_full[target_class_idx]
        importances = [original_score - pred for pred in predictions]
        
        return {
            'units': units,
            'importances': importances,
            'perturbations': perturbations,
            'predictions': predictions
        }
    
    def _generate_microstate_units(self, microstate_seq: np.ndarray) -> List[Dict[str, Any]]:
        """生成微状态单元"""
        units = []
        
        # 找到连续的微状态区间
        current_state = microstate_seq[0]
        start_idx = 0
        
        for i in range(1, len(microstate_seq)):
            if microstate_seq[i] != current_state:
                # 创建单元
                unit = {
                    'type': 'microstate',
                    'state': current_state,
                    'channels': list(range(22)),  # 添加所有通道（假设22通道EEG）
                    'time_range': (start_idx, i-1),  # 添加time_range字段
                    'start_time': start_idx,
                    'end_time': i-1,
                    'duration': i - start_idx,
                    'primitive_id': f'generated_microstate_{current_state}_{start_idx}_{i-1}'  # 添加基元ID
                }
                units.append(unit)
                
                current_state = microstate_seq[i]
                start_idx = i
                
        # 添加最后一个单元
        unit = {
            'type': 'microstate',
            'state': current_state,
            'channels': list(range(22)),  # 添加所有通道（假设22通道EEG）
            'time_range': (start_idx, len(microstate_seq)-1),  # 添加time_range字段
            'start_time': start_idx,
            'end_time': len(microstate_seq)-1,
            'duration': len(microstate_seq) - start_idx,
            'primitive_id': f'generated_microstate_{current_state}_{start_idx}_{len(microstate_seq)-1}'  # 添加基元ID
        }
        units.append(unit)
        
        return units
    
    def _generate_shapelet_units(self, shapelet_matches: np.ndarray) -> List[Dict[str, Any]]:
        """生成shapelet单元"""
        units = []
        
        # 基于shapelet匹配生成单元
        for i, match_score in enumerate(shapelet_matches):
            if match_score > 0.5:  # 阈值可配置
                # 安全地获取维度信息
                if isinstance(shapelet_matches, np.ndarray):
                    if shapelet_matches.ndim >= 2:
                        n_channels = shapelet_matches.shape[0]
                        n_timepoints = shapelet_matches.shape[1]
                    else:
                        n_channels = 1
                        n_timepoints = len(shapelet_matches) if len(shapelet_matches) > 0 else 100
                else:
                    n_channels = 8  # 默认值
                    n_timepoints = 100  # 默认值
                
                unit = {
                    'type': 'shapelet',
                    'shapelet_id': i,
                    'match_score': match_score,
                    'channels': list(range(n_channels)),
                    'time_range': (0, n_timepoints)
                }
                units.append(unit)
                
        return units
    
    def _generate_timefreq_units(self, timefreq_repr) -> List[Dict[str, Any]]:
        """生成时频单元"""
        units = []
        
        # 处理不同类型的时频表示
        if isinstance(timefreq_repr, dict):
            items = timefreq_repr.items()
        elif isinstance(timefreq_repr, np.ndarray):
            # 如果是数组，创建默认频带
            items = [('default', timefreq_repr)]
        else:
            # 其他情况，返回空列表
            return units
        
        for band_name, band_data in items:
            # 安全地获取维度
            if isinstance(band_data, np.ndarray):
                if band_data.ndim >= 2:
                    n_channels, n_times = band_data.shape[:2]
                elif band_data.ndim == 1:
                    n_channels = 1
                    n_times = len(band_data)
                else:
                    n_channels = 8  # 默认值
                    n_times = 100  # 默认值
            else:
                n_channels = 8  # 默认值
                n_times = 100  # 默认值
            
            # 简化：每个时间窗口作为一个单元
            window_size = min(50, n_times // 4)  # 可配置
            
            for t_start in range(0, n_times - window_size, window_size // 2):
                t_end = min(t_start + window_size, n_times)
                
                unit = {
                    'type': 'timefreq',
                    'freq_band': band_name,
                    'time_start': t_start,
                    'time_end': t_end,
                    'channels': list(range(n_channels)),
                    'power': np.mean(band_data[..., t_start:t_end]) if band_data.ndim > 1 else np.mean(band_data[t_start:t_end])
                }
                units.append(unit)
                
        return units
    
    def _generate_causal_perturbations(self, 
                                     x: np.ndarray, 
                                     unit: Dict[str, Any], 
                                     unit_type: str) -> List[np.ndarray]:
        """生成因果一致性扰动"""
        perturbations = []
        
        for _ in range(min(10, self.config.n_perturbations // 100)):  # 简化处理
            perturbed_x = self.causal_engine.perturb_units(
                x, [unit], np.array([0.1])
            )
            perturbations.append(perturbed_x)
            
        return perturbations
    
    def _combine_explanations(self, 
                            units: List[Dict[str, Any]], 
                            importances: List[float],
                            x: np.ndarray) -> Dict[str, Any]:
        """综合多种基元的解释"""
        # 使用局部回归选择器
        # 创建特征矩阵
        feature_matrix = np.array([[i] for i in range(len(units))])
        importance_scores = np.array(importances)
        
        # 清理NaN值
        valid_mask = ~np.isnan(importance_scores)
        if not np.any(valid_mask):
            # 如果所有值都是NaN，使用默认值
            importance_scores = np.ones(len(units))
            valid_mask = np.ones(len(units), dtype=bool)
        else:
            # 用均值填充NaN
            mean_val = np.nanmean(importance_scores)
            importance_scores[~valid_mask] = mean_val
        
        # 使用fit_and_select方法
        try:
            self.local_regressor.fit_and_select(
                feature_matrix, importance_scores, units, units
            )
        except Exception as e:
            print(f"局部回归失败: {e}")
            # 使用简单的top-k选择作为备选
            selected_indices = list(range(min(len(units), self.config.n_features)))
            weights = np.ones(len(selected_indices))
            selected_units = [units[i] for i in selected_indices]
            selected_importances = [importances[i] for i in selected_indices]
            
            return {
                'selected_units': selected_units,
                'selected_importances': selected_importances,
                'selection_weights': weights,
                'selection_indices': selected_indices
            }
        
        # 获取选择结果
        explanation = self.local_regressor.get_explanation()
        selected_indices = explanation.get('selected_features', list(range(min(len(units), self.config.n_features))))
        weights = explanation.get('feature_importance', np.ones(len(selected_indices)))
        
        selected_units = [units[i] for i in selected_indices]
        selected_importances = [importances[i] for i in selected_indices]
        
        return {
            'selected_units': selected_units,
            'selected_importances': selected_importances,
            'selection_weights': weights,
            'selection_indices': selected_indices
        }
    
    def _predict_single(self, x: np.ndarray) -> np.ndarray:
        """单样本预测"""
        # 处理维度
        if x.ndim == 2:
            # 检查是否需要转置：如果第二维度更大，可能是(n_timepoints, n_channels)
            if x.shape[1] > x.shape[0] and x.shape[0] <= 64:  # 假设通道数不超过64
                # (n_timepoints, n_channels) -> (1, n_channels, n_timepoints)
                x = x.T.reshape(1, x.shape[0], x.shape[1])
            else:
                # (n_channels, n_timepoints) -> (1, n_channels, n_timepoints)
                x = x.reshape(1, x.shape[0], x.shape[1])
        elif x.ndim == 1:
            # 1D数据，根据训练数据形状重塑
            if self.training_data is not None:
                n_channels = self.training_data[0].shape[1]
                n_timepoints = len(x) // n_channels
                x = x.reshape(1, n_channels, n_timepoints)
            else:
                x = x.reshape(1, 1, -1)
        elif x.ndim == 3 and x.shape[0] == 1:
            # 强制检查维度：如果是(1, n_timepoints, n_channels)格式，转换为(1, n_channels, n_timepoints)
            if x.shape[1] > 100 and x.shape[2] <= 64:  # 时间点数>100且通道数<=64
                # (1, n_timepoints, n_channels) -> (1, n_channels, n_timepoints)
                x = x.transpose(0, 2, 1)
            # 否则已经是正确的3D格式 (1, n_channels, n_timepoints)
        elif x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 1:
            # 4D格式 (1, 1, n_channels, n_timepoints) -> (1, n_channels, n_timepoints)
            x = x.reshape(1, x.shape[2], x.shape[3])
            
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(x)[0]
        elif hasattr(self.model, 'predict'):
            pred = self.model.predict(x)[0]
            if isinstance(pred, (int, np.integer)):
                # 分类结果转为概率
                n_classes = len(np.unique(self.training_data[1]))
                proba = np.zeros(n_classes)
                proba[pred] = 1.0
                return proba
            return pred
        else:
            # PyTorch模型
            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x)
                output = self.model(x_tensor)
                
                # 确保输出是概率分布
                if output.dim() > 1 and output.size(1) > 1:
                    # 多类分类，应用softmax
                    probs = torch.nn.functional.softmax(output, dim=1)
                    return probs[0].detach().cpu().numpy()
                else:
                    # 二分类或回归，直接返回
                    result = output[0].detach().cpu().numpy()
                    if isinstance(result, np.ndarray) and result.ndim == 0:
                        # 标量结果，转换为数组
                        result = np.array([result])
                    return result
    
    def visualize_explanation(self, 
                            explanation: Dict[str, Any], 
                            x: np.ndarray,
                            save_path: Optional[str] = None):
        """可视化解释结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CMS-LIME解释结果', fontsize=16)
        
        # 1. 原始信号
        axes[0, 0].plot(x.T)
        axes[0, 0].set_title('Original EEG Signal')
        axes[0, 0].set_xlabel('Time Points')
        axes[0, 0].set_ylabel('Amplitude')
        
        # 2. 基元重要性
        if explanation['combined_explanation'] is not None:
            importances = explanation['combined_explanation']['selected_importances']
            unit_labels = [f"Unit {i}" for i in range(len(importances))]
            
            axes[0, 1].bar(unit_labels, importances)
            axes[0, 1].set_title('Primitive Importance')
            axes[0, 1].set_xlabel('Primitives')
            axes[0, 1].set_ylabel('Importance Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 不同基元类型的贡献
        unit_types = list(explanation['unit_explanations'].keys())
        type_importances = []
        for unit_type in unit_types:
            importances = explanation['unit_explanations'][unit_type]['importances']
            mean_importance = np.mean(np.abs(importances))
            # 处理NaN值
            if np.isnan(mean_importance) or mean_importance == 0:
                mean_importance = 0.001  # 设置一个很小的值避免饼图出错
            type_importances.append(mean_importance)
            
        # 确保所有值都是有效的
        type_importances = np.array(type_importances)
        if np.any(np.isnan(type_importances)) or np.sum(type_importances) == 0:
            type_importances = np.ones(len(unit_types)) / len(unit_types)  # 均匀分布
            
        axes[1, 0].pie(type_importances, labels=unit_types, autopct='%1.1f%%')
        axes[1, 0].set_title('Primitive Type Contribution Distribution')
        
        # 4. 预测置信度
        pred = explanation['original_prediction']
        class_labels = [f"Class {i}" for i in range(len(pred))]
        
        axes[1, 1].bar(class_labels, pred)
        axes[1, 1].set_title('Prediction Confidence')
        axes[1, 1].set_xlabel('Classes')
        axes[1, 1].set_ylabel('Probability')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def get_explanation_summary(self, explanation: Dict[str, Any]) -> str:
        """获取解释摘要"""
        summary = []
        summary.append("=== CMS-LIME Explanation Summary ===")
        
        # 预测信息
        pred = explanation['original_prediction']
        target_class = explanation['target_class']
        # 确保target_class是有效的整数索引
        try:
            target_class_idx = int(target_class) if isinstance(target_class, (str, np.str_)) else target_class
        except (ValueError, TypeError):
            target_class_idx = np.argmax(pred)
        
        if target_class_idx >= len(pred):
            target_class_idx = np.argmax(pred)
            
        confidence = pred[target_class_idx]
        
        summary.append(f"Predicted Class: {target_class}")
        summary.append(f"Prediction Confidence: {confidence:.3f}")
        
        # 基元解释信息
        summary.append("\n基元解释:")
        for unit_type, unit_exp in explanation['unit_explanations'].items():
            n_units = len(unit_exp['units'])
            avg_importance = np.mean(np.abs(unit_exp['importances']))
            summary.append(f"  {unit_type}: {n_units}个基元, 平均重要性={avg_importance:.3f}")
        
        # 综合解释信息
        if explanation['combined_explanation'] is not None:
            combined = explanation['combined_explanation']
            n_selected = len(combined['selected_units'])
            top_importance = max(np.abs(combined['selected_importances']))
            summary.append(f"\n综合解释: 选择了{n_selected}个关键基元")
            summary.append(f"最高重要性: {top_importance:.3f}")
        
        return "\n".join(summary)

# 示例使用代码
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    n_samples, n_channels, n_timepoints = 100, 32, 1000
    X_train = np.random.randn(n_samples, n_channels, n_timepoints)
    y_train = np.random.randint(0, 3, n_samples)
    
    # 创建简单的模型（示例）
    class SimpleEEGModel:
        def predict_proba(self, x):
            # 简单的随机预测（实际应用中应使用真实模型）
            return np.random.dirichlet([1, 1, 1], x.shape[0])
    
    model = SimpleEEGModel()
    
    # 创建和拟合解释器
    config = CMSLimeConfig(
        n_microstates=4,
        n_shapelets=50,
        n_perturbations=100,
        verbose=True
    )
    
    explainer = CMSLimeExplainer(config)
    
    print("拟合CMS-LIME解释器...")
    explainer.fit(X_train, y_train, model)
    
    # 解释单个实例
    test_sample = np.random.randn(n_channels, n_timepoints)
    
    print("\n生成解释...")
    explanation = explainer.explain_instance(
        test_sample, 
        unit_types=['microstate', 'shapelet', 'timefreq']
    )
    
    # 打印解释摘要
    print("\n" + explainer.get_explanation_summary(explanation))
    
    # 可视化解释
    explainer.visualize_explanation(explanation, test_sample)
    
    print("\nCMS-LIME解释完成!")