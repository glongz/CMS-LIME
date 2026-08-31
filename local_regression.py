#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
局部回归与选择模块

实现CMS-LIME的局部回归和特征选择，包括：
- 多维权重核函数（时频距离、微状态距离、因果一致性）
- 稀疏回归方法（Ridge、Lasso、Elastic Net、多任务学习）
- 确定点过程（DPP）特征选择
- 子模块优化选择
- 解释质量评估

Author: CMS-LIME Framework
Date: 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import det, inv, pinv
from sklearn.linear_model import Ridge, Lasso, ElasticNet, MultiTaskLasso, MultiTaskElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


class WeightKernel:
    """多维权重核函数类"""
    
    def __init__(self, time_weight: float = 1.0, freq_weight: float = 1.0,
                 microstate_weight: float = 1.0, causal_weight: float = 1.0,
                 kernel_type: str = 'rbf', bandwidth: float = 1.0):
        """
        初始化权重核函数
        
        Parameters:
        -----------
        time_weight : float
            时间距离权重
        freq_weight : float
            频率距离权重
        microstate_weight : float
            微状态距离权重
        causal_weight : float
            因果一致性权重
        kernel_type : str
            核函数类型 ('rbf', 'linear', 'polynomial', 'exponential')
        bandwidth : float
            核函数带宽
        """
        self.time_weight = time_weight
        self.freq_weight = freq_weight
        self.microstate_weight = microstate_weight
        self.causal_weight = causal_weight
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth
    
    def compute_time_distance(self, units1: List, units2: List) -> np.ndarray:
        """
        计算时间距离
        
        Parameters:
        -----------
        units1, units2 : list
            单元列表
        
        Returns:
        --------
        distances : np.ndarray
            时间距离矩阵
        """
        n1, n2 = len(units1), len(units2)
        distances = np.zeros((n1, n2))
        
        for i, unit1 in enumerate(units1):
            for j, unit2 in enumerate(units2):
                if hasattr(unit1, 'time_range') and hasattr(unit2, 'time_range'):
                    t1_center = (unit1.time_range[0] + unit1.time_range[1]) / 2
                    t2_center = (unit2.time_range[0] + unit2.time_range[1]) / 2
                    distances[i, j] = abs(t1_center - t2_center)
                else:
                    distances[i, j] = 0
        
        return distances
    
    def compute_freq_distance(self, units1: List, units2: List) -> np.ndarray:
        """
        计算频率距离
        
        Parameters:
        -----------
        units1, units2 : list
            单元列表
        
        Returns:
        --------
        distances : np.ndarray
            频率距离矩阵
        """
        n1, n2 = len(units1), len(units2)
        distances = np.zeros((n1, n2))
        
        for i, unit1 in enumerate(units1):
            for j, unit2 in enumerate(units2):
                if hasattr(unit1, 'freq_band') and hasattr(unit2, 'freq_band'):
                    f1_center = (unit1.freq_band[0] + unit1.freq_band[1]) / 2
                    f2_center = (unit2.freq_band[0] + unit2.freq_band[1]) / 2
                    distances[i, j] = abs(f1_center - f2_center)
                elif hasattr(unit1, 'frequency') and hasattr(unit2, 'frequency'):
                    distances[i, j] = abs(unit1.frequency - unit2.frequency)
                else:
                    distances[i, j] = 0
        
        return distances
    
    def compute_microstate_distance(self, units1: List, units2: List) -> np.ndarray:
        """
        计算微状态距离
        
        Parameters:
        -----------
        units1, units2 : list
            单元列表
        
        Returns:
        --------
        distances : np.ndarray
            微状态距离矩阵
        """
        n1, n2 = len(units1), len(units2)
        distances = np.zeros((n1, n2))
        
        for i, unit1 in enumerate(units1):
            for j, unit2 in enumerate(units2):
                if hasattr(unit1, 'microstate_label') and hasattr(unit2, 'microstate_label'):
                    # 微状态标签距离
                    if unit1.microstate_label == unit2.microstate_label:
                        distances[i, j] = 0
                    else:
                        distances[i, j] = 1
                elif hasattr(unit1, 'topography') and hasattr(unit2, 'topography'):
                    # 拓扑距离
                    distances[i, j] = np.linalg.norm(unit1.topography - unit2.topography)
                else:
                    distances[i, j] = 0
        
        return distances
    
    def compute_causal_consistency(self, units1: List, units2: List, 
                                 causal_graph=None) -> np.ndarray:
        """
        计算因果一致性分数
        
        Parameters:
        -----------
        units1, units2 : list
            单元列表
        causal_graph : CausalGraph, optional
            因果图
        
        Returns:
        --------
        consistency : np.ndarray
            因果一致性分数矩阵
        """
        n1, n2 = len(units1), len(units2)
        consistency = np.ones((n1, n2))  # 默认一致性为1
        
        if causal_graph is None:
            return consistency
        
        for i, unit1 in enumerate(units1):
            for j, unit2 in enumerate(units2):
                if hasattr(unit1, 'channels') and hasattr(unit2, 'channels'):
                    # 检查因果关系
                    channels1 = unit1.channels if isinstance(unit1.channels, list) else [unit1.channels]
                    channels2 = unit2.channels if isinstance(unit2.channels, list) else [unit2.channels]
                    
                    # 计算因果连接强度
                    causal_strength = 0
                    total_pairs = 0
                    
                    for ch1 in channels1:
                        for ch2 in channels2:
                            if (0 <= ch1 < causal_graph.n_nodes and 
                                0 <= ch2 < causal_graph.n_nodes):
                                causal_strength += causal_graph.strength_matrix[ch1, ch2]
                                causal_strength += causal_graph.strength_matrix[ch2, ch1]
                                total_pairs += 2
                    
                    if total_pairs > 0:
                        consistency[i, j] = causal_strength / total_pairs
        
        return consistency
    
    def compute_weights(self, target_units: List, candidate_units: List,
                       causal_graph=None) -> np.ndarray:
        """
        计算权重矩阵
        
        Parameters:
        -----------
        target_units : list
            目标单元列表
        candidate_units : list
            候选单元列表
        causal_graph : CausalGraph, optional
            因果图
        
        Returns:
        --------
        weights : np.ndarray
            权重矩阵
        """
        # 计算各种距离
        time_dist = self.compute_time_distance(target_units, candidate_units)
        freq_dist = self.compute_freq_distance(target_units, candidate_units)
        microstate_dist = self.compute_microstate_distance(target_units, candidate_units)
        causal_consistency = self.compute_causal_consistency(target_units, candidate_units, causal_graph)
        
        # 归一化距离
        time_dist_norm = self._normalize_distance(time_dist)
        freq_dist_norm = self._normalize_distance(freq_dist)
        microstate_dist_norm = self._normalize_distance(microstate_dist)
        
        # 计算综合距离
        combined_distance = (self.time_weight * time_dist_norm +
                           self.freq_weight * freq_dist_norm +
                           self.microstate_weight * microstate_dist_norm)
        
        # 应用因果一致性
        combined_distance = combined_distance * (2 - causal_consistency)  # 一致性高时距离小
        
        # 应用核函数
        weights = self._apply_kernel(combined_distance)
        
        return weights
    
    def _normalize_distance(self, distances: np.ndarray) -> np.ndarray:
        """
        归一化距离矩阵
        
        Parameters:
        -----------
        distances : np.ndarray
            距离矩阵
        
        Returns:
        --------
        normalized : np.ndarray
            归一化距离矩阵
        """
        max_dist = np.max(distances)
        if max_dist > 0:
            return distances / max_dist
        else:
            return distances
    
    def _apply_kernel(self, distances: np.ndarray) -> np.ndarray:
        """
        应用核函数
        
        Parameters:
        -----------
        distances : np.ndarray
            距离矩阵
        
        Returns:
        --------
        weights : np.ndarray
            权重矩阵
        """
        if self.kernel_type == 'rbf':
            return np.exp(-distances**2 / (2 * self.bandwidth**2))
        elif self.kernel_type == 'exponential':
            return np.exp(-distances / self.bandwidth)
        elif self.kernel_type == 'linear':
            return np.maximum(0, 1 - distances / self.bandwidth)
        elif self.kernel_type == 'polynomial':
            return (1 + distances / self.bandwidth) ** (-2)
        else:
            return np.exp(-distances**2 / (2 * self.bandwidth**2))  # 默认RBF


class SparseRegressor:
    """稀疏回归器类"""
    
    def __init__(self, method: str = 'ridge', alpha: float = 1.0, 
                 l1_ratio: float = 0.5, max_iter: int = 1000,
                 multi_task: bool = False):
        """
        初始化稀疏回归器
        
        Parameters:
        -----------
        method : str
            回归方法 ('ridge', 'lasso', 'elastic_net')
        alpha : float
            正则化强度
        l1_ratio : float
            L1正则化比例（仅用于Elastic Net）
        max_iter : int
            最大迭代次数
        multi_task : bool
            是否使用多任务学习
        """
        self.method = method
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.multi_task = multi_task
        self.regressor = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def _create_regressor(self):
        """
        创建回归器
        
        Returns:
        --------
        regressor : sklearn regressor
            回归器对象
        """
        if self.multi_task:
            if self.method == 'lasso':
                return MultiTaskLasso(alpha=self.alpha, max_iter=self.max_iter)
            elif self.method == 'elastic_net':
                return MultiTaskElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, 
                                         max_iter=self.max_iter)
            else:
                # 多任务Ridge
                return Ridge(alpha=self.alpha, max_iter=self.max_iter)
        else:
            if self.method == 'ridge':
                return Ridge(alpha=self.alpha, max_iter=self.max_iter)
            elif self.method == 'lasso':
                return Lasso(alpha=self.alpha, max_iter=self.max_iter)
            elif self.method == 'elastic_net':
                return ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, 
                                max_iter=self.max_iter)
            else:
                return Ridge(alpha=self.alpha, max_iter=self.max_iter)
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> 'SparseRegressor':
        """
        拟合回归模型
        
        Parameters:
        -----------
        X : np.ndarray
            特征矩阵，形状为 (n_samples, n_features)
        y : np.ndarray
            目标变量，形状为 (n_samples,) 或 (n_samples, n_targets)
        sample_weight : np.ndarray, optional
            样本权重
        
        Returns:
        --------
        self : SparseRegressor
        """
        # 数据有效性检查
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("警告: 特征矩阵包含NaN或Inf值，进行清理")
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print("警告: 目标变量包含NaN或Inf值，进行清理")
            y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 检查数据方差
        X_std = np.std(X, axis=0)
        low_var_features = np.where(X_std < 1e-8)[0]
        if len(low_var_features) > 0:
            print(f"警告: {len(low_var_features)}个特征方差过小，添加噪声")
            X[:, low_var_features] += np.random.normal(0, 1e-6, (X.shape[0], len(low_var_features)))
        
        try:
            # 标准化特征
            X_scaled = self.scaler_X.fit_transform(X)
            
            # 标准化目标变量
            if y.ndim == 1:
                y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
            else:
                y_scaled = self.scaler_y.fit_transform(y)
            
            # 创建并拟合回归器
            self.regressor = self._create_regressor()
            
            if sample_weight is not None:
                # 权重有效性检查
                if np.any(np.isnan(sample_weight)) or np.any(np.isinf(sample_weight)):
                    print("警告: 样本权重包含无效值，使用均匀权重")
                    sample_weight = np.ones(len(sample_weight)) / len(sample_weight)
                self.regressor.fit(X_scaled, y_scaled, sample_weight=sample_weight)
            else:
                self.regressor.fit(X_scaled, y_scaled)
                
        except Exception as e:
            print(f"回归拟合失败: {e}，使用备用方法")
            # 使用更稳定的Ridge回归
            from sklearn.linear_model import Ridge
            self.regressor = Ridge(alpha=1.0, fit_intercept=True)
            if sample_weight is not None:
                self.regressor.fit(X_scaled, y_scaled, sample_weight=sample_weight)
            else:
                self.regressor.fit(X_scaled, y_scaled)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Parameters:
        -----------
        X : np.ndarray
            特征矩阵
        
        Returns:
        --------
        predictions : np.ndarray
            预测结果
        """
        if self.regressor is None:
            raise ValueError("模型尚未拟合")
        
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.regressor.predict(X_scaled)
        
        # 反标准化
        if y_pred_scaled.ndim == 1:
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        else:
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred
    
    def get_coefficients(self) -> np.ndarray:
        """
        获取回归系数
        
        Returns:
        --------
        coefficients : np.ndarray
            回归系数
        """
        if self.regressor is None:
            raise ValueError("模型尚未拟合")
        
        return self.regressor.coef_
    
    def get_feature_importance(self) -> np.ndarray:
        """
        获取特征重要性
        
        Returns:
        --------
        importance : np.ndarray
            特征重要性
        """
        coefficients = self.get_coefficients()
        
        if coefficients.ndim == 1:
            return np.abs(coefficients)
        else:
            return np.mean(np.abs(coefficients), axis=0)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算R²分数
        
        Parameters:
        -----------
        X : np.ndarray
            特征矩阵
        y : np.ndarray
            真实目标变量
        
        Returns:
        --------
        score : float
            R²分数
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


class DPPSelector:
    """确定点过程（DPP）特征选择器"""
    
    def __init__(self, diversity_weight: float = 1.0, quality_weight: float = 1.0):
        """
        初始化DPP选择器
        
        Parameters:
        -----------
        diversity_weight : float
            多样性权重
        quality_weight : float
            质量权重
        """
        self.diversity_weight = diversity_weight
        self.quality_weight = quality_weight
    
    def select_features(self, feature_matrix: np.ndarray, quality_scores: np.ndarray,
                       k: int, max_iter: int = 100) -> List[int]:
        """
        使用DPP选择特征
        
        Parameters:
        -----------
        feature_matrix : np.ndarray
            特征矩阵，形状为 (n_samples, n_features)
        quality_scores : np.ndarray
            质量分数，形状为 (n_features,)
        k : int
            选择的特征数量
        max_iter : int
            最大迭代次数
        
        Returns:
        --------
        selected_indices : list
            选择的特征索引
        """
        n_features = feature_matrix.shape[1]
        
        # 计算相似性矩阵
        similarity_matrix = self._compute_similarity_matrix(feature_matrix)
        
        # 构建DPP核矩阵
        kernel_matrix = self._build_dpp_kernel(similarity_matrix, quality_scores)
        
        # 使用贪心算法近似DPP采样
        selected_indices = self._greedy_dpp_sampling(kernel_matrix, k, max_iter)
        
        return selected_indices
    
    def _compute_similarity_matrix(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        计算特征相似性矩阵
        
        Parameters:
        -----------
        feature_matrix : np.ndarray
            特征矩阵
        
        Returns:
        --------
        similarity_matrix : np.ndarray
            相似性矩阵
        """
        # 确保特征矩阵至少是2D
        if feature_matrix.ndim == 1:
            feature_matrix = feature_matrix.reshape(1, -1)
        elif feature_matrix.ndim == 0:
            # 如果是标量，创建1x1矩阵
            return np.array([[1.0]])
        
        # 如果只有一个特征，返回单位矩阵
        if feature_matrix.shape[1] == 1:
            return np.array([[1.0]])
        
        try:
            # 计算特征间的相关系数
            correlation_matrix = np.corrcoef(feature_matrix.T)
            
            # 如果结果是标量（只有一个特征），转换为矩阵
            if correlation_matrix.ndim == 0:
                correlation_matrix = np.array([[correlation_matrix]])
            
            # 处理NaN值
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
            
            # 转换为相似性（取绝对值）
            similarity_matrix = np.abs(correlation_matrix)
            
        except Exception as e:
            print(f"计算相似性矩阵失败: {e}，使用单位矩阵")
            n_features = feature_matrix.shape[1]
            similarity_matrix = np.eye(n_features)
        
        return similarity_matrix
    
    def _build_dpp_kernel(self, similarity_matrix: np.ndarray, 
                         quality_scores: np.ndarray) -> np.ndarray:
        """
        构建DPP核矩阵
        
        Parameters:
        -----------
        similarity_matrix : np.ndarray
            相似性矩阵
        quality_scores : np.ndarray
            质量分数
        
        Returns:
        --------
        kernel_matrix : np.ndarray
            DPP核矩阵
        """
        n_features = len(quality_scores)
        
        # 归一化质量分数
        quality_normalized = (quality_scores - np.min(quality_scores)) / \
                           (np.max(quality_scores) - np.min(quality_scores) + 1e-10)
        
        # 构建质量矩阵
        quality_matrix = np.diag(quality_normalized)
        
        # 构建多样性矩阵（反相似性）
        diversity_matrix = 1 - similarity_matrix
        np.fill_diagonal(diversity_matrix, 1)  # 对角线设为1
        
        # 组合质量和多样性
        kernel_matrix = (self.quality_weight * quality_matrix + 
                        self.diversity_weight * diversity_matrix)
        
        # 确保矩阵为正定
        eigenvals, eigenvecs = np.linalg.eigh(kernel_matrix)
        eigenvals = np.maximum(eigenvals, 1e-10)  # 确保正定
        kernel_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return kernel_matrix
    
    def _greedy_dpp_sampling(self, kernel_matrix: np.ndarray, k: int, 
                           max_iter: int) -> List[int]:
        """
        贪心DPP采样
        
        Parameters:
        -----------
        kernel_matrix : np.ndarray
            DPP核矩阵
        k : int
            选择数量
        max_iter : int
            最大迭代次数
        
        Returns:
        --------
        selected_indices : list
            选择的索引
        """
        n_features = kernel_matrix.shape[0]
        selected_indices = []
        remaining_indices = list(range(n_features))
        
        for _ in range(min(k, n_features)):
            if not remaining_indices:
                break
            
            best_score = -np.inf
            best_idx = None
            
            for idx in remaining_indices:
                # 计算添加该特征后的DPP分数
                temp_selected = selected_indices + [idx]
                score = self._compute_dpp_score(kernel_matrix, temp_selected)
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        return selected_indices
    
    def _compute_dpp_score(self, kernel_matrix: np.ndarray, 
                          indices: List[int]) -> float:
        """
        计算DPP分数
        
        Parameters:
        -----------
        kernel_matrix : np.ndarray
            核矩阵
        indices : list
            特征索引
        
        Returns:
        --------
        score : float
            DPP分数
        """
        if not indices:
            return 0.0
        
        # 提取子矩阵
        sub_matrix = kernel_matrix[np.ix_(indices, indices)]
        
        # 计算行列式（多样性度量）
        try:
            score = det(sub_matrix)
            return max(0, score)  # 确保非负
        except:
            return 0.0


class SubmodularSelector:
    """子模块优化选择器"""
    
    def __init__(self, diversity_weight: float = 1.0):
        """
        初始化子模块选择器
        
        Parameters:
        -----------
        diversity_weight : float
            多样性权重
        """
        self.diversity_weight = diversity_weight
    
    def select_features(self, feature_matrix: np.ndarray, quality_scores: np.ndarray,
                       k: int) -> List[int]:
        """
        使用子模块优化选择特征
        
        Parameters:
        -----------
        feature_matrix : np.ndarray
            特征矩阵
        quality_scores : np.ndarray
            质量分数
        k : int
            选择数量
        
        Returns:
        --------
        selected_indices : list
            选择的特征索引
        """
        n_features = feature_matrix.shape[1]
        selected_indices = []
        remaining_indices = list(range(n_features))
        
        # 计算相似性矩阵
        similarity_matrix = np.corrcoef(feature_matrix.T)
        similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)
        
        for _ in range(min(k, n_features)):
            if not remaining_indices:
                break
            
            best_gain = -np.inf
            best_idx = None
            
            for idx in remaining_indices:
                # 计算边际增益
                gain = self._compute_marginal_gain(idx, selected_indices, 
                                                 quality_scores, similarity_matrix)
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        return selected_indices
    
    def _compute_marginal_gain(self, new_idx: int, selected_indices: List[int],
                              quality_scores: np.ndarray, 
                              similarity_matrix: np.ndarray) -> float:
        """
        计算边际增益
        
        Parameters:
        -----------
        new_idx : int
            新特征索引
        selected_indices : list
            已选择的特征索引
        quality_scores : np.ndarray
            质量分数
        similarity_matrix : np.ndarray
            相似性矩阵
        
        Returns:
        --------
        gain : float
            边际增益
        """
        # 质量增益
        quality_gain = quality_scores[new_idx]
        
        # 多样性增益（与已选择特征的最大相似性的负值）
        if selected_indices:
            max_similarity = max(abs(similarity_matrix[new_idx, idx]) 
                               for idx in selected_indices)
            diversity_gain = -self.diversity_weight * max_similarity
        else:
            diversity_gain = 0
        
        return quality_gain + diversity_gain


class LocalRegressionSelector:
    """局部回归选择器主类"""
    
    def __init__(self, weight_kernel: WeightKernel = None,
                 regressor: SparseRegressor = None,
                 selector_type: str = 'dpp',
                 max_features: int = 10):
        """
        初始化局部回归选择器
        
        Parameters:
        -----------
        weight_kernel : WeightKernel, optional
            权重核函数
        regressor : SparseRegressor, optional
            稀疏回归器
        selector_type : str
            选择器类型 ('dpp', 'submodular', 'greedy')
        max_features : int
            最大特征数量
        """
        self.weight_kernel = weight_kernel or WeightKernel()
        self.regressor = regressor or SparseRegressor()
        self.selector_type = selector_type
        self.max_features = max_features
        
        if selector_type == 'dpp':
            self.selector = DPPSelector()
        elif selector_type == 'submodular':
            self.selector = SubmodularSelector()
        else:
            self.selector = None
        
        self.selected_features = []
        self.feature_importance = None
        self.explanation_quality = {}
    
    def fit_and_select(self, X: np.ndarray, y: np.ndarray, 
                      target_units: List, candidate_units: List,
                      causal_graph=None) -> 'LocalRegressionSelector':
        """
        拟合模型并选择特征
        
        Parameters:
        -----------
        X : np.ndarray
            特征矩阵
        y : np.ndarray
            目标变量
        target_units : list
            目标单元
        candidate_units : list
            候选单元
        causal_graph : CausalGraph, optional
            因果图
        
        Returns:
        --------
        self : LocalRegressionSelector
        """
        # 计算样本权重
        weights = self.weight_kernel.compute_weights(target_units, candidate_units, causal_graph)
        sample_weights = np.mean(weights, axis=0)  # 平均权重作为样本权重
        
        # 数值稳定性检查
        if np.any(np.isnan(sample_weights)) or np.any(np.isinf(sample_weights)):
            print("警告: 样本权重包含NaN或Inf值，使用均匀权重")
            sample_weights = np.ones(len(sample_weights)) / len(sample_weights)
        
        # 确保权重非负且归一化
        sample_weights = np.abs(sample_weights)
        if np.sum(sample_weights) == 0:
            sample_weights = np.ones(len(sample_weights)) / len(sample_weights)
        else:
            sample_weights = sample_weights / np.sum(sample_weights)
        
        # 数据预处理检查
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("警告: 特征矩阵包含NaN或Inf值")
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print("警告: 目标变量包含NaN或Inf值")
            y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 拟合回归模型
        try:
            self.regressor.fit(X, y, sample_weights)
        except Exception as e:
            print(f"回归拟合失败: {e}，尝试使用默认参数")
            # 使用更稳定的回归参数
            backup_regressor = SparseRegressor(method='ridge', alpha=1.0)
            backup_regressor.fit(X, y, sample_weights)
            self.regressor = backup_regressor
        
        # 获取特征重要性
        try:
            self.feature_importance = self.regressor.get_feature_importance()
        except Exception as e:
            print(f"获取特征重要性失败: {e}，使用随机重要性")
            self.feature_importance = np.random.rand(X.shape[1]) * 0.1
        
        # 特征重要性数值稳定性检查
        if np.any(np.isnan(self.feature_importance)) or np.any(np.isinf(self.feature_importance)):
            print("警告: 特征重要性包含NaN或Inf值，使用随机重要性")
            self.feature_importance = np.random.rand(len(self.feature_importance)) * 0.1
        
        # 如果所有重要性都为0，添加小的随机扰动
        if np.all(np.abs(self.feature_importance) < 1e-10):
            print("警告: 所有特征重要性接近0，添加随机扰动")
            self.feature_importance += np.random.rand(len(self.feature_importance)) * 1e-3
        
        # 确保重要性非负
        self.feature_importance = np.abs(self.feature_importance)
        
        # 特征选择
        try:
            if self.selector is not None:
                self.selected_features = self.selector.select_features(
                    X, self.feature_importance, self.max_features
                )
            else:
                # 贪心选择：按重要性排序
                sorted_indices = np.argsort(self.feature_importance)[::-1]
                self.selected_features = sorted_indices[:self.max_features].tolist()
        except Exception as e:
            print(f"特征选择失败: {e}，使用前{self.max_features}个特征")
            self.selected_features = list(range(min(self.max_features, X.shape[1])))
        
        # 评估解释质量
        self.explanation_quality = self._evaluate_explanation_quality(X, y, sample_weights)
        
        return self
    
    def _evaluate_explanation_quality(self, X: np.ndarray, y: np.ndarray,
                                     sample_weights: np.ndarray) -> Dict[str, float]:
        """
        评估解释质量
        
        Parameters:
        -----------
        X : np.ndarray
            特征矩阵
        y : np.ndarray
            目标变量
        sample_weights : np.ndarray
            样本权重
        
        Returns:
        --------
        quality_metrics : dict
            质量指标字典
        """
        metrics = {}
        
        # 1. 拟合质量
        y_pred = self.regressor.predict(X)
        metrics['r2_score'] = r2_score(y, y_pred, sample_weight=sample_weights)
        metrics['mse'] = mean_squared_error(y, y_pred, sample_weight=sample_weights)
        
        # 2. 稀疏性
        non_zero_features = np.sum(self.feature_importance > 1e-6)
        metrics['sparsity'] = 1 - non_zero_features / len(self.feature_importance)
        
        # 3. 稳定性（使用交叉验证）
        try:
            cv_scores = cross_val_score(self.regressor.regressor, X, y, cv=3, 
                                      scoring='r2')
            metrics['stability'] = np.std(cv_scores)
        except:
            metrics['stability'] = 0.0
        
        # 4. 多样性（选择特征间的平均相关性）
        if len(self.selected_features) > 1:
            selected_X = X[:, self.selected_features]
            corr_matrix = np.corrcoef(selected_X.T)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            
            # 计算上三角矩阵的平均相关性
            upper_triangle = np.triu(corr_matrix, k=1)
            n_pairs = len(self.selected_features) * (len(self.selected_features) - 1) // 2
            if n_pairs > 0:
                avg_correlation = np.sum(np.abs(upper_triangle)) / n_pairs
                metrics['diversity'] = 1 - avg_correlation
            else:
                metrics['diversity'] = 1.0
        else:
            metrics['diversity'] = 1.0
        
        # 5. 综合质量分数
        metrics['overall_quality'] = (
            0.4 * metrics['r2_score'] +
            0.2 * metrics['sparsity'] +
            0.2 * (1 - metrics['stability']) +
            0.2 * metrics['diversity']
        )
        
        return metrics
    
    def get_explanation(self) -> Dict:
        """
        获取解释结果
        
        Returns:
        --------
        explanation : dict
            解释结果字典
        """
        if not self.selected_features:
            return {}
        
        explanation = {
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance[self.selected_features],
            'coefficients': self.regressor.get_coefficients(),
            'quality_metrics': self.explanation_quality,
            'n_selected_features': len(self.selected_features)
        }
        
        return explanation
    
    def predict_with_explanation(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        预测并提供解释
        
        Parameters:
        -----------
        X : np.ndarray
            特征矩阵
        
        Returns:
        --------
        predictions : np.ndarray
            预测结果
        explanation : dict
            解释信息
        """
        predictions = self.regressor.predict(X)
        explanation = self.get_explanation()
        
        # 添加特征贡献
        if len(self.selected_features) > 0:
            selected_X = X[:, self.selected_features]
            coefficients = self.regressor.get_coefficients()
            
            if coefficients.ndim == 1:
                feature_contributions = selected_X * coefficients[self.selected_features]
            else:
                feature_contributions = selected_X * coefficients.mean(axis=0)[self.selected_features]
            
            explanation['feature_contributions'] = feature_contributions
        
        return predictions, explanation


def create_local_regressor(kernel_params: Dict = None, regressor_params: Dict = None,
                          selector_type: str = 'dpp', max_features: int = 10) -> LocalRegressionSelector:
    """
    创建局部回归选择器的便捷函数
    
    Parameters:
    -----------
    kernel_params : dict, optional
        核函数参数
    regressor_params : dict, optional
        回归器参数
    selector_type : str
        选择器类型
    max_features : int
        最大特征数
    
    Returns:
    --------
    selector : LocalRegressionSelector
        局部回归选择器
    """
    kernel_params = kernel_params or {}
    regressor_params = regressor_params or {}
    
    weight_kernel = WeightKernel(**kernel_params)
    regressor = SparseRegressor(**regressor_params)
    
    return LocalRegressionSelector(
        weight_kernel=weight_kernel,
        regressor=regressor,
        selector_type=selector_type,
        max_features=max_features
    )


if __name__ == "__main__":
    # 示例使用
    print("局部回归与选择模块已加载")
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 200
    n_features = 50
    
    # 生成特征矩阵
    X = np.random.randn(n_samples, n_features)
    
    # 生成目标变量（只有前10个特征有用）
    true_coefficients = np.zeros(n_features)
    true_coefficients[:10] = np.random.randn(10)
    y = X @ true_coefficients + 0.1 * np.random.randn(n_samples)
    
    # 创建模拟单元
    class MockUnit:
        def __init__(self, channels, time_range, freq_band):
            self.channels = channels
            self.time_range = time_range
            self.freq_band = freq_band
    
    target_units = [MockUnit([0], (0, 100), (8, 13))]
    candidate_units = [MockUnit([i], (i*10, (i+1)*10), (8, 13)) for i in range(n_samples)]
    
    # 创建局部回归选择器
    selector = create_local_regressor(
        regressor_params={'method': 'lasso', 'alpha': 0.1},
        selector_type='dpp',
        max_features=15
    )
    
    # 拟合和选择
    selector.fit_and_select(X, y, target_units, candidate_units)
    
    # 获取解释
    explanation = selector.get_explanation()
    print(f"选择的特征数量: {explanation['n_selected_features']}")
    print(f"选择的特征索引: {explanation['selected_features'][:10]}")
    print(f"质量指标: {explanation['quality_metrics']}")
    
    # 预测
    X_test = np.random.randn(10, n_features)
    predictions, test_explanation = selector.predict_with_explanation(X_test)
    print(f"测试预测形状: {predictions.shape}")