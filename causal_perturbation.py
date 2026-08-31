#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因果一致性扰动模块

实现EEG信号的因果分析和一致性扰动，包括：
- Granger因果分析
- PCMCI因果发现
- NOTEARS结构学习
- 因果图构建和分析
- 因果一致性扰动生成
- 扰动质量评估

Author: CMS-LIME Framework
Date: 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Set
from scipy import stats, signal
from scipy.linalg import pinv
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import networkx as nx
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


class CausalGraph:
    """因果图类"""
    
    def __init__(self, n_nodes: int, node_names: List[str] = None):
        """
        初始化因果图
        
        Parameters:
        -----------
        n_nodes : int
            节点数量
        node_names : list, optional
            节点名称列表
        """
        self.n_nodes = n_nodes
        self.node_names = node_names or [f"Node_{i}" for i in range(n_nodes)]
        self.adjacency_matrix = np.zeros((n_nodes, n_nodes))
        self.lag_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
        self.strength_matrix = np.zeros((n_nodes, n_nodes))
        self.graph = nx.DiGraph()
        
        # 添加节点
        for i, name in enumerate(self.node_names):
            self.graph.add_node(i, name=name)
    
    def add_edge(self, source: int, target: int, strength: float = 1.0, lag: int = 1):
        """
        添加因果边
        
        Parameters:
        -----------
        source : int
            源节点
        target : int
            目标节点
        strength : float
            因果强度
        lag : int
            时间滞后
        """
        self.adjacency_matrix[source, target] = 1
        self.strength_matrix[source, target] = strength
        self.lag_matrix[source, target] = lag
        
        self.graph.add_edge(source, target, weight=strength, lag=lag)
    
    def remove_edge(self, source: int, target: int):
        """
        移除因果边
        
        Parameters:
        -----------
        source : int
            源节点
        target : int
            目标节点
        """
        self.adjacency_matrix[source, target] = 0
        self.strength_matrix[source, target] = 0
        self.lag_matrix[source, target] = 0
        
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)
    
    def get_parents(self, node: int, max_lag: int = 5) -> List[Tuple[int, int]]:
        """
        获取节点的父节点
        
        Parameters:
        -----------
        node : int
            目标节点
        max_lag : int
            最大滞后
        
        Returns:
        --------
        parents : list
            父节点列表 [(parent_node, lag), ...]
        """
        parents = []
        for parent in range(self.n_nodes):
            if self.adjacency_matrix[parent, node] > 0:
                lag = int(self.lag_matrix[parent, node])
                if lag <= max_lag:
                    parents.append((parent, lag))
        return parents
    
    def get_children(self, node: int, max_lag: int = 5) -> List[Tuple[int, int]]:
        """
        获取节点的子节点
        
        Parameters:
        -----------
        node : int
            源节点
        max_lag : int
            最大滞后
        
        Returns:
        --------
        children : list
            子节点列表 [(child_node, lag), ...]
        """
        children = []
        for child in range(self.n_nodes):
            if self.adjacency_matrix[node, child] > 0:
                lag = int(self.lag_matrix[node, child])
                if lag <= max_lag:
                    children.append((child, lag))
        return children
    
    def get_causal_closure(self, nodes: List[int], max_lag: int = 5) -> Set[Tuple[int, int]]:
        """
        获取节点集合的因果闭包
        
        Parameters:
        -----------
        nodes : list
            节点列表
        max_lag : int
            最大滞后
        
        Returns:
        --------
        closure : set
            因果闭包 {(node, lag_offset), ...}
        """
        closure = set()
        
        # 添加原始节点
        for node in nodes:
            closure.add((node, 0))
        
        # 添加父节点和子节点
        for node in nodes:
            # 父节点
            parents = self.get_parents(node, max_lag)
            for parent, lag in parents:
                closure.add((parent, -lag))  # 负滞后表示过去
            
            # 子节点
            children = self.get_children(node, max_lag)
            for child, lag in children:
                closure.add((child, lag))  # 正滞后表示未来
        
        return closure
    
    def is_acyclic(self) -> bool:
        """
        检查图是否为无环图
        
        Returns:
        --------
        is_acyclic : bool
            是否无环
        """
        return nx.is_directed_acyclic_graph(self.graph)
    
    def get_topological_order(self) -> List[int]:
        """
        获取拓扑排序
        
        Returns:
        --------
        order : list
            拓扑排序
        """
        if self.is_acyclic():
            return list(nx.topological_sort(self.graph))
        else:
            return list(range(self.n_nodes))


class GrangerCausalityAnalyzer:
    """Granger因果分析器"""
    
    def __init__(self, max_lag: int = 5, significance_level: float = 0.05):
        """
        初始化Granger因果分析器
        
        Parameters:
        -----------
        max_lag : int
            最大滞后阶数
        significance_level : float
            显著性水平
        """
        self.max_lag = max_lag
        self.significance_level = significance_level
        self.causal_matrix = None
        self.pvalue_matrix = None
    
    def fit(self, data: np.ndarray) -> 'GrangerCausalityAnalyzer':
        """
        拟合Granger因果模型
        
        Parameters:
        -----------
        data : np.ndarray
            时间序列数据，形状为 (n_timepoints, n_variables)
        
        Returns:
        --------
        self : GrangerCausalityAnalyzer
        """
        if data.ndim == 3:
            # 多个样本，取平均
            data = data.mean(axis=0).T
        elif data.ndim == 2 and data.shape[0] < data.shape[1]:
            # 转置为 (n_timepoints, n_variables)
            data = data.T
        
        n_timepoints, n_variables = data.shape
        
        self.causal_matrix = np.zeros((n_variables, n_variables))
        self.pvalue_matrix = np.ones((n_variables, n_variables))
        
        # 标准化数据
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # 对每对变量进行Granger因果检验
        for i in range(n_variables):
            for j in range(n_variables):
                if i != j:
                    causality, pvalue = self._granger_test(data_scaled[:, i], data_scaled[:, j])
                    self.causal_matrix[i, j] = causality
                    self.pvalue_matrix[i, j] = pvalue
        
        return self
    
    def _granger_test(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        执行Granger因果检验
        
        Parameters:
        -----------
        x : np.ndarray
            原因变量
        y : np.ndarray
            结果变量
        
        Returns:
        --------
        causality : float
            因果强度
        pvalue : float
            p值
        """
        n = len(y)
        
        # 构建滞后矩阵
        def build_lag_matrix(series, max_lag):
            n = len(series)
            X = np.zeros((n - max_lag, max_lag))
            for lag in range(1, max_lag + 1):
                X[:, lag - 1] = series[max_lag - lag:-lag]
            return X
        
        # 受限模型：只用y的滞后项预测y
        y_lags = build_lag_matrix(y, self.max_lag)
        y_target = y[self.max_lag:]
        
        try:
            # 拟合受限模型
            reg_restricted = LinearRegression().fit(y_lags, y_target)
            y_pred_restricted = reg_restricted.predict(y_lags)
            rss_restricted = np.sum((y_target - y_pred_restricted) ** 2)
            
            # 非受限模型：用x和y的滞后项预测y
            x_lags = build_lag_matrix(x, self.max_lag)
            X_full = np.hstack([y_lags, x_lags])
            
            reg_full = LinearRegression().fit(X_full, y_target)
            y_pred_full = reg_full.predict(X_full)
            rss_full = np.sum((y_target - y_pred_full) ** 2)
            
            # F统计量
            n_obs = len(y_target)
            df1 = self.max_lag  # x的滞后项数量
            df2 = n_obs - 2 * self.max_lag  # 残差自由度
            
            if df2 > 0 and rss_full < rss_restricted:
                f_stat = ((rss_restricted - rss_full) / df1) / (rss_full / df2)
                pvalue = 1 - stats.f.cdf(f_stat, df1, df2)
                causality = f_stat if pvalue < self.significance_level else 0
            else:
                causality = 0
                pvalue = 1.0
            
        except Exception:
            causality = 0
            pvalue = 1.0
        
        return causality, pvalue
    
    def get_causal_graph(self, node_names: List[str] = None) -> CausalGraph:
        """
        获取因果图
        
        Parameters:
        -----------
        node_names : list, optional
            节点名称
        
        Returns:
        --------
        graph : CausalGraph
            因果图
        """
        if self.causal_matrix is None:
            raise ValueError("模型尚未拟合")
        
        n_nodes = self.causal_matrix.shape[0]
        graph = CausalGraph(n_nodes, node_names)
        
        # 添加显著的因果边
        for i in range(n_nodes):
            for j in range(n_nodes):
                if (self.causal_matrix[i, j] > 0 and 
                    self.pvalue_matrix[i, j] < self.significance_level):
                    graph.add_edge(i, j, self.causal_matrix[i, j], lag=1)
        
        return graph


class NOTEARSAnalyzer:
    """NOTEARS结构学习分析器"""
    
    def __init__(self, lambda_reg: float = 0.1, max_iter: int = 100, 
                 tolerance: float = 1e-6):
        """
        初始化NOTEARS分析器
        
        Parameters:
        -----------
        lambda_reg : float
            正则化参数
        max_iter : int
            最大迭代次数
        tolerance : float
            收敛容差
        """
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.adjacency_matrix = None
    
    def fit(self, data: np.ndarray) -> 'NOTEARSAnalyzer':
        """
        拟合NOTEARS模型
        
        Parameters:
        -----------
        data : np.ndarray
            数据矩阵，形状为 (n_samples, n_variables)
        
        Returns:
        --------
        self : NOTEARSAnalyzer
        """
        if data.ndim == 3:
            # 多个样本，重塑为 (n_samples * n_timepoints, n_variables)
            n_samples, n_variables, n_timepoints = data.shape
            data = data.transpose(0, 2, 1).reshape(-1, n_variables)
        elif data.ndim == 2 and data.shape[0] < data.shape[1]:
            data = data.T
        
        # 标准化数据
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # 简化的NOTEARS算法实现
        self.adjacency_matrix = self._notears_linear(data_scaled)
        
        return self
    
    def _notears_linear(self, X: np.ndarray) -> np.ndarray:
        """
        NOTEARS线性模型
        
        Parameters:
        -----------
        X : np.ndarray
            数据矩阵
        
        Returns:
        --------
        W : np.ndarray
            邻接矩阵
        """
        n, d = X.shape
        
        # 初始化权重矩阵
        W = np.random.uniform(-0.1, 0.1, (d, d))
        np.fill_diagonal(W, 0)
        
        # 计算样本协方差矩阵
        S = np.cov(X.T)
        
        for iteration in range(self.max_iter):
            W_old = W.copy()
            
            # 更新权重矩阵
            for i in range(d):
                for j in range(d):
                    if i != j:
                        # 计算梯度
                        grad = self._compute_gradient(W, S, i, j)
                        
                        # 梯度下降更新
                        W[i, j] -= 0.01 * grad
                        
                        # L1正则化（软阈值）
                        W[i, j] = self._soft_threshold(W[i, j], self.lambda_reg * 0.01)
            
            # 检查收敛
            if np.linalg.norm(W - W_old) < self.tolerance:
                break
        
        # 阈值化得到二值邻接矩阵
        threshold = np.std(W) * 0.5
        adjacency = (np.abs(W) > threshold).astype(float)
        
        return adjacency
    
    def _compute_gradient(self, W: np.ndarray, S: np.ndarray, i: int, j: int) -> float:
        """
        计算梯度
        
        Parameters:
        -----------
        W : np.ndarray
            权重矩阵
        S : np.ndarray
            协方差矩阵
        i, j : int
            位置索引
        
        Returns:
        --------
        grad : float
            梯度值
        """
        d = W.shape[0]
        I = np.eye(d)
        
        try:
            # 计算 (I - W)^{-1}
            IW_inv = np.linalg.inv(I - W)
            
            # 计算梯度
            grad = -2 * S[i, j] + 2 * np.sum(W[i, :] * S[:, j])
            
            # 添加无环约束的梯度
            acyclicity_grad = 2 * np.trace(IW_inv @ IW_inv.T) * IW_inv[j, i]
            grad += acyclicity_grad
            
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，使用简化梯度
            grad = -2 * S[i, j] + 2 * np.sum(W[i, :] * S[:, j])
        
        return grad
    
    def _soft_threshold(self, x: float, threshold: float) -> float:
        """
        软阈值函数
        
        Parameters:
        -----------
        x : float
            输入值
        threshold : float
            阈值
        
        Returns:
        --------
        result : float
            阈值化结果
        """
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0.0
    
    def get_causal_graph(self, node_names: List[str] = None) -> CausalGraph:
        """
        获取因果图
        
        Parameters:
        -----------
        node_names : list, optional
            节点名称
        
        Returns:
        --------
        graph : CausalGraph
            因果图
        """
        if self.adjacency_matrix is None:
            raise ValueError("模型尚未拟合")
        
        n_nodes = self.adjacency_matrix.shape[0]
        graph = CausalGraph(n_nodes, node_names)
        
        # 添加因果边
        for i in range(n_nodes):
            for j in range(n_nodes):
                if self.adjacency_matrix[i, j] > 0:
                    graph.add_edge(i, j, self.adjacency_matrix[i, j], lag=1)
        
        return graph


class CausalPerturbationEngine:
    """因果一致性扰动引擎"""
    
    def __init__(self, causal_graph: CausalGraph, perturbation_strength: float = 0.1,
                 preserve_statistics: bool = True):
        """
        初始化扰动引擎
        
        Parameters:
        -----------
        causal_graph : CausalGraph
            因果图
        perturbation_strength : float
            扰动强度
        preserve_statistics : bool
            是否保持统计特性
        """
        self.causal_graph = causal_graph
        self.perturbation_strength = perturbation_strength
        self.preserve_statistics = preserve_statistics
        self.original_stats = {}
    
    def compute_original_statistics(self, data: np.ndarray):
        """
        计算原始数据的统计特性
        
        Parameters:
        -----------
        data : np.ndarray
            原始数据
        """
        if data.ndim == 3:
            data = data[0]  # 取第一个样本
        
        if data.shape[0] < data.shape[1]:
            data = data.T  # 转置为 (n_timepoints, n_channels)
        
        self.original_stats = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'power_spectrum': self._compute_power_spectrum(data),
            'cross_correlation': np.corrcoef(data.T)
        }
    
    def _compute_power_spectrum(self, data: np.ndarray) -> np.ndarray:
        """
        计算功率谱
        
        Parameters:
        -----------
        data : np.ndarray
            时间序列数据
        
        Returns:
        --------
        power_spectrum : np.ndarray
            功率谱
        """
        n_channels = data.shape[1]
        power_spectra = []
        
        for ch in range(n_channels):
            freqs, psd = signal.welch(data[:, ch], nperseg=min(256, len(data)//4))
            power_spectra.append(psd)
        
        return np.array(power_spectra)
    
    def perturb_units(self, data: np.ndarray, units: List, 
                     perturbation_mask: np.ndarray) -> np.ndarray:
        """
        对指定单元进行因果一致性扰动
        
        Parameters:
        -----------
        data : np.ndarray
            原始数据
        units : list
            要扰动的单元列表
        perturbation_mask : np.ndarray
            扰动掩码
        
        Returns:
        --------
        perturbed_data : np.ndarray
            扰动后的数据
        """
        if data.ndim == 3:
            data = data[0]  # 取第一个样本
        
        if data.shape[0] < data.shape[1]:
            data = data.T  # 转置为 (n_timepoints, n_channels)
        
        perturbed_data = data.copy()
        
        # 计算原始统计特性
        if not self.original_stats:
            self.compute_original_statistics(data)
        
        # 对每个要扰动的单元
        for i, unit in enumerate(units):
            if perturbation_mask[i] > 0:
                perturbed_data = self._perturb_unit(perturbed_data, unit, perturbation_mask[i])
        
        # 如果需要保持统计特性，进行后处理
        if self.preserve_statistics:
            perturbed_data = self._preserve_statistics(perturbed_data)
        
        # 确保返回格式为 (n_channels, n_timepoints)
        # 如果第一维度很大且第二维度较小（<=64），可能是(n_timepoints, n_channels)格式
        if perturbed_data.shape[0] > 100 and perturbed_data.shape[1] <= 64:
            perturbed_data = perturbed_data.T
        
        return perturbed_data
    
    def _perturb_unit(self, data: np.ndarray, unit, strength: float) -> np.ndarray:
        """
        扰动单个单元
        
        Parameters:
        -----------
        data : np.ndarray
            数据
        unit : object
            要扰动的单元
        strength : float
            扰动强度
        
        Returns:
        --------
        perturbed_data : np.ndarray
            扰动后的数据
        """
        # 确保输入数据是numpy数组
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        perturbed_data = data.copy()
        
        # 确保perturbed_data是numpy数组且数据类型正确
        if not isinstance(perturbed_data, np.ndarray):
            perturbed_data = np.array(perturbed_data)
        
        # 确保数据类型是float
        if perturbed_data.dtype not in [np.float32, np.float64]:
            perturbed_data = perturbed_data.astype(np.float64)
        
        # 调试信息已移除
        
        # 获取单元的通道和时间范围
        if hasattr(unit, 'channels') and hasattr(unit, 'time_range'):
            channels = unit.channels
            time_start, time_end = unit.time_range
        elif isinstance(unit, dict):
            channels = unit.get('channels', [0])
            time_range = unit.get('time_range', (0, data.shape[0]))
            time_start, time_end = time_range
        else:
            # 默认处理
            channels = [0]
            time_start, time_end = 0, data.shape[0]
        
        # 确保channels是整数列表（处理字符串类型的channels）
        if channels:
            try:
                # 处理各种可能的channels格式
                processed_channels = []
                for ch in channels:
                    if isinstance(ch, (str, np.str_)):
                        # 尝试转换字符串为整数
                        try:
                            processed_channels.append(int(ch))
                        except ValueError:
                            # 如果转换失败，跳过这个通道
                            continue
                    elif isinstance(ch, (int, np.integer)):
                        processed_channels.append(int(ch))
                    elif isinstance(ch, (float, np.floating)):
                        # 浮点数转整数
                        processed_channels.append(int(ch))
                    else:
                        # 其他类型，尝试转换
                        try:
                            processed_channels.append(int(ch))
                        except (ValueError, TypeError):
                            continue
                channels = processed_channels if processed_channels else [0]
            except (ValueError, TypeError):
                channels = [0]  # 默认使用第一个通道
        else:
            channels = [0]  # 空列表时使用默认通道
        
        # 检测数据格式并确保索引在有效范围内
        # 如果第一维度很大且第二维度较小（<=64），可能是(n_timepoints, n_channels)格式
        if data.shape[0] > 100 and data.shape[1] <= 64:
            # 数据格式为 (n_timepoints, n_channels)
            time_start = max(0, min(int(time_start), data.shape[0] - 1))
            time_end = max(time_start + 1, min(int(time_end), data.shape[0]))
            # 确保channels是整数且在有效范围内
            channels = [ch for ch in channels if isinstance(ch, (int, np.integer)) and 0 <= ch < data.shape[1]]
            data_format = 'time_channel'
        else:
            # 数据格式为 (n_channels, n_timepoints)
            time_start = max(0, min(int(time_start), data.shape[1] - 1))
            time_end = max(time_start + 1, min(int(time_end), data.shape[1]))
            # 确保channels是整数且在有效范围内
            channels = [ch for ch in channels if isinstance(ch, (int, np.integer)) and 0 <= ch < data.shape[0]]
            data_format = 'channel_time'
        
        if not channels:
            return perturbed_data
        
        # 获取因果闭包
        try:
            # 确保channels是整数列表
            int_channels = [int(ch) for ch in channels]
            closure = self.causal_graph.get_causal_closure(int_channels)
        except Exception as e:
            # 静默处理错误，使用空闭包作为备选
            closure = set()
        
        # 对因果闭包中的所有节点进行一致性扰动
        for node, lag_offset in closure:
            try:
                # 确保node是整数
                node_int = int(node)
                
                # 处理因果闭包中的节点
                
                # 根据数据格式检查节点索引有效性并应用扰动
                if data_format == 'time_channel':
                    if 0 <= node_int < data.shape[1]:
                        # 计算实际时间范围
                        actual_start = max(0, time_start + lag_offset)
                        actual_end = min(data.shape[0], time_end + lag_offset)
                        
                        if actual_start < actual_end:
                            # 生成扰动
                            segment = perturbed_data[actual_start:actual_end, node_int]
                            perturbation = self._generate_perturbation(segment, strength)
                            
                            # 应用扰动
                            perturbed_data[actual_start:actual_end, node_int] += perturbation
                else:  # channel_time format
                    if 0 <= node_int < data.shape[0]:
                        # 计算实际时间范围
                        actual_start = max(0, time_start + lag_offset)
                        actual_end = min(data.shape[1], time_end + lag_offset)
                        
                        if actual_start < actual_end:
                            # 生成扰动
                            segment = perturbed_data[node_int, actual_start:actual_end]
                            perturbation = self._generate_perturbation(segment, strength)
                            
                            # 应用扰动
                            perturbed_data[node_int, actual_start:actual_end] += perturbation
            except Exception as e:
                # 跳过有问题的节点
                continue
        
        return perturbed_data
    
    def _generate_perturbation(self, segment: np.ndarray, strength: float) -> np.ndarray:
        """
        生成扰动信号
        
        Parameters:
        -----------
        segment : np.ndarray
            原始信号段
        strength : float
            扰动强度
        
        Returns:
        --------
        perturbation : np.ndarray
            扰动信号
        """
        # 方法1：高斯噪声扰动
        noise_perturbation = np.random.normal(0, strength * np.std(segment), len(segment))
        
        # 方法2：基于AR模型的扰动
        try:
            # 简单AR(1)模型
            if len(segment) > 2:
                ar_coef = np.corrcoef(segment[:-1], segment[1:])[0, 1]
                ar_noise = np.random.normal(0, strength * np.std(segment), len(segment))
                ar_perturbation = np.zeros_like(segment)
                ar_perturbation[0] = ar_noise[0]
                
                for i in range(1, len(segment)):
                    ar_perturbation[i] = ar_coef * ar_perturbation[i-1] + ar_noise[i]
                
                # 混合两种扰动
                perturbation = 0.7 * noise_perturbation + 0.3 * ar_perturbation
            else:
                perturbation = noise_perturbation
        except:
            perturbation = noise_perturbation
        
        return perturbation
    
    def _preserve_statistics(self, data: np.ndarray) -> np.ndarray:
        """
        保持统计特性
        
        Parameters:
        -----------
        data : np.ndarray
            扰动后的数据
        
        Returns:
        --------
        adjusted_data : np.ndarray
            调整后的数据
        """
        adjusted_data = data.copy()
        
        # 调整均值和标准差
        current_mean = np.mean(adjusted_data, axis=0)
        current_std = np.std(adjusted_data, axis=0)
        
        target_mean = self.original_stats['mean']
        target_std = self.original_stats['std']
        
        # 标准化并重新缩放
        for ch in range(adjusted_data.shape[1]):
            if current_std[ch] > 1e-10:
                adjusted_data[:, ch] = (adjusted_data[:, ch] - current_mean[ch]) / current_std[ch]
                adjusted_data[:, ch] = adjusted_data[:, ch] * target_std[ch] + target_mean[ch]
        
        return adjusted_data
    
    def evaluate_perturbation_quality(self, original_data: np.ndarray, 
                                     perturbed_data: np.ndarray) -> Dict[str, float]:
        """
        评估扰动质量
        
        Parameters:
        -----------
        original_data : np.ndarray
            原始数据
        perturbed_data : np.ndarray
            扰动后数据
        
        Returns:
        --------
        quality_metrics : dict
            质量指标字典
        """
        if original_data.ndim == 3:
            original_data = original_data[0]
        if perturbed_data.ndim == 3:
            perturbed_data = perturbed_data[0]
        
        if original_data.shape[0] < original_data.shape[1]:
            original_data = original_data.T
        if perturbed_data.shape[0] < perturbed_data.shape[1]:
            perturbed_data = perturbed_data.T
        
        metrics = {}
        
        # 1. 均值保持度
        orig_mean = np.mean(original_data, axis=0)
        pert_mean = np.mean(perturbed_data, axis=0)
        metrics['mean_preservation'] = 1 - np.mean(np.abs(orig_mean - pert_mean) / (np.abs(orig_mean) + 1e-10))
        
        # 2. 标准差保持度
        orig_std = np.std(original_data, axis=0)
        pert_std = np.std(perturbed_data, axis=0)
        metrics['std_preservation'] = 1 - np.mean(np.abs(orig_std - pert_std) / (orig_std + 1e-10))
        
        # 3. 相关性保持度
        orig_corr = np.corrcoef(original_data.T)
        pert_corr = np.corrcoef(perturbed_data.T)
        metrics['correlation_preservation'] = np.corrcoef(orig_corr.flatten(), pert_corr.flatten())[0, 1]
        
        # 4. 功率谱保持度
        orig_psd = self._compute_power_spectrum(original_data)
        pert_psd = self._compute_power_spectrum(perturbed_data)
        
        psd_similarity = []
        for ch in range(min(orig_psd.shape[0], pert_psd.shape[0])):
            corr = np.corrcoef(orig_psd[ch], pert_psd[ch])[0, 1]
            if not np.isnan(corr):
                psd_similarity.append(corr)
        
        metrics['power_spectrum_preservation'] = np.mean(psd_similarity) if psd_similarity else 0
        
        # 5. 整体质量分数
        metrics['overall_quality'] = np.mean([
            metrics['mean_preservation'],
            metrics['std_preservation'],
            metrics['correlation_preservation'],
            metrics['power_spectrum_preservation']
        ])
        
        return metrics


def create_causal_analyzer(method: str = 'granger', **kwargs):
    """
    创建因果分析器的便捷函数
    
    Parameters:
    -----------
    method : str
        分析方法 ('granger', 'notears')
    **kwargs : dict
        其他参数
    
    Returns:
    --------
    analyzer : object
        因果分析器
    """
    if method == 'granger':
        return GrangerCausalityAnalyzer(**kwargs)
    elif method == 'notears':
        return NOTEARSAnalyzer(**kwargs)
    else:
        raise ValueError(f"未知的分析方法: {method}")


if __name__ == "__main__":
    # 示例使用
    print("因果一致性扰动模块已加载")
    
    # 创建示例数据
    np.random.seed(42)
    n_timepoints = 1280
    n_channels = 4
    
    # 生成具有因果关系的数据
    data = np.zeros((n_timepoints, n_channels))
    data[:, 0] = np.random.randn(n_timepoints)
    
    for t in range(1, n_timepoints):
        data[t, 1] = 0.5 * data[t-1, 0] + np.random.randn() * 0.5  # 0 -> 1
        data[t, 2] = 0.3 * data[t-1, 1] + np.random.randn() * 0.5  # 1 -> 2
        data[t, 3] = 0.4 * data[t-1, 0] + 0.2 * data[t-1, 2] + np.random.randn() * 0.5  # 0,2 -> 3
    
    # Granger因果分析
    granger_analyzer = GrangerCausalityAnalyzer(max_lag=3)
    granger_analyzer.fit(data)
    granger_graph = granger_analyzer.get_causal_graph([f"Ch_{i}" for i in range(n_channels)])
    
    print(f"Granger因果矩阵:")
    print(granger_analyzer.causal_matrix)
    
    # NOTEARS分析
    notears_analyzer = NOTEARSAnalyzer(lambda_reg=0.1)
    notears_analyzer.fit(data)
    notears_graph = notears_analyzer.get_causal_graph([f"Ch_{i}" for i in range(n_channels)])
    
    print(f"NOTEARS邻接矩阵:")
    print(notears_analyzer.adjacency_matrix)
    
    # 创建扰动引擎
    perturbation_engine = CausalPerturbationEngine(granger_graph)
    
    # 模拟单元扰动
    class MockUnit:
        def __init__(self, channels, time_range):
            self.channels = channels
            self.time_range = time_range
    
    units = [MockUnit([0], (100, 200)), MockUnit([1], (150, 250))]
    perturbation_mask = np.array([1.0, 0.5])  # 扰动强度
    
    perturbed_data = perturbation_engine.perturb_units(data.T, units, perturbation_mask)
    
    # 评估扰动质量
    quality = perturbation_engine.evaluate_perturbation_quality(data.T, perturbed_data)
    print(f"扰动质量评估: {quality}")