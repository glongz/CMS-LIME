#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shapelet分析模块

实现EEG时间序列的shapelet发现和分析，包括：
- 基于信息增益的shapelet发现
- 学习型shapelet方法
- Shapelet变换和特征提取
- 判别性shapelet选择

Author: CMS-LIME Framework
Date: 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional, Union
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')


class ShapeletCandidate:
    """Shapelet候选者类"""
    
    def __init__(self, data: np.ndarray, start_pos: int, length: int, 
                 channel: int, series_id: int, class_label: Optional[int] = None):
        """
        初始化shapelet候选者
        
        Parameters:
        -----------
        data : np.ndarray
            shapelet数据
        start_pos : int
            起始位置
        length : int
            长度
        channel : int
            通道索引
        series_id : int
            时间序列ID
        class_label : int, optional
            类别标签
        """
        self.data = data
        self.start_pos = start_pos
        self.length = length
        self.channel = channel
        self.series_id = series_id
        self.class_label = class_label
        self.information_gain = 0.0
        self.quality_score = 0.0
    
    def compute_distance(self, other_data: np.ndarray) -> float:
        """
        计算与其他数据的距离
        
        Parameters:
        -----------
        other_data : np.ndarray
            其他时间序列数据
        
        Returns:
        --------
        distance : float
            欧几里得距离
        """
        if len(other_data) != self.length:
            return float('inf')
        
        return euclidean(self.data, other_data)
    
    def compute_subsequence_distance(self, time_series: np.ndarray) -> float:
        """
        计算与时间序列中最相似子序列的距离
        
        Parameters:
        -----------
        time_series : np.ndarray
            目标时间序列
        
        Returns:
        --------
        min_distance : float
            最小距离
        """
        if len(time_series) < self.length:
            return float('inf')
        
        min_distance = float('inf')
        
        for i in range(len(time_series) - self.length + 1):
            subsequence = time_series[i:i + self.length]
            distance = self.compute_distance(subsequence)
            min_distance = min(min_distance, distance)
        
        return min_distance


class InformationGainShapeletFinder:
    """基于信息增益的Shapelet发现器"""
    
    def __init__(self, min_length: int = 10, max_length: int = 100, 
                 length_step: int = 5, max_candidates: int = 100):
        """
        初始化shapelet发现器
        
        Parameters:
        -----------
        min_length : int
            最小shapelet长度
        max_length : int
            最大shapelet长度
        length_step : int
            长度步长
        max_candidates : int
            最大候选者数量
        """
        self.min_length = min_length
        self.max_length = max_length
        self.length_step = length_step
        self.max_candidates = max_candidates
        self.shapelets = []
    
    def find_shapelets(self, X: np.ndarray, y: np.ndarray, 
                      n_shapelets: int = 50) -> List[ShapeletCandidate]:
        """
        发现判别性shapelets
        
        Parameters:
        -----------
        X : np.ndarray
            时间序列数据，形状为 (n_samples, n_channels, n_timepoints)
        y : np.ndarray
            类别标签
        n_shapelets : int
            要发现的shapelet数量
        
        Returns:
        --------
        best_shapelets : list
            最佳shapelets列表
        """
        # print(f"开始发现shapelets，数据形状: {X.shape}")  # 注释掉
        
        # 生成候选shapelets
        candidates = self._generate_candidates(X, y)
        # print(f"生成了{len(candidates)}个候选shapelets")  # 注释掉
        
        # 计算信息增益
        for i, candidate in enumerate(candidates):
            # if i % 100 == 0:
            #     print(f"处理候选者 {i}/{len(candidates)}")  # 注释掉
            
            candidate.information_gain = self._compute_information_gain(candidate, X, y)
        
        # 选择最佳shapelets
        candidates.sort(key=lambda x: x.information_gain, reverse=True)
        best_shapelets = candidates[:n_shapelets]
        
        self.shapelets = best_shapelets
        # print(f"发现了{len(best_shapelets)}个高质量shapelets")  # 注释掉
        
        return best_shapelets
    
    def _generate_candidates(self, X: np.ndarray, y: np.ndarray) -> List[ShapeletCandidate]:
        """
        生成候选shapelet
        
        Parameters:
        -----------
        X : np.ndarray
            时间序列数据，形状为 (n_samples, n_channels, n_timepoints) 或 (n_samples, 1, n_channels, n_timepoints)
        y : np.ndarray
            标签
        
        Returns:
        --------
        candidates : list
            候选者列表
        """
        candidates = []
        
        # Handle different input dimensions
        if len(X.shape) == 4:
            # (n_samples, 1, n_channels, n_timepoints) -> (n_samples, n_channels, n_timepoints)
            X = X.squeeze(1)
        
        n_samples, n_channels, n_timepoints = X.shape
        
        # 计算每个长度的候选者数量
        lengths = list(range(self.min_length, 
                           min(self.max_length, n_timepoints), 
                           self.length_step))
        
        candidates_per_length = self.max_candidates // len(lengths)
        
        for length in lengths:
            length_candidates = 0
            
            for sample_idx in range(n_samples):
                if length_candidates >= candidates_per_length:
                    break
                
                for channel_idx in range(n_channels):
                    if length_candidates >= candidates_per_length:
                        break
                    
                    # 在该通道的时间序列中采样
                    max_start = n_timepoints - length
                    if max_start <= 0:
                        continue
                    
                    # 随机采样起始位置
                    n_samples_per_series = min(5, max_start + 1)
                    start_positions = np.random.choice(
                        max_start + 1, 
                        size=min(n_samples_per_series, max_start + 1), 
                        replace=False
                    )
                    
                    for start_pos in start_positions:
                        if length_candidates >= candidates_per_length:
                            break
                        
                        shapelet_data = X[sample_idx, channel_idx, start_pos:start_pos + length]
                        
                        # 标准化shapelet
                        if np.std(shapelet_data) > 0:
                            shapelet_data = (shapelet_data - np.mean(shapelet_data)) / np.std(shapelet_data)
                        
                        candidate = ShapeletCandidate(
                            data=shapelet_data,
                            start_pos=start_pos,
                            length=length,
                            channel=channel_idx,
                            series_id=sample_idx,
                            class_label=y[sample_idx]
                        )
                        
                        candidates.append(candidate)
                        length_candidates += 1
        
        return candidates
    
    def _compute_information_gain(self, candidate: ShapeletCandidate, 
                                X: np.ndarray, y: np.ndarray) -> float:
        """
        计算shapelet的信息增益
        
        Parameters:
        -----------
        candidate : ShapeletCandidate
            候选shapelet
        X : np.ndarray
            时间序列数据
        y : np.ndarray
            类别标签
        
        Returns:
        --------
        information_gain : float
            信息增益值
        """
        n_samples = X.shape[0]
        
        # 计算每个样本到shapelet的距离
        distances = np.zeros(n_samples)
        
        for i in range(n_samples):
            time_series = X[i, candidate.channel, :]
            distances[i] = candidate.compute_subsequence_distance(time_series)
        
        # 尝试不同的阈值来分割数据
        unique_distances = np.unique(distances)
        if len(unique_distances) < 2:
            return 0.0
        
        best_gain = 0.0
        
        # 尝试多个阈值
        n_thresholds = min(20, len(unique_distances) - 1)
        threshold_indices = np.linspace(0, len(unique_distances) - 2, n_thresholds, dtype=int)
        
        for idx in threshold_indices:
            threshold = (unique_distances[idx] + unique_distances[idx + 1]) / 2
            
            # 根据阈值分割数据
            left_mask = distances <= threshold
            right_mask = distances > threshold
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            # 计算信息增益
            gain = self._calculate_information_gain(y, left_mask, right_mask)
            best_gain = max(best_gain, gain)
        
        return best_gain
    
    def _calculate_information_gain(self, y: np.ndarray, left_mask: np.ndarray, 
                                  right_mask: np.ndarray) -> float:
        """
        计算信息增益
        
        Parameters:
        -----------
        y : np.ndarray
            类别标签
        left_mask : np.ndarray
            左分支掩码
        right_mask : np.ndarray
            右分支掩码
        
        Returns:
        --------
        gain : float
            信息增益
        """
        # 计算原始熵
        original_entropy = self._compute_entropy(y)
        
        # 计算分割后的加权熵
        n_total = len(y)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        left_entropy = self._compute_entropy(y[left_mask])
        right_entropy = self._compute_entropy(y[right_mask])
        
        weighted_entropy = (n_left / n_total) * left_entropy + (n_right / n_total) * right_entropy
        
        return original_entropy - weighted_entropy
    
    def _compute_entropy(self, labels: np.ndarray) -> float:
        """
        计算标签的熵
        
        Parameters:
        -----------
        labels : np.ndarray
            类别标签
        
        Returns:
        --------
        entropy_value : float
            熵值
        """
        if len(labels) == 0:
            return 0.0
        
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        
        return entropy(probabilities, base=2)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        将时间序列转换为shapelet特征
        
        Parameters:
        -----------
        X : np.ndarray
            时间序列数据，形状为 (n_samples, n_channels, n_timepoints) 或 (n_samples, 1, n_channels, n_timepoints)
        
        Returns:
        --------
        features : np.ndarray
            shapelet特征矩阵
        """
        if not self.shapelets:
            raise ValueError("尚未发现shapelets，请先调用find_shapelets方法")
        
        # Handle different input dimensions
        if len(X.shape) == 4:
            # (n_samples, 1, n_channels, n_timepoints) -> (n_samples, n_channels, n_timepoints)
            X = X.squeeze(1)
        
        n_samples = X.shape[0]
        n_shapelets = len(self.shapelets)
        features = np.zeros((n_samples, n_shapelets))
        
        for i, shapelet in enumerate(self.shapelets):
            for j in range(n_samples):
                time_series = X[j, shapelet.channel, :]
                features[j, i] = shapelet.compute_subsequence_distance(time_series)
        
        return features


class LearnableShapelet(nn.Module):
    """可学习的Shapelet层"""
    
    def __init__(self, n_shapelets: int, shapelet_length: int, n_channels: int):
        """
        初始化可学习shapelet层
        
        Parameters:
        -----------
        n_shapelets : int
            shapelet数量
        shapelet_length : int
            shapelet长度
        n_channels : int
            通道数量
        """
        super(LearnableShapelet, self).__init__()
        
        self.n_shapelets = n_shapelets
        self.shapelet_length = shapelet_length
        self.n_channels = n_channels
        
        # 初始化shapelet参数
        self.shapelets = nn.Parameter(
            torch.randn(n_shapelets, n_channels, shapelet_length)
        )
        
        # 可学习的缩放参数
        self.scales = nn.Parameter(torch.ones(n_shapelets))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Parameters:
        -----------
        x : torch.Tensor
            输入时间序列，形状为 (batch_size, n_channels, n_timepoints)
        
        Returns:
        --------
        distances : torch.Tensor
            到各shapelet的最小距离，形状为 (batch_size, n_shapelets)
        """
        batch_size, n_channels, n_timepoints = x.shape
        
        # 计算每个shapelet的距离
        min_distances = torch.zeros(batch_size, self.n_shapelets, device=x.device)
        
        for i in range(self.n_shapelets):
            shapelet = self.shapelets[i]  # (n_channels, shapelet_length)
            
            # 计算滑动窗口距离
            distances = []
            
            for start in range(n_timepoints - self.shapelet_length + 1):
                subsequence = x[:, :, start:start + self.shapelet_length]
                
                # 计算欧几里得距离
                diff = subsequence - shapelet.unsqueeze(0)
                distance = torch.sqrt(torch.sum(diff ** 2, dim=(1, 2)))
                distances.append(distance)
            
            if distances:
                distances = torch.stack(distances, dim=1)  # (batch_size, n_positions)
                min_distances[:, i] = torch.min(distances, dim=1)[0]
            else:
                min_distances[:, i] = torch.full((batch_size,), float('inf'), device=x.device)
        
        # 应用缩放
        scaled_distances = min_distances * self.scales.unsqueeze(0)
        
        return scaled_distances


class LearnableShapeletClassifier(nn.Module):
    """基于可学习shapelet的分类器"""
    
    def __init__(self, n_shapelets: int, shapelet_length: int, 
                 n_channels: int, n_classes: int):
        """
        初始化分类器
        
        Parameters:
        -----------
        n_shapelets : int
            shapelet数量
        shapelet_length : int
            shapelet长度
        n_channels : int
            通道数量
        n_classes : int
            类别数量
        """
        super(LearnableShapeletClassifier, self).__init__()
        
        self.shapelet_layer = LearnableShapelet(n_shapelets, shapelet_length, n_channels)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(n_shapelets, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Parameters:
        -----------
        x : torch.Tensor
            输入时间序列
        
        Returns:
        --------
        logits : torch.Tensor
            分类logits
        """
        # 获取shapelet特征
        shapelet_features = self.shapelet_layer(x)
        
        # 分类
        logits = self.classifier(shapelet_features)
        
        return logits
    
    def get_shapelets(self) -> np.ndarray:
        """
        获取学习到的shapelets
        
        Returns:
        --------
        shapelets : np.ndarray
            shapelet数组
        """
        return self.shapelet_layer.shapelets.detach().cpu().numpy()


class ShapeletAnalyzer:
    """Shapelet分析器主类"""
    
    def __init__(self, method: str = 'information_gain', 
                 min_length: int = 10, max_length: int = 100,
                 n_shapelets: int = 50):
        """
        初始化shapelet分析器
        
        Parameters:
        -----------
        method : str
            方法类型 ('information_gain', 'learnable')
        min_length : int
            最小shapelet长度
        max_length : int
            最大shapelet长度
        n_shapelets : int
            shapelet数量
        """
        self.method = method
        self.min_length = min_length
        self.max_length = max_length
        self.n_shapelets = n_shapelets
        
        if method == 'information_gain':
            self.finder = InformationGainShapeletFinder(
                min_length=min_length,
                max_length=max_length,
                max_candidates=100
            )
        elif method == 'learnable':
            self.model = None
        else:
            raise ValueError(f"未知的方法: {method}")
        
        self.fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'ShapeletAnalyzer':
        """
        拟合shapelet分析器
        
        Parameters:
        -----------
        X : np.ndarray
            时间序列数据，形状为 (n_samples, n_channels, n_timepoints) 或 (n_samples, 1, n_channels, n_timepoints)
        y : np.ndarray
            类别标签
        **kwargs : dict
            其他参数
        
        Returns:
        --------
        self : ShapeletAnalyzer
        """
        # print(f"开始发现shapelets，数据形状: {X.shape}")  # 注释掉
        
        # Handle different input dimensions
        if len(X.shape) == 4:
            # (n_samples, 1, n_channels, n_timepoints) -> (n_samples, n_channels, n_timepoints)
            X = X.squeeze(1)
            # print(f"调整后的数据形状: {X.shape}")  # 注释掉
        
        if self.method == 'information_gain':
            self.shapelets = self.finder.find_shapelets(X, y, self.n_shapelets)
        
        elif self.method == 'learnable':
            self._fit_learnable_shapelets(X, y, **kwargs)
        
        self.fitted = True
        return self
    
    def _fit_learnable_shapelets(self, X: np.ndarray, y: np.ndarray, 
                                epochs: int = 100, lr: float = 0.001,
                                batch_size: int = 32) -> None:
        """
        训练可学习shapelet模型
        
        Parameters:
        -----------
        X : np.ndarray
            训练数据
        y : np.ndarray
            标签
        epochs : int
            训练轮数
        lr : float
            学习率
        batch_size : int
            批次大小
        """
        n_samples, n_channels, n_timepoints = X.shape
        n_classes = len(np.unique(y))
        
        # 选择合适的shapelet长度
        shapelet_length = min(self.max_length, n_timepoints // 4)
        
        # 创建模型
        self.model = LearnableShapeletClassifier(
            n_shapelets=self.n_shapelets,
            shapelet_length=shapelet_length,
            n_channels=n_channels,
            n_classes=n_classes
        )
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            if (epoch + 1) % 20 == 0:
                accuracy = 100 * correct / total
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%')
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        转换数据为shapelet特征
        
        Parameters:
        -----------
        X : np.ndarray
            时间序列数据，形状为 (n_samples, n_channels, n_timepoints) 或 (n_samples, 1, n_channels, n_timepoints)
        
        Returns:
        --------
        features : np.ndarray
            shapelet特征
        """
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        # Handle different input dimensions
        if len(X.shape) == 4:
            # (n_samples, 1, n_channels, n_timepoints) -> (n_samples, n_channels, n_timepoints)
            X = X.squeeze(1)
        
        if self.method == 'information_gain':
            return self.finder.transform(X)
        
        elif self.method == 'learnable':
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                features = self.model.shapelet_layer(X_tensor)
                return features.numpy()
    
    def get_shapelets(self) -> List:
        """
        获取发现的shapelets
        
        Returns:
        --------
        shapelets : list
            shapelet列表
        """
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        if self.method == 'information_gain':
            return self.shapelets
        elif self.method == 'learnable':
            return self.model.get_shapelets()
    
    def evaluate_shapelets(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        评估shapelet质量
        
        Parameters:
        -----------
        X : np.ndarray
            测试数据
        y : np.ndarray
            测试标签
        
        Returns:
        --------
        metrics : dict
            评估指标
        """
        features = self.transform(X)
        
        # 使用决策树评估特征质量
        clf = DecisionTreeClassifier(random_state=42)
        scores = cross_val_score(clf, features, y, cv=5)
        
        metrics = {
            'mean_accuracy': np.mean(scores),
            'std_accuracy': np.std(scores),
            'n_features': features.shape[1]
        }
        
        return metrics


def create_shapelet_analyzer(method: str = 'information_gain', **kwargs) -> ShapeletAnalyzer:
    """
    创建shapelet分析器的便捷函数
    
    Parameters:
    -----------
    method : str
        分析方法
    **kwargs : dict
        其他参数
    
    Returns:
    --------
    analyzer : ShapeletAnalyzer
        shapelet分析器
    """
    return ShapeletAnalyzer(method=method, **kwargs)


if __name__ == "__main__":
    # 示例使用
    print("Shapelet分析模块已加载")
    
    # 创建示例数据
    np.random.seed(42)
    n_samples, n_channels, n_timepoints = 100, 8, 200
    X = np.random.randn(n_samples, n_channels, n_timepoints)
    y = np.random.randint(0, 3, n_samples)
    
    # 创建分析器
    analyzer = ShapeletAnalyzer(method='information_gain', n_shapelets=20)
    
    # 拟合和转换
    analyzer.fit(X, y)
    features = analyzer.transform(X)
    
    print(f"Shapelet特征形状: {features.shape}")
    
    # 评估
    metrics = analyzer.evaluate_shapelets(X, y)
    print(f"评估结果: {metrics}")