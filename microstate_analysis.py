#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微状态分析模块

实现EEG微状态分析的核心算法，包括：
- K-means聚类方法
- Atomize-Agglomerate算法
- 微状态序列分割和标记
- 拓扑地图分析

Author: CMS-LIME Framework
Date: 2025
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')


class MicrostateAnalyzer:
    """微状态分析器"""
    
    def __init__(self, n_microstates: int = 4, method: str = 'kmeans', 
                 gfp_threshold: float = 0.1, min_duration: int = 3):
        """
        初始化微状态分析器
        
        Parameters:
        -----------
        n_microstates : int
            微状态数量
        method : str
            分析方法 ('kmeans', 'atomize_agglomerate')
        gfp_threshold : float
            全局场功率阈值
        min_duration : int
            最小微状态持续时间（样本点数）
        """
        self.n_microstates = n_microstates
        self.method = method
        self.gfp_threshold = gfp_threshold
        self.min_duration = min_duration
        self.microstate_maps = None
        self.fitted = False
    
    def fit(self, data: np.ndarray, sampling_rate: float = 250.0) -> 'MicrostateAnalyzer':
        """
        拟合微状态模型
        
        Parameters:
        -----------
        data : np.ndarray
            EEG数据，形状为 (n_channels, n_timepoints) 或 (n_epochs, n_channels, n_timepoints)
        sampling_rate : float
            采样率
        
        Returns:
        --------
        self : MicrostateAnalyzer
        """
        # Handle different input dimensions
        if len(data.shape) == 4:
            # (batch, channels, time, extra) -> (channels, time)
            # Take the first sample and squeeze extra dimensions
            data = data[0].squeeze()
            if len(data.shape) == 3:
                data = data.reshape(data.shape[0], -1)
        elif len(data.shape) == 3:
            n0, n1, n2 = data.shape
            # CMS-LIME 传入 (n_samples, n_channels, n_timepoints)；时间维通常最大。
            # 旧逻辑用 n0 < n1 区分 layout，在 n_samples(如 24) > n_channels(22) 时会错展成 (24, C*T)，
            # 导致微状态维数与解释时 (C,T) 不一致（如 24 vs 22）。
            if n2 >= n0 and n2 >= n1:
                data = np.transpose(data, (1, 0, 2)).reshape(n1, n0 * n2)
            elif n0 < n1:
                data = data.reshape(n1, -1)
            else:
                data = data.reshape(n0, -1)
        
        # 预处理数据
        preprocessed_data = self._preprocess_data(data)
        
        # 计算全局场功率
        gfp = self._compute_gfp(preprocessed_data)
        
        # 选择GFP峰值点
        peak_indices = self._find_gfp_peaks(gfp)
        peak_data = preprocessed_data[:, peak_indices]
        
        # 根据方法进行微状态分析
        if self.method == 'kmeans':
            self.microstate_maps = self._kmeans_clustering(peak_data)
        elif self.method == 'atomize_agglomerate':
            self.microstate_maps = self._atomize_agglomerate(peak_data)
        else:
            raise ValueError(f"未知的微状态分析方法: {self.method}")
        
        self.fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        将数据转换为微状态序列
        
        Parameters:
        -----------
        data : np.ndarray
            EEG数据，形状为 (n_channels, n_timepoints)
        
        Returns:
        --------
        microstate_sequence : np.ndarray
            微状态标签序列
        correlation_values : np.ndarray
            与各微状态的相关性值
        microstate_stats : dict
            微状态统计信息
        """
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        if len(data.shape) == 3:
            n0, n1, n2 = data.shape
            if n2 >= n0 and n2 >= n1:
                data = np.transpose(data, (1, 0, 2)).reshape(n1, n0 * n2)
            else:
                data = data.reshape(n1, -1)
        
        # 预处理数据
        preprocessed_data = self._preprocess_data(data)
        
        # 计算与各微状态的相关性
        correlations = self._compute_correlations(preprocessed_data)
        
        # 分配微状态标签
        microstate_sequence = np.argmax(correlations, axis=0)
        
        # 应用最小持续时间约束
        microstate_sequence = self._apply_min_duration(microstate_sequence)
        
        # 计算统计信息
        microstate_stats = self._compute_microstate_statistics(
            microstate_sequence, correlations
        )
        
        return microstate_sequence, correlations, microstate_stats
    
    def fit_transform(self, data: np.ndarray, sampling_rate: float = 250.0) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        拟合模型并转换数据
        """
        self.fit(data, sampling_rate)
        return self.transform(data)
    
    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        预处理EEG数据
        
        Parameters:
        -----------
        data : np.ndarray
            原始EEG数据
        
        Returns:
        --------
        preprocessed_data : np.ndarray
            预处理后的数据
        """
        # 处理NaN值
        if np.any(np.isnan(data)):
            # 用0替换NaN值
            data = np.nan_to_num(data, nan=0.0)
        
        # Z-score标准化
        preprocessed_data = zscore(data, axis=1)
        
        # 再次检查并处理可能的NaN值（zscore可能产生NaN）
        if np.any(np.isnan(preprocessed_data)):
            preprocessed_data = np.nan_to_num(preprocessed_data, nan=0.0)
        
        # 移除均值（重新参考）
        preprocessed_data = preprocessed_data - np.mean(preprocessed_data, axis=0, keepdims=True)
        
        return preprocessed_data
    
    def _compute_gfp(self, data: np.ndarray) -> np.ndarray:
        """
        计算全局场功率 (Global Field Power)
        
        Parameters:
        -----------
        data : np.ndarray
            EEG数据，形状为 (n_channels, n_timepoints)
        
        Returns:
        --------
        gfp : np.ndarray
            全局场功率序列
        """
        # GFP = 标准差 across channels
        gfp = np.std(data, axis=0)
        return gfp
    
    def _find_gfp_peaks(self, gfp: np.ndarray) -> np.ndarray:
        """
        寻找GFP峰值点
        
        Parameters:
        -----------
        gfp : np.ndarray
            全局场功率序列
        
        Returns:
        --------
        peak_indices : np.ndarray
            峰值点索引
        """
        # 简单的峰值检测：局部最大值且超过阈值
        threshold = np.percentile(gfp, 90)  # 使用90%分位数作为阈值
        
        peak_indices = []
        for i in range(1, len(gfp) - 1):
            if (gfp[i] > gfp[i-1] and gfp[i] > gfp[i+1] and 
                gfp[i] > threshold):
                peak_indices.append(i)
        
        # 确保返回整数类型的数组
        peak_indices = np.array(peak_indices, dtype=np.int32)
        
        # 如果没有找到峰值，至少返回一些均匀分布的点
        if len(peak_indices) == 0:
            n_points = min(100, len(gfp) // 10)
            peak_indices = np.linspace(0, len(gfp)-1, n_points, dtype=np.int32)
        
        return peak_indices
    
    def _kmeans_clustering(self, peak_data: np.ndarray) -> np.ndarray:
        """
        使用K-means聚类进行微状态分析
        
        Parameters:
        -----------
        peak_data : np.ndarray
            峰值点数据，形状为 (n_channels, n_peaks)
        
        Returns:
        --------
        microstate_maps : np.ndarray
            微状态拓扑地图，形状为 (n_microstates, n_channels)
        """
        # 转置数据以适应sklearn格式
        X = peak_data.T  # (n_peaks, n_channels)
        
        # 使用多次随机初始化选择最佳结果
        best_score = -1
        best_maps = None
        
        for random_state in range(10):
            kmeans = KMeans(
                n_clusters=self.n_microstates,
                random_state=random_state,
                n_init=10,
                max_iter=300
            )
            labels = kmeans.fit_predict(X)
            
            # 计算轮廓系数
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_maps = kmeans.cluster_centers_.T  # 转置回 (n_channels, n_microstates)
        
        if best_maps is None:
            # 如果所有尝试都失败，使用简单的随机初始化
            kmeans = KMeans(n_clusters=self.n_microstates, random_state=42)
            kmeans.fit(X)
            best_maps = kmeans.cluster_centers_.T
        
        # 标准化微状态地图
        for i in range(self.n_microstates):
            best_maps[:, i] = best_maps[:, i] / np.linalg.norm(best_maps[:, i])
        
        return best_maps
    
    def _atomize_agglomerate(self, peak_data: np.ndarray) -> np.ndarray:
        """
        使用Atomize-Agglomerate算法进行微状态分析
        
        Parameters:
        -----------
        peak_data : np.ndarray
            峰值点数据，形状为 (n_channels, n_peaks)
        
        Returns:
        --------
        microstate_maps : np.ndarray
            微状态拓扑地图，形状为 (n_channels, n_microstates)
        """
        n_channels, n_peaks = peak_data.shape
        
        # 标准化每个时间点的数据
        normalized_data = np.zeros_like(peak_data)
        for i in range(n_peaks):
            normalized_data[:, i] = peak_data[:, i] / np.linalg.norm(peak_data[:, i])
        
        # 计算相关性矩阵
        correlation_matrix = np.corrcoef(normalized_data.T)
        
        # 初始化：每个峰值点作为一个原子
        clusters = [[i] for i in range(n_peaks)]
        cluster_representatives = normalized_data.copy()
        
        # 聚合过程
        while len(clusters) > self.n_microstates:
            # 找到最相似的两个聚类
            max_similarity = -1
            merge_indices = (0, 1)
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # 计算聚类间的平均相关性
                    similarities = []
                    for idx1 in clusters[i]:
                        for idx2 in clusters[j]:
                            similarities.append(correlation_matrix[idx1, idx2])
                    
                    avg_similarity = np.mean(similarities)
                    if avg_similarity > max_similarity:
                        max_similarity = avg_similarity
                        merge_indices = (i, j)
            
            # 合并最相似的两个聚类
            i, j = merge_indices
            merged_cluster = clusters[i] + clusters[j]
            
            # 计算新的代表性拓扑
            cluster_data = normalized_data[:, merged_cluster]
            representative = np.mean(cluster_data, axis=1)
            representative = representative / np.linalg.norm(representative)
            
            # 更新聚类列表
            new_clusters = []
            new_representatives = []
            
            for k, cluster in enumerate(clusters):
                if k != i and k != j:
                    new_clusters.append(cluster)
                    new_representatives.append(cluster_representatives[:, k])
            
            new_clusters.append(merged_cluster)
            new_representatives.append(representative)
            
            clusters = new_clusters
            cluster_representatives = np.column_stack(new_representatives)
        
        return cluster_representatives
    
    def _compute_correlations(self, data: np.ndarray) -> np.ndarray:
        """
        计算数据与微状态地图的相关性
        
        Parameters:
        -----------
        data : np.ndarray
            EEG数据，形状为 (n_channels, n_timepoints)
        
        Returns:
        --------
        correlations : np.ndarray
            相关性矩阵，形状为 (n_microstates, n_timepoints)
        """
        n_channels, n_timepoints = data.shape
        if self.microstate_maps is not None and self.microstate_maps.shape[0] != n_channels:
            raise ValueError(
                f"通道数与微状态图不一致: 数据为 (n_ch={n_channels}, T)，"
                f"microstate_maps 第一维为 {self.microstate_maps.shape[0]}。"
                " 请用与 CMS-LIME 相同的 (n_samples, n_channels, n_time) 对 MicrostateAnalyzer.fit 拟合，"
                "transform 为 (C, T) 或 (1, C, T)。"
            )
        correlations = np.zeros((self.n_microstates, n_timepoints))
        
        for t in range(n_timepoints):
            # 标准化当前时间点的数据
            current_data = data[:, t]
            if np.linalg.norm(current_data) > 0:
                current_data = current_data / np.linalg.norm(current_data)
                
                # 计算与各微状态的相关性
                for ms in range(self.n_microstates):
                    correlations[ms, t] = np.abs(np.dot(current_data, self.microstate_maps[:, ms]))
        
        return correlations
    
    def _apply_min_duration(self, sequence: np.ndarray) -> np.ndarray:
        """
        应用最小持续时间约束
        
        Parameters:
        -----------
        sequence : np.ndarray
            原始微状态序列
        
        Returns:
        --------
        filtered_sequence : np.ndarray
            应用约束后的序列
        """
        filtered_sequence = sequence.copy()
        
        # 找到所有微状态段
        segments = []
        current_state = sequence[0]
        start_idx = 0
        
        for i in range(1, len(sequence)):
            if sequence[i] != current_state:
                segments.append((current_state, start_idx, i))
                current_state = sequence[i]
                start_idx = i
        
        # 添加最后一段
        segments.append((current_state, start_idx, len(sequence)))
        
        # 处理短于最小持续时间的段
        for state, start, end in segments:
            duration = end - start
            if duration < self.min_duration:
                # 用相邻段的状态替换
                if start > 0 and end < len(sequence):
                    # 选择持续时间更长的相邻状态
                    prev_state = filtered_sequence[start - 1]
                    next_state = filtered_sequence[end] if end < len(sequence) else prev_state
                    
                    # 简单策略：用前一个状态替换
                    filtered_sequence[start:end] = prev_state
                elif start == 0 and end < len(sequence):
                    # 开始段，用下一个状态替换
                    next_state = filtered_sequence[end]
                    filtered_sequence[start:end] = next_state
                elif start > 0 and end == len(sequence):
                    # 结束段，用前一个状态替换
                    prev_state = filtered_sequence[start - 1]
                    filtered_sequence[start:end] = prev_state
        
        return filtered_sequence
    
    def _compute_microstate_statistics(self, sequence: np.ndarray, 
                                     correlations: np.ndarray) -> Dict:
        """
        计算微状态统计信息
        
        Parameters:
        -----------
        sequence : np.ndarray
            微状态序列
        correlations : np.ndarray
            相关性矩阵
        
        Returns:
        --------
        stats : dict
            统计信息字典
        """
        stats = {}
        
        # 计算各微状态的覆盖率
        coverage = np.zeros(self.n_microstates)
        for ms in range(self.n_microstates):
            coverage[ms] = np.sum(sequence == ms) / len(sequence)
        stats['coverage'] = coverage
        
        # 计算平均持续时间
        durations = {ms: [] for ms in range(self.n_microstates)}
        current_state = sequence[0]
        current_duration = 1
        
        for i in range(1, len(sequence)):
            if sequence[i] == current_state:
                current_duration += 1
            else:
                durations[current_state].append(current_duration)
                current_state = sequence[i]
                current_duration = 1
        
        # 添加最后一段
        durations[current_state].append(current_duration)
        
        mean_durations = np.zeros(self.n_microstates)
        for ms in range(self.n_microstates):
            if durations[ms]:
                mean_durations[ms] = np.mean(durations[ms])
        stats['mean_duration'] = mean_durations
        
        # 计算转移概率矩阵
        transition_matrix = np.zeros((self.n_microstates, self.n_microstates))
        for i in range(len(sequence) - 1):
            from_state = sequence[i]
            to_state = sequence[i + 1]
            transition_matrix[from_state, to_state] += 1
        
        # 标准化
        for ms in range(self.n_microstates):
            total_transitions = np.sum(transition_matrix[ms, :])
            if total_transitions > 0:
                transition_matrix[ms, :] /= total_transitions
        
        stats['transition_matrix'] = transition_matrix
        
        # 计算全局解释方差
        total_variance = 0
        explained_variance = 0
        
        for t in range(correlations.shape[1]):
            assigned_state = sequence[t]
            total_variance += 1  # 标准化数据的方差为1
            explained_variance += correlations[assigned_state, t] ** 2
        
        stats['explained_variance'] = explained_variance / total_variance
        
        return stats
    
    def get_microstate_maps(self) -> np.ndarray:
        """
        获取微状态拓扑地图
        
        Returns:
        --------
        microstate_maps : np.ndarray
            微状态地图，形状为 (n_channels, n_microstates)
        """
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        return self.microstate_maps
    
    def plot_microstate_maps(self, channel_names: Optional[List[str]] = None):
        """
        绘制微状态拓扑地图
        
        Parameters:
        -----------
        channel_names : list, optional
            通道名称列表
        """
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, self.n_microstates, figsize=(4*self.n_microstates, 4))
            if self.n_microstates == 1:
                axes = [axes]
            
            for i in range(self.n_microstates):
                ax = axes[i]
                
                # 简单的拓扑图：显示各通道的权重
                weights = self.microstate_maps[:, i]
                
                if channel_names:
                    ax.bar(range(len(weights)), weights)
                    ax.set_xticks(range(len(weights)))
                    ax.set_xticklabels(channel_names, rotation=45)
                else:
                    ax.bar(range(len(weights)), weights)
                    ax.set_xlabel('Channel Index')
                
                ax.set_title(f'Microstate {i+1}')
                ax.set_ylabel('Weight')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib未安装，无法绘制图形")


def analyze_microstate_sequence(sequence: np.ndarray, sampling_rate: float = 250.0) -> Dict:
    """
    分析微状态序列的时间动力学特性
    
    Parameters:
    -----------
    sequence : np.ndarray
        微状态序列
    sampling_rate : float
        采样率
    
    Returns:
    --------
    dynamics : dict
        时间动力学特性
    """
    dynamics = {}
    
    # 计算微状态出现频率
    unique_states, counts = np.unique(sequence, return_counts=True)
    dynamics['occurrence_rate'] = dict(zip(unique_states, counts / len(sequence)))
    
    # 计算平均持续时间（毫秒）
    durations = {}
    current_state = sequence[0]
    current_duration = 1
    
    for i in range(1, len(sequence)):
        if sequence[i] == current_state:
            current_duration += 1
        else:
            if current_state not in durations:
                durations[current_state] = []
            durations[current_state].append(current_duration * 1000 / sampling_rate)
            current_state = sequence[i]
            current_duration = 1
    
    # 添加最后一段
    if current_state not in durations:
        durations[current_state] = []
    durations[current_state].append(current_duration * 1000 / sampling_rate)
    
    dynamics['mean_duration_ms'] = {state: np.mean(durs) for state, durs in durations.items()}
    dynamics['std_duration_ms'] = {state: np.std(durs) for state, durs in durations.items()}
    
    # 计算转移率
    n_transitions = len(sequence) - 1
    actual_transitions = np.sum(sequence[:-1] != sequence[1:])
    dynamics['transition_rate'] = actual_transitions / n_transitions
    
    return dynamics


if __name__ == "__main__":
    # 示例使用
    print("微状态分析模块已加载")
    
    # 创建示例数据
    np.random.seed(42)
    n_channels, n_timepoints = 22, 1280
    sample_data = np.random.randn(n_channels, n_timepoints)
    
    # 创建分析器
    analyzer = MicrostateAnalyzer(n_microstates=4, method='kmeans')
    
    # 拟合和转换
    sequence, correlations, stats = analyzer.fit_transform(sample_data)
    
    print(f"微状态序列长度: {len(sequence)}")
    print(f"微状态覆盖率: {stats['coverage']}")
    print(f"平均持续时间: {stats['mean_duration']}")
    print(f"解释方差: {stats['explained_variance']:.3f}")