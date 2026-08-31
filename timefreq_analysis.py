#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时频分析模块

实现EEG信号的时频域分析，包括：
- 连续小波变换 (CWT)
- 短时傅里叶变换 (STFT)
- 时频单元构造和特征提取
- 频带功率分析
- 时频图可视化

Author: CMS-LIME Framework
Date: 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import signal
from scipy.signal import hilbert, butter, filtfilt
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class TimeFreqUnit:
    """时频基元单位类"""
    
    def __init__(self, unit_id: str, channels: List[int], 
                 time_range: Tuple[int, int], freq_band: Tuple[float, float],
                 freq_band_name: str, power_values: np.ndarray = None,
                 phase_values: np.ndarray = None, metadata: Dict = None):
        """
        初始化时频单元
        
        Parameters:
        -----------
        unit_id : str
            单元唯一标识
        channels : list
            相关通道列表
        time_range : tuple
            时间范围 (start, end)
        freq_band : tuple
            频率范围 (low_freq, high_freq)
        freq_band_name : str
            频带名称
        power_values : np.ndarray, optional
            功率值
        phase_values : np.ndarray, optional
            相位值
        metadata : dict, optional
            元数据
        """
        self.unit_id = unit_id
        self.channels = channels
        self.time_range = time_range
        self.freq_band = freq_band
        self.freq_band_name = freq_band_name
        self.power_values = power_values
        self.phase_values = phase_values
        self.metadata = metadata or {}
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """
        提取时频特征
        
        Parameters:
        -----------
        data : np.ndarray
            EEG数据，形状为 (n_channels, n_timepoints)
        
        Returns:
        --------
        features : np.ndarray
            提取的特征
        """
        # 确保channels是整数列表
        try:
            channels_int = [int(ch) for ch in self.channels]
        except (ValueError, TypeError) as e:
            print(f"Warning: 无法将channels转换为整数: {self.channels}, 错误: {e}")
            channels_int = [0]  # 默认使用第一个通道
        
        segment = data[channels_int, self.time_range[0]:self.time_range[1]]
        
        # 计算功率谱密度
        freqs, psd = signal.welch(segment, axis=1, nperseg=min(256, segment.shape[1]))
        
        # 提取目标频带功率
        freq_mask = (freqs >= self.freq_band[0]) & (freqs <= self.freq_band[1])
        if np.any(freq_mask):
            band_power = psd[:, freq_mask].mean(axis=1)
        else:
            band_power = np.zeros(len(self.channels))
        
        # 计算相对功率
        total_power = psd.mean(axis=1)
        relative_power = band_power / (total_power + 1e-10)
        
        # 计算功率变异性
        power_var = np.var(band_power)
        
        # 组合特征
        features = np.concatenate([
            band_power,
            relative_power,
            [power_var, np.mean(band_power), np.std(band_power)]
        ])
        
        return features
    
    def compute_distance(self, other: 'TimeFreqUnit', data: np.ndarray) -> float:
        """
        计算与其他时频单元的距离
        
        Parameters:
        -----------
        other : TimeFreqUnit
            其他时频单元
        data : np.ndarray
            EEG数据
        
        Returns:
        --------
        distance : float
            距离值
        """
        # 时间距离
        time_dist = abs(self.time_range[0] - other.time_range[0])
        
        # 频率距离
        freq_dist = abs(self.freq_band[0] - other.freq_band[0]) + \
                   abs(self.freq_band[1] - other.freq_band[1])
        
        # 通道距离
        channel_overlap = len(set(self.channels).intersection(set(other.channels)))
        channel_dist = len(self.channels) + len(other.channels) - 2 * channel_overlap
        
        # 特征距离
        features1 = self.extract_features(data)
        features2 = other.extract_features(data)
        feature_dist = np.linalg.norm(features1 - features2)
        
        # 综合距离
        total_dist = time_dist + freq_dist + channel_dist + feature_dist
        
        return total_dist


class CWTAnalyzer:
    """连续小波变换分析器"""
    
    def __init__(self, wavelet: str = 'cmor1.5-1.0', scales: np.ndarray = None,
                 sampling_rate: float = 250.0):
        """
        初始化CWT分析器
        
        Parameters:
        -----------
        wavelet : str
            小波类型
        scales : np.ndarray, optional
            尺度数组
        sampling_rate : float
            采样率
        """
        self.wavelet = wavelet
        self.sampling_rate = sampling_rate
        
        if scales is None:
            # 默认尺度，对应1-100Hz
            freqs = np.logspace(0, 2, 50)  # 1-100Hz
            self.scales = pywt.frequency2scale(self.wavelet, freqs / sampling_rate)
        else:
            self.scales = scales
        
        self.freqs = pywt.scale2frequency(self.wavelet, self.scales) * sampling_rate
    
    def transform(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行连续小波变换
        
        Parameters:
        -----------
        data : np.ndarray
            输入信号，形状为 (n_channels, n_timepoints) 或 (n_timepoints,)
        
        Returns:
        --------
        coefficients : np.ndarray
            小波系数
        freqs : np.ndarray
            频率数组
        power : np.ndarray
            功率谱
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        n_channels, n_timepoints = data.shape
        n_scales = len(self.scales)
        
        coefficients = np.zeros((n_channels, n_scales, n_timepoints), dtype=complex)
        
        for ch in range(n_channels):
            coef, _ = pywt.cwt(data[ch], self.scales, self.wavelet)
            coefficients[ch] = coef
        
        # 计算功率
        power = np.abs(coefficients) ** 2
        
        return coefficients, self.freqs, power
    
    def extract_band_power(self, data: np.ndarray, 
                          freq_bands: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
        """
        提取各频带功率
        
        Parameters:
        -----------
        data : np.ndarray
            输入信号
        freq_bands : dict
            频带定义字典
        
        Returns:
        --------
        band_powers : dict
            各频带功率字典
        """
        _, freqs, power = self.transform(data)
        
        band_powers = {}
        
        for band_name, (low_freq, high_freq) in freq_bands.items():
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(freq_mask):
                band_power = power[:, freq_mask, :].mean(axis=1)
            else:
                band_power = np.zeros((power.shape[0], power.shape[2]))
            
            band_powers[band_name] = band_power
        
        return band_powers


class STFTAnalyzer:
    """短时傅里叶变换分析器"""
    
    def __init__(self, window: str = 'hann', nperseg: int = 256, 
                 noverlap: Optional[int] = None, sampling_rate: float = 250.0):
        """
        初始化STFT分析器
        
        Parameters:
        -----------
        window : str
            窗函数类型
        nperseg : int
            每段长度
        noverlap : int, optional
            重叠长度
        sampling_rate : float
            采样率
        """
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        self.sampling_rate = sampling_rate
    
    def transform(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行短时傅里叶变换
        
        Parameters:
        -----------
        data : np.ndarray
            输入信号，形状为 (n_channels, n_timepoints) 或 (n_timepoints,)
        
        Returns:
        --------
        freqs : np.ndarray
            频率数组
        times : np.ndarray
            时间数组
        spectrogram : np.ndarray
            频谱图
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        n_channels, n_timepoints = data.shape
        
        # 调整nperseg以适应数据长度
        nperseg = min(self.nperseg, n_timepoints)
        noverlap = min(self.noverlap, nperseg - 1)
        
        spectrograms = []
        
        for ch in range(n_channels):
            freqs, times, Zxx = signal.stft(
                data[ch], 
                fs=self.sampling_rate,
                window=self.window,
                nperseg=nperseg,
                noverlap=noverlap
            )
            spectrograms.append(Zxx)
        
        spectrogram = np.array(spectrograms)
        
        return freqs, times, spectrogram
    
    def extract_band_power(self, data: np.ndarray, 
                          freq_bands: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
        """
        提取各频带功率
        
        Parameters:
        -----------
        data : np.ndarray
            输入信号
        freq_bands : dict
            频带定义字典
        
        Returns:
        --------
        band_powers : dict
            各频带功率字典
        """
        freqs, times, spectrogram = self.transform(data)
        power = np.abs(spectrogram) ** 2
        
        band_powers = {}
        
        for band_name, (low_freq, high_freq) in freq_bands.items():
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(freq_mask):
                band_power = power[:, freq_mask, :].mean(axis=1)
            else:
                band_power = np.zeros((power.shape[0], power.shape[2]))
            
            band_powers[band_name] = band_power
        
        return band_powers


class TimeFreqAnalyzer:
    """时频分析器主类"""
    
    def __init__(self, method: str = 'stft', sampling_rate: float = 250.0,
                 freq_bands: Dict[str, Tuple[float, float]] = None,
                 time_window_size: int = 50, overlap_ratio: float = 0.5):
        """
        初始化时频分析器
        
        Parameters:
        -----------
        method : str
            分析方法 ('stft', 'cwt')
        sampling_rate : float
            采样率
        freq_bands : dict, optional
            频带定义
        time_window_size : int
            时间窗口大小
        overlap_ratio : float
            重叠比例
        """
        self.method = method
        self.sampling_rate = sampling_rate
        self.time_window_size = time_window_size
        self.overlap_ratio = overlap_ratio
        
        if freq_bands is None:
            self.freq_bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 100)
            }
        else:
            self.freq_bands = freq_bands
        
        # 初始化分析器
        if method == 'stft':
            self.analyzer = STFTAnalyzer(sampling_rate=sampling_rate)
        elif method == 'cwt':
            self.analyzer = CWTAnalyzer(sampling_rate=sampling_rate)
        else:
            raise ValueError(f"未知的分析方法: {method}")
        
        self.units = []
        self.fitted = False
    
    def fit(self, data: np.ndarray) -> 'TimeFreqAnalyzer':
        """
        拟合时频分析器
        
        Parameters:
        -----------
        data : np.ndarray
            EEG数据，形状为 (n_channels, n_timepoints) 或 (n_samples, n_channels, n_timepoints)
        
        Returns:
        --------
        self : TimeFreqAnalyzer
        """
        if data.ndim == 3:
            # 多个样本，取第一个样本进行拟合
            data = data[0]
        
        # 构造时频单元
        self.units = self._construct_timefreq_units(data)
        
        self.fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        转换数据为时频特征
        
        Parameters:
        -----------
        data : np.ndarray
            EEG数据
        
        Returns:
        --------
        features : np.ndarray
            时频特征矩阵
        """
        if not self.fitted:
            raise ValueError("分析器尚未拟合，请先调用fit方法")
        
        if data.ndim == 3:
            # 多个样本
            n_samples = data.shape[0]
            all_features = []
            
            for i in range(n_samples):
                sample_features = self._extract_sample_features(data[i])
                all_features.append(sample_features)
            
            return np.array(all_features)
        else:
            # 单个样本
            return self._extract_sample_features(data)
    
    def _extract_sample_features(self, data: np.ndarray) -> np.ndarray:
        """
        提取单个样本的时频特征
        
        Parameters:
        -----------
        data : np.ndarray
            单个样本数据，形状为 (n_channels, n_timepoints)
        
        Returns:
        --------
        features : np.ndarray
            特征向量
        """
        all_features = []
        
        for unit in self.units:
            unit_features = unit.extract_features(data)
            all_features.append(unit_features)
        
        return np.concatenate(all_features)
    
    def _construct_timefreq_units(self, data: np.ndarray) -> List[TimeFreqUnit]:
        """
        构造时频单元
        
        Parameters:
        -----------
        data : np.ndarray
            EEG数据，形状为 (n_channels, n_timepoints)
        
        Returns:
        --------
        units : list
            时频单元列表
        """
        n_channels, n_timepoints = data.shape
        units = []
        
        # 计算时间窗口
        step_size = int(self.time_window_size * (1 - self.overlap_ratio))
        n_windows = max(1, (n_timepoints - self.time_window_size) // step_size + 1)
        
        unit_id = 0
        
        # 为每个频带、时间窗口和通道组合创建单元
        for freq_name, freq_band in self.freq_bands.items():
            for window_idx in range(n_windows):
                start_time = window_idx * step_size
                end_time = min(start_time + self.time_window_size, n_timepoints)
                
                if end_time - start_time < self.time_window_size // 2:
                    continue  # 跳过太短的窗口
                
                # 创建单通道单元
                for channel_idx in range(n_channels):
                    unit = TimeFreqUnit(
                        unit_id=f"tf_{freq_name}_{window_idx}_{channel_idx}",
                        channels=[channel_idx],
                        time_range=(start_time, end_time),
                        freq_band=freq_band,
                        freq_band_name=freq_name,
                        metadata={
                            'window_idx': window_idx,
                            'channel_idx': channel_idx
                        }
                    )
                    units.append(unit)
                    unit_id += 1
                
                # 创建多通道单元（通道组合）
                if n_channels > 1:
                    # 创建相邻通道组合
                    for ch_start in range(0, n_channels - 1, 4):  # 每4个通道一组
                        ch_end = min(ch_start + 4, n_channels)
                        channels = list(range(ch_start, ch_end))
                        
                        unit = TimeFreqUnit(
                            unit_id=f"tf_{freq_name}_{window_idx}_group_{ch_start}_{ch_end}",
                            channels=channels,
                            time_range=(start_time, end_time),
                            freq_band=freq_band,
                            freq_band_name=freq_name,
                            metadata={
                                'window_idx': window_idx,
                                'channel_group': (ch_start, ch_end),
                                'is_group': True
                            }
                        )
                        units.append(unit)
                        unit_id += 1
        
        return units
    
    def get_spectrogram(self, data: np.ndarray, channel: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取指定通道的频谱图
        
        Parameters:
        -----------
        data : np.ndarray
            EEG数据
        channel : int
            通道索引
        
        Returns:
        --------
        freqs : np.ndarray
            频率数组
        times : np.ndarray
            时间数组
        spectrogram : np.ndarray
            频谱图
        """
        if data.ndim == 3:
            data = data[0]  # 取第一个样本
        
        channel_data = data[channel:channel+1, :]
        
        if self.method == 'stft':
            freqs, times, spec = self.analyzer.transform(channel_data)
            return freqs, times, np.abs(spec[0]) ** 2
        elif self.method == 'cwt':
            _, freqs, power = self.analyzer.transform(channel_data)
            times = np.arange(data.shape[1]) / self.sampling_rate
            return freqs, times, power[0]
    
    def extract_connectivity_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取通道间连接性特征
        
        Parameters:
        -----------
        data : np.ndarray
            EEG数据
        
        Returns:
        --------
        connectivity_features : dict
            连接性特征字典
        """
        if data.ndim == 3:
            data = data[0]  # 取第一个样本
        
        n_channels = data.shape[0]
        connectivity_features = {}
        
        # 提取各频带功率
        band_powers = self.analyzer.extract_band_power(data, self.freq_bands)
        
        for band_name, power in band_powers.items():
            # 计算通道间相关性
            correlations = np.corrcoef(power)
            
            # 提取上三角矩阵（去除对角线）
            triu_indices = np.triu_indices(n_channels, k=1)
            connectivity_features[f'{band_name}_correlation'] = correlations[triu_indices]
            
            # 计算相位同步
            if hasattr(self.analyzer, 'transform'):
                if self.method == 'stft':
                    _, _, spec = self.analyzer.transform(data)
                    phases = np.angle(spec)
                    
                    # 计算相位锁定值 (PLV)
                    plv_matrix = np.zeros((n_channels, n_channels))
                    for i in range(n_channels):
                        for j in range(i+1, n_channels):
                            phase_diff = phases[i] - phases[j]
                            plv = np.abs(np.mean(np.exp(1j * phase_diff), axis=-1)).mean()
                            plv_matrix[i, j] = plv
                            plv_matrix[j, i] = plv
                    
                    connectivity_features[f'{band_name}_plv'] = plv_matrix[triu_indices]
        
        return connectivity_features
    
    def get_units(self) -> List[TimeFreqUnit]:
        """
        获取构造的时频单元
        
        Returns:
        --------
        units : list
            时频单元列表
        """
        return self.units
    
    def plot_spectrogram(self, data: np.ndarray, channel: int = 0, 
                        title: str = None):
        """
        绘制频谱图
        
        Parameters:
        -----------
        data : np.ndarray
            EEG数据
        channel : int
            通道索引
        title : str, optional
            图标题
        """
        try:
            import matplotlib.pyplot as plt
            
            freqs, times, spec = self.get_spectrogram(data, channel)
            
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(times, freqs, 10 * np.log10(spec), shading='gouraud')
            plt.colorbar(label='Power (dB)')
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (s)')
            
            if title:
                plt.title(title)
            else:
                plt.title(f'Spectrogram - Channel {channel}')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib未安装，无法绘制图形")


def create_timefreq_analyzer(method: str = 'stft', **kwargs) -> TimeFreqAnalyzer:
    """
    创建时频分析器的便捷函数
    
    Parameters:
    -----------
    method : str
        分析方法
    **kwargs : dict
        其他参数
    
    Returns:
    --------
    analyzer : TimeFreqAnalyzer
        时频分析器
    """
    return TimeFreqAnalyzer(method=method, **kwargs)


if __name__ == "__main__":
    # 示例使用
    print("时频分析模块已加载")
    
    # 创建示例数据
    np.random.seed(42)
    sampling_rate = 250.0
    duration = 4.0  # 4秒
    n_channels = 8
    n_timepoints = int(sampling_rate * duration)
    
    # 生成包含多个频率成分的信号
    t = np.linspace(0, duration, n_timepoints)
    data = np.zeros((n_channels, n_timepoints))
    
    for ch in range(n_channels):
        # 添加不同频率成分
        data[ch] = (np.sin(2 * np.pi * 10 * t) +  # alpha
                   0.5 * np.sin(2 * np.pi * 20 * t) +  # beta
                   0.3 * np.sin(2 * np.pi * 40 * t) +  # gamma
                   0.1 * np.random.randn(n_timepoints))  # noise
    
    # 创建分析器
    analyzer = TimeFreqAnalyzer(method='stft', sampling_rate=sampling_rate)
    
    # 拟合和转换
    analyzer.fit(data)
    features = analyzer.transform(data)
    
    print(f"时频特征形状: {features.shape}")
    print(f"构造的时频单元数量: {len(analyzer.get_units())}")
    
    # 提取连接性特征
    connectivity = analyzer.extract_connectivity_features(data)
    print(f"连接性特征: {list(connectivity.keys())}")