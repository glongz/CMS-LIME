#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生物标志物统计分析可视化工具

对已发现的生物标志物进行全面的统计验证和可视化分析

Author: CMS-LIME Framework
Date: 2025
"""

import numpy as np
import json
import os
import glob
import re
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, Counter
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import pandas as pd
from scipy import stats
import gc
from pathlib import Path

# 设置英文字体
def setup_english_font():
    """设置英文字体"""
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 禁用字体警告
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 初始化字体设置
setup_english_font()

# 设置seaborn风格
sns.set_style("whitegrid")
sns.set_palette("husl")


class BiomarkerData:
    """生物标志物数据类"""
    
    def __init__(self, filepath: str):
        """
        初始化生物标志物数据
        
        Parameters:
        -----------
        filepath : str
            npz文件路径
        """
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.data = None
        self.meta = None
        self.sample_index = None
        self.batch_index = None
        self.primitive_type = None
        self.primitive_id = None
        self.channels = None
        self.time_range = None
        self.shape = None
        self.dtype = None
        
        self._load_data()
        self._parse_filename()
    
    def _load_data(self):
        """加载npz文件数据"""
        try:
            npz = np.load(self.filepath, allow_pickle=False)
            self.data = npz["data"]
            self.meta = json.loads(npz["meta_json"].item())
            
            # 提取元数据
            self.sample_index = self.meta.get("sample_index")
            self.primitive_type = self.meta.get("primitive_type")
            self.primitive_id = self.meta.get("primitive_id")
            self.channels = self.meta.get("channels", [])
            self.time_range = self.meta.get("time_range", [0, 0])
            self.shape = self.meta.get("shape")
            self.dtype = self.meta.get("dtype")
            
        except Exception as e:
            print(f"加载文件 {self.filepath} 失败: {e}")
            raise
    
    def _parse_filename(self):
        """解析文件名提取样本和批次信息"""
        # 文件名格式: s000207_b0000_shapelet_shapelet_63.npz
        pattern = r's(\d+)_b(\d+)_(\w+)_'
        match = re.search(pattern, self.filename)
        if match:
            self.sample_index = int(match.group(1))
            self.batch_index = int(match.group(2))
            # primitive_type 可能已经在meta中，但也可以从文件名提取
    
    def get_duration(self) -> int:
        """获取持续时间（样本点数）"""
        if self.time_range and len(self.time_range) >= 2:
            return self.time_range[1] - self.time_range[0]
        return 0
    
    def get_statistics(self) -> Dict[str, float]:
        """获取数据统计信息"""
        if self.data is None:
            return {}
        
        data_flat = self.data.flatten()
        return {
            'mean': float(np.mean(data_flat)),
            'std': float(np.std(data_flat)),
            'min': float(np.min(data_flat)),
            'max': float(np.max(data_flat)),
            'median': float(np.median(data_flat)),
            'skewness': float(stats.skew(data_flat)),
            'kurtosis': float(stats.kurtosis(data_flat)),
            'duration': self.get_duration()
        }


def load_biomarker_data(data_dir: str) -> List[BiomarkerData]:
    """
    加载所有生物标志物数据
    
    Parameters:
    -----------
    data_dir : str
        数据目录路径
    
    Returns:
    --------
    biomarkers : List[BiomarkerData]
        生物标志物数据列表
    """
    print(f"正在加载数据从: {data_dir}")
    
    # 查找所有npz文件
    pattern = os.path.join(data_dir, "*.npz")
    files = glob.glob(pattern)
    
    print(f"找到 {len(files)} 个文件")
    
    biomarkers = []
    failed_files = []
    
    for filepath in files:
        try:
            biomarker = BiomarkerData(filepath)
            biomarkers.append(biomarker)
        except Exception as e:
            failed_files.append((filepath, str(e)))
    
    if failed_files:
        print(f"警告: {len(failed_files)} 个文件加载失败")
        for filepath, error in failed_files[:5]:  # 只显示前5个错误
            print(f"  {os.path.basename(filepath)}: {error}")
    
    print(f"成功加载 {len(biomarkers)} 个生物标志物")
    return biomarkers


def basic_statistics(biomarkers: List[BiomarkerData]) -> Dict[str, Any]:
    """
    基本统计分析
    
    Parameters:
    -----------
    biomarkers : List[BiomarkerData]
        生物标志物数据列表
    
    Returns:
    --------
    stats_dict : Dict[str, Any]
        统计结果字典
    """
    stats_dict = {}
    
    # 总数量
    stats_dict['total_count'] = len(biomarkers)
    
    # 类型分布
    type_counts = Counter([b.primitive_type for b in biomarkers])
    stats_dict['type_distribution'] = dict(type_counts)
    
    # 样本分布
    sample_counts = Counter([b.sample_index for b in biomarkers])
    stats_dict['sample_distribution'] = dict(sample_counts)
    stats_dict['unique_samples'] = len(sample_counts)
    
    # 批次分布
    batch_counts = Counter([b.batch_index for b in biomarkers])
    stats_dict['batch_distribution'] = dict(batch_counts)
    stats_dict['unique_batches'] = len(batch_counts)
    
    # 每个样本的平均标志物数量
    if sample_counts:
        stats_dict['avg_biomarkers_per_sample'] = np.mean(list(sample_counts.values()))
    
    return stats_dict


def temporal_analysis(biomarkers: List[BiomarkerData]) -> Dict[str, Any]:
    """
    时间特征分析
    
    Parameters:
    -----------
    biomarkers : List[BiomarkerData]
        生物标志物数据列表
    
    Returns:
    --------
    temporal_stats : Dict[str, Any]
        时间特征统计结果
    """
    temporal_stats = {}
    
    # 持续时间分布
    durations = [b.get_duration() for b in biomarkers]
    temporal_stats['durations'] = durations
    temporal_stats['duration_mean'] = np.mean(durations) if durations else 0
    temporal_stats['duration_std'] = np.std(durations) if durations else 0
    temporal_stats['duration_min'] = np.min(durations) if durations else 0
    temporal_stats['duration_max'] = np.max(durations) if durations else 0
    
    # 起始时间分布
    start_times = [b.time_range[0] if b.time_range and len(b.time_range) >= 1 else 0 
                   for b in biomarkers]
    temporal_stats['start_times'] = start_times
    temporal_stats['start_time_mean'] = np.mean(start_times) if start_times else 0
    
    # 结束时间分布
    end_times = [b.time_range[1] if b.time_range and len(b.time_range) >= 2 else 0 
                 for b in biomarkers]
    temporal_stats['end_times'] = end_times
    temporal_stats['end_time_mean'] = np.mean(end_times) if end_times else 0
    
    # 时间位置分类（早期/中期/晚期）
    if start_times:
        max_time = max(end_times) if end_times else max(start_times)
        if max_time > 0:
            early_threshold = max_time / 3
            late_threshold = max_time * 2 / 3
            
            early_count = sum(1 for st in start_times if st < early_threshold)
            mid_count = sum(1 for st in start_times if early_threshold <= st < late_threshold)
            late_count = sum(1 for st in start_times if st >= late_threshold)
            
            temporal_stats['time_position'] = {
                'early': early_count,
                'mid': mid_count,
                'late': late_count
            }
    
    return temporal_stats


def spatial_analysis(biomarkers: List[BiomarkerData]) -> Dict[str, Any]:
    """
    空间特征分析
    
    Parameters:
    -----------
    biomarkers : List[BiomarkerData]
        生物标志物数据列表
    
    Returns:
    --------
    spatial_stats : Dict[str, Any]
        空间特征统计结果
    """
    spatial_stats = {}
    
    # 通道使用频率
    channel_usage = Counter()
    for b in biomarkers:
        if b.channels:
            for ch in b.channels:
                channel_usage[ch] += 1
    
    spatial_stats['channel_usage'] = dict(channel_usage)
    spatial_stats['unique_channels'] = len(channel_usage)
    spatial_stats['most_used_channels'] = dict(channel_usage.most_common(10))
    
    # 多通道标志物统计
    multi_channel_count = sum(1 for b in biomarkers if b.channels and len(b.channels) > 1)
    spatial_stats['multi_channel_count'] = multi_channel_count
    spatial_stats['multi_channel_ratio'] = multi_channel_count / len(biomarkers) if biomarkers else 0
    
    # 通道-类型关联
    channel_type_map = defaultdict(lambda: defaultdict(int))
    for b in biomarkers:
        if b.channels and b.primitive_type:
            for ch in b.channels:
                channel_type_map[ch][b.primitive_type] += 1
    
    spatial_stats['channel_type_association'] = {
        str(ch): dict(type_counts) 
        for ch, type_counts in channel_type_map.items()
    }
    
    return spatial_stats


def quality_assessment(biomarkers: List[BiomarkerData]) -> Dict[str, Any]:
    """
    数据质量评估
    
    Parameters:
    -----------
    biomarkers : List[BiomarkerData]
        生物标志物数据列表
    
    Returns:
    --------
    quality_stats : Dict[str, Any]
        质量评估结果
    """
    quality_stats = {}
    
    # 数据完整性检查
    complete_count = sum(1 for b in biomarkers if b.data is not None and b.meta is not None)
    quality_stats['completeness'] = complete_count / len(biomarkers) if biomarkers else 0
    
    # 异常值检测（基于统计信息）
    all_stats = [b.get_statistics() for b in biomarkers]
    
    if all_stats:
        # 提取各项统计指标
        means = [s.get('mean', 0) for s in all_stats]
        stds = [s.get('std', 0) for s in all_stats]
        
        # 使用3-sigma规则检测异常值
        if means and stds:
            mean_mean = np.mean(means)
            std_mean = np.std(means)
            
            outliers = []
            for i, b in enumerate(biomarkers):
                stats_i = all_stats[i]
                mean_i = stats_i.get('mean', 0)
                if abs(mean_i - mean_mean) > 3 * std_mean:
                    outliers.append(i)
            
            quality_stats['outlier_count'] = len(outliers)
            quality_stats['outlier_ratio'] = len(outliers) / len(biomarkers) if biomarkers else 0
        
        # 数据分布特征汇总
        quality_stats['statistics_summary'] = {
            'mean_of_means': np.mean(means) if means else 0,
            'std_of_means': np.std(means) if means else 0,
            'mean_of_stds': np.mean(stds) if stds else 0,
            'std_of_stds': np.std(stds) if stds else 0
        }
    
    # 缺失值统计（检查是否有NaN或Inf）
    nan_count = 0
    inf_count = 0
    for b in biomarkers:
        if b.data is not None:
            if np.isnan(b.data).any():
                nan_count += 1
            if np.isinf(b.data).any():
                inf_count += 1
    
    quality_stats['nan_count'] = nan_count
    quality_stats['inf_count'] = inf_count
    
    return quality_stats


def pattern_analysis(biomarkers: List[BiomarkerData]) -> Dict[str, Any]:
    """
    模式分析
    
    Parameters:
    -----------
    biomarkers : List[BiomarkerData]
        生物标志物数据列表
    
    Returns:
    --------
    pattern_stats : Dict[str, Any]
        模式分析结果
    """
    pattern_stats = {}
    
    # 按类型分组
    by_type = defaultdict(list)
    for b in biomarkers:
        if b.primitive_type:
            by_type[b.primitive_type].append(b)
    
    pattern_stats['by_type'] = {k: len(v) for k, v in by_type.items()}
    
    # Shapelet模式特征
    shapelets = [b for b in biomarkers if b.primitive_type == 'shapelet']
    if shapelets:
        shapelet_lengths = [len(b.data.flatten()) for b in shapelets if b.data is not None]
        pattern_stats['shapelet_lengths'] = {
            'mean': np.mean(shapelet_lengths) if shapelet_lengths else 0,
            'std': np.std(shapelet_lengths) if shapelet_lengths else 0,
            'min': np.min(shapelet_lengths) if shapelet_lengths else 0,
            'max': np.max(shapelet_lengths) if shapelet_lengths else 0
        }
        
        # Shapelet波形特征（均值、方差等）
        shapelet_stats = [b.get_statistics() for b in shapelets]
        pattern_stats['shapelet_waveform_features'] = {
            'mean_amplitude': np.mean([s.get('mean', 0) for s in shapelet_stats]),
            'mean_variance': np.mean([s.get('std', 0)**2 for s in shapelet_stats]),
            'mean_skewness': np.mean([s.get('skewness', 0) for s in shapelet_stats]),
            'mean_kurtosis': np.mean([s.get('kurtosis', 0) for s in shapelet_stats])
        }
    
    # Timefreq特征
    timefreqs = [b for b in biomarkers if b.primitive_type == 'timefreq']
    if timefreqs:
        pattern_stats['timefreq_count'] = len(timefreqs)
        timefreq_stats = [b.get_statistics() for b in timefreqs]
        pattern_stats['timefreq_features'] = {
            'mean_amplitude': np.mean([s.get('mean', 0) for s in timefreq_stats]),
            'mean_variance': np.mean([s.get('std', 0)**2 for s in timefreq_stats])
        }
    
    # Microstate特征
    microstates = [b for b in biomarkers if b.primitive_type == 'microstate']
    if microstates:
        pattern_stats['microstate_count'] = len(microstates)
    
    return pattern_stats


def validation_analysis(biomarkers: List[BiomarkerData]) -> Dict[str, Any]:
    """
    验证性分析
    
    Parameters:
    -----------
    biomarkers : List[BiomarkerData]
        生物标志物数据列表
    
    Returns:
    --------
    validation_stats : Dict[str, Any]
        验证性分析结果
    """
    validation_stats = {}
    
    # 跨样本一致性检查（相同primitive_id在不同样本中的出现）
    primitive_id_samples = defaultdict(set)
    for b in biomarkers:
        if b.primitive_id and b.sample_index is not None:
            primitive_id_samples[b.primitive_id].add(b.sample_index)
    
    # 计算每个primitive_id出现的样本数
    id_sample_counts = {pid: len(samples) for pid, samples in primitive_id_samples.items()}
    
    # 重复出现的标志物（在多个样本中出现）
    repeated_ids = {pid: count for pid, count in id_sample_counts.items() if count > 1}
    validation_stats['repeated_biomarkers'] = len(repeated_ids)
    validation_stats['repeated_biomarker_ratio'] = len(repeated_ids) / len(id_sample_counts) if id_sample_counts else 0
    
    # 最常见的标志物（跨样本出现频率最高）
    if id_sample_counts:
        most_common = sorted(id_sample_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        validation_stats['most_common_biomarkers'] = dict(most_common)
    
    # 标志物重复性统计（相同primitive_id的总出现次数）
    primitive_id_counts = Counter([b.primitive_id for b in biomarkers if b.primitive_id])
    validation_stats['primitive_id_frequency'] = dict(primitive_id_counts.most_common(20))
    
    # 样本间标志物数量一致性
    sample_counts = Counter([b.sample_index for b in biomarkers])
    if sample_counts:
        counts_list = list(sample_counts.values())
        validation_stats['sample_consistency'] = {
            'mean': np.mean(counts_list),
            'std': np.std(counts_list),
            'cv': np.std(counts_list) / np.mean(counts_list) if np.mean(counts_list) > 0 else 0  # 变异系数
        }
    
    return validation_stats


def plot_overview(biomarkers: List[BiomarkerData], save_dir: str):
    """
    生成总览图
    
    Parameters:
    -----------
    biomarkers : List[BiomarkerData]
        生物标志物数据列表
    save_dir : str
        保存目录
    """
    print("生成总览图...")
    
    # 确保字体设置正确
    setup_english_font()
    
    # 计算统计数据
    basic_stats = basic_statistics(biomarkers)
    temporal_stats = temporal_analysis(biomarkers)
    spatial_stats = spatial_analysis(biomarkers)
    
    # 创建图形
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('Biomarker Statistical Analysis Overview', fontsize=18, fontweight='bold', y=0.98)
    
    # 1. 类型分布饼图
    ax1 = fig.add_subplot(gs[0, 0])
    type_dist = basic_stats['type_distribution']
    if type_dist:
        colors = {'shapelet': '#4ECDC4', 'timefreq': '#45B7D1', 'microstate': '#FF6B6B'}
        type_colors = [colors.get(t, '#96CEB4') for t in type_dist.keys()]
        ax1.pie(type_dist.values(), labels=type_dist.keys(), autopct='%1.1f%%',
               colors=type_colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax1.set_title('Type Distribution', fontsize=14, fontweight='bold', pad=15)
    
    # 2. 样本分布柱状图
    ax2 = fig.add_subplot(gs[0, 1])
    sample_dist = basic_stats['sample_distribution']
    if sample_dist:
        samples = sorted(sample_dist.keys())[:20]  # 只显示前20个样本
        counts = [sample_dist[s] for s in samples]
        ax2.bar(range(len(samples)), counts, color='#4ECDC4', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Biomarkers', fontsize=12, fontweight='bold')
        ax2.set_title(f'Sample Distribution (Top 20 of {len(sample_dist)} samples)', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(samples)))
        ax2.set_xticklabels([f's{s}' for s in samples], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 持续时间分布直方图
    ax3 = fig.add_subplot(gs[0, 2])
    durations = temporal_stats['durations']
    if durations:
        ax3.hist(durations, bins=30, color='#45B7D1', alpha=0.7, edgecolor='black')
        ax3.axvline(temporal_stats['duration_mean'], color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {temporal_stats["duration_mean"]:.1f}')
        ax3.set_xlabel('Duration (samples)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('Duration Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 通道使用热图
    ax4 = fig.add_subplot(gs[1, :2])
    channel_usage = spatial_stats['channel_usage']
    if channel_usage:
        channels = sorted(channel_usage.keys())
        usage_counts = [channel_usage[ch] for ch in channels]
        
        # 创建热图数据
        max_ch = max(channels) if channels else 0
        heatmap_data = np.zeros((1, max_ch + 1))
        for ch, count in channel_usage.items():
            heatmap_data[0, ch] = count
        
        im = ax4.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax4.set_xlabel('Channel Index', fontsize=12, fontweight='bold')
        ax4.set_ylabel('', fontsize=12)
        ax4.set_title('Channel Usage Frequency Heatmap', fontsize=14, fontweight='bold')
        ax4.set_yticks([])
        plt.colorbar(im, ax=ax4, label='Usage Count')
    
    # 5. 时间位置分布
    ax5 = fig.add_subplot(gs[1, 2])
    time_position = temporal_stats.get('time_position', {})
    if time_position:
        positions = list(time_position.keys())
        counts = list(time_position.values())
        colors_pos = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        ax5.bar(positions, counts, color=colors_pos[:len(positions)], alpha=0.8, edgecolor='black')
        ax5.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax5.set_title('Temporal Position Distribution', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 统计摘要表格
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    summary_data = [
        ['Total Biomarkers', f"{basic_stats['total_count']}"],
        ['Unique Samples', f"{basic_stats['unique_samples']}"],
        ['Unique Batches', f"{basic_stats['unique_batches']}"],
        ['Avg Biomarkers per Sample', f"{basic_stats.get('avg_biomarkers_per_sample', 0):.2f}"],
        ['Mean Duration', f"{temporal_stats['duration_mean']:.1f} samples"],
        ['Unique Channels', f"{spatial_stats['unique_channels']}"],
        ['Multi-channel Ratio', f"{spatial_stats['multi_channel_ratio']*100:.1f}%"]
    ]
    
    table = ax6.table(cellText=summary_data,
                     colLabels=['Statistic', 'Value'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.4, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # 样式化表格
    for i in range(len(summary_data) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax6.set_title('Statistical Summary', fontsize=14, fontweight='bold', pad=20)
    
    # 使用subplots_adjust替代tight_layout以避免警告
    plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.06, hspace=0.3, wspace=0.3)
    save_path = os.path.join(save_dir, 'biomarker_overview.png')
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png', facecolor='white')
    except Exception as e:
        print(f"保存总览图时出错: {e}")
    plt.close('all')
    gc.collect()
    
    print(f"总览图已保存: {save_path}")


def plot_individual_biomarker(biomarker: BiomarkerData, save_path: str, top_n: int = 20):
    """
    可视化单个生物标志物
    
    Parameters:
    -----------
    biomarker : BiomarkerData
        生物标志物数据
    save_path : str
        保存路径
    top_n : int
        如果保存多个，显示前N个
    """
    if biomarker.data is None:
        return
    
    # 确保字体设置正确
    setup_english_font()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Biomarker Details: {biomarker.primitive_id}', fontsize=16, fontweight='bold')
    
    # 1. 时间序列波形图
    ax1 = axes[0]
    data_flat = biomarker.data.flatten()
    time_points = np.arange(len(data_flat))
    
    ax1.plot(time_points, data_flat, 'b-', linewidth=2, alpha=0.8, label='Waveform')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # 标注统计信息
    stats_dict = biomarker.get_statistics()
    mean_val = stats_dict.get('mean', 0)
    std_val = stats_dict.get('std', 0)
    
    ax1.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5, 
               label=f'Mean: {mean_val:.3f}')
    ax1.fill_between(time_points, mean_val - std_val, mean_val + std_val, 
                     alpha=0.2, color='red', label=f'±1 Std Dev')
    
    ax1.set_xlabel('Time Points', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax1.set_title(f'{biomarker.primitive_type.upper()} Waveform - Channel {biomarker.channels}', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. 统计信息展示
    ax2 = axes[1]
    ax2.axis('off')
    
    info_text = f"""
    Basic Information:
    - Sample Index: {biomarker.sample_index}
    - Batch Index: {biomarker.batch_index}
    - Type: {biomarker.primitive_type}
    - Primitive ID: {biomarker.primitive_id}
    - Channels: {biomarker.channels}
    - Time Range: {biomarker.time_range}
    - Duration: {stats_dict.get('duration', 0)} samples
    
    Statistical Features:
    - Mean: {stats_dict.get('mean', 0):.4f}
    - Std Dev: {stats_dict.get('std', 0):.4f}
    - Min: {stats_dict.get('min', 0):.4f}
    - Max: {stats_dict.get('max', 0):.4f}
    - Median: {stats_dict.get('median', 0):.4f}
    - Skewness: {stats_dict.get('skewness', 0):.4f}
    - Kurtosis: {stats_dict.get('kurtosis', 0):.4f}
    """
    
    ax2.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 使用subplots_adjust替代tight_layout
    plt.subplots_adjust(left=0.1, right=0.95, top=0.93, bottom=0.1, hspace=0.3)
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png', facecolor='white')
    except Exception as e:
        print(f"保存单个标志物图时出错: {e}")
    plt.close('all')
    gc.collect()


def plot_comparison(biomarkers: List[BiomarkerData], save_dir: str):
    """
    生成对比分析图
    
    Parameters:
    -----------
    biomarkers : List[BiomarkerData]
        生物标志物数据列表
    save_dir : str
        保存目录
    """
    print("生成对比分析图...")
    
    # 确保字体设置正确
    setup_english_font()
    
    # 按类型分组
    by_type = defaultdict(list)
    for b in biomarkers:
        if b.primitive_type:
            by_type[b.primitive_type].append(b)
    
    # 按样本分组
    by_sample = defaultdict(list)
    for b in biomarkers:
        if b.sample_index is not None:
            by_sample[b.sample_index].append(b)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    fig.suptitle('Biomarker Comparison Analysis', fontsize=18, fontweight='bold', y=0.98)
    
    # 1. 不同类型标志物数量对比
    ax1 = fig.add_subplot(gs[0, 0])
    if by_type:
        types = list(by_type.keys())
        counts = [len(by_type[t]) for t in types]
        colors = {'shapelet': '#4ECDC4', 'timefreq': '#45B7D1', 'microstate': '#FF6B6B'}
        type_colors = [colors.get(t, '#96CEB4') for t in types]
        bars = ax1.bar(types, counts, color=type_colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Biomarker Count by Type', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 不同样本标志物数量对比（前20个样本）
    ax2 = fig.add_subplot(gs[0, 1])
    if by_sample:
        samples = sorted(by_sample.keys())[:20]
        counts = [len(by_sample[s]) for s in samples]
        ax2.bar(range(len(samples)), counts, color='#4ECDC4', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Biomarkers', fontsize=12, fontweight='bold')
        ax2.set_title(f'Biomarker Count by Sample (Top 20)', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(samples)))
        ax2.set_xticklabels([f's{s}' for s in samples], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 不同类型标志物的平均持续时间对比
    ax3 = fig.add_subplot(gs[1, 0])
    if by_type:
        type_durations = {}
        for t, b_list in by_type.items():
            durations = [b.get_duration() for b in b_list]
            if durations:
                type_durations[t] = np.mean(durations)
        
        if type_durations:
            types = list(type_durations.keys())
            avg_durations = list(type_durations.values())
            colors = {'shapelet': '#4ECDC4', 'timefreq': '#45B7D1', 'microstate': '#FF6B6B'}
            type_colors = [colors.get(t, '#96CEB4') for t in types]
            bars = ax3.bar(types, avg_durations, color=type_colors, alpha=0.8, edgecolor='black')
            ax3.set_ylabel('Mean Duration (samples)', fontsize=12, fontweight='bold')
            ax3.set_title('Mean Duration by Type', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar, dur in zip(bars, avg_durations):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{dur:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Top 20 标志物（按出现频率）
    ax4 = fig.add_subplot(gs[1, 1])
    primitive_id_counts = Counter([b.primitive_id for b in biomarkers if b.primitive_id])
    top_20 = primitive_id_counts.most_common(20)
    
    if top_20:
        ids = [item[0][:15] + '...' if len(item[0]) > 15 else item[0] for item in top_20]
        counts = [item[1] for item in top_20]
        
        ax4.barh(range(len(ids)), counts, color='#45B7D1', alpha=0.8, edgecolor='black')
        ax4.set_yticks(range(len(ids)))
        ax4.set_yticklabels(ids, fontsize=9)
        ax4.set_xlabel('Occurrence Count', fontsize=12, fontweight='bold')
        ax4.set_title('Top 20 Biomarkers (by Frequency)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
    
    # 使用subplots_adjust替代tight_layout
    plt.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.08, hspace=0.3, wspace=0.3)
    save_path = os.path.join(save_dir, 'biomarker_comparison.png')
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png', facecolor='white')
    except Exception as e:
        print(f"保存对比分析图时出错: {e}")
    plt.close('all')
    gc.collect()
    
    print(f"对比分析图已保存: {save_path}")


def plot_top_biomarkers(biomarkers: List[BiomarkerData], save_dir: str, top_n: int = 10):
    """
    可视化Top N标志物
    
    Parameters:
    -----------
    biomarkers : List[BiomarkerData]
        生物标志物数据列表
    save_dir : str
        保存目录
    top_n : int
        显示前N个标志物
    """
    print(f"生成Top {top_n}标志物可视化...")
    
    # 确保字体设置正确
    setup_english_font()
    
    # 按primitive_id分组，选择出现频率最高的
    primitive_id_groups = defaultdict(list)
    for b in biomarkers:
        if b.primitive_id:
            primitive_id_groups[b.primitive_id].append(b)
    
    # 选择出现频率最高的N个
    sorted_ids = sorted(primitive_id_groups.items(), key=lambda x: len(x[1]), reverse=True)[:top_n]
    
    if not sorted_ids:
        print("没有找到可用的标志物")
        return
    
    # 创建图形（每个标志物一个子图）
    n_cols = 3
    n_rows = (top_n + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(18, 6 * n_rows))
    fig.suptitle(f'Top {top_n} Biomarkers Visualization', fontsize=18, fontweight='bold', y=0.995)
    
    for idx, (primitive_id, b_list) in enumerate(sorted_ids):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        
        # 选择第一个作为代表
        b = b_list[0]
        if b.data is not None:
            data_flat = b.data.flatten()
            time_points = np.arange(len(data_flat))
            
            ax.plot(time_points, data_flat, 'b-', linewidth=2, alpha=0.8)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            stats_dict = b.get_statistics()
            mean_val = stats_dict.get('mean', 0)
            ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5)
            
            ax.set_title(f'{primitive_id}\n(Occurred {len(b_list)} times)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time Points', fontsize=10)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.grid(True, alpha=0.3)
    
    # 使用subplots_adjust替代tight_layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.96, bottom=0.05, hspace=0.4, wspace=0.3)
    save_path = os.path.join(save_dir, f'top_{top_n}_biomarkers.png')
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png', facecolor='white')
    except Exception as e:
        print(f"保存Top {top_n}标志物图时出错: {e}")
    plt.close('all')
    gc.collect()
    
    print(f"Top {top_n}标志物可视化已保存: {save_path}")


def generate_report(biomarkers: List[BiomarkerData], save_dir: str):
    """
    生成综合统计报告
    
    Parameters:
    -----------
    biomarkers : List[BiomarkerData]
        生物标志物数据列表
    save_dir : str
        保存目录
    """
    print("生成综合统计报告...")
    
    # 执行所有分析
    basic_stats = basic_statistics(biomarkers)
    temporal_stats = temporal_analysis(biomarkers)
    spatial_stats = spatial_analysis(biomarkers)
    quality_stats = quality_assessment(biomarkers)
    pattern_stats = pattern_analysis(biomarkers)
    validation_stats = validation_analysis(biomarkers)
    
    # 生成报告文本
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("生物标志物统计分析报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    report_lines.append("1. 基本统计")
    report_lines.append("-" * 80)
    report_lines.append(f"  总标志物数量: {basic_stats['total_count']}")
    report_lines.append(f"  唯一样本数: {basic_stats['unique_samples']}")
    report_lines.append(f"  唯一批次数: {basic_stats['unique_batches']}")
    report_lines.append(f"  平均每样本标志物数: {basic_stats.get('avg_biomarkers_per_sample', 0):.2f}")
    report_lines.append("")
    report_lines.append("  类型分布:")
    for t, count in basic_stats['type_distribution'].items():
        report_lines.append(f"    {t}: {count} ({count/basic_stats['total_count']*100:.1f}%)")
    report_lines.append("")
    
    report_lines.append("2. 时间特征分析")
    report_lines.append("-" * 80)
    report_lines.append(f"  平均持续时间: {temporal_stats['duration_mean']:.1f} 样本点")
    report_lines.append(f"  持续时间标准差: {temporal_stats['duration_std']:.1f}")
    report_lines.append(f"  最短持续时间: {temporal_stats['duration_min']}")
    report_lines.append(f"  最长持续时间: {temporal_stats['duration_max']}")
    if 'time_position' in temporal_stats:
        tp = temporal_stats['time_position']
        report_lines.append(f"  时间位置分布: 早期={tp.get('early', 0)}, 中期={tp.get('mid', 0)}, 晚期={tp.get('late', 0)}")
    report_lines.append("")
    
    report_lines.append("3. 空间特征分析")
    report_lines.append("-" * 80)
    report_lines.append(f"  唯一通道数: {spatial_stats['unique_channels']}")
    report_lines.append(f"  多通道标志物数量: {spatial_stats['multi_channel_count']}")
    report_lines.append(f"  多通道标志物比例: {spatial_stats['multi_channel_ratio']*100:.1f}%")
    report_lines.append("")
    report_lines.append("  使用频率最高的10个通道:")
    for ch, count in list(spatial_stats['most_used_channels'].items())[:10]:
        report_lines.append(f"    通道 {ch}: {count} 次")
    report_lines.append("")
    
    report_lines.append("4. 数据质量评估")
    report_lines.append("-" * 80)
    report_lines.append(f"  数据完整性: {quality_stats['completeness']*100:.1f}%")
    report_lines.append(f"  异常值数量: {quality_stats.get('outlier_count', 0)}")
    report_lines.append(f"  异常值比例: {quality_stats.get('outlier_ratio', 0)*100:.1f}%")
    report_lines.append(f"  包含NaN的文件数: {quality_stats.get('nan_count', 0)}")
    report_lines.append(f"  包含Inf的文件数: {quality_stats.get('inf_count', 0)}")
    report_lines.append("")
    
    report_lines.append("5. 模式分析")
    report_lines.append("-" * 80)
    if 'shapelet_lengths' in pattern_stats:
        sl = pattern_stats['shapelet_lengths']
        report_lines.append(f"  Shapelet平均长度: {sl['mean']:.1f} 样本点")
        report_lines.append(f"  Shapelet长度范围: {sl['min']:.0f} - {sl['max']:.0f}")
    if 'shapelet_waveform_features' in pattern_stats:
        swf = pattern_stats['shapelet_waveform_features']
        report_lines.append(f"  Shapelet平均幅值: {swf['mean_amplitude']:.4f}")
        report_lines.append(f"  Shapelet平均方差: {swf['mean_variance']:.4f}")
    report_lines.append("")
    
    report_lines.append("6. 验证性分析")
    report_lines.append("-" * 80)
    report_lines.append(f"  跨样本重复的标志物数: {validation_stats['repeated_biomarkers']}")
    report_lines.append(f"  重复标志物比例: {validation_stats['repeated_biomarker_ratio']*100:.1f}%")
    if 'sample_consistency' in validation_stats:
        sc = validation_stats['sample_consistency']
        report_lines.append(f"  样本间一致性 (变异系数): {sc['cv']:.3f}")
    report_lines.append("")
    report_lines.append("  最常见的10个标志物 (跨样本出现):")
    if 'most_common_biomarkers' in validation_stats:
        for pid, count in list(validation_stats['most_common_biomarkers'].items())[:10]:
            report_lines.append(f"    {pid}: {count} 个样本")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    
    # 保存报告
    report_text = "\n".join(report_lines)
    report_path = os.path.join(save_dir, 'biomarker_statistics_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # 同时打印到控制台
    print("\n" + report_text)
    print(f"\n报告已保存: {report_path}")
    
    # 保存CSV统计摘要
    csv_data = {
        '统计项': [
            '总标志物数量', '唯一样本数', '唯一批次数', '平均每样本标志物数',
            '平均持续时间', '唯一通道数', '多通道标志物比例',
            '数据完整性', '异常值比例', '重复标志物比例'
        ],
        '数值': [
            basic_stats['total_count'],
            basic_stats['unique_samples'],
            basic_stats['unique_batches'],
            f"{basic_stats.get('avg_biomarkers_per_sample', 0):.2f}",
            f"{temporal_stats['duration_mean']:.1f}",
            spatial_stats['unique_channels'],
            f"{spatial_stats['multi_channel_ratio']*100:.1f}%",
            f"{quality_stats['completeness']*100:.1f}%",
            f"{quality_stats.get('outlier_ratio', 0)*100:.1f}%",
            f"{validation_stats['repeated_biomarker_ratio']*100:.1f}%"
        ]
    }
    
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(save_dir, 'biomarker_statistics_summary.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV统计摘要已保存: {csv_path}")


def main():
    """主函数"""
    # 数据目录
    data_dir = r"D:\2025_important_projects\data\chbmit_biomarkers\06\raw_eeg"
    
    # 输出目录
    output_dir = r"D:\2025_important_projects\paper-main\biomarker_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("生物标志物统计分析可视化")
    print("=" * 80)
    print()
    
    # 1. 加载数据
    biomarkers = load_biomarker_data(data_dir)
    
    if not biomarkers:
        print("错误: 没有加载到任何生物标志物数据")
        return
    
    print()
    
    # 2. 生成总览图
    plot_overview(biomarkers, output_dir)
    print()
    
    # 3. 生成对比分析图
    plot_comparison(biomarkers, output_dir)
    print()
    
    # 4. 生成Top 10标志物可视化
    plot_top_biomarkers(biomarkers, output_dir, top_n=10)
    print()
    
    # 5. 生成综合统计报告
    generate_report(biomarkers, output_dir)
    print()
    
    # 6. 保存前20个标志物的详细可视化
    print("生成前20个标志物详细可视化...")
    top_20_dir = os.path.join(output_dir, 'individual_biomarkers')
    os.makedirs(top_20_dir, exist_ok=True)
    
    # 按出现频率排序
    primitive_id_counts = Counter([b.primitive_id for b in biomarkers if b.primitive_id])
    top_20_ids = [item[0] for item in primitive_id_counts.most_common(20)]
    
    for idx, primitive_id in enumerate(top_20_ids):
        # 找到第一个匹配的标志物
        biomarker = next((b for b in biomarkers if b.primitive_id == primitive_id), None)
        if biomarker:
            save_path = os.path.join(top_20_dir, f'{idx+1:02d}_{primitive_id}.png')
            plot_individual_biomarker(biomarker, save_path)
    
    print(f"前20个标志物详细可视化已保存到: {top_20_dir}")
    print()
    
    print("=" * 80)
    print("分析完成！所有结果已保存到:", output_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
