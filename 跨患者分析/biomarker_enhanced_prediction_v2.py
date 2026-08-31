#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生物标志物增强预测模块

利用生物标志物文件来提高癫痫预测任务的敏感性和特异性

核心思想：
1. 加载并分析生物标志物文件
2. 计算生物标志物在测试样本中的出现情况
3. 结合模型预测概率和生物标志物置信度进行综合判断
4. 针对低置信度样本采用特殊处理策略
"""

import numpy as np
import json
import os
import glob
import warnings
import shutil
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, Counter
from tqdm import tqdm
import torch
import re
# from model.EEGInception_SE import EEGInception
# from braindecode.models import Deep4Net
# from braindecode.models import EEGNetv4
import datetime
from pathlib import Path
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from linear_attention_transformer import LinearAttentionTransformer
# from braindecode.models.atcnet import ATCNet
# from braindecode.models.eegconformer import EEGConformer
from collections import OrderedDict, Counter



class BiomarkerMatcher:
    """生物标志物匹配器"""

    def __init__(self, biomarker_dir: str, filtered_biomarkers: Optional[List[Dict]] = None):
        """
        初始化生物标志物匹配器

        Parameters:
        -----------
        biomarker_dir : str
            生物标志物文件目录路径
        filtered_biomarkers : List[Dict], optional
            筛选后的生物标志物列表，如果提供则使用此列表而不是加载所有标志物
        """
        self.biomarker_dir = biomarker_dir
        self.biomarkers = []
        self.biomarker_index = defaultdict(list)  # primitive_id -> [biomarker_objects]
        if filtered_biomarkers is not None:
            self.biomarkers = filtered_biomarkers
            # 重建索引
            for biomarker in self.biomarkers:
                primitive_id = biomarker.get('primitive_id')
                if primitive_id:
                    self.biomarker_index[primitive_id].append(biomarker)
        else:
            self._load_biomarkers()

    def _load_biomarkers(self):
        """加载所有生物标志物文件"""
        if not os.path.exists(self.biomarker_dir):
            print(f"警告: 生物标志物目录不存在: {self.biomarker_dir}")
            return

        pattern = os.path.join(self.biomarker_dir, "*.npz")
        files = glob.glob(pattern)

        # print(f"正在加载生物标志物文件: 找到 {len(files)} 个文件")

        for filepath in files:
            try:
                biomarker = self._load_single_biomarker(filepath)
                if biomarker:
                    self.biomarkers.append(biomarker)
                    primitive_id = biomarker.get('primitive_id')
                    if primitive_id:
                        self.biomarker_index[primitive_id].append(biomarker)
            except Exception as e:
                print(f"加载文件失败 {os.path.basename(filepath)}: {e}")

        # print(f"成功加载 {len(self.biomarkers)} 个生物标志物")
        # print(f"唯一标志物类型: {len(self.biomarker_index)} 种")

    def _load_single_biomarker(self, filepath: str) -> Optional[Dict]:
        """加载单个生物标志物文件"""
        try:
            npz = np.load(filepath, allow_pickle=False)
            data = npz["data"]
            meta = json.loads(npz["meta_json"].item())

            return {
                'filepath': filepath,
                'data': data,
                'meta': meta,
                'primitive_type': meta.get('primitive_type'),
                'primitive_id': meta.get('primitive_id'),
                'channels': meta.get('channels', []),
                'time_range': meta.get('time_range', [0, 0]),
                'sample_index': meta.get('sample_index'),
                'shape': data.shape
            }
        except Exception as e:
            return None

    def find_matching_biomarkers(self, sample_data: np.ndarray,
                                 sample_index: Optional[int] = None,
                                 channels: Optional[List[int]] = None) -> List[Dict]:
        """
        在样本数据中查找匹配的生物标志物

        Parameters:
        -----------
        sample_data : np.ndarray
            样本EEG数据，形状为 (n_channels, n_timepoints)
        sample_index : int, optional
            样本索引，用于过滤特定样本的生物标志物
        channels : List[int], optional
            通道列表，用于过滤特定通道的生物标志物

        Returns:
        --------
        matches : List[Dict]
            匹配的生物标志物列表，每个元素包含匹配信息和置信度
        """
        matches = []

        # 如果指定了sample_index，只检查该样本的生物标志物
        if sample_index is not None:
            relevant_biomarkers = [b for b in self.biomarkers
                                   if b.get('sample_index') == sample_index]
        else:
            relevant_biomarkers = self.biomarkers

        for biomarker in relevant_biomarkers:
            match_info = self._check_biomarker_match(biomarker, sample_data, channels)
            if match_info['matched']:
                # print(match_info)
                matches.append(match_info)

        return matches

    def _check_biomarker_match(self, biomarker: Dict,
                               sample_data: np.ndarray,
                               channels: Optional[List[int]] = None) -> Dict:
        """
        检查生物标志物是否在样本数据中匹配。
        """
        biomarker_data = biomarker['data']
        biomarker_channels = biomarker.get('channels', [])
        time_range = biomarker.get('time_range', [0, 0])

        # 检查通道匹配
        if channels is not None and biomarker_channels:
            if not any(ch in channels for ch in biomarker_channels):
                return {'matched': False, 'confidence': 0.0}

        # 时间窗口长度
        window_len = time_range[1] - time_range[0] if time_range[1] - time_range[0] > 0 else biomarker_data.shape[1]
        n_sample_time = sample_data.shape[2]

        # 返回错误条件
        if window_len > n_sample_time:
            return {'matched': False, 'confidence': 0.0}

        # 确定通道索引与生物标志物切片
        if biomarker_channels:
            channel_indices = [i for i, ch in enumerate(biomarker_channels) if ch < sample_data.shape[0]]
            if not channel_indices:
                return {'matched': False, 'confidence': 0.0}
            biomarker_slice = biomarker_data[channel_indices, :]
            if biomarker_slice.shape[1] != window_len:
                window_len = biomarker_slice.shape[1]
        else:
            channel_indices = np.arange(sample_data.shape[0])
            biomarker_slice = biomarker_data
            if biomarker_slice.ndim == 2:
                window_len = biomarker_slice.shape[1]
            else:
                return {'matched': False, 'confidence': 0.0}

        # 初始化
        best_correlation = -1.0
        best_start = 0.1

        # 确保 biomarker_slice 和 sample_data 是一维数组
        biomarker_flat = biomarker_slice.flatten()
        std_biomarker = np.std(biomarker_flat)
        for start in range(0, n_sample_time - window_len + 1, window_len):
            end = start + window_len
            sample_slice = sample_data[:, channel_indices, start:end]

            if sample_slice.shape[-1] != biomarker_slice.shape[-1]:
                continue

            sample_flat = sample_slice.flatten()
            sample_flat = sample_flat.numpy() if hasattr(sample_flat, 'numpy') else np.asarray(sample_flat)

            # 检查方差，避免 corrcoef 除以零导致 RuntimeWarning
            std_sample = np.std(sample_flat)
            std_invalid = (std_sample < 1e-8 or std_biomarker < 1e-8 or
                           np.isnan(std_sample) or np.isnan(std_biomarker))
            if std_invalid:
                corr = 0.0  # 常数序列或无效数据，无有效相关性
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'invalid value encountered in divide',
                                            RuntimeWarning)
                    corr = np.corrcoef(sample_flat, biomarker_flat)[0, 1]
                if np.isnan(corr):
                    corr = 0.0

            if corr > best_correlation:
                best_correlation = corr
                best_start = start

        # 最终匹配结果
        correlation = max(0.0, best_correlation)
        matched = correlation > 0.5
        confidence = correlation

        return {
            'matched': matched,
            'confidence': confidence,
            'correlation': correlation,
            'best_start': best_start,
            'window_len': window_len,
            'biomarker': biomarker,
            'primitive_id': biomarker.get('primitive_id'),
            'primitive_type': biomarker.get('primitive_type')
        }


class BiomarkerEnhancedPredictor:
    """生物标志物增强预测器"""

    def __init__(self, biomarker_dir: str,
                 base_model_predictor,
                 biomarker_weight: float,
                 min_biomarker_confidence: float = 0.6,
                 filtered_biomarkers: Optional[List[Dict]] = None,
                 no_match_penalty: float = 0.0):
        """
        初始化生物标志物增强预测器

        Parameters:
        -----------
        biomarker_dir : str
            生物标志物文件目录
        base_model_predictor : callable
            基础模型预测函数，输入样本返回概率
        biomarker_weight : float
            生物标志物权重（0-1之间）
        min_biomarker_confidence : float
            生物标志物最小置信度阈值
        filtered_biomarkers : List[Dict], optional
            筛选后的生物标志物列表
        no_match_penalty : float, optional
            当未匹配到任何生物标志物时，将预测概率向 [0.5, 0.5] 拉近的程度（0~1）。
            0 表示不惩罚；0.2~0.4 表示适度降低置信度。用于“无生物标志物支持时降低模型置信度”。
        """
        self.biomarker_matcher = BiomarkerMatcher(biomarker_dir, filtered_biomarkers=filtered_biomarkers)
        self.base_model_predictor = base_model_predictor
        self.biomarker_weight = biomarker_weight
        self.min_biomarker_confidence = min_biomarker_confidence
        self.no_match_penalty = max(0.0, min(1.0, no_match_penalty))

        # 统计信息
        self.stats = {
            'total_samples': 0,
            'biomarker_matched_samples': 0,
            'biomarker_boosted_predictions': 0
        }

    def predict(self, sample_data: np.ndarray,
                sample_index: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        使用生物标志物增强的预测

        Parameters:
        -----------
        sample_data : np.ndarray
            样本EEG数据，形状为 (n_channels, n_timepoints)
        sample_index : int, optional
            样本索引

        Returns:
        --------
        enhanced_probs : np.ndarray
            增强后的预测概率 [P(Interictal), P(Preictal)]
        info : Dict
            预测信息，包含生物标志物匹配情况等
        """
        self.stats['total_samples'] += 1

        # 1. 基础模型预测
        base_probs = self.base_model_predictor(sample_data)

        # 处理不同的返回格式
        base_probs = np.asarray(base_probs)

        # 如果是标量或形状为 (1,) 的数组（类别预测），转换为概率
        if base_probs.ndim == 0 or (base_probs.ndim == 1 and base_probs.shape[0] == 1):
            # 如果是类别预测，转换为概率数组
            pred_class = int(base_probs.item() if base_probs.ndim == 0 else base_probs[0])
            base_probs = np.array([1.0 - pred_class, pred_class])  # [P(Interictal), P(Preictal)]
        elif base_probs.ndim == 1:
            # 如果已经是概率数组，确保长度为2
            if base_probs.shape[0] == 1:
                # 只有一个值，假设是Preictal的概率
                pred_val = float(base_probs[0])
                base_probs = np.array([1.0 - pred_val, pred_val])
            elif base_probs.shape[0] != 2:
                # 长度不是2，尝试reshape或取前两个
                if base_probs.shape[0] > 2:
                    base_probs = base_probs[:2]
                else:
                    # 如果只有一个值，假设是Preictal的概率
                    pred_val = float(base_probs[0])
                    base_probs = np.array([1.0 - pred_val, pred_val])
        elif base_probs.ndim == 2:
            # 如果是 (batch_size, n_classes) 形状，取第一个样本
            base_probs = base_probs[0]
            if base_probs.shape[0] != 2:
                # 如果长度不是2，处理同上
                if base_probs.shape[0] == 1:
                    pred_val = float(base_probs[0])
                    base_probs = np.array([1.0 - pred_val, pred_val])
                else:
                    base_probs = base_probs[:2]

        # 确保是长度为2的概率数组
        if base_probs.shape[0] != 2:
            # 最后的兜底处理
            if len(base_probs) == 1:
                pred_val = float(base_probs[0])
                base_probs = np.array([1.0 - pred_val, pred_val])
            else:
                base_probs = np.array([0.5, 0.5])  # 默认等概率

        # 2. 查找匹配的生物标志物
        matches = self.biomarker_matcher.find_matching_biomarkers(
            sample_data, sample_index=sample_index
        )

        # 3. 计算生物标志物增强因子
        biomarker_boost = self._calculate_biomarker_boost(matches)
        # print(biomarker_boost)

        # 4. 结合基础预测和生物标志物信息
        enhanced_probs = self._combine_predictions(base_probs, biomarker_boost)

        # 更新统计信息
        if matches:
            self.stats['biomarker_matched_samples'] += 1
        if biomarker_boost['boost_strength'] > 0:
            # print('有了'* 8 )
            self.stats['biomarker_boosted_predictions'] += 1

        info = {
            'base_probs': base_probs,
            'enhanced_probs': enhanced_probs,
            'biomarker_matches': len(matches),
            'biomarker_boost': biomarker_boost,
            'sample_index': sample_index
        }

        return enhanced_probs, info

    def _calculate_biomarker_boost(self, matches: List[Dict]) -> Dict:
        """
        计算生物标志物增强因子

        Parameters:
        -----------
        matches : List[Dict]
            匹配的生物标志物列表

        Returns:
        --------
        boost_info : Dict
            增强信息，包含增强强度和方向
        """
        if not matches:
            return {
                'boost_strength': 0.0,
                'boost_direction': 0,  # 0: 无增强, 1: 增强Preictal, -1: 增强Interictal
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'n_matches': 0
            }

        # 计算平均置信度和最大置信度
        confidences = [m['confidence'] for m in matches if 'confidence' in m]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        max_confidence = np.max(confidences) if confidences else 0.0

        # 如果置信度足够高，增强Preictal预测
        # 因为生物标志物通常出现在Preictal阶段
        if max_confidence >= self.min_biomarker_confidence:
            boost_strength = min(1.0, max_confidence * self.biomarker_weight)
            boost_direction = 1  # 增强Preictal
        else:
            boost_strength = 0.0
            boost_direction = 0

        return {
            'boost_strength': boost_strength,
            'boost_direction': boost_direction,
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'n_matches': len(matches),
        }

    def _combine_predictions(self, base_probs: np.ndarray,
                             biomarker_boost: Dict) -> np.ndarray:
        """
        结合基础预测和生物标志物增强。

        - 若匹配到高置信度生物标志物：增强 Preictal 概率。
        - 若未匹配到任何生物标志物且 no_match_penalty > 0：将预测向 [0.5, 0.5] 拉近，降低置信度。
        """
        enhanced_probs = base_probs.copy()

        boost_strength = biomarker_boost['boost_strength']
        boost_direction = biomarker_boost['boost_direction']
        n_matches = biomarker_boost.get('n_matches', 0)

        if boost_strength > 0 and boost_direction == 1:
            # 增强Preictal概率
            transfer = boost_strength * base_probs[0]
            enhanced_probs[0] = base_probs[0] - transfer
            enhanced_probs[1] = base_probs[1] + transfer
        elif n_matches == 0 and self.no_match_penalty > 0:
            # 无任何生物标志物匹配时：向均匀分布拉近，降低模型置信度
            uniform = np.array([0.5, 0.5])
            enhanced_probs = (1.0 - self.no_match_penalty) * base_probs + self.no_match_penalty * uniform

        # 确保概率归一化
        enhanced_probs = np.clip(enhanced_probs, 0.0, 1.0)
        enhanced_probs = enhanced_probs / (enhanced_probs.sum() + 1e-10)

        return enhanced_probs

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        if stats['total_samples'] > 0:
            stats['biomarker_match_rate'] = stats['biomarker_matched_samples'] / stats['total_samples']
            stats['biomarker_boost_rate'] = stats['biomarker_boosted_predictions'] / stats['total_samples']
        else:
            stats['biomarker_match_rate'] = 0.0
            stats['biomarker_boost_rate'] = 0.0
        return stats

    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'total_samples': 0,
            'biomarker_matched_samples': 0,
            'biomarker_boosted_predictions': 0
        }


def filter_biomarkers_by_segments(biomarker_dir: str,
                                  data_root_dir: str,
                                  segment_info_path: str,
                                  match_threshold: float = 0.7,
                                  sample_window_size: int = 15360) -> List[Dict]:
    """
    筛选生物标志物：只保留在前期片段匹配但在间期片段不匹配的生物标志物
    
    Parameters:
    -----------
    biomarker_dir : str
        生物标志物文件目录
    data_root_dir : str
        数据根目录
    segment_info_path : str
        分段信息JSON文件路径
    match_threshold : float
        匹配阈值（相关系数）
    min_pre_matches : int
        在前期片段中的最小匹配次数
    max_inter_matches : int
        在间期片段中的最大匹配次数（超过此值则排除）
    sample_window_size : int
        采样窗口大小
    n_samples_per_segment : int
        每个片段中采样的窗口数量
    
    Returns:
    --------
    filtered_biomarkers : List[Dict]
        筛选后的生物标志物列表
    """
    print("=" * 80)
    print("开始筛选生物标志物...")
    # 加载所有生物标志物
    matcher = BiomarkerMatcher(biomarker_dir)
    all_biomarkers = matcher.biomarkers
    print(f"总共加载了 {len(all_biomarkers)} 个生物标志物")
    
    # 加载分段信息
    with open(segment_info_path, 'r') as f:
        files_list = json.load(f)
    
    # 分离前期和间期片段
    pre_segments = [f for f in files_list if re.match("Pre", f.get("Label", ""))]
    inter_segments = [f for f in files_list if re.match("Inter", f.get("Label", ""))]
    
    print(f"找到 {len(pre_segments)} 个前期片段, {len(inter_segments)} 个间期片段")
    
    # 评估每个生物标志物
    filtered_biomarkers = []
    biomarker_stats = []
    
    for idx, biomarker in enumerate(tqdm(all_biomarkers, desc="评估生物标志物")):
        pre_match_count = 0
        inter_match_count = 0
        
        # 在前期片段中测试
        for seg in pre_segments[:min(len(pre_segments), len(inter_segments))]:
            try:
                data = np.load(os.path.join(data_root_dir, seg['File']))
                data = np.squeeze(data)
                
                # 采样几个窗口进行测试
                j = 0
                tested = 0
                while (j + 1) * sample_window_size <= data.shape[-1] and tested < 15:
                    split_data = data[:, j * sample_window_size:(j + 1) * sample_window_size]
                    # 转换为 (1, channels, timepoints) 格式以匹配_check_biomarker_match的期望
                    split_data = split_data[np.newaxis, :, :]
                    
                    match_info = matcher._check_biomarker_match(biomarker, split_data)
                    if match_info['matched'] and match_info['confidence'] >= match_threshold:
                        pre_match_count += 1
                        # break  # 只要匹配一次就计数
                    
                    j += 1
                    tested += 1
            except Exception as e:
                continue
        
        # 在间期片段中测试
        for seg in inter_segments[:min(len(inter_segments), len(pre_segments))]:
            try:
                data = np.load(os.path.join(data_root_dir, seg['File']))
                data = np.squeeze(data)
                
                # 采样几个窗口进行测试
                j = 0
                tested = 0
                while (j + 1) * sample_window_size <= data.shape[-1] and tested < 15:
                    split_data = data[:, j * sample_window_size:(j + 1) * sample_window_size]
                    # 转换为 (1, channels, timepoints) 格式
                    split_data = split_data[np.newaxis, :, :]
                    
                    match_info = matcher._check_biomarker_match(biomarker, split_data)
                    if match_info['matched'] and match_info['confidence'] >= match_threshold:
                        inter_match_count += 1
                        # break  # 只要匹配一次就计数
                    
                    j += 1
                    tested += 1
            except Exception as e:
                continue
        
        # 判断是否保留
        if pre_match_count >= 2 *  inter_match_count:
            # print('前期数量%d' % pre_match_count, '间期数量%d' % inter_match_count)
            filtered_biomarkers.append(biomarker)
            biomarker_stats.append({
                'biomarker_idx': idx,
                'pre_matches': pre_match_count,
                'inter_matches': inter_match_count,
                'primitive_id': biomarker.get('primitive_id')
            })
    
    print(f"\n筛选结果:")
    print(f"  原始生物标志物数量: {len(all_biomarkers)}")
    print(f"  筛选后数量: {len(filtered_biomarkers)}")
    print(f"  筛选率: {len(filtered_biomarkers)/len(all_biomarkers)*100:.2f}%")
    
    if biomarker_stats:
        avg_pre = np.mean([s['pre_matches'] for s in biomarker_stats])
        avg_inter = np.mean([s['inter_matches'] for s in biomarker_stats])
        print(f"  平均前期匹配次数: {avg_pre:.2f}")
        print(f"  平均间期匹配次数: {avg_inter:.2f}")
    
    print("=" * 80)
    return filtered_biomarkers, biomarker_stats


def save_filtered_biomarkers(filtered_biomarkers: List[Dict],
                             biomarker_stats: List[Dict],
                             save_dir: str,
                             original_biomarker_dir: str):
    """
    保存筛选后的生物标志物到指定目录
    
    Parameters:
    -----------
    filtered_biomarkers : List[Dict]
        筛选后的生物标志物列表
    biomarker_stats : List[Dict]
        生物标志物统计信息
    save_dir : str
        保存目录路径
    original_biomarker_dir : str
        原始生物标志物目录（用于复制文件）
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n保存筛选后的生物标志物到: {save_dir}")
    
    # 保存统计信息
    stats_file = os.path.join(save_dir, 'filtering_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_filtered': len(filtered_biomarkers),
            'biomarker_stats': biomarker_stats,
            'filtering_criteria': {
                'description': '只保留在前期片段匹配次数 >= 2 * 间期片段匹配次数的生物标志物'
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"  统计信息已保存到: {stats_file}")
    
    # 复制筛选后的生物标志物文件
    copied_count = 0
    for biomarker in tqdm(filtered_biomarkers, desc="复制生物标志物文件"):
        original_filepath = biomarker.get('filepath')
        if original_filepath and os.path.exists(original_filepath):
            filename = os.path.basename(original_filepath)
            dest_filepath = os.path.join(save_dir, filename)
            
            try:
                # 复制文件
                shutil.copy2(original_filepath, dest_filepath)
                copied_count += 1
            except Exception as e:
                print(f"  警告: 复制文件失败 {filename}: {e}")
    
    print(f"  成功复制 {copied_count}/{len(filtered_biomarkers)} 个生物标志物文件")
    
    # 保存筛选后的生物标志物列表（pickle格式，包含完整数据）
    pickle_file = os.path.join(save_dir, 'filtered_biomarkers.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(filtered_biomarkers, f)
    print(f"  完整数据已保存到: {pickle_file}")
    
    print(f"保存完成！筛选后的生物标志物已保存到: {save_dir}")


def load_filtered_biomarkers(save_dir: str) -> List[Dict]:
    """
    加载已保存的筛选后的生物标志物
    
    Parameters:
    -----------
    save_dir : str
        保存目录路径
    
    Returns:
    --------
    filtered_biomarkers : List[Dict]
        筛选后的生物标志物列表
    """
    pickle_file = os.path.join(save_dir, 'filtered_biomarkers.pkl')
    
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"未找到筛选后的生物标志物文件: {pickle_file}")
    
    print(f"从 {pickle_file} 加载筛选后的生物标志物...")
    with open(pickle_file, 'rb') as f:
        filtered_biomarkers = pickle.load(f)
    
    print(f"成功加载 {len(filtered_biomarkers)} 个筛选后的生物标志物")
    return filtered_biomarkers


def create_biomarker_enhanced_predictor(biomarker_dir: str,
                                        model_predictor,
                                        biomarker_weight: float,
                                        filtered_biomarkers: Optional[List[Dict]] = None,
                                        no_match_penalty: float = 0.0) -> BiomarkerEnhancedPredictor:
    """
    创建生物标志物增强预测器的便捷函数

    Parameters:
    -----------
    biomarker_dir : str
        生物标志物文件目录
    model_predictor : callable
        基础模型预测函数
    biomarker_weight : float
        生物标志物权重
    filtered_biomarkers : List[Dict], optional
        筛选后的生物标志物列表
    no_match_penalty : float, optional
        无生物标志物匹配时降低置信度的强度（0~1），默认 0 不惩罚

    Returns:
    --------
    predictor : BiomarkerEnhancedPredictor
        生物标志物增强预测器
    """
    return BiomarkerEnhancedPredictor(
        biomarker_dir=biomarker_dir,
        base_model_predictor=model_predictor,
        biomarker_weight=biomarker_weight,
        filtered_biomarkers=filtered_biomarkers,
        no_match_penalty=no_match_penalty
    )


def model_predictor(pre_data):
    """模型预测函数，返回类别（0或1）"""
    predict_cla = model(pre_data.to(device)).argmax(dim=1).cpu().numpy()
    return predict_cla


def model_predictor_with_probs(pre_data):
    """模型预测函数，返回概率数组 [P(Interictal), P(Preictal)]"""
    with torch.no_grad():
        logits = model(pre_data.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        if probs.ndim == 2:
            probs = probs[0]  # 取第一个样本
        return probs


def evaluate_accuracy(data_windows, net, device_acc, bool_use, biomarker_dir, filtered_biomarkers=None, no_match_penalty=0.0):
    pre_counts = 0
    with torch.no_grad():
        slide_i = 0
        while slide_i < data_windows.shape[0]:
            if isinstance(net, torch.nn.Module):
                X = data_windows[slide_i, :, :]
                slide_i += 1
                net.eval()  # 评估模式, 这会关闭dropout
                X = torch.from_numpy(X).to(device_acc)
                X = X.unsqueeze(0)
                X = X.to(torch.float32)
                if X.shape[-1] != 1280:
                    print(X.shape)
                    continue
                if bool_use:
                    # 创建一个返回概率的包装函数
                    def prob_predictor(data):
                        with torch.no_grad():
                            logits = net(data.to(device_acc))
                            probs = torch.softmax(logits, dim=1).cpu().numpy()
                            if probs.ndim == 2:
                                probs = probs[0]  # 取第一个样本
                            return probs

                    # 使用筛选后的生物标志物（如果已筛选）；无匹配时可降低置信度
                    predictor = create_biomarker_enhanced_predictor(
                        biomarker_dir=biomarker_dir,
                        model_predictor=prob_predictor,
                        biomarker_weight=0.7,
                        filtered_biomarkers=filtered_biomarkers,
                        no_match_penalty=no_match_penalty
                    )
                    enhanced_probs, info = predictor.predict(X)
                    # 将概率转换为类别：如果 Preictal 概率 >= 0.5，预测为1，否则为0
                    predict_cla = np.argmax(enhanced_probs)
                num_of_ones = 1 if predict_cla == 1 else 0
                pre_counts += num_of_ones
        # print(data_windows.shape[-1])
        # print('\033[91m________________________________________\033[0m')
        return pre_counts


# 设置K-of-N方法且设置伪不应期（无法盘到下一个文件）
def find_subarrays(arr):
    i = 0
    while i < len(arr):
        if arr[i] > 6:
            arr[i] = 1
            # Set the next 30 elements to 1, if they exist
            for j in range(i + 1, min(i + 32, len(arr))):
                arr[j] = 1
            for j in range(min(i + 32, len(arr)), min(i + 62, len(arr))):  # 设置不应期
                arr[j] = 0
            i += 61  # Skip the next 30 elements as they are already set to 1
        elif arr[i] <= 6:
            arr[i] = 0
        # If the element is exactly 6, do nothing and move to the next element
        i += 1
    return arr


def natural_keys(text):
    """Helper function for natural sorting of filenames"""

    def atoi(text):
        return int(text) if text.isdigit() else text

    return [atoi(c) for c in re.split(r'(\d+)', text)]


def load_model_robust(model_path, n_chans, n_classes=2, device='cpu'):
    """Robust model loading with proper EEGInception loading

    Args:
        model_path: Path to model weights file
        n_chans: Number of EEG channels (should be 22 for user's data)
        n_classes: Number of output classes
        device: Device to load model on

    Returns:
        Loaded model ready for inference
    """

    try:
        # Check if model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # Try to load EEGInception model
        print(f"Attempting to load EEGInception model with {n_chans} channels...")
        model = EEGInception(input_time=1280, fs=256, ncha=n_chans, n_classes=n_classes)

        # Load weights - handle different checkpoint formats
        try:
            checkpoint = torch.load(model_path, map_location=device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Load state dict with strict=False to handle minor mismatches
            model.load_state_dict(state_dict, strict=False)

        except Exception as load_error:
            print(f"Error loading checkpoint: {load_error}")
            print("Attempting to load as direct state dict...")

            # Try loading as direct state dict
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)

        model.to(device)
        model.eval()  # Set to evaluation mode

        print(f"Successfully loaded EEGInception model from {model_path}")
        return model

    except Exception as e:
        print(f"Error loading EEGInception model: {e}")
        print(f"Falling back to SimpleEEGModel with {n_chans} channels...")

        # Create simple model as fallback with correct channel count


def process_data(dict_of_tuples):
    current_sequence = []
    # 初始化时间段列表
    time_periods = {}

    # 遍历字典中的每个键值对
    start_time = 0
    pre_key = 0
    for i in range(len(dict_of_tuples)):
        key = list(dict_of_tuples.keys())[i]
        tuples_array = dict_of_tuples[key]
        if i == 0:
            current_max_time = len(tuples_array)
            last_max_time = current_max_time
        else:
            lat_key = list(dict_of_tuples.keys())[i - 1]
            last_max_time = dict_of_tuples[lat_key]
            last_max_time = len(last_max_time)
        for minute, value in tuples_array:
            if value == 1:
                # 如果当前值是1，添加到当前连续1的序列
                if not current_sequence:
                    # 如果当前序列为空，记录开始时间
                    start_time = minute
                    pre_key = key
                current_sequence.append((minute, value))
            elif current_sequence:
                # 如果当前值是0且当前序列非空，结束当前连续1的序列
                end_time = current_sequence[-1][0]  # 获取结束时间
                if end_time == last_max_time:
                    post_key = list(dict_of_tuples.keys())[i - 1]
                else:
                    post_key = key
                span_key = (pre_key, post_key)
                span_time = (start_time, end_time)
                # print(key, minute, end_time)
                # 如果键已经在字典中，添加值到对应的列表
                if span_key in time_periods:
                    time_periods[span_key].append(span_time)
                else:
                    # 否则，创建一个新列表作为这个键的值
                    time_periods[span_key] = [span_time]
                current_sequence = []  # 重置当前连续1的序列

    # 检查循环结束后是否还有剩余的连续1的序列
    if current_sequence:
        # 获取字典中的最后一个键
        last_key = list(dict_of_tuples.keys())[-1]
        # 获取最后一个键对应的值
        last_value = dict_of_tuples[last_key]
        # 获取值的长度
        value_length = len(last_value)
        if len(current_sequence) > value_length:
            n = len(current_sequence) // value_length
            # 使用取模运算符检查余数
            remainder = len(current_sequence) % value_length
            # 判断是否有余数
            if remainder == 0:
                n = n
            else:
                n += 1
            last_last_key = list(dict_of_tuples.keys())[-n]
            final_key_name = (last_last_key, last_key)
            end_time = current_sequence[-1][0]  # 获取结束时间
            span_time = (start_time, end_time)
            # 如果键已经在字典中，添加值到对应的列表
            if final_key_name in time_periods:
                time_periods[final_key_name].append(span_time)
            else:
                # 否则，创建一个新列表作为这个键的值
                time_periods[final_key_name] = [span_time]
        else:
            final_key_name = (last_key, last_key)
            end_time = current_sequence[-1][0]  # 获取结束时间
            span_time = (start_time, end_time)
            # 如果键已经在字典中，添加值到对应的列表
            if final_key_name in time_periods:
                time_periods[final_key_name].append(span_time)
            else:
                # 否则，创建一个新列表作为这个键的值
                time_periods[final_key_name] = [span_time]

    return time_periods


def get_onset_info():
    dt_fmt = '%Y-%m-%d %H:%M:%S'
    onset_record_info = []
    onsetInfo = []
    recordInfo = []
    with open(os.path.join(data_root_dir, 'datetime_info.json'), 'r') as f:
        record_lst = json.load(f)
        total_time = 0
        for record in record_lst:
            edf_name = record['File Name']
            start_str, end_str = record['Record Datetimes']
            start_dt, end_dt = datetime.datetime.strptime(start_str, dt_fmt), datetime.datetime.strptime(end_str,
                                                                                                         dt_fmt)
            time_delta = end_dt - start_dt
            # 将时间差转换为分钟
            hours = time_delta.total_seconds() / 3600
            # 将分钟数转换为整数
            hours = int(hours)
            total_time += hours
            recordInfo.append((start_dt, end_dt, edf_name))
            for nsz, sz_span in enumerate(record['Seizures']):
                sz_start_sec, sz_end_sec = sz_span
                sz_start_dt, sz_end_dt = start_dt + datetime.timedelta(
                    seconds=sz_start_sec), start_dt + datetime.timedelta(seconds=sz_end_sec)
                onsetInfo.append((edf_name, sz_start_dt, sz_end_dt))
                onset_record_info.append({'Label': 'Onset%d' % len(onsetInfo),
                                          'File': edf_name[:-3] + 'npy',
                                          'Span': [sz_start_sec // 60, sz_end_sec // 60]})
    return onset_record_info, total_time


def extract_number(file_name):
    # 这个正则表达式匹配最后一个下划线后面的数字序列
    match = re.search(r'_(\d+)[^_\d]*\.\w+$', file_name)
    return int(match.group(1)) if match else None


def calculate_scores(results_time_i):
    n_predict_seizures = len(results_time_i)
    counts_true = 0
    counts_false = 0
    all_aver_time_length = 0
    time_length = 0
    for key, value in results_time_i.items():
        # print(f"{key}: {value}")
        # 提取元组中文件名的数字
        start_num = extract_number(key[0])
        end_num = extract_number(key[1])
        for value_i in value:
            start_time = value_i[0]
            end_time = value_i[1]
            time_length = end_num * 60 + end_time - (start_num * 60 + start_time)
            # 检查 'chb01_03.npy' 的数字是否在这两个数字之间
            counts_predict_true_seizures = 0
            for onset_seg in pat_onset_segs:
                # print(onset_seg['File'])
                file_to_check = onset_seg['File']
                file_to_span = onset_seg['Span']
                check_num = extract_number(file_to_check)
                if start_num < check_num < end_num:
                    counts_predict_true_seizures += 1
                elif check_num == end_num and check_num != start_num:
                    if file_to_span[1] <= end_time:
                        counts_predict_true_seizures += 1
                elif check_num == start_num and check_num != end_num:
                    if start_time <= file_to_span[0]:
                        counts_predict_true_seizures += 1
                elif check_num == start_num and check_num == end_num:
                    if start_time <= file_to_span[0] and file_to_span[1] <= end_time:
                        counts_predict_true_seizures += 1
            if counts_predict_true_seizures:
                counts_true += counts_predict_true_seizures
            else:
                counts_false += 1
        all_aver_time_length += time_length
    sensitivity_events = counts_true / ture_n
    aver_time_rate = all_aver_time_length / total_time_hours
    FDR_h = counts_false / total_time_hours
    return n_predict_seizures, aver_time_rate, sensitivity_events, FDR_h


def sort_key(filename):
    # This regular expression will find the numbers preceded by "FDR" and followed by "-" in the filename
    match = re.search(r'(\d+)-(\d+)', filename)
    if match:
        # Generate a tuple (first_number, second_number)
        return int(match.group(1)), int(match.group(2))
    else:
        return filename


class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size: int = 256, n_freq: int = 101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, freq, time)
        Returns:
            (batch, time, emb_size)
        """
        x = x.permute(0, 2, 1)          # -> (batch, time, freq)
        return self.projection(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # shape (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int, n_classes: int):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls_head(x)


class BIOTEncoder(nn.Module):
    def __init__(
        self,
        emb_size: int = 256,
        heads: int = 8,
        depth: int = 4,
        n_channels: int = 16,
        n_fft: int = 200,
        hop_length: int = 100,
        **kwargs
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1
        )
        self.positional_encoding = PositionalEncoding(emb_size)

        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )

        # learnable channel tokens
        self.channel_tokens = nn.Embedding(n_channels, emb_size)
        self.index = nn.Parameter(
            torch.arange(n_channels, dtype=torch.long), requires_grad=False
        )

    # --- STFT ---
    def stft(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sample: (batch, 1, ts)
        Returns:
            magnitude spec: (batch, freq, time)
        """
        spec = torch.stft(
            input=sample.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=False,
            onesided=True,
            return_complex=True,
        )
        return spec.abs()

    # --- forward ---
    def forward(self, x: torch.Tensor,
                n_channel_offset: int = 0,
                perturb: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch, channel, ts)
        Returns:
            (batch, emb_size)
        """
        emb_seq = []
        B, C, _ = x.shape

        for i in range(C):
            # (B, freq, time)
            spec = self.stft(x[:, i:i + 1, :])
            # (B, time, emb)
            patch_emb = self.patch_embedding(spec)

            # channel token
            token = self.channel_tokens(
                self.index[i + n_channel_offset]
            ).unsqueeze(0).unsqueeze(0)        # (1,1,emb)
            token = token.expand(B, patch_emb.size(1), -1)

            # add & positional encoding
            channel_emb = self.positional_encoding(patch_emb + token)

            # optional random temporal crop (perturb)
            if perturb:
                ts = channel_emb.size(1)
                ts_new = np.random.randint(ts // 2, ts)
                sel = np.random.choice(ts, ts_new, replace=False)
                channel_emb = channel_emb[:, sel]

            emb_seq.append(channel_emb)

        # concat over channels  → (B, C*ts, emb)
        emb = torch.cat(emb_seq, dim=1)
        # transformer  → (B, seq, emb) → take mean
        return self.transformer(emb).mean(dim=1)

class BIOTClassifier(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4,
                 n_classes=6, **kwargs):
        super().__init__()
        self.biot = BIOTEncoder(emb_size, heads, depth, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.biot(x))


class UnsupervisedPretrain(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4,
                 n_channels=18, **kwargs):
        super().__init__()
        self.biot = BIOTEncoder(emb_size, heads, depth, n_channels, **kwargs)
        self.prediction = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Linear(emb_size, emb_size),
        )

    def forward(self, x: torch.Tensor,
                n_channel_offset: int = 0):
        emb1 = self.biot(x, n_channel_offset, perturb=True)
        emb1 = self.prediction(emb1)
        emb2 = self.biot(x, n_channel_offset)
        return emb1, emb2


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate,
                 bn_size, drop_rate, conv_bias, batch_norm):
        super().__init__()
        if batch_norm:
            self.add_module('norm1', nn.BatchNorm1d(num_input_features))
        self.add_module('elu1', nn.ELU())
        self.add_module('conv1', nn.Conv1d(num_input_features,
                                           bn_size * growth_rate,
                                           kernel_size=1, stride=1,
                                           bias=conv_bias))
        if batch_norm:
            self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate))
        self.add_module('elu2', nn.ELU())
        self.add_module('conv2', nn.Conv1d(bn_size * growth_rate,
                                           growth_rate,
                                           kernel_size=3, stride=1,
                                           padding=1, bias=conv_bias))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                      p=self.drop_rate,
                                      training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features,
                 bn_size, growth_rate, drop_rate, conv_bias, batch_norm):
        super().__init__()
        for i in range(num_layers):
            self.add_module(f'denselayer{i+1}',
                _DenseLayer(num_input_features + i * growth_rate,
                            growth_rate, bn_size, drop_rate,
                            conv_bias, batch_norm))


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features,
                 conv_bias, batch_norm):
        super().__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm1d(num_input_features))
        self.add_module('elu', nn.ELU())
        self.add_module('conv', nn.Conv1d(num_input_features,
                                          num_output_features,
                                          kernel_size=1, stride=1,
                                          bias=conv_bias))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))


class DenseNetEnconder(nn.Module):
    def __init__(self, growth_rate=32,
                 block_config=(4, 4, 4, 4, 4, 4, 4),
                 in_channels=16, num_init_features=64,
                 bn_size=4, drop_rate=0.2,
                 conv_bias=True, batch_norm=False):

        super().__init__()

        first_conv = OrderedDict([
            ('conv0', nn.Conv1d(in_channels, num_init_features,
                                kernel_size=7, stride=2, padding=3,
                                bias=conv_bias))
        ])
        if batch_norm:
            first_conv['norm0'] = nn.BatchNorm1d(num_init_features)
        first_conv['elu0'] = nn.ELU()
        first_conv['pool0'] = nn.MaxPool1d(kernel_size=3,
                                           stride=2, padding=1)

        self.densenet = nn.Sequential(first_conv)

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            self.densenet.add_module(
                f'denseblock{i+1}',
                _DenseBlock(num_layers, num_features, bn_size, growth_rate,
                            drop_rate, conv_bias, batch_norm)
            )
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                self.densenet.add_module(
                    f'transition{i+1}',
                    _Transition(num_features, num_features // 2,
                                conv_bias, batch_norm)
                )
                num_features //= 2

        if batch_norm:
            self.densenet.add_module(f'norm{len(block_config)+1}',
                                     nn.BatchNorm1d(num_features))

        self.densenet.add_module(f'relu{len(block_config)+1}', nn.ReLU())

        # ★★★ 关键修改：固定池化 → 自适应池化 ★★★
        self.densenet.add_module(
            f'pool{len(block_config)+1}',
            nn.AdaptiveAvgPool1d(1)          # 无论输入多短都能输出长度 1
        )

        self.num_features = num_features

        # 官方初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.densenet(x)
        return features.view(features.size(0), -1)   # (B, num_features)


class DenseNetClassifier(nn.Module):
    def __init__(self, growth_rate=32,
                 block_config=(4, 4, 4, 4, 4, 4, 4),
                 in_channels=16, num_init_features=64,
                 bn_size=4, drop_rate=0.2,
                 conv_bias=True, batch_norm=False,
                 drop_fc=0.5, num_classes=6):

        super().__init__()

        self.features = DenseNetEnconder(growth_rate, block_config,
                                         in_channels, num_init_features,
                                         bn_size, drop_rate, conv_bias,
                                         batch_norm)

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_fc),
            nn.Linear(self.features.num_features, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits


class ResBlock(nn.Module):
    """Convolutional Residual Block 2D
    This block stacks two convolutional layers with batch normalization,
    max pooling, dropout, and residual connection.
    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        stride: stride of the convolutional layers.
        downsample: whether to use a downsampling residual connection.
        pooling: whether to use max pooling.
    Example:
        >>> import torch
        >>> from pyhealth.models import ResBlock2D
        >>>
        >>> model = ResBlock2D(6, 16, 1, True, True)
        >>> input_ = torch.randn((16, 6, 28, 150))  # (batch, channel, height, width)
        >>> output = model(input_)
        >>> output.shape
        torch.Size([16, 16, 14, 75])
    """

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=False, pooling=False
    ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(3, stride=stride, padding=1)
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out


class ContraWR(nn.Module):
    """The encoder model of ContraWR (a supervised model, STFT + 2D CNN layers)
    Yang, Chaoqi, Danica Xiao, M. Brandon Westover, and Jimeng Sun.
    "Self-supervised eeg representation learning for automatic sleep staging."
    arXiv preprint arXiv:2110.15278 (2021).

    @article{yang2021self,
        title={Self-supervised eeg representation learning for automatic sleep staging},
        author={Yang, Chaoqi and Xiao, Danica and Westover, M Brandon and Sun, Jimeng},
        journal={arXiv preprint arXiv:2110.15278},
        year={2021}
    }
    """

    def __init__(self, in_channels=16, n_classes=6, fft=200, steps=20):
        super(ContraWR, self).__init__()
        self.fft = fft
        self.steps = steps
        self.conv1 = ResBlock(in_channels, 32, 2, True, True)
        self.conv2 = ResBlock(32, 64, 2, True, True)
        self.conv3 = ResBlock(64, 128, 2, True, True)
        self.conv4 = ResBlock(128, 256, 2, True, True)

        self.classifier = nn.Sequential(
            nn.ELU(),
            nn.Linear(256, n_classes),
        )

    def torch_stft(self, x):
        signal = []
        for s in range(x.shape[1]):
            spectral = torch.stft(
                x[:, s, :],
                n_fft=self.fft,
                hop_length=self.fft // self.steps,
                win_length=self.fft,
                normalized=True,
                center=True,
                onesided=True,
                return_complex=True,
            )
            signal.append(spectral)
        stacked = torch.stack(signal).permute(1, 0, 2, 3)
        return torch.abs(stacked)

    def forward(self, x):
        x = self.torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x).squeeze(-1).squeeze(-1)
        return self.classifier(x)



def extract_model_scores(patient_w, test_dataset, model_name, no_match_penalty=0.25):
    """
    提取模型在测试集上的分数（每段 12 个窗口的 Preictal 计数）。

    no_match_penalty: 当未匹配到任何生物标志物时，将预测概率向 0.5 拉近的程度（0~1）。
                      例如 0.25 表示适度降低置信度，0 表示不惩罚（仅做生物标志物增强）。
    """
    # model_name = 'eeginception'
    # model_name = 'eegnet'
    # biomarker_dir = r"D:\2025_important_projects\data\chbmit_biomarkers\%02d\filtered_biomarkers" % patient_w
    biomarker_dir = r'D:\2025_important_projects\data\chbmit_biomarkers\shared'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_model_path = r'D:\public_data\CHBMIT\weight\跨患者微调\%s' % model_name
    # weight_model_path = r'E:\public database\CHBMIT\weight\LOOCV\%s' % model_name
    weight_sub_dirs = os.listdir(weight_model_path)
    weight_sub_dirs = sorted(weight_sub_dirs, key=natural_keys)
    patient = patient_folder = patient_w
    # print("now is patient%d" % patient)
    if patient > 12:
        patient_folder = patient - 1
        print("now is patient%d" % patient)
    elif patient == 12:
        print('exclude 12,now is patient13')
        patient += 1
        patient_folder = patient - 1
    model_path = os.path.join(weight_model_path, weight_sub_dirs[patient_folder - 1])
    print('model_path is %s' % model_path)
    data_root_dir = r'D:\public_data\CHBMIT\1_data_clean_18channels\chb%02d' % patient
    # ________________________________________________________________________________________________________________________
    data_sub_dirs = os.listdir(data_root_dir)
    data_sub_dirs = list(filter(lambda x: x.endswith('.npy'), data_sub_dirs))
    # ___________________________为了找到通道数___________________________________________________________________________________
    example = data_sub_dirs[0]
    example_sub_patient_dirs = os.path.join(data_root_dir, example)
    example_data = np.load(example_sub_patient_dirs)
    example_data = np.squeeze(example_data)
    n_chans = example_data.shape[0]
    n_classes = 2
    if model_name == 'eeginception':
        from braindecode.models.eegconformer import EEGConformer
        model = EEGInception(input_time=1280, fs=256, ncha=n_chans, n_classes=n_classes)
    elif model_name == 'eegnet':
        from braindecode.models import EEGNetv4
        model = EEGNetv4(n_chans, n_classes, final_conv_length='auto', n_times=1280)
    elif model_name == 'deep4':
        from braindecode.models import Deep4Net
        model = Deep4Net(n_chans=18, n_outputs=2, n_times=1280)
    elif model_name == 'BIOT':
        model = BIOTClassifier(n_fft=200, hop_length=200,depth=4, heads=8,n_channels=n_chans, n_classes=2).to(device)
    elif model_name == 'ATCNet':
        from braindecode.models.atcnet import ATCNet
        model = ATCNet(n_chans=n_chans, n_outputs=2, sfreq=256,input_window_seconds=5)
    elif model_name == 'EEGConformer':
        from braindecode.models.eegconformer import EEGConformer
        model = EEGConformer(n_chans = n_chans, n_outputs = 2, n_times=1280, final_fc_length='auto')
    elif model_name == 'SPaRCNet':
        model = DenseNetClassifier(in_channels=n_chans, num_classes=2).to(device)  #  SPaRCNet
    elif model_name == 'ContraWR':
        model = ContraWR(in_channels=n_chans, n_classes=2, fft=200, steps=20)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    # print('parameters:', sum(param.numel() for param in model.parameters() if param.requires_grad))
    counts = list()
    for X, y in tqdm(test_dataset):
        X = X.to(device)
        # X = X.unsqueeze(dim=1)
        # y = y.to(device)
        X = X.to(torch.float32)
        # 将 tensor 转换为 numpy ndarray
        X = X.numpy()
        data_counts = evaluate_accuracy(X, net=model, device_acc=device,
                                        bool_use=True, biomarker_dir=biomarker_dir,
                                        filtered_biomarkers=None,
                                        no_match_penalty=no_match_penalty)
        counts.append(data_counts)
    return counts


if __name__ == "__main__":
    # 示例用法
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("生物标志物增强预测模块")
    print("=" * 80)
    for patient in range(6,7):
        biomarker_dir = r"D:\2025_important_projects\data\chbmit_biomarkers\%02d\preprocess_eeg"% patient
        data_root_dir = r'D:\public_data\CHBMIT\1_data_clean\chb%02d' % patient
        weight_model_path = r'D:\public_data\CHBMIT\weight\eeginception+se'
        segment_info_template = r'D:\public_data\CHBMIT\segment_clean_18channels\30-1-240\chb%02d\segment_info.json' % patient
        model = None

        # 筛选生物标志物：只保留在前期匹配但不在间期匹配的标志物
        print("\n开始筛选生物标志物...")
        filtered_biomarkers, biomarker_stats = filter_biomarkers_by_segments(
            biomarker_dir=biomarker_dir,
            data_root_dir=data_root_dir,
            segment_info_path=segment_info_template,
            match_threshold=0.7,
            sample_window_size=15360
        )
        if len(filtered_biomarkers) == 0:
            print('未找到合适标志物')
            break
        
        # 保存筛选后的生物标志物
        filtered_biomarkers_dir = r"D:\2025_important_projects\data\chbmit_biomarkers\%02d\filtered_biomarkers" % patient
        save_filtered_biomarkers(
            filtered_biomarkers=filtered_biomarkers,
            biomarker_stats=biomarker_stats,
            save_dir=filtered_biomarkers_dir,
            original_biomarker_dir=biomarker_dir
        )
        continue
        data_sub_dirs = os.listdir(data_root_dir)
        data_sub_dirs = list(filter(lambda x: x.endswith('.npy'), data_sub_dirs))
        example = data_sub_dirs[0]
        example_sub_patient_dirs = os.path.join(data_root_dir, example)
        example_data = np.load(example_sub_patient_dirs)
        example_data = np.squeeze(example_data)
        n_chans = example_data.shape[0]
        if os.path.exists(weight_model_path):
            weight_files = [f for f in os.listdir(weight_model_path) if f.endswith('.pth')]
            if weight_files:
                weight_files = sorted(weight_files, key=natural_keys)
                model_path = os.path.join(weight_model_path, weight_files[patient - 1])
                print(f"找到模型文件: {model_path}")
                try:
                    model = load_model_robust(model_path, n_chans=n_chans, n_classes=2, device=device)
                except Exception as e:
                    print(f"模型加载失败: {e}")

        each_patient_subarrays = {}
        with open(segment_info_template) as f:
            files_list = json.load(f)
        for dict_file in files_list:
            result = re.match("Pre", dict_file["Label"])
            if result:
                data = np.load(os.path.join(data_root_dir, dict_file['File']))
                data = np.squeeze(data)
                n_chans = data.shape[1]
                counts = list()
                j = 0
                while (j + 1) * 15360 <= data.shape[-1]:
                    split_data = data[:, j * 15360:(j + 1) * 15360]
                    data_counts = evaluate_accuracy(split_data, net=model, device_acc=device,
                                                    bool_use=True, biomarker_dir=biomarker_dir,
                                                    filtered_biomarkers=filtered_biomarkers)
                    counts.append(data_counts)
                    j += 1

                # 调用函数
                subarrays = find_subarrays(counts)
                subarrays = [(index + 1, value) for index, value in enumerate(subarrays)]
                # 将键和值添加到字典中
                each_patient_subarrays[dict_file["File"]] = subarrays
        print(each_patient_subarrays)
        # 调用函数并打印结果
        results_time = process_data(each_patient_subarrays)
        pat_onset_segs, total_time_hours = get_onset_info()
        ture_n = len(pat_onset_segs)
        n_predict, time_length_aver, sensitivity, fdr_h = calculate_scores(results_time)
        row_csv = [n_predict, time_length_aver, sensitivity, fdr_h]
        print('n_predict:%02d次, time_length_aver:%.3f%%, sensitivity:%.3f, fdr_h:%.3f,ture_n:%02d次' % (
            n_predict, time_length_aver, sensitivity, fdr_h, ture_n))
        print('------------------------------------------------------------------')

    # 测试预测
    # sample_data = np.random.randn(22, 1280)  # 示例数据
    # enhanced_probs, info = predictor.predict(sample_data, sample_index=0)

    # print(f"基础预测: {info['base_probs']}")
    # print(f"增强预测: {enhanced_probs}")
    # print(f"生物标志物匹配数: {info['biomarker_matches']}")
    # print(f"统计信息: {predictor.get_statistics()}")
    # print("模块已加载，请根据实际需求配置使用")
