#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基元库管理模块

该模块实现了一个共享的基元库，用于存储和重用所有样本的基元解释，
避免每个样本都重新生成新的基元，提高解释效率和一致性。

作者: CMS-LIME Framework
日期: 2025-01-27
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
from collections import defaultdict
import hashlib
from primitive_matrix_storage import PrimitiveMatrixStorage, create_matrix_storage

@dataclass
class PrimitiveInfo:
    """基元信息类"""
    primitive_type: str  # 'microstate', 'shapelet', 'timefreq'
    primitive_id: str    # 唯一标识符
    channels: List[int]  # 涉及的通道
    time_range: Tuple[int, int]  # 时间范围
    freq_range: Optional[Tuple[float, float]] = None  # 频率范围（时频基元）
    microstate_id: Optional[int] = None  # 微状态ID（微状态基元）
    shapelet_pattern: Optional[np.ndarray] = None  # shapelet模式（shapelet基元）
    quality_score: float = 0.0  # 质量分数
    usage_count: int = 0  # 使用次数
    importance_history: List[float] = None  # 历史重要性分数
    
    # 新增：矩阵数据存储标识
    has_stored_matrices: bool = False  # 是否已存储矩阵数据
    matrix_info: Optional[Dict[str, Any]] = None  # 矩阵信息
    
    def __post_init__(self):
        if self.importance_history is None:
            self.importance_history = []
        if self.matrix_info is None:
            self.matrix_info = {}
        
        # 确保channels是整数列表（处理从JSON加载的字符串类型）
        if self.channels:
            try:
                self.channels = [int(ch) if isinstance(ch, (str, np.str_)) else ch for ch in self.channels]
            except (ValueError, TypeError):
                # 如果转换失败，使用默认值
                self.channels = [0]
        else:
            self.channels = []

class PrimitiveLibrary:
    """基元库类"""
    
    def __init__(self, library_path: str = "primitive_library.pkl"):
        """
        初始化基元库
        
        Args:
            library_path: 基元库保存路径
        """
        self.library_path = library_path
        self.primitives: Dict[str, PrimitiveInfo] = {}
        self.type_index: Dict[str, List[str]] = defaultdict(list)  # 按类型索引
        self.channel_index: Dict[int, List[str]] = defaultdict(list)  # 按通道索引
        self.quality_index: List[Tuple[float, str]] = []  # 按质量排序的索引
        
        # 初始化矩阵存储管理器
        library_dir = os.path.dirname(library_path) if os.path.dirname(library_path) else "."
        self.matrix_storage = create_matrix_storage(library_dir)
        
        # 加载已有的基元库
        self.load_library()
    
    def add_primitive(self, primitive: PrimitiveInfo) -> str:
        """
        添加基元到库中
        
        Args:
            primitive: 基元信息
            
        Returns:
            primitive_id: 基元唯一标识符
        """
        # 生成唯一ID
        if primitive.primitive_id is None or primitive.primitive_id == "":
            primitive.primitive_id = self._generate_primitive_id(primitive)
        
        # 检查是否已存在相似基元
        similar_id = self._find_similar_primitive(primitive)
        if similar_id:
            # 更新已有基元的使用次数和质量分数
            existing = self.primitives[similar_id]
            existing.usage_count += 1
            existing.quality_score = max(existing.quality_score, primitive.quality_score)
            return similar_id
        
        # 添加新基元
        self.primitives[primitive.primitive_id] = primitive
        
        # 更新索引
        self.type_index[primitive.primitive_type].append(primitive.primitive_id)
        for channel in primitive.channels:
            self.channel_index[channel].append(primitive.primitive_id)
        self.quality_index.append((primitive.quality_score, primitive.primitive_id))
        self.quality_index.sort(reverse=True)  # 按质量降序排列
        
        return primitive.primitive_id
    
    def add_primitive_with_matrices(self, primitive: PrimitiveInfo, 
                                  matrix_data: Optional[Dict[str, np.ndarray]] = None) -> str:
        """
        添加基元到库中并保存矩阵数据（增强版）
        
        Args:
            primitive: 基元信息
            matrix_data: 矩阵数据字典
            
        Returns:
            primitive_id: 基元唯一标识符
        """
        # 数据质量预检查
        if matrix_data:
            quality_check_result = self._perform_data_quality_check(primitive, matrix_data)
            if not quality_check_result['passed']:
                print(f"基元 {primitive.primitive_id} 数据质量检查失败: {quality_check_result['reason']}")
                # 尝试数据修复
                repaired_matrices = self._attempt_data_repair(matrix_data, quality_check_result)
                if repaired_matrices is None:
                    print(f"基元 {primitive.primitive_id} 数据修复失败")
                else:
                    matrix_data = repaired_matrices
                    print(f"基元 {primitive.primitive_id} 数据已修复")
        
        # 首先添加基元元数据
        primitive_id = self.add_primitive(primitive)
        
        # 如果有矩阵数据，使用渐进式保存策略
        if matrix_data and any(v is not None for v in matrix_data.values()):
            save_result = self._progressive_matrix_save(primitive_id, primitive.primitive_type, matrix_data)
            
            if save_result['success']:
                # 更新基元的矩阵存储状态
                if primitive_id in self.primitives:
                    self.primitives[primitive_id].has_stored_matrices = True
                    self.primitives[primitive_id].matrix_info = {
                        'saved_keys': save_result['saved_keys'],
                        'failed_keys': save_result['failed_keys'],
                        'total_size': save_result['total_size']
                    }
                    print(f"基元 {primitive_id} 的矩阵数据保存成功")
                    print(f"  保存的矩阵: {save_result['saved_keys']}")
                    if save_result['failed_keys']:
                        print(f"  保存失败的矩阵: {save_result['failed_keys']}")
            else:
                print(f"基元 {primitive_id} 的矩阵数据保存失败: {save_result['error']}")
        
        return primitive_id
    
    def _perform_data_quality_check(self, primitive: PrimitiveInfo, matrix_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        执行数据质量检查
        
        Args:
            primitive: 基元信息
            matrix_data: 矩阵数据
            
        Returns:
            检查结果字典
        """
        result = {'passed': True, 'reason': '', 'issues': []}
        
        try:
            for key, matrix in matrix_data.items():
                if matrix is None:
                    continue
                    
                # 检查数据类型
                if not isinstance(matrix, np.ndarray):
                    result['issues'].append(f"矩阵 {key} 不是numpy数组")
                    continue
                
                # 检查数据形状
                if matrix.size == 0:
                    result['issues'].append(f"矩阵 {key} 为空")
                    continue
                
                # 检查数据有效性
                if np.any(np.isnan(matrix)):
                    result['issues'].append(f"矩阵 {key} 包含NaN值")
                
                if np.any(np.isinf(matrix)):
                    result['issues'].append(f"矩阵 {key} 包含无穷值")
                
                # 检查数据范围
                if np.max(np.abs(matrix)) > 1e10:
                    result['issues'].append(f"矩阵 {key} 包含异常大的值")
            
            if result['issues']:
                result['passed'] = False
                result['reason'] = '; '.join(result['issues'])
                
        except Exception as e:
            result['passed'] = False
            result['reason'] = f"数据质量检查异常: {e}"
        
        return result
    
    def _attempt_data_repair(self, matrix_data: Dict[str, np.ndarray], 
                           quality_check_result: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        """
        尝试修复数据问题
        
        Args:
            matrix_data: 原始矩阵数据
            quality_check_result: 质量检查结果
            
        Returns:
            修复后的矩阵数据，如果无法修复则返回None
        """
        try:
            repaired_data = {}
            
            for key, matrix in matrix_data.items():
                if matrix is None:
                    continue
                
                repaired_matrix = matrix.copy()
                
                # 修复NaN值
                if np.any(np.isnan(repaired_matrix)):
                    repaired_matrix = np.nan_to_num(repaired_matrix, nan=0.0)
                
                # 修复无穷值
                if np.any(np.isinf(repaired_matrix)):
                    repaired_matrix = np.nan_to_num(repaired_matrix, posinf=1e6, neginf=-1e6)
                
                # 限制数据范围
                if np.max(np.abs(repaired_matrix)) > 1e10:
                    repaired_matrix = np.clip(repaired_matrix, -1e6, 1e6)
                
                repaired_data[key] = repaired_matrix
            
            return repaired_data
            
        except Exception as e:
            print(f"数据修复失败: {e}")
            return None
    
    def _progressive_matrix_save(self, primitive_id: str, primitive_type: str, 
                               matrix_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        渐进式矩阵保存策略
        
        Args:
            primitive_id: 基元ID
            primitive_type: 基元类型
            matrix_data: 矩阵数据
            
        Returns:
            保存结果字典
        """
        result = {
            'success': False,
            'saved_keys': [],
            'failed_keys': [],
            'total_size': 0,
            'error': ''
        }
        
        try:
            # 按矩阵大小排序，优先保存小矩阵
            sorted_items = sorted(matrix_data.items(), 
                                key=lambda x: x[1].nbytes if x[1] is not None else 0)
            
            for key, matrix in sorted_items:
                if matrix is None:
                    continue
                
                try:
                    # 尝试保存单个矩阵
                    single_matrix_data = {key: matrix}
                    success = self.matrix_storage.save_primitive_matrix(
                        primitive_id, primitive_type, single_matrix_data
                    )
                    
                    if success:
                        result['saved_keys'].append(key)
                        result['total_size'] += matrix.nbytes
                    else:
                        result['failed_keys'].append(key)
                        
                except Exception as e:
                    result['failed_keys'].append(key)
                    print(f"保存矩阵 {key} 失败: {e}")
            
            # 如果至少有一个矩阵保存成功，认为整体成功
            result['success'] = len(result['saved_keys']) > 0
            
            if not result['success']:
                result['error'] = "所有矩阵保存都失败"
                
        except Exception as e:
            result['error'] = f"渐进式保存异常: {e}"
        
        return result
    
    def get_primitive_matrices(self, primitive_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        获取基元的矩阵数据
        
        Args:
            primitive_id: 基元ID
            
        Returns:
            矩阵数据字典
        """
        if primitive_id not in self.primitives:
            return None
        
        primitive = self.primitives[primitive_id]
        if not primitive.has_stored_matrices:
            return None
        
        return self.matrix_storage.load_primitive_matrix(
            primitive_id, primitive.primitive_type
        )

    def get_primitives_by_type(self, primitive_type: str, 
                              min_quality: float = 0.0,
                              max_count: Optional[int] = None) -> List[PrimitiveInfo]:
        """
        按类型获取基元
        
        Args:
            primitive_type: 基元类型
            min_quality: 最小质量分数
            max_count: 最大返回数量
            
        Returns:
            基元列表
        """
        primitive_ids = self.type_index.get(primitive_type, [])
        primitives = []
        
        for pid in primitive_ids:
            primitive = self.primitives[pid]
            if primitive.quality_score >= min_quality:
                primitives.append(primitive)
        
        # 按质量分数排序
        primitives.sort(key=lambda x: x.quality_score, reverse=True)
        
        if max_count:
            primitives = primitives[:max_count]
            
        return primitives
    
    def get_primitives_by_channels(self, channels: List[int],
                                  min_quality: float = 0.0) -> List[PrimitiveInfo]:
        """
        按通道获取基元
        
        Args:
            channels: 通道列表
            min_quality: 最小质量分数
            
        Returns:
            基元列表
        """
        relevant_ids = set()
        for channel in channels:
            relevant_ids.update(self.channel_index.get(channel, []))
        
        primitives = []
        for pid in relevant_ids:
            primitive = self.primitives[pid]
            if primitive.quality_score >= min_quality:
                primitives.append(primitive)
        
        return primitives
    
    def get_top_primitives(self, n: int = 50) -> List[PrimitiveInfo]:
        """
        获取质量最高的N个基元
        
        Args:
            n: 返回数量
            
        Returns:
            基元列表
        """
        top_ids = [pid for _, pid in self.quality_index[:n]]
        return [self.primitives[pid] for pid in top_ids]
    
    def update_primitive_importance(self, primitive_id: str, importance: float):
        """
        更新基元的重要性历史
        
        Args:
            primitive_id: 基元ID
            importance: 重要性分数
        """
        if primitive_id in self.primitives:
            primitive = self.primitives[primitive_id]
            primitive.importance_history.append(importance)
            primitive.usage_count += 1
            
            # 更新平均质量分数
            if primitive.importance_history:
                primitive.quality_score = np.mean(np.abs(primitive.importance_history))
    
    def get_primitive_statistics(self) -> Dict[str, Any]:
        """
        获取基元库统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_primitives': len(self.primitives),
            'by_type': {},
            'quality_distribution': {},
            'usage_distribution': {},
            'top_primitives': [],
            'matrix_storage': {}
        }
        
        # 按类型统计
        for ptype, pids in self.type_index.items():
            stats['by_type'][ptype] = len(pids)
        
        # 质量分布
        qualities = [p.quality_score for p in self.primitives.values()]
        if qualities:
            stats['quality_distribution'] = {
                'mean': np.mean(qualities),
                'std': np.std(qualities),
                'min': np.min(qualities),
                'max': np.max(qualities)
            }
        
        # 使用次数分布
        usage_counts = [p.usage_count for p in self.primitives.values()]
        if usage_counts:
            stats['usage_distribution'] = {
                'mean': np.mean(usage_counts),
                'std': np.std(usage_counts),
                'min': np.min(usage_counts),
                'max': np.max(usage_counts)
            }
        
        # 顶级基元
        top_primitives = self.get_top_primitives(10)
        stats['top_primitives'] = [
            {
                'id': p.primitive_id,
                'type': p.primitive_type,
                'quality': p.quality_score,
                'usage': p.usage_count,
                'has_matrices': p.has_stored_matrices
            }
            for p in top_primitives
        ]
        
        # 矩阵存储统计
        matrix_stats = self.matrix_storage.get_storage_statistics()
        stats['matrix_storage'] = matrix_stats
        
        # 矩阵存储覆盖率
        primitives_with_matrices = sum(1 for p in self.primitives.values() if p.has_stored_matrices)
        stats['matrix_coverage'] = {
            'total_with_matrices': primitives_with_matrices,
            'coverage_rate': primitives_with_matrices / len(self.primitives) if self.primitives else 0.0
        }
        
        return stats
    
    def save_library(self):
        """
        保存基元库到文件
        """
        try:
            with open(self.library_path, 'wb') as f:
                pickle.dump({
                    'primitives': self.primitives,
                    'type_index': dict(self.type_index),
                    'channel_index': dict(self.channel_index),
                    'quality_index': self.quality_index
                }, f)
            print(f"基元库已保存到: {self.library_path}")
        except Exception as e:
            print(f"保存基元库失败: {e}")
    
    def load_library(self):
        """
        从文件加载基元库
        """
        if os.path.exists(self.library_path):
            try:
                with open(self.library_path, 'rb') as f:
                    data = pickle.load(f)
                    self.primitives = data['primitives']
                    self.type_index = defaultdict(list, data['type_index'])
                    self.channel_index = defaultdict(list, data['channel_index'])
                    self.quality_index = data['quality_index']
                print(f"基元库已从 {self.library_path} 加载，包含 {len(self.primitives)} 个基元")
            except Exception as e:
                print(f"加载基元库失败: {e}，将创建新的基元库")
                self._initialize_empty_library()
        else:
            print("基元库文件不存在，将创建新的基元库")
            self._initialize_empty_library()
    
    def _initialize_empty_library(self):
        """初始化空的基元库"""
        self.primitives = {}
        self.type_index = defaultdict(list)
        self.channel_index = defaultdict(list)
        self.quality_index = []
    
    def _generate_primitive_id(self, primitive: PrimitiveInfo) -> str:
        """
        生成基元唯一ID
        
        Args:
            primitive: 基元信息
            
        Returns:
            唯一ID字符串
        """
        # 创建基于基元特征的哈希
        content = f"{primitive.primitive_type}_{primitive.channels}_{primitive.time_range}"
        if primitive.freq_range:
            content += f"_{primitive.freq_range}"
        if primitive.microstate_id is not None:
            content += f"_ms{primitive.microstate_id}"
        
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _find_similar_primitive(self, primitive: PrimitiveInfo) -> Optional[str]:
        """
        查找相似的基元
        
        Args:
            primitive: 基元信息
            
        Returns:
            相似基元的ID，如果没有则返回None
        """
        # 获取同类型的基元
        candidate_ids = self.type_index.get(primitive.primitive_type, [])
        
        for pid in candidate_ids:
            existing = self.primitives[pid]
            
            # 检查通道重叠
            channel_overlap = len(set(primitive.channels) & set(existing.channels))
            if channel_overlap == 0:
                continue
            
            # 检查时间重叠
            time_overlap = self._compute_time_overlap(primitive.time_range, existing.time_range)
            if time_overlap < 0.5:  # 至少50%重叠
                continue
            
            # 对于特定类型的额外检查
            if primitive.primitive_type == 'microstate':
                if primitive.microstate_id == existing.microstate_id:
                    return pid
            elif primitive.primitive_type == 'timefreq':
                if primitive.freq_range and existing.freq_range:
                    freq_overlap = self._compute_freq_overlap(primitive.freq_range, existing.freq_range)
                    if freq_overlap > 0.8:  # 80%频率重叠
                        return pid
            else:  # shapelet
                # 对于shapelet，可以比较模式相似性
                if (primitive.shapelet_pattern is not None and 
                    existing.shapelet_pattern is not None):
                    similarity = self._compute_pattern_similarity(
                        primitive.shapelet_pattern, existing.shapelet_pattern
                    )
                    if similarity > 0.9:  # 90%相似性
                        return pid
        
        return None
    
    def _compute_time_overlap(self, range1: Tuple[int, int], range2: Tuple[int, int]) -> float:
        """计算时间范围重叠比例"""
        start1, end1 = range1
        start2, end2 = range2
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_length = overlap_end - overlap_start
        total_length = max(end1 - start1, end2 - start2)
        
        return overlap_length / total_length if total_length > 0 else 0.0
    
    def _compute_freq_overlap(self, range1: Tuple[float, float], range2: Tuple[float, float]) -> float:
        """计算频率范围重叠比例"""
        start1, end1 = range1
        start2, end2 = range2
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_length = overlap_end - overlap_start
        total_length = max(end1 - start1, end2 - start2)
        
        return overlap_length / total_length if total_length > 0 else 0.0
    
    def _compute_pattern_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """计算模式相似性"""
        try:
            # 归一化模式
            p1_norm = (pattern1 - np.mean(pattern1)) / (np.std(pattern1) + 1e-8)
            p2_norm = (pattern2 - np.mean(pattern2)) / (np.std(pattern2) + 1e-8)
            
            # 计算相关系数
            correlation = np.corrcoef(p1_norm.flatten(), p2_norm.flatten())[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0


def create_primitive_library(library_path: str = "primitive_library.pkl") -> PrimitiveLibrary:
    """
    创建基元库的便捷函数
    
    Args:
        library_path: 基元库保存路径
        
    Returns:
        基元库实例
    """
    return PrimitiveLibrary(library_path)


if __name__ == "__main__":
    # 示例使用
    print("基元库管理模块测试")
    
    # 创建基元库
    library = create_primitive_library("test_primitive_library.pkl")
    
    # 创建示例基元
    primitive1 = PrimitiveInfo(
        primitive_type="microstate",
        primitive_id="",
        channels=[0, 1, 2],
        time_range=(100, 200),
        microstate_id=1,
        quality_score=0.8
    )
    
    primitive2 = PrimitiveInfo(
        primitive_type="shapelet",
        primitive_id="",
        channels=[3, 4],
        time_range=(150, 250),
        shapelet_pattern=np.random.randn(50),
        quality_score=0.6
    )
    
    # 添加基元
    id1 = library.add_primitive(primitive1)
    id2 = library.add_primitive(primitive2)
    
    print(f"添加基元: {id1}, {id2}")
    
    # 获取统计信息
    stats = library.get_primitive_statistics()
    print(f"基元库统计: {stats}")
    
    # 保存基元库
    library.save_library()
    
    print("基元库测试完成")