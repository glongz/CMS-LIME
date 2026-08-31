#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基元矩阵存储模块

该模块负责将基元的真实矩阵值存储到独立文件中，
与pkl文件中的元数据分离，提高存储效率和可读性。

作者: CMS-LIME Framework
日期: 2025-01-27
"""

import numpy as np
import os
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pickle
from pathlib import Path

class PrimitiveMatrixStorage:
    """基元矩阵存储管理器"""
    
    def __init__(self, base_path: str):
        """
        初始化矩阵存储管理器
        
        Args:
            base_path: 基元库基础路径
        """
        self.base_path = Path(base_path)
        self.matrices_dir = self.base_path / "matrices"
        self.matrices_dir.mkdir(exist_ok=True)
        
        # 为不同类型的基元创建子目录
        self.microstate_dir = self.matrices_dir / "microstate"
        self.shapelet_dir = self.matrices_dir / "shapelet"
        self.timefreq_dir = self.matrices_dir / "timefreq"
        
        for dir_path in [self.microstate_dir, self.shapelet_dir, self.timefreq_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def save_primitive_matrix(self, primitive_id: str, primitive_type: str, 
                            matrix_data: Dict[str, np.ndarray]) -> bool:
        """
        保存基元的矩阵数据
        
        Args:
            primitive_id: 基元唯一标识符
            primitive_type: 基元类型 ('microstate', 'shapelet', 'timefreq')
            matrix_data: 矩阵数据字典
            
        Returns:
            是否保存成功
        """
        try:
            # 选择对应的目录
            if primitive_type == 'microstate':
                save_dir = self.microstate_dir
            elif primitive_type == 'shapelet':
                save_dir = self.shapelet_dir
            elif primitive_type == 'timefreq':
                save_dir = self.timefreq_dir
            else:
                print(f"未知的基元类型: {primitive_type}")
                return False
            
            # 创建基元专用文件夹
            primitive_dir = save_dir / primitive_id
            primitive_dir.mkdir(exist_ok=True)
            
            # 保存每个矩阵
            matrix_info = {}
            for matrix_name, matrix_array in matrix_data.items():
                if matrix_array is not None and isinstance(matrix_array, np.ndarray):
                    # 保存矩阵到.npy文件
                    matrix_file = primitive_dir / f"{matrix_name}.npy"
                    np.save(matrix_file, matrix_array)
                    
                    # 记录矩阵信息
                    matrix_info[matrix_name] = {
                        'shape': matrix_array.shape,
                        'dtype': str(matrix_array.dtype),
                        'file': str(matrix_file.relative_to(self.base_path))
                    }
            
            # 保存矩阵信息到JSON文件
            info_file = primitive_dir / "matrix_info.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(matrix_info, f, indent=2, ensure_ascii=False)
            
            print(f"基元 {primitive_id} 的矩阵数据已保存到 {primitive_dir}")
            return True
            
        except Exception as e:
            print(f"保存基元 {primitive_id} 的矩阵数据失败: {e}")
            return False
    
    def load_primitive_matrix(self, primitive_id: str, primitive_type: str) -> Optional[Dict[str, np.ndarray]]:
        """
        加载基元的矩阵数据
        
        Args:
            primitive_id: 基元唯一标识符
            primitive_type: 基元类型
            
        Returns:
            矩阵数据字典，如果不存在则返回None
        """
        try:
            # 选择对应的目录
            if primitive_type == 'microstate':
                load_dir = self.microstate_dir
            elif primitive_type == 'shapelet':
                load_dir = self.shapelet_dir
            elif primitive_type == 'timefreq':
                load_dir = self.timefreq_dir
            else:
                return None
            
            primitive_dir = load_dir / primitive_id
            info_file = primitive_dir / "matrix_info.json"
            
            if not info_file.exists():
                return None
            
            # 加载矩阵信息
            with open(info_file, 'r', encoding='utf-8') as f:
                matrix_info = json.load(f)
            
            # 加载所有矩阵
            matrix_data = {}
            for matrix_name, info in matrix_info.items():
                matrix_file = self.base_path / info['file']
                if matrix_file.exists():
                    matrix_data[matrix_name] = np.load(matrix_file)
            
            return matrix_data
            
        except Exception as e:
            print(f"加载基元 {primitive_id} 的矩阵数据失败: {e}")
            return None
    
    def get_primitive_matrix_info(self, primitive_id: str, primitive_type: str) -> Optional[Dict[str, Any]]:
        """
        获取基元矩阵信息（不加载实际数据）
        
        Args:
            primitive_id: 基元唯一标识符
            primitive_type: 基元类型
            
        Returns:
            矩阵信息字典
        """
        try:
            if primitive_type == 'microstate':
                load_dir = self.microstate_dir
            elif primitive_type == 'shapelet':
                load_dir = self.shapelet_dir
            elif primitive_type == 'timefreq':
                load_dir = self.timefreq_dir
            else:
                return None
            
            primitive_dir = load_dir / primitive_id
            info_file = primitive_dir / "matrix_info.json"
            
            if not info_file.exists():
                return None
            
            with open(info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            print(f"获取基元 {primitive_id} 的矩阵信息失败: {e}")
            return None
    
    def list_stored_primitives(self) -> Dict[str, List[str]]:
        """
        列出所有已存储矩阵的基元
        
        Returns:
            按类型分组的基元ID列表
        """
        stored_primitives = {
            'microstate': [],
            'shapelet': [],
            'timefreq': []
        }
        
        for primitive_type, type_dir in [
            ('microstate', self.microstate_dir),
            ('shapelet', self.shapelet_dir),
            ('timefreq', self.timefreq_dir)
        ]:
            if type_dir.exists():
                for primitive_dir in type_dir.iterdir():
                    if primitive_dir.is_dir() and (primitive_dir / "matrix_info.json").exists():
                        stored_primitives[primitive_type].append(primitive_dir.name)
        
        return stored_primitives
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Returns:
            存储统计信息
        """
        stored_primitives = self.list_stored_primitives()
        
        stats = {
            'total_stored_primitives': sum(len(ids) for ids in stored_primitives.values()),
            'by_type': {ptype: len(ids) for ptype, ids in stored_primitives.items()},
            'storage_path': str(self.matrices_dir),
            'disk_usage': self._calculate_disk_usage()
        }
        
        return stats
    
    def _calculate_disk_usage(self) -> Dict[str, int]:
        """
        计算磁盘使用量
        
        Returns:
            磁盘使用量信息（字节）
        """
        usage = {'total': 0, 'by_type': {}}
        
        for primitive_type, type_dir in [
            ('microstate', self.microstate_dir),
            ('shapelet', self.shapelet_dir),
            ('timefreq', self.timefreq_dir)
        ]:
            type_usage = 0
            if type_dir.exists():
                for file_path in type_dir.rglob('*'):
                    if file_path.is_file():
                        type_usage += file_path.stat().st_size
            
            usage['by_type'][primitive_type] = type_usage
            usage['total'] += type_usage
        
        return usage

def create_matrix_storage(base_path: str) -> PrimitiveMatrixStorage:
    """
    创建基元矩阵存储管理器的便捷函数
    
    Args:
        base_path: 基元库基础路径
        
    Returns:
        矩阵存储管理器实例
    """
    return PrimitiveMatrixStorage(base_path)

if __name__ == "__main__":
    # 测试矩阵存储功能
    print("基元矩阵存储模块测试")
    
    # 创建存储管理器
    storage = create_matrix_storage("test_primitive_library")
    
    # 创建测试矩阵数据
    test_matrices = {
        'topography': np.random.randn(64),  # 微状态拓扑图
        'pattern': np.random.randn(100, 3),  # shapelet模式
        'power_spectrum': np.random.randn(50, 10)  # 时频功率谱
    }
    
    # 保存测试数据
    success = storage.save_primitive_matrix(
        primitive_id="test_primitive_001",
        primitive_type="microstate",
        matrix_data={'topography': test_matrices['topography']}
    )
    
    print(f"保存测试: {success}")
    
    # 加载测试数据
    loaded_data = storage.load_primitive_matrix(
        primitive_id="test_primitive_001",
        primitive_type="microstate"
    )
    
    print(f"加载测试: {loaded_data is not None}")
    if loaded_data:
        print(f"加载的矩阵形状: {loaded_data['topography'].shape}")
    
    # 获取统计信息
    stats = storage.get_storage_statistics()
    print(f"存储统计: {stats}")
    
    print("矩阵存储测试完成")