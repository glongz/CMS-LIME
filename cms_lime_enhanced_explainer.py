#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版CMS-LIME解释器

集成基元库功能，提供更高效和一致的解释结果。
解决了原版本中基元选择数量少、重要性低等问题。

主要改进：
1. 集成基元库，避免重复生成基元
2. 改进基元选择策略，增加选择数量
3. 优化重要性计算，提高解释质量
4. 添加基元质量评估机制

作者: CMS-LIME Framework
日期: 2025-01-27
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

# 导入原有模块
from cms_lime_explainer import CMSLimeExplainer, CMSLimeConfig
from primitive_library import PrimitiveLibrary, PrimitiveInfo, create_primitive_library
from microstate_analysis import MicrostateAnalyzer
from shapelet_analysis import ShapeletAnalyzer
from timefreq_analysis import TimeFreqAnalyzer
from causal_perturbation import CausalPerturbationEngine
from local_regression import LocalRegressionSelector, WeightKernel, SparseRegressor

warnings.filterwarnings('ignore')

@dataclass
class EnhancedCMSLimeConfig(CMSLimeConfig):
    """增强版CMS-LIME配置"""
    # 基元库配置
    use_primitive_library: bool = True
    library_path: str = "cms_lime_primitive_library.pkl"
    min_primitive_quality: float = 0.01  # 最小基元质量阈值
    max_library_primitives: int = 200  # 基元库最大基元数
    
    # 改进的选择策略
    adaptive_selection: bool = True  # 自适应选择
    min_selected_primitives: int = 5  # 最小选择基元数
    max_selected_primitives: int = 20  # 最大选择基元数
    quality_weight: float = 0.4  # 质量权重（降低以平衡重要性）
    diversity_weight: float = 0.6  # 多样性/重要性权重（提高以优先考虑重要性）
    
    # 类型特定限制
    max_microstate_primitives: int = 50  # 微状态基元最大数量
    max_shapelet_primitives: int = 60  # shapelet基元最大数量
    max_timefreq_primitives: int = 70  # 时频基元最大数量
    
    # 质量阈值
    microstate_quality_threshold: float = 0.05  # 微状态基元质量阈值
    shapelet_quality_threshold: float = 0.03  # shapelet基元质量阈值
    timefreq_quality_threshold: float = 0.02  # 时频基元质量阈值
    
    # 重要性计算改进
    importance_aggregation: str = 'weighted_mean'  # 'mean', 'weighted_mean', 'max'
    perturbation_strength: float = 0.5  # 扰动强度
    stability_iterations: int = 3  # 稳定性迭代次数

class EnhancedCMSLimeExplainer(CMSLimeExplainer):
    """增强版CMS-LIME解释器"""
    
    def __init__(self, config: EnhancedCMSLimeConfig = None):
        # 使用增强配置
        if config is None:
            config = EnhancedCMSLimeConfig()
        
        super().__init__(config)
        self.enhanced_config = config
        
        # 初始化基元库
        if self.enhanced_config.use_primitive_library:
            self.primitive_library = create_primitive_library(config.library_path)
            print(f"基元库已初始化，当前包含 {len(self.primitive_library.primitives)} 个基元")
        else:
            self.primitive_library = None
        
        # 基元缓存
        self.cached_primitives = {
            'microstate': [],
            'shapelet': [],
            'timefreq': []
        }
        
        # 解释历史
        self.explanation_history = []
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, model: Any):
        """拟合解释器并构建/更新基元库"""
        print("开始拟合增强版CMS-LIME解释器...")
        
        # 调用父类的fit方法
        super().fit(X_train, y_train, model)
        
        # 如果使用基元库，预先构建高质量基元
        if self.primitive_library is not None:
            self._build_primitive_library(X_train, y_train)
        
        print("增强版CMS-LIME解释器拟合完成!")
    
    def _build_primitive_library(self, X_train: np.ndarray, y_train: np.ndarray):
        """构建基元库"""
        print("构建基元库...")
        
        # 确保shapelet分析器已拟合
        if not hasattr(self.shapelet_analyzer, 'fitted') or not self.shapelet_analyzer.fitted:
            print("重新拟合shapelet分析器...")
            self.shapelet_analyzer.fit(X_train, y_train)
        
        # 统计各类型基元数量
        type_counts = {'microstate': 0, 'shapelet': 0, 'timefreq': 0}
        
        # 从训练数据中提取高质量基元
        n_samples = min(10, len(X_train))  # 使用前10个样本构建基元库
        
        for i in range(n_samples):
            if self.config.verbose:
                print(f"从样本 {i} 提取基元...")
            
            sample = X_train[i]
            
            # 提取微状态基元（应用类型特定限制）
            if type_counts['microstate'] < self.enhanced_config.max_microstate_primitives:
                microstate_primitives = self._extract_microstate_primitives(sample)
                for primitive in microstate_primitives:
                    if (primitive.quality_score >= self.enhanced_config.microstate_quality_threshold and 
                        type_counts['microstate'] < self.enhanced_config.max_microstate_primitives):
                        # 准备微状态矩阵数据
                        matrix_data = {}
                        if hasattr(self.microstate_analyzer, 'get_microstate_maps') and primitive.microstate_id is not None:
                            try:
                                microstate_maps = self.microstate_analyzer.get_microstate_maps()
                                if microstate_maps is not None and primitive.microstate_id < microstate_maps.shape[1]:
                                    # 获取特定微状态的拓扑地图
                                    topography = microstate_maps[:, primitive.microstate_id]
                                    matrix_data['topography'] = topography
                                    matrix_data['microstate_id'] = primitive.microstate_id
                            except Exception as e:
                                if self.config.verbose:
                                    print(f"获取微状态拓扑地图失败: {e}")
                                pass
                        
                        # 使用带矩阵存储的方法添加基元
                        self.primitive_library.add_primitive_with_matrices(primitive, matrix_data)
                        type_counts['microstate'] += 1
            
            # 提取shapelet基元（应用类型特定限制）
            if type_counts['shapelet'] < self.enhanced_config.max_shapelet_primitives:
                shapelet_primitives = self._extract_shapelet_primitives(sample)
                for primitive in shapelet_primitives:
                    if (primitive.quality_score >= self.enhanced_config.shapelet_quality_threshold and 
                        type_counts['shapelet'] < self.enhanced_config.max_shapelet_primitives):
                        # 准备shapelet矩阵数据
                        matrix_data = {}
                        if primitive.shapelet_pattern is not None:
                            matrix_data['pattern'] = primitive.shapelet_pattern
                        
                        # 使用带矩阵存储的方法添加基元
                        self.primitive_library.add_primitive_with_matrices(primitive, matrix_data)
                        type_counts['shapelet'] += 1
            
            # 提取时频基元（应用类型特定限制）
            if type_counts['timefreq'] < self.enhanced_config.max_timefreq_primitives:
                timefreq_primitives = self._extract_time_frequency_primitives(sample)
                for primitive in timefreq_primitives:
                    if (primitive.quality_score >= self.enhanced_config.timefreq_quality_threshold and 
                        type_counts['timefreq'] < self.enhanced_config.max_timefreq_primitives):
                        # 准备时频矩阵数据
                        matrix_data = {}
                        try:
                            # 从样本中提取时频模式
                            if primitive.channels and primitive.time_range and primitive.freq_range:
                                pattern = self._extract_timefreq_segment(
                                    sample, 
                                    primitive.channels[0], 
                                    primitive.time_range[0], 
                                    primitive.time_range[1], 
                                    primitive.freq_range
                                )
                                if pattern is not None:
                                    matrix_data['pattern'] = pattern
                        except:
                            pass
                        
                        # 使用带矩阵存储的方法添加基元
                        self.primitive_library.add_primitive_with_matrices(primitive, matrix_data)
                        type_counts['timefreq'] += 1
        
        # 保存基元库
        self.primitive_library.save_library()
        
        # 打印统计信息
        stats = self.primitive_library.get_primitive_statistics()
        print(f"基元库构建完成: {stats['total_primitives']} 个基元")
        print(f"按类型分布: {stats['by_type']}")
        print(f"各类型基元数量: {type_counts}")
    
    def _extract_microstate_primitives(self, x: np.ndarray) -> List[PrimitiveInfo]:
        """从样本中提取微状态基元"""
        primitives = []
        
        try:
            # 获取微状态序列
            microstate_seq = self.microstate_analyzer.transform(x.reshape(1, *x.shape))[0]
            
            # 生成微状态基元
            units = self._generate_microstate_units(microstate_seq)
            
            for unit in units:
                # 确保channels字段包含正确的整数列表
                channels = unit.get('channels', [])
                if not channels:  # 如果channels为空，使用默认的通道列表
                    channels = list(range(x.shape[0]))  # 使用实际的通道数
                
                primitive = PrimitiveInfo(
                    primitive_type="microstate",
                    primitive_id="",
                    channels=channels,
                    time_range=(unit.get('start_time', 0), unit.get('end_time', 0)),
                    microstate_id=unit.get('state', 0),  # 使用'state'字段作为microstate_id
                    quality_score=unit.get('duration', 1) / 10.0  # 基于持续时间计算质量分数
                )
                primitives.append(primitive)
        except Exception as e:
            if self.config.verbose:
                print(f"微状态基元提取失败: {e}")
        
        return primitives
    
    def _extract_shapelet_primitives(self, x: np.ndarray) -> List[PrimitiveInfo]:
        """从样本中提取shapelet基元"""
        primitives = []
        extraction_stats = {
            'total_attempts': 0,
            'successful_extractions': 0,
            'failed_validations': 0,
            'analyzer_shapelets': 0,
            'generated_shapelets': 0
        }
        
        try:
            if self.config.verbose:
                print("开始shapelet基元提取...")
            
            # 策略1: 检查shapelet分析器是否已拟合
            analyzer_fitted = False
            discovered_shapelets = []
            
            if hasattr(self.shapelet_analyzer, 'fitted') and self.shapelet_analyzer.fitted:
                analyzer_fitted = True
                try:
                    discovered_shapelets = self.shapelet_analyzer.get_shapelets()
                    if self.config.verbose:
                        print(f"从已拟合的shapelet分析器获取到 {len(discovered_shapelets)} 个shapelets")
                except Exception as e:
                    if self.config.verbose:
                        print(f"获取已发现shapelets失败: {e}")
                    discovered_shapelets = []
            
            # 处理分析器发现的shapelets
            if analyzer_fitted and discovered_shapelets and len(discovered_shapelets) > 0:
                for i, shapelet in enumerate(discovered_shapelets):
                    extraction_stats['total_attempts'] += 1
                    
                    try:
                        # 增强的pattern数据验证
                        if not hasattr(shapelet, 'data') or shapelet.data is None:
                            extraction_stats['failed_validations'] += 1
                            if self.config.verbose:
                                print(f"Shapelet {i}: 缺少data属性")
                            continue
                        
                        # 验证shapelet数据
                        if not self._validate_shapelet_pattern(shapelet.data):
                            extraction_stats['failed_validations'] += 1
                            if self.config.verbose:
                                print(f"Shapelet {i}: pattern验证失败")
                            continue
                        
                        # 标准化处理
                        normalized_pattern = self._robust_normalize_pattern(shapelet.data)
                        if normalized_pattern is None:
                            extraction_stats['failed_validations'] += 1
                            if self.config.verbose:
                                print(f"Shapelet {i}: 标准化失败")
                            continue
                        
                        # 重新整形pattern数据
                        pattern_data = normalized_pattern.reshape(1, -1) if normalized_pattern.ndim == 1 else normalized_pattern
                        
                        # 验证最终的pattern数据
                        if not self._validate_final_pattern_data(pattern_data):
                            extraction_stats['failed_validations'] += 1
                            if self.config.verbose:
                                print(f"Shapelet {i}: 最终pattern数据验证失败")
                            continue
                        
                        primitive = PrimitiveInfo(
                            primitive_type="shapelet",
                            primitive_id=f"analyzer_shapelet_{i}",
                            channels=[shapelet.channel] if hasattr(shapelet, 'channel') else [0],
                            time_range=(shapelet.start_pos, shapelet.start_pos + shapelet.length) if hasattr(shapelet, 'start_pos') and hasattr(shapelet, 'length') else (0, len(shapelet.data)),
                            shapelet_pattern=pattern_data,
                            quality_score=shapelet.information_gain if hasattr(shapelet, 'information_gain') else 0.5
                        )
                        primitives.append(primitive)
                        extraction_stats['successful_extractions'] += 1
                        extraction_stats['analyzer_shapelets'] += 1
                        
                    except Exception as e:
                        extraction_stats['failed_validations'] += 1
                        if self.config.verbose:
                            print(f"处理analyzer shapelet {i} 失败: {e}")
                        continue
            
            # 策略2: 从当前样本生成shapelet基元（主要策略）
            if self.config.verbose:
                print("从当前样本生成shapelet基元...")
            
            try:
                units = self._generate_shapelet_units_enhanced(x)
                
                for unit in units:
                    extraction_stats['total_attempts'] += 1
                    
                    try:
                        # 验证unit数据完整性
                        if not self._validate_unit_data(unit):
                            extraction_stats['failed_validations'] += 1
                            if self.config.verbose:
                                print(f"Unit {unit.get('id', 'unknown')}: 数据完整性验证失败")
                            continue
                        
                        # 获取并验证pattern数据
                        pattern_data = unit.get('pattern', None)
                        if pattern_data is None:
                            extraction_stats['failed_validations'] += 1
                            if self.config.verbose:
                                print(f"Unit {unit.get('id', 'unknown')}: pattern数据为空")
                            continue
                        
                        # 最终验证pattern数据
                        if not self._validate_final_pattern_data(pattern_data):
                            extraction_stats['failed_validations'] += 1
                            if self.config.verbose:
                                print(f"Unit {unit.get('id', 'unknown')}: 最终pattern数据验证失败")
                            continue
                        
                        primitive = PrimitiveInfo(
                            primitive_type="shapelet",
                            primitive_id=unit.get('id', f'generated_shapelet_{len(primitives)}'),
                            channels=unit.get('channels', []),
                            time_range=unit.get('time_range', (0, 0)),
                            shapelet_pattern=pattern_data,
                            quality_score=unit.get('quality', 0.5)
                        )
                        primitives.append(primitive)
                        extraction_stats['successful_extractions'] += 1
                        extraction_stats['generated_shapelets'] += 1
                        
                    except Exception as e:
                        extraction_stats['failed_validations'] += 1
                        if self.config.verbose:
                            print(f"处理generated unit {unit.get('id', 'unknown')} 失败: {e}")
                        continue
                        
            except Exception as e:
                if self.config.verbose:
                    print(f"生成shapelet基元失败: {e}")
            
            # 策略3: 备用提取策略（如果前面的策略都失败了）
            if len(primitives) == 0:
                if self.config.verbose:
                    print("启用备用shapelet提取策略...")
                backup_primitives = self._backup_shapelet_extraction(x)
                primitives.extend(backup_primitives)
                extraction_stats['successful_extractions'] += len(backup_primitives)
            
            # 详细的调试信息
            if self.config.verbose:
                print(f"Shapelet基元提取完成:")
                print(f"  总尝试次数: {extraction_stats['total_attempts']}")
                print(f"  成功提取: {extraction_stats['successful_extractions']}")
                print(f"  验证失败: {extraction_stats['failed_validations']}")
                print(f"  分析器shapelets: {extraction_stats['analyzer_shapelets']}")
                print(f"  生成shapelets: {extraction_stats['generated_shapelets']}")
                print(f"  最终基元数量: {len(primitives)}")
                
                # 质量分布统计
                if primitives:
                    qualities = [p.quality_score for p in primitives]
                    print(f"  质量分数统计: 平均={np.mean(qualities):.3f}, 最大={np.max(qualities):.3f}, 最小={np.min(qualities):.3f}")
                    
        except Exception as e:
            if self.config.verbose:
                print(f"Shapelet基元提取失败: {e}")
                import traceback
                traceback.print_exc()
        
        return primitives
    
    def _validate_unit_data(self, unit: dict) -> bool:
        """验证unit数据的完整性"""
        try:
            required_keys = ['id', 'channels', 'time_range', 'pattern', 'quality']
            for key in required_keys:
                if key not in unit:
                    return False
            
            # 验证channels
            channels = unit.get('channels', [])
            if not isinstance(channels, list) or len(channels) == 0:
                return False
            
            # 验证time_range
            time_range = unit.get('time_range', (0, 0))
            if not isinstance(time_range, tuple) or len(time_range) != 2:
                return False
            if time_range[1] <= time_range[0]:
                return False
            
            # 验证quality
            quality = unit.get('quality', 0)
            if not isinstance(quality, (int, float)) or quality < 0:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_final_pattern_data(self, pattern_data: np.ndarray) -> bool:
        """验证最终的pattern数据"""
        try:
            if pattern_data is None:
                return False
            
            if not isinstance(pattern_data, np.ndarray):
                return False
            
            if pattern_data.size == 0:
                return False
            
            if pattern_data.ndim == 0:
                return False
            
            # 检查是否包含有效数值
            if np.all(np.isnan(pattern_data)) or np.all(np.isinf(pattern_data)):
                return False
            
            # 检查数据范围是否合理
            if np.any(np.abs(pattern_data) > 1e6):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _backup_shapelet_extraction(self, x: np.ndarray) -> List[PrimitiveInfo]:
        """备用shapelet提取策略"""
        primitives = []
        
        try:
            if self.config.verbose:
                print("执行备用shapelet提取策略...")
            
            n_channels, n_timepoints = x.shape
            backup_count = 0
            max_backup = min(5, self.config.max_shapelet_primitives // 2)  # 最多生成一半数量的备用基元
            
            # 简单策略：从每个通道提取固定长度的片段
            for channel in range(n_channels):
                if backup_count >= max_backup:
                    break
                
                try:
                    # 选择通道中间部分的数据
                    start_pos = n_timepoints // 4
                    end_pos = 3 * n_timepoints // 4
                    
                    if end_pos - start_pos < 10:  # 确保有足够的数据点
                        continue
                    
                    # 提取片段
                    segment = x[channel, start_pos:end_pos]
                    
                    # 基本验证
                    if len(segment) < 5:
                        continue
                    
                    # 简单标准化
                    if np.std(segment) > 1e-8:
                        normalized_segment = (segment - np.mean(segment)) / np.std(segment)
                    else:
                        normalized_segment = segment - np.mean(segment)
                    
                    # 验证标准化结果
                    if not self._validate_final_pattern_data(normalized_segment.reshape(1, -1)):
                        continue
                    
                    # 计算简单质量分数
                    quality = min(0.3, np.std(normalized_segment) * 0.5)  # 备用策略的质量分数较低
                    
                    primitive = PrimitiveInfo(
                        primitive_type="shapelet",
                        primitive_id=f"backup_shapelet_{backup_count}",
                        channels=[channel],
                        time_range=(start_pos, end_pos),
                        shapelet_pattern=normalized_segment.reshape(1, -1),
                        quality_score=quality
                    )
                    
                    primitives.append(primitive)
                    backup_count += 1
                    
                    if self.config.verbose:
                        print(f"生成备用shapelet {backup_count}: 通道{channel}, 长度{len(segment)}, 质量{quality:.3f}")
                        
                except Exception as e:
                    if self.config.verbose:
                        print(f"备用提取通道{channel}失败: {e}")
                    continue
            
            if self.config.verbose:
                print(f"备用策略生成了 {len(primitives)} 个shapelet基元")
                
        except Exception as e:
            if self.config.verbose:
                print(f"备用shapelet提取策略失败: {e}")
        
        return primitives
    
    def _extract_time_frequency_primitives(self, x: np.ndarray) -> List[PrimitiveInfo]:
        """从样本中提取时频基元"""
        primitives = []
        
        try:
            # 检查时频分析器是否已拟合
            if hasattr(self.timefreq_analyzer, 'fitted') and self.timefreq_analyzer.fitted:
                # 从已拟合的分析器获取高质量时频基元
                timefreq_units = self.timefreq_analyzer.get_units()
                
                for unit in timefreq_units[:50]:  # 限制数量
                    # 计算基元质量
                    quality = self._evaluate_timefreq_unit_quality(unit, x)
                    
                    if quality > self.enhanced_config.min_primitive_quality:
                        # 提取时频模式数据
                        timefreq_pattern = self._extract_timefreq_pattern(unit, x)
                        
                        primitive = PrimitiveInfo(
                            primitive_type="timefreq",
                            primitive_id=f"tf_{unit.unit_id}",
                            channels=unit.channels,
                            time_range=unit.time_range,
                            freq_range=unit.freq_band,
                            quality_score=quality,
                            # 存储时频模式数据
                            shapelet_pattern=timefreq_pattern  # 复用字段存储时频数据
                        )
                        primitives.append(primitive)
            else:
                # 如果分析器未拟合，生成基元
                units = self._generate_timefreq_units_enhanced(x)
                
                for unit in units:
                    primitive = PrimitiveInfo(
                        primitive_type="timefreq",
                        primitive_id=unit.get('id', ''),
                        channels=unit.get('channels', []),
                        time_range=unit.get('time_range', (0, 0)),
                        freq_range=unit.get('freq_range', None),
                        quality_score=unit.get('quality', 0.5),
                        shapelet_pattern=unit.get('pattern', None)
                    )
                    primitives.append(primitive)
                    
        except Exception as e:
            if self.config.verbose:
                print(f"时频基元提取失败: {e}")
        
        return primitives
    
    def explain_instance(self, 
                        x: np.ndarray, 
                        target_class: Optional[int] = None,
                        unit_types: List[str] = ['microstate', 'shapelet', 'timefreq']) -> Dict[str, Any]:
        """增强版实例解释"""
        if not self.is_fitted:
            raise ValueError("解释器尚未拟合，请先调用fit()方法")
            
        if self.config.verbose:
            print(f"开始增强解释实例，使用基元类型: {unit_types}")
            
        # 获取原始预测
        original_pred = self._predict_single(x)
        if target_class is None:
            target_class = np.argmax(original_pred)
            
        explanation = {
            'original_prediction': original_pred,
            'target_class': target_class,
            'unit_explanations': {},
            'combined_explanation': None,
            'library_usage': {},
            'explanation_quality': {}
        }
        
        all_units = []
        all_importances = []
        
        # 使用基元库或生成新基元
        for unit_type in unit_types:
            if self.config.verbose:
                print(f"处理{unit_type}基元...")
            
            if self.primitive_library is not None:
                # 从基元库获取相关基元
                library_primitives = self.primitive_library.get_primitives_by_type(
                    unit_type, 
                    min_quality=self.enhanced_config.min_primitive_quality,
                    max_count=50
                )
                
                if library_primitives:
                    # 使用库中的基元
                    unit_exp = self._explain_with_library_primitives(x, target_class, library_primitives)
                    explanation['library_usage'][unit_type] = len(library_primitives)
                else:
                    # 库中没有合适基元，生成新的
                    unit_exp = self._explain_unit_type_enhanced(x, target_class, unit_type)
                    explanation['library_usage'][unit_type] = 0
            else:
                # 不使用基元库，直接生成
                unit_exp = self._explain_unit_type_enhanced(x, target_class, unit_type)
                explanation['library_usage'][unit_type] = 0
            
            explanation['unit_explanations'][unit_type] = unit_exp
            all_units.extend(unit_exp['units'])
            all_importances.extend(unit_exp['importances'])
        
        # 增强的综合解释选择
        if len(all_units) > 0:
            if self.config.verbose:
                print("进行增强综合解释选择...")
            combined_exp = self._enhanced_combine_explanations(all_units, all_importances, x)
            explanation['combined_explanation'] = combined_exp
            
            # 计算解释质量
            explanation['explanation_quality'] = self._evaluate_explanation_quality(
                explanation, x, target_class
            )
        
        # 更新基元库中基元的使用情况
        if self.primitive_library is not None:
            self._update_primitive_usage(explanation)
            # 保存基元库更新
            try:
                self.primitive_library.save_library()
            except Exception as e:
                if self.config.verbose:
                    print(f"保存基元库失败: {e}")
        
        # 记录解释历史
        self.explanation_history.append(explanation)
        
        return explanation
    
    def _explain_with_library_primitives(self, x: np.ndarray, target_class: int, 
                                       primitives: List[PrimitiveInfo]) -> Dict[str, Any]:
        """使用基元库中的基元进行解释"""
        units = []
        importances = []
        
        for primitive in primitives:
            # 将基元信息转换为单元格式
            # 确保channels字段为整数列表
            channels = primitive.channels
            if isinstance(channels, (list, tuple)):
                channels = [int(ch) if isinstance(ch, (str, np.str_)) else ch for ch in channels]
            elif isinstance(channels, (str, np.str_)):
                channels = [int(channels)]
            
            unit = {
                'type': primitive.primitive_type,
                'channels': channels,
                'time_range': primitive.time_range,
                'freq_range': primitive.freq_range,
                'microstate_id': primitive.microstate_id,
                'pattern': primitive.shapelet_pattern,
                'quality': primitive.quality_score,
                'primitive_id': primitive.primitive_id  # 添加基元ID字段
            }
            
            # 计算该基元的重要性
            importance = self._compute_primitive_importance(x, unit, target_class)
            
            units.append(unit)
            importances.append(importance)
        
        return {
            'units': units,
            'importances': importances,
            'source': 'library'
        }
    
    def _explain_unit_type_enhanced(self, x: np.ndarray, target_class: int, unit_type: str) -> Dict[str, Any]:
        """增强的单一类型基元解释"""
        if unit_type == 'microstate':
            return self._explain_microstate_units_enhanced(x, target_class)
        elif unit_type == 'shapelet':
            return self._explain_shapelet_units_enhanced(x, target_class)
        elif unit_type == 'timefreq':
            return self._explain_timefreq_units_enhanced(x, target_class)
        else:
            raise ValueError(f"未知的基元类型: {unit_type}")
    
    def _explain_microstate_units_enhanced(self, x: np.ndarray, target_class: int) -> Dict[str, Any]:
        """增强的微状态基元解释"""
        # 调用原有方法
        base_result = self._explain_microstate_units(x, target_class)
        
        # 添加质量评估
        for i, unit in enumerate(base_result['units']):
            quality = self._assess_unit_quality(unit, base_result['importances'][i])
            unit['quality'] = quality
        
        base_result['source'] = 'generated'
        return base_result
    
    def _explain_shapelet_units_enhanced(self, x: np.ndarray, target_class: int) -> Dict[str, Any]:
        """增强的shapelet基元解释"""
        units = []
        importances = []
        
        try:
            # 生成包含pattern的shapelet基元
            shapelet_units = self._generate_shapelet_units_enhanced(x)
            
            # 计算每个基元的重要性
            original_pred_full = self._predict_single(x)
            # 确保target_class是有效的整数索引
            try:
                target_class_idx = int(target_class) if isinstance(target_class, (str, np.str_)) else target_class
            except (ValueError, TypeError):
                target_class_idx = np.argmax(original_pred_full)
            
            if target_class_idx >= len(original_pred_full):
                target_class_idx = np.argmax(original_pred_full)
                
            original_pred = original_pred_full[target_class_idx]
            
            for unit in shapelet_units:
                # 生成扰动样本
                perturbed_samples = self._generate_shapelet_perturbations(x, unit)
                
                # 计算扰动后的预测
                perturbed_preds = []
                for perturbed_x in perturbed_samples:
                    pred_full = self._predict_single(perturbed_x)
                    pred = pred_full[target_class_idx]
                    perturbed_preds.append(pred)
                
                # 计算重要性（原始预测 - 扰动后平均预测）
                importance = original_pred - np.mean(perturbed_preds)
                
                # 添加质量评估
                quality = self._assess_unit_quality(unit, importance)
                unit['quality'] = quality
                
                units.append(unit)
                importances.append(importance)
                
        except Exception as e:
            if self.config.verbose:
                print(f"增强shapelet解释失败: {e}")
            # 回退到原有方法
            base_result = self._explain_shapelet_units(x, target_class)
            for i, unit in enumerate(base_result['units']):
                quality = self._assess_unit_quality(unit, base_result['importances'][i])
                unit['quality'] = quality
            base_result['source'] = 'generated'
            return base_result
        
        return {
            'units': units,
            'importances': importances,
            'source': 'generated'
        }
    
    def _generate_shapelet_units_enhanced(self, x: np.ndarray) -> List[Dict[str, Any]]:
        """从当前样本生成增强的shapelet基元"""
        units = []
        
        try:
            n_channels, n_timepoints = x.shape
            
            # 配置参数
            min_length = 10
            max_length = min(100, n_timepoints // 4)
            n_shapelets = 20
            
            # 1. 信号质量预评估 - 评估每个通道的信号质量
            channel_qualities = self._assess_channel_qualities(x)
            
            # 2. 智能通道选择 - 基于质量分数选择最佳通道
            best_channels = self._select_best_channels(channel_qualities, min(n_channels, 10))
            
            if self.config.verbose:
                print(f"选择了 {len(best_channels)} 个高质量通道用于shapelet提取")
            
            successful_extractions = 0
            attempts = 0
            max_attempts = n_shapelets * 3  # 允许更多尝试以确保成功提取
            
            while successful_extractions < n_shapelets and attempts < max_attempts:
                attempts += 1
                
                try:
                    # 3. 增强异常处理机制 - 智能选择通道和时间段
                    if best_channels:
                        # 优先从高质量通道中选择
                        channel_idx = np.random.choice(best_channels)
                    else:
                        # 备用策略：随机选择
                        channel_idx = np.random.randint(0, n_channels)
                    
                    # 动态调整长度范围
                    adaptive_min_length = max(min_length, int(n_timepoints * 0.02))
                    adaptive_max_length = min(max_length, int(n_timepoints * 0.3))
                    
                    if adaptive_min_length >= adaptive_max_length:
                        adaptive_min_length = min_length
                        adaptive_max_length = max_length
                    
                    length = np.random.randint(adaptive_min_length, adaptive_max_length + 1)
                    
                    # 智能选择起始时间 - 避免边界效应
                    margin = max(1, length // 10)
                    safe_start_range = max(1, n_timepoints - length - margin)
                    start_time = np.random.randint(margin, safe_start_range)
                    
                    # 提取shapelet模式
                    shapelet_pattern = x[channel_idx, start_time:start_time + length].copy()
                    
                    # 数据有效性检查
                    if not self._validate_shapelet_pattern(shapelet_pattern):
                        continue
                    
                    # 改进的标准化处理
                    normalized_pattern = self._robust_normalize_pattern(shapelet_pattern)
                    if normalized_pattern is None:
                        continue
                    
                    # 增强的质量评估
                    quality = self._compute_enhanced_shapelet_quality(
                        normalized_pattern, channel_qualities[channel_idx], length, adaptive_max_length
                    )
                    
                    # 质量阈值检查
                    if quality < 0.01:  # 最低质量阈值
                        continue
                    
                    unit = {
                        'id': f'shapelet_{successful_extractions}',
                        'type': 'shapelet',
                        'channels': [channel_idx],
                        'time_range': (start_time, start_time + length),
                        'pattern': normalized_pattern.reshape(1, -1),
                        'quality': quality,
                        'length': length,
                        'channel_quality': channel_qualities[channel_idx],
                        'extraction_attempt': attempts,
                        'primitive_id': f'generated_shapelet_{successful_extractions}_{channel_idx}_{start_time}'  # 添加基元ID
                    }
                    units.append(unit)
                    successful_extractions += 1
                    
                except Exception as e:
                    if self.config.verbose and attempts % 10 == 0:
                        print(f"Shapelet提取尝试 {attempts} 失败: {e}")
                    continue
            
            if self.config.verbose:
                print(f"成功提取 {successful_extractions}/{n_shapelets} 个shapelet基元，共尝试 {attempts} 次")
                
        except Exception as e:
            if self.config.verbose:
                print(f"生成shapelet基元失败: {e}")
                import traceback
                traceback.print_exc()
        
        return units

    def _assess_channel_qualities(self, x: np.ndarray) -> np.ndarray:
        """评估每个通道的信号质量"""
        n_channels, n_timepoints = x.shape
        qualities = np.zeros(n_channels)
        
        for ch in range(n_channels):
            signal = x[ch, :]
            
            # 多维度质量评估
            # 1. 信号方差（反映信号活跃度）
            variance_score = np.var(signal)
            
            # 2. 信号能量
            energy_score = np.mean(signal ** 2)
            
            # 3. 频域复杂度（通过FFT评估）
            try:
                fft_signal = np.fft.fft(signal)
                freq_complexity = np.std(np.abs(fft_signal))
            except:
                freq_complexity = 0
            
            # 4. 时域复杂度（一阶差分的方差）
            time_complexity = np.var(np.diff(signal))
            
            # 5. 信噪比估计（基于高频成分）
            try:
                # 简单的信噪比估计
                signal_power = np.var(signal)
                high_freq = signal[1:] - signal[:-1]
                noise_power = np.var(high_freq)
                snr = signal_power / (noise_power + 1e-8)
            except:
                snr = 1.0
            
            # 综合质量分数
            qualities[ch] = (
                0.3 * min(1.0, variance_score / (np.mean(np.var(x, axis=1)) + 1e-8)) +
                0.2 * min(1.0, energy_score / (np.mean(np.mean(x**2, axis=1)) + 1e-8)) +
                0.2 * min(1.0, freq_complexity / (np.mean([np.std(np.abs(np.fft.fft(x[i, :]))) for i in range(n_channels)]) + 1e-8)) +
                0.2 * min(1.0, time_complexity / (np.mean([np.var(np.diff(x[i, :])) for i in range(n_channels)]) + 1e-8)) +
                0.1 * min(1.0, snr / 10.0)
            )
        
        return qualities
    
    def _select_best_channels(self, channel_qualities: np.ndarray, max_channels: int) -> List[int]:
        """基于质量分数选择最佳通道"""
        # 选择质量分数最高的通道
        sorted_indices = np.argsort(channel_qualities)[::-1]
        
        # 确保至少选择一些通道
        n_select = min(max_channels, len(sorted_indices))
        
        # 选择前N个最佳通道，但也包含一些随机性
        best_channels = sorted_indices[:max(1, n_select // 2)].tolist()
        
        # 添加一些中等质量的通道以增加多样性
        if n_select > len(best_channels):
            remaining_channels = sorted_indices[len(best_channels):]
            additional_channels = np.random.choice(
                remaining_channels, 
                size=min(n_select - len(best_channels), len(remaining_channels)),
                replace=False
            )
            best_channels.extend(additional_channels.tolist())
        
        return best_channels
    
    def _validate_shapelet_pattern(self, pattern: np.ndarray) -> bool:
        """验证shapelet模式的有效性"""
        if pattern is None or len(pattern) == 0:
            return False
        
        # 检查是否包含NaN或无穷大值
        if np.any(np.isnan(pattern)) or np.any(np.isinf(pattern)):
            return False
        
        # 检查是否为常数序列
        if np.std(pattern) < 1e-10:
            return False
        
        # 检查长度是否合理
        if len(pattern) < 5:
            return False
        
        return True
    
    def _robust_normalize_pattern(self, pattern: np.ndarray) -> Optional[np.ndarray]:
        """鲁棒的模式标准化"""
        try:
            if not self._validate_shapelet_pattern(pattern):
                return None
            
            # 计算统计量
            mean_val = np.mean(pattern)
            std_val = np.std(pattern)
            
            # 避免除零错误
            if std_val < 1e-10:
                return None
            
            # Z-score标准化
            normalized = (pattern - mean_val) / std_val
            
            # 检查标准化结果
            if np.any(np.isnan(normalized)) or np.any(np.isinf(normalized)):
                return None
            
            return normalized
            
        except Exception:
            return None
    
    def _compute_enhanced_shapelet_quality(self, pattern: np.ndarray, channel_quality: float, 
                                         length: int, max_length: int) -> float:
        """计算增强的shapelet质量分数"""
        try:
            # 1. 基础方差分数
            variance_score = min(1.0, np.var(pattern))
            
            # 2. 长度归一化分数
            length_score = length / max_length
            
            # 3. 通道质量分数
            channel_score = channel_quality
            
            # 4. 复杂度分数（基于自相关）
            try:
                autocorr = np.correlate(pattern, pattern, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr / autocorr[0]  # 归一化
                complexity_score = 1.0 - np.mean(np.abs(autocorr[1:min(10, len(autocorr))]))  # 低自相关表示高复杂度
                complexity_score = max(0.0, min(1.0, complexity_score))
            except:
                complexity_score = 0.5
            
            # 5. 频域特征分数
            try:
                fft_pattern = np.fft.fft(pattern)
                freq_energy = np.sum(np.abs(fft_pattern[1:len(fft_pattern)//2]))
                total_energy = np.sum(np.abs(fft_pattern))
                freq_score = freq_energy / (total_energy + 1e-8)
                freq_score = min(1.0, freq_score)
            except:
                freq_score = 0.5
            
            # 综合质量分数
            quality = (
                0.3 * variance_score +
                0.2 * length_score +
                0.2 * channel_score +
                0.15 * complexity_score +
                0.15 * freq_score
            )
            
            return max(0.0, min(1.0, quality))
            
        except Exception:
            return 0.1  # 默认最低质量分数

    def _generate_shapelet_perturbations(self, x: np.ndarray, unit: Dict[str, Any]) -> List[np.ndarray]:
        """为shapelet基元生成扰动样本"""
        perturbations = []
        n_perturbations = 10
        
        try:
            channels = unit.get('channels', [0])
            time_range = unit.get('time_range', (0, x.shape[1]))
            start_time, end_time = time_range
            
            for _ in range(n_perturbations):
                perturbed_x = x.copy()
                
                # 对指定区域进行扰动
                for channel in channels:
                    if channel < x.shape[0] and start_time < x.shape[1] and end_time <= x.shape[1]:
                        # 添加噪声扰动
                        noise_level = 0.1 * np.std(x[channel, start_time:end_time])
                        noise = np.random.normal(0, noise_level, end_time - start_time)
                        perturbed_x[channel, start_time:end_time] += noise
                
                perturbations.append(perturbed_x)
                
        except Exception as e:
            # 静默处理错误，返回原始样本作为备选
            perturbations = [x.copy() for _ in range(n_perturbations)]
        
        return perturbations
    
    def _generate_timefreq_units_enhanced(self, x: np.ndarray) -> List[Dict[str, Any]]:
        """从当前样本生成增强的时频基元（优化选择策略）"""
        units = []
        
        try:
            n_channels, n_timepoints = x.shape
            
            # 智能频带选择（基于EEG重要性）
            freq_bands = {
                'alpha': (8, 13, 1.0),   # 最重要
                'beta': (13, 30, 0.9),   # 很重要
                'theta': (4, 8, 0.7),    # 重要
                'delta': (0.5, 4, 0.5),  # 中等重要
                'gamma': (30, 50, 0.6)   # 中等重要
            }
            
            # 自适应时间窗口选择
            base_window_sizes = [64, 128]  # 减少到2个基础窗口
            
            # 预先计算通道活跃度
            channel_activity = []
            for ch in range(n_channels):
                activity = np.var(x[ch]) + np.mean(np.abs(np.diff(x[ch])))
                channel_activity.append((ch, activity))
            
            # 按活跃度排序，选择前50%的通道
            channel_activity.sort(key=lambda x: x[1], reverse=True)
            active_channels = [ch for ch, _ in channel_activity[:max(3, n_channels//2)]]
            
            unit_count = 0
            max_units_per_band = 4  # 每个频带最多4个单元
            
            for freq_name, (freq_low, freq_high, importance_weight) in freq_bands.items():
                band_units = 0
                
                for window_size in base_window_sizes:
                    if band_units >= max_units_per_band:
                        break
                        
                    # 智能时间窗口选择（基于信号变化）
                    n_windows = min(3, (n_timepoints - window_size) // (window_size // 2))
                    
                    for i in range(n_windows):
                        if band_units >= max_units_per_band:
                            break
                            
                        start_time = i * (window_size // 2)
                        end_time = start_time + window_size
                        
                        if end_time > n_timepoints:
                            break
                        
                        # 选择最活跃的通道
                        best_channel = None
                        best_quality = 0
                        best_pattern = None
                        
                        for channel_idx in active_channels[:3]:  # 只检查前3个最活跃通道
                            pattern = self._extract_timefreq_segment(x, channel_idx, start_time, end_time, (freq_low, freq_high))
                            
                            if len(pattern) > 0:
                                quality = self._compute_timefreq_quality(pattern, (freq_low, freq_high))
                                quality *= importance_weight  # 应用频带重要性权重
                                
                                if quality > best_quality:
                                    best_quality = quality
                                    best_channel = channel_idx
                                    best_pattern = pattern
                        
                        # 只保留高质量单元
                        if best_quality > self.enhanced_config.timefreq_quality_threshold:
                            unit = {
                                'id': f'timefreq_{freq_name}_{i}_{best_channel}',
                                'type': 'timefreq',
                                'channels': [best_channel],
                                'time_range': (start_time, end_time),
                                'freq_range': (freq_low, freq_high),
                                'freq_band_name': freq_name,
                                'pattern': best_pattern,
                                'quality': best_quality,
                                'window_idx': i,
                                'importance_weight': importance_weight,
                                'primitive_id': f'generated_timefreq_{freq_name}_{i}_{best_channel}_{start_time}'  # 添加基元ID
                            }
                            units.append(unit)
                            band_units += 1
                            unit_count += 1
                            
                            # 全局单元数量限制
                            if unit_count >= 15:  # 总共最多15个时频单元
                                return units
            
            # 按质量排序，保留最好的
            units.sort(key=lambda u: u['quality'], reverse=True)
            return units[:12]  # 最多保留12个最高质量的单元
                                
        except Exception as e:
            if self.config.verbose:
                print(f"时频基元生成失败: {e}")
        
        return units
    
    def _explain_timefreq_units_enhanced(self, x: np.ndarray, target_class: int) -> Dict[str, Any]:
        """增强的时频基元解释"""
        try:
            # 生成包含pattern的时频基元
            units = self._generate_timefreq_units_enhanced(x)
            
            # 计算重要性
            importances = []
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
                
            original_score = original_pred_full[target_class_idx]
            
            for unit in units:
                # 生成扰动样本
                perturbed_samples = self._generate_timefreq_perturbations(x, unit)
                
                # 计算重要性
                unit_predictions = []
                for perturbed_x in perturbed_samples:
                    pred = self._predict_single(perturbed_x)
                    unit_predictions.append(pred[target_class_idx])
                
                importance = original_score - np.mean(unit_predictions)
                importances.append(importance)
                perturbations.append(perturbed_samples)
                predictions.append(np.mean(unit_predictions))
                
                # 更新质量评估
                unit['quality'] = self._assess_unit_quality(unit, importance)
            
            return {
                'units': units,
                'importances': importances,
                'perturbations': perturbations,
                'predictions': predictions,
                'source': 'enhanced_generated'
            }
            
        except Exception as e:
            if self.config.verbose:
                print(f"增强时频基元解释失败: {e}")
            # 回退到原有方法
            base_result = self._explain_timefreq_units(x, target_class)
            for i, unit in enumerate(base_result['units']):
                quality = self._assess_unit_quality(unit, base_result['importances'][i])
                unit['quality'] = quality
            base_result['source'] = 'fallback'
            return base_result
    
    def _enhanced_combine_explanations(self, 
                                     units: List[Dict[str, Any]], 
                                     importances: List[float],
                                     x: np.ndarray) -> Dict[str, Any]:
        """增强的综合解释选择"""
        if len(units) == 0:
            return None
        
        # 计算基元质量分数
        quality_scores = []
        for i, unit in enumerate(units):
            # 基础质量分数（来自基元库或重要性）
            base_quality = unit.get('quality', abs(importances[i]))
            # 当前重要性调节因子：重要性为0时大幅降低质量分数
            importance_factor = max(0.1, abs(importances[i]) * 2)  # 最小保留10%，重要性高时可以超过基础质量
            # 最终质量分数 = 基础质量 * 重要性调节因子
            adjusted_quality = base_quality * importance_factor
            quality_scores.append(adjusted_quality)
        
        # 自适应选择数量
        if self.enhanced_config.adaptive_selection:
            # 根据质量分布确定选择数量
            quality_array = np.array(quality_scores)
            quality_threshold = np.percentile(quality_array, 70)  # 选择前30%
            
            candidate_count = np.sum(quality_array >= quality_threshold)
            n_select = max(
                self.enhanced_config.min_selected_primitives,
                min(candidate_count, self.enhanced_config.max_selected_primitives)
            )
        else:
            n_select = self.config.n_features
        
        # 使用基于重要性的动态选择策略
        try:
            # 清理重要性评分中的NaN值
            importance_scores = np.array(importances)
            valid_mask = ~np.isnan(importance_scores)
            if not np.any(valid_mask):
                importance_scores = np.ones(len(units))
            else:
                mean_val = np.nanmean(importance_scores)
                importance_scores[~valid_mask] = mean_val
            
            # 计算重要性分布统计
            importance_mean = np.mean(np.abs(importance_scores))
            importance_std = np.std(np.abs(importance_scores))
            importance_threshold = importance_mean + 0.5 * importance_std
            
            # 基于重要性的动态选择数量
            high_importance_count = np.sum(np.abs(importance_scores) >= importance_threshold)
            dynamic_n_select = max(
                self.enhanced_config.min_selected_primitives,
                min(high_importance_count + 2, self.enhanced_config.max_selected_primitives)
            )
            
            # 计算综合评分（降低质量权重，提高重要性权重）
            importance_weight = 0.6  # 提高重要性权重
            quality_weight = 0.4     # 降低质量权重
            
            combined_scores = []
            for i in range(len(units)):
                # 标准化重要性评分和质量分数
                norm_importance = abs(importance_scores[i]) / (importance_mean + 1e-8)
                norm_quality = quality_scores[i] / (np.mean(quality_scores) + 1e-8)
                
                # 计算综合评分
                combined_score = importance_weight * norm_importance + quality_weight * norm_quality
                combined_scores.append(combined_score)
            
            # 选择综合评分最高的基元
            selected_indices = np.argsort(combined_scores)[::-1][:dynamic_n_select].tolist()
            
            # 质量分数提升机制：为被选中且重要性高的基元提升质量分数
            for idx in selected_indices:
                # 只有当重要性评分真正大于阈值且阈值大于0时才提升质量分数
                if importance_threshold > 0 and abs(importance_scores[idx]) > importance_threshold:
                    # 提升质量分数（最多提升20%）
                    boost_factor = min(1.2, 1.0 + 0.1 * (abs(importance_scores[idx]) / importance_threshold))
                    units[idx]['quality'] = units[idx].get('quality', quality_scores[idx]) * boost_factor
                    
                    # 更新基元库中的质量分数
                    if hasattr(self, 'primitive_library') and self.primitive_library:
                        primitive_id = units[idx].get('id')
                        if primitive_id:
                            self.primitive_library.update_primitive_importance(
                                primitive_id, importance_scores[idx]
                            )
            
        except Exception as e:
            if self.config.verbose:
                print(f"动态选择失败，使用备选方案: {e}")
            # 备选方案：基于质量和重要性的简单选择
            combined_scores = [
                0.4 * quality_scores[i] + 0.6 * abs(importances[i])
                for i in range(len(units))
            ]
            selected_indices = np.argsort(combined_scores)[::-1][:n_select].tolist()
        
        # 构建结果
        selected_units = [units[i] for i in selected_indices]
        selected_importances = [importances[i] for i in selected_indices]
        selected_qualities = [quality_scores[i] for i in selected_indices]
        
        return {
            'selected_units': selected_units,
            'selected_importances': selected_importances,
            'selected_qualities': selected_qualities,
            'selection_indices': selected_indices,
            'total_candidates': len(units),
            'selection_method': 'enhanced'
        }
    
    def _compute_primitive_importance(self, x: np.ndarray, unit: Dict[str, Any], target_class: int) -> float:
        """计算基元重要性"""
        try:
            # 生成扰动样本
            perturbed_samples = self._generate_causal_perturbations(x, unit, unit['type'])
            
            # 多次预测取平均（提高稳定性）
            importance_scores = []
            # 确保target_class是有效的整数索引
            original_pred_full = self._predict_single(x)
            try:
                target_class_idx = int(target_class) if isinstance(target_class, (str, np.str_)) else target_class
            except (ValueError, TypeError):
                target_class_idx = np.argmax(original_pred_full)
            
            if target_class_idx >= len(original_pred_full):
                target_class_idx = np.argmax(original_pred_full)
                
            original_score = original_pred_full[target_class_idx]
            
            for _ in range(self.enhanced_config.stability_iterations):
                unit_predictions = []
                for perturbed_x in perturbed_samples:
                    pred = self._predict_single(perturbed_x)
                    unit_predictions.append(pred[target_class_idx])
                
                # 计算重要性
                if self.enhanced_config.importance_aggregation == 'mean':
                    avg_pred = np.mean(unit_predictions)
                elif self.enhanced_config.importance_aggregation == 'weighted_mean':
                    # 根据扰动强度加权
                    weights = np.ones(len(unit_predictions))
                    avg_pred = np.average(unit_predictions, weights=weights)
                else:  # max
                    avg_pred = np.max(unit_predictions)
                
                importance = original_score - avg_pred
                importance_scores.append(importance)
            
            # 返回稳定的重要性分数
            return np.mean(importance_scores)
            
        except Exception as e:
            # 静默处理索引错误和其他异常，不显示错误信息
            return 0.0
    
    def _assess_unit_quality(self, unit: Dict[str, Any], importance: float) -> float:
        """评估基元质量"""
        quality_factors = []
        
        # 重要性因子
        importance_factor = min(abs(importance) * 10, 1.0)  # 归一化到[0,1]
        quality_factors.append(importance_factor)
        
        # 覆盖范围因子
        time_range = unit.get('time_range', (0, 0))
        time_span = time_range[1] - time_range[0]
        coverage_factor = min(time_span / 100.0, 1.0)  # 假设100是合理的时间跨度
        quality_factors.append(coverage_factor)
        
        # 通道数因子
        channels = unit.get('channels', [])
        channel_factor = min(len(channels) / 5.0, 1.0)  # 假设5个通道是合理的
        quality_factors.append(channel_factor)
        
        # 类型特定因子
        if unit.get('type') == 'microstate':
            # 微状态的稳定性
            microstate_factor = 0.8  # 默认值
            quality_factors.append(microstate_factor)
        elif unit.get('type') == 'shapelet':
            # Shapelet的匹配分数
            match_score = unit.get('match_score', 0.5)
            quality_factors.append(match_score)
        elif unit.get('type') == 'timefreq':
            # 时频的能量集中度
            energy_factor = unit.get('energy_concentration', 0.5)
            quality_factors.append(energy_factor)
        
        # 综合质量分数
        return np.mean(quality_factors)
    
    def _evaluate_explanation_quality(self, explanation: Dict[str, Any], 
                                    x: np.ndarray, target_class: int) -> Dict[str, float]:
        """评估解释质量"""
        quality_metrics = {}
        
        if explanation['combined_explanation'] is not None:
            combined = explanation['combined_explanation']
            
            # 选择基元数量
            n_selected = len(combined['selected_units'])
            quality_metrics['n_selected_primitives'] = n_selected
            
            # 平均重要性
            importances = combined['selected_importances']
            if importances:
                quality_metrics['mean_importance'] = np.mean(np.abs(importances))
                quality_metrics['max_importance'] = np.max(np.abs(importances))
                quality_metrics['importance_std'] = np.std(importances)
            
            # 平均质量
            qualities = combined.get('selected_qualities', [])
            if qualities:
                quality_metrics['mean_quality'] = np.mean(qualities)
                quality_metrics['min_quality'] = np.min(qualities)
            
            # 覆盖率
            total_candidates = combined.get('total_candidates', 0)
            if total_candidates > 0:
                quality_metrics['selection_ratio'] = n_selected / total_candidates
        
        return quality_metrics
    
    def _update_primitive_usage(self, explanation: Dict[str, Any]):
        """更新基元库中基元的使用情况"""
        if explanation['combined_explanation'] is not None:
            selected_units = explanation['combined_explanation']['selected_units']
            selected_importances = explanation['combined_explanation']['selected_importances']
            
            for unit, importance in zip(selected_units, selected_importances):
                primitive_id = unit.get('primitive_id')
                
                if primitive_id:
                    # 如果基元已存在于库中，更新其使用情况
                    if primitive_id in self.primitive_library.primitives:
                        self.primitive_library.update_primitive_importance(primitive_id, importance)
                    # 如果是新生成的基元且质量足够高，添加到库中
                    elif primitive_id.startswith('generated_') and unit.get('quality', 0) >= self.enhanced_config.min_primitive_quality:
                        try:
                            # 创建PrimitiveInfo对象
                            primitive = PrimitiveInfo(
                                primitive_type=unit.get('type', 'unknown'),
                                primitive_id=primitive_id,
                                channels=unit.get('channels', []),
                                time_range=unit.get('time_range', (0, 0)),
                                freq_range=unit.get('freq_range', None),
                                microstate_id=unit.get('state', None),
                                shapelet_pattern=unit.get('pattern', None),
                                quality_score=unit.get('quality', 0.0)
                            )
                            
                            # 添加到基元库
                            actual_id = self.primitive_library.add_primitive(primitive)
                            # 更新重要性
                            self.primitive_library.update_primitive_importance(actual_id, importance)
                            
                            if self.config.verbose:
                                print(f"新基元已添加到库中: {actual_id} (质量: {unit.get('quality', 0):.3f}, 重要性: {importance:.3f})")
                                
                        except Exception as e:
                            if self.config.verbose:
                                print(f"添加新基元到库中失败: {e}")
                            pass
    
    def _generate_timefreq_perturbations(self, x: np.ndarray, unit: Dict[str, Any]) -> List[np.ndarray]:
        """为时频基元生成扰动样本"""
        perturbations = []
        
        try:
            channels = unit.get('channels', [])
            time_range = unit.get('time_range', (0, 0))
            freq_range = unit.get('freq_range', (0, 100))
            
            n_perturbations = min(10, self.config.n_perturbations // 100)
            
            for _ in range(n_perturbations):
                perturbed_x = x.copy()
                
                # 时频域扰动：在指定频带和时间范围内添加噪声
                for ch in channels:
                    if ch < perturbed_x.shape[0]:
                        segment = perturbed_x[ch, time_range[0]:time_range[1]]
                        
                        # 生成频带特定的噪声
                        noise_strength = 0.1 * np.std(segment)
                        noise = np.random.normal(0, noise_strength, len(segment))
                        
                        # 应用频带滤波（简化版）
                        if len(segment) > 10:
                            # 简单的频带加权
                            freq_weight = 1.0 if freq_range[0] <= 20 <= freq_range[1] else 0.5
                            noise *= freq_weight
                        
                        perturbed_x[ch, time_range[0]:time_range[1]] = segment + noise
                
                perturbations.append(perturbed_x)
                
        except Exception as e:
            # 静默处理错误，备选方案：简单的高斯噪声
            for _ in range(5):
                noise = np.random.normal(0, 0.1 * np.std(x), x.shape)
                perturbations.append(x + noise)
        
        return perturbations
    
    def _select_representative_channels(self, x: np.ndarray, channel_list: List[int], max_channels: int) -> List[int]:
        """选择代表性通道"""
        if len(channel_list) <= max_channels:
            return channel_list
        
        # 基于信号方差选择最活跃的通道
        channel_variances = []
        for ch in channel_list:
            try:
                # 确保ch是整数
                ch_int = int(ch)
                if ch_int < x.shape[0]:
                    variance = np.var(x[ch_int])
                    channel_variances.append((ch_int, variance))
            except (ValueError, TypeError):
                print(f"通道索引转换失败: {ch}")
                continue
        
        # 按方差排序并选择前max_channels个
        channel_variances.sort(key=lambda x: x[1], reverse=True)
        selected = [ch for ch, _ in channel_variances[:max_channels]]
        
        return selected
    
    def _extract_timefreq_segment(self, x: np.ndarray, channel_idx: int, start_time: int, end_time: int, freq_band: Tuple[float, float]) -> np.ndarray:
        """提取时频段模式"""
        try:
            if channel_idx >= x.shape[0] or start_time >= end_time:
                return np.array([])
            
            segment = x[channel_idx, start_time:end_time]
            
            # 简化的频带特征提取（使用功率谱密度）
            from scipy import signal
            
            if len(segment) > 10:
                freqs, psd = signal.welch(segment, nperseg=min(len(segment), 64))
                
                # 提取目标频带的功率
                freq_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
                if np.any(freq_mask):
                    band_power = psd[freq_mask]
                    # 标准化
                    if len(band_power) > 0 and np.std(band_power) > 0:
                        band_power = (band_power - np.mean(band_power)) / np.std(band_power)
                    return band_power
            
            # 备选方案：返回原始段的标准化版本
            if len(segment) > 0 and np.std(segment) > 0:
                return (segment - np.mean(segment)) / np.std(segment)
            else:
                return segment
                
        except Exception as e:
            return np.array([])
    
    def _compute_timefreq_quality(self, pattern: np.ndarray, freq_band: Tuple[float, float]) -> float:
        """计算时频基元质量"""
        if len(pattern) == 0:
            return 0.0
        
        try:
            # 基于信号特征的质量评估
            quality_factors = []
            
            # 1. 信号变异性
            if np.std(pattern) > 0:
                variability = min(np.var(pattern), 1.0)
                quality_factors.append(variability)
            
            # 2. 频带重要性权重
            freq_center = (freq_band[0] + freq_band[1]) / 2
            if 8 <= freq_center <= 30:  # alpha和beta频带更重要
                freq_weight = 1.0
            elif 4 <= freq_center <= 8:  # theta频带
                freq_weight = 0.8
            else:
                freq_weight = 0.6
            quality_factors.append(freq_weight)
            
            # 3. 模式长度因子
            length_factor = min(len(pattern) / 20.0, 1.0)
            quality_factors.append(length_factor)
            
            # 4. 信号强度
            if len(pattern) > 0:
                strength = min(np.mean(np.abs(pattern)), 1.0)
                quality_factors.append(strength)
            
            return np.mean(quality_factors)
            
        except Exception:
            return 0.5  # 默认质量
    
    def _evaluate_timefreq_unit_quality(self, unit, x: np.ndarray) -> float:
        """评估时频单元质量"""
        try:
            # 提取单元特征
            features = unit.extract_features(x)
            
            if len(features) == 0:
                return 0.0
            
            # 基于特征的质量评估
            quality_factors = []
            
            # 特征强度
            feature_strength = np.mean(np.abs(features))
            quality_factors.append(min(feature_strength, 1.0))
            
            # 特征变异性
            if len(features) > 1:
                feature_var = np.var(features)
                quality_factors.append(min(feature_var, 1.0))
            
            # 频带权重
            freq_band = getattr(unit, 'freq_band', (0, 100))
            freq_center = (freq_band[0] + freq_band[1]) / 2
            if 8 <= freq_center <= 30:
                freq_weight = 1.0
            elif 4 <= freq_center <= 8:
                freq_weight = 0.8
            else:
                freq_weight = 0.6
            quality_factors.append(freq_weight)
            
            return np.mean(quality_factors)
            
        except Exception:
            return 0.5
    
    def _extract_timefreq_pattern(self, unit, x: np.ndarray) -> np.ndarray:
        """提取时频基元的模式数据"""
        try:
            # 提取单元特征作为模式
            features = unit.extract_features(x)
            
            if len(features) > 0:
                return features
            else:
                # 备选方案：提取原始时间段数据
                channels = getattr(unit, 'channels', [])
                time_range = getattr(unit, 'time_range', (0, 0))
                
                if len(channels) > 0:
                    try:
                        # 确保channel是整数
                        ch_int = int(channels[0])
                        if ch_int < x.shape[0]:
                            segment = x[ch_int, time_range[0]:time_range[1]]
                            if len(segment) > 0:
                                return segment
                    except (ValueError, TypeError):
                        pass
                
                return np.array([])
                
        except Exception:
            return np.array([])
    
    def get_enhanced_explanation_summary(self, explanation: Dict[str, Any]) -> str:
        """获取增强解释摘要 - 修复版"""
        summary = []
        summary.append("=== 增强版CMS-LIME解释摘要 ===")
        
        # 基本信息
        pred = explanation['original_prediction']
        target_class = explanation['target_class']
        
        # 确保pred是1D数组
        if len(pred.shape) > 1:
            pred = pred.flatten()
        
        # 计算自然预测类别（概率最高的类别）
        natural_predicted_class = np.argmax(pred)
        
        # 获取各类别的概率
        class_probs = [f"类别{i}: {pred[i]:.3f}" for i in range(len(pred))]
        
        # 目标类别的置信度
        try:
            # 尝试直接转换为整数
            target_class_int = int(target_class) if target_class is not None else 0
        except (ValueError, TypeError):
            # 如果target_class是字符串标签，使用自然预测类别
            target_class_int = natural_predicted_class
        
        target_confidence = pred[target_class_int] if target_class_int < len(pred) else 0.0
        
        # 添加详细的预测信息
        summary.append("\n=== 预测信息 ===")
        summary.append(f"模型预测概率分布: [{', '.join(class_probs)}]")
        summary.append(f"自然预测类别 (概率最高): {natural_predicted_class}")
        summary.append(f"自然预测置信度: {pred[natural_predicted_class]:.3f}")
        
        # 检查是否存在不一致
        if target_class_int != natural_predicted_class:
            summary.append(f"\n⚠️  注意：解释目标类别 ({target_class}) 与自然预测类别 ({natural_predicted_class}) 不一致")
            summary.append(f"解释目标类别: {target_class} (映射为类别{target_class_int})")
            summary.append(f"解释目标置信度: {target_confidence:.3f}")
            
            if target_confidence < 0.5:
                summary.append(f"⚠️  警告：目标类别置信度 ({target_confidence:.3f}) 低于0.5，这可能造成混淆")
                summary.append(f"建议：考虑使用自然预测类别 {natural_predicted_class} (置信度: {pred[natural_predicted_class]:.3f})")
        else:
            summary.append(f"\n✅ 解释目标类别与自然预测一致")
            summary.append(f"预测类别: {target_class}")
            summary.append(f"预测置信度: {target_confidence:.3f}")
        
        # 基元解释信息
        summary.append("\n=== 基元解释 ===")
        for unit_type, unit_exp in explanation['unit_explanations'].items():
            n_units = len(unit_exp['units'])
            importances = unit_exp['importances']
            if importances:
                avg_importance = np.mean(np.abs(importances))
                max_importance = np.max(np.abs(importances))
            else:
                avg_importance = 0.0
                max_importance = 0.0
            
            source = unit_exp.get('source', 'unknown')
            summary.append(f"  {unit_type}: {n_units}个基元, 平均重要性={avg_importance:.3f}, 最大重要性={max_importance:.3f}, 来源={source}")
        
        # 综合解释信息
        if explanation['combined_explanation'] is not None:
            combined = explanation['combined_explanation']
            n_selected = len(combined['selected_units'])
            selected_importances = combined['selected_importances']
            
            if selected_importances:
                top_importance = max(np.abs(selected_importances))
                avg_selected_importance = np.mean(np.abs(selected_importances))
            else:
                top_importance = 0.0
                avg_selected_importance = 0.0
            
            total_candidates = combined.get('total_candidates', 0)
            
            summary.append(f"\n=== 综合解释 ===")
            summary.append(f"从{total_candidates}个候选中选择了{n_selected}个关键基元")
            summary.append(f"最高重要性: {top_importance:.3f}")
            summary.append(f"平均重要性: {avg_selected_importance:.3f}")
        
        # 基元库使用情况
        if explanation.get('library_usage'):
            summary.append("\n=== 基元库使用 ===")
            for unit_type, count in explanation['library_usage'].items():
                summary.append(f"  {unit_type}: 使用了{count}个库中基元")
        
        # 解释质量指标
        if explanation.get('explanation_quality'):
            quality = explanation['explanation_quality']
            summary.append("\n=== 解释质量指标 ===")
            for metric, value in quality.items():
                if isinstance(value, (int, float)):
                    summary.append(f"  {metric}: {value:.3f}")
                else:
                    summary.append(f"  {metric}: {value}")
        
        # 添加调试信息（可选）
        if explanation.get('debug_info'):
            summary.append("\n=== 调试信息 ===")
            debug_info = explanation['debug_info']
            if 'prediction_consistency' in debug_info:
                consistency = debug_info['prediction_consistency']
                summary.append(f"  预测一致性检查: {consistency}")
            if 'target_class_source' in debug_info:
                source = debug_info['target_class_source']
                summary.append(f"  目标类别来源: {source}")
        
        return "\n".join(summary)
    
    def get_primitive_library_stats(self) -> Dict[str, Any]:
        """获取基元库统计信息"""
        if self.primitive_library is not None:
            return self.primitive_library.get_primitive_statistics()
        else:
            return {"message": "未使用基元库"}


# 便捷创建函数
def create_enhanced_cms_lime_explainer(config_dict: Dict = None) -> EnhancedCMSLimeExplainer:
    """
    创建增强版CMS-LIME解释器的便捷函数
    
    Args:
        config_dict: 配置字典
        
    Returns:
        增强版解释器实例
    """
    if config_dict:
        config = EnhancedCMSLimeConfig(**config_dict)
    else:
        config = EnhancedCMSLimeConfig()
    
    return EnhancedCMSLimeExplainer(config)


if __name__ == "__main__":
    # 示例使用
    print("增强版CMS-LIME解释器测试")
    
    # 创建配置
    config = EnhancedCMSLimeConfig(
        use_primitive_library=True,
        min_selected_primitives=5,
        max_selected_primitives=15,
        adaptive_selection=True,
        verbose=True
    )
    
    # 创建解释器
    explainer = EnhancedCMSLimeExplainer(config)
    
    print("增强版CMS-LIME解释器创建完成")
    print(f"基元库状态: {'启用' if config.use_primitive_library else '禁用'}")
    print(f"自适应选择: {'启用' if config.adaptive_selection else '禁用'}")
    print(f"选择基元数量范围: {config.min_selected_primitives}-{config.max_selected_primitives}")