#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一基元选拔框架

整合基元提取和重要性评估，对每个样本提取基元后立即评估重要性

Author: CMS-LIME Framework
Date: 2025
"""

import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
import json
import os
import sys
from datetime import datetime

# 支持相对导入和绝对导入
try:
    from .config import UnifiedSelectionConfig, create_default_config
    from .primitive_importance_evaluator import PrimitiveImportanceEvaluator
    from .unified_primitive_extractor import UnifiedPrimitiveExtractor, BiomarkerUnit
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from config import UnifiedSelectionConfig, create_default_config
    from primitive_importance_evaluator import PrimitiveImportanceEvaluator
    from unified_primitive_extractor import UnifiedPrimitiveExtractor, BiomarkerUnit

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        return iterable


class UnifiedPrimitiveSelection:
    """统一基元选拔框架"""
    
    def __init__(self,
                 config: Optional[UnifiedSelectionConfig] = None,
                 model: Optional[Callable] = None):
        """
        初始化统一选拔框架
        
        Parameters:
        -----------
        config : UnifiedSelectionConfig, optional
            配置对象，如果为None则使用默认配置
        model : Callable, optional
            预测模型，接受 (n_channels, n_timepoints) 输入，返回概率分布
        """
        self.config = config or create_default_config()
        self.model = model
        
        # 如果启用因果扰动但没有提供因果图，尝试自动创建
        if (self.config.evaluation_config.use_causal_perturbation and 
            self.config.evaluation_config.causal_graph is None):
            n_channels = getattr(self.config.prior_config, 'n_channels', None)
            if n_channels is not None and n_channels > 0:
                self.config.evaluation_config.causal_graph = (
                    self.config.evaluation_config.get_or_create_causal_graph(
                        n_channels=n_channels,
                        max_lag=5
                    )
                )
                if self.config.evaluation_config.causal_graph is None:
                    self.config.evaluation_config.use_causal_perturbation = False
        
        # 初始化重要性评估器
        self.importance_evaluator = None
        if model is not None:
            self.importance_evaluator = PrimitiveImportanceEvaluator(
                model=model,
                n_perturbations=self.config.evaluation_config.n_perturbations,
                stability_iterations=self.config.evaluation_config.stability_iterations,
                use_causal_perturbation=self.config.evaluation_config.use_causal_perturbation,
                causal_graph=self.config.evaluation_config.causal_graph
            )
        
        # 初始化统一提取器
        self.extractor = None
        if self.importance_evaluator is not None:
            self.extractor = UnifiedPrimitiveExtractor(
                config=self.config,
                importance_evaluator=self.importance_evaluator
            )
        
        # 存储结果
        self.all_biomarkers: List[BiomarkerUnit] = []  # 所有样本的生物标志物
        self.sample_biomarkers: Dict[int, List[BiomarkerUnit]] = {}  # 每个样本的生物标志物
    
    def create_causal_graph_from_data(self, X_data: np.ndarray, method: str = 'auto', max_lag: int = 5):
        """
        根据数据创建因果图
        
        Parameters:
        -----------
        X_data : np.ndarray
            训练数据，形状为 (n_samples, n_channels, n_timepoints)
        method : str
            因果分析方法：'auto', 'granger', 'simple'
            - 'auto': 尝试使用 GrangerCausalityAnalyzer，失败则使用简单方法
            - 'granger': 使用 GrangerCausalityAnalyzer
            - 'simple': 使用简单的默认因果图（空图）
        max_lag : int
            最大滞后时间
        
        Returns:
        --------
        causal_graph : CausalGraph
            创建的因果图对象
        """
        n_channels = X_data.shape[1] if X_data.ndim == 3 else X_data.shape[0]
        
        if method == 'auto' or method == 'granger':
            try:
                # 尝试使用 GrangerCausalityAnalyzer
                from causal_perturbation import GrangerCausalityAnalyzer
                
                causal_analyzer = GrangerCausalityAnalyzer(
                    max_lag=max_lag,
                    significance_level=0.05
                )
                causal_analyzer.fit(X_data)
                causal_graph = causal_analyzer.get_causal_graph()
                
                # 更新配置和评估器
                self.config.evaluation_config.causal_graph = causal_graph
                if self.importance_evaluator is not None:
                    self.importance_evaluator.causal_graph = causal_graph
                    self.importance_evaluator.use_causal_perturbation = True
                
                return causal_graph
                
            except ImportError:
                if method == 'granger':
                    raise ImportError("无法导入 GrangerCausalityAnalyzer，请确保 causal_perturbation.py 可用")
                # 如果 method == 'auto'，继续使用简单方法
        
        # 使用简单方法：创建空的因果图
        causal_graph = self.config.evaluation_config.create_default_causal_graph(
            n_channels=n_channels,
            max_lag=max_lag
        )
        
        # 更新配置和评估器
        self.config.evaluation_config.causal_graph = causal_graph
        if self.importance_evaluator is not None:
            self.importance_evaluator.causal_graph = causal_graph
            self.importance_evaluator.use_causal_perturbation = True
        
        return causal_graph
        
        # 结果保存路径
        self.results_folder: Optional[str] = None
    
    def set_model(self, model: Callable):
        """设置预测模型"""
        self.model = model
        self.importance_evaluator = PrimitiveImportanceEvaluator(
            model=model,
            n_perturbations=self.config.evaluation_config.n_perturbations,
            stability_iterations=self.config.evaluation_config.stability_iterations
        )
        self.extractor = UnifiedPrimitiveExtractor(
            config=self.config,
            importance_evaluator=self.importance_evaluator
        )
    
    def process_samples(self,
                       X_samples: np.ndarray,
                       y_samples: np.ndarray,
                       sample_indices: Optional[List[int]] = None) -> Dict[int, List[BiomarkerUnit]]:
        """
        处理多个样本，提取并评估生物标志物
        
        Parameters:
        -----------
        X_samples : np.ndarray
            样本数据，形状为 (n_samples, n_channels, n_timepoints)
        y_samples : np.ndarray
            样本标签
        sample_indices : list, optional
            要处理的样本索引列表，如果为None则处理所有样本
        
        Returns:
        --------
        sample_biomarkers : dict
            每个样本的生物标志物列表，键为样本索引
        """
        if self.extractor is None:
            raise ValueError("模型未设置，无法进行提取和评估。请先设置模型。")
        
        n_samples = len(X_samples)
        if sample_indices is None:
            sample_indices = list(range(n_samples))
        
        print("=" * 80)
        print("统一基元选拔：提取并评估生物标志物")
        print("=" * 80)
        
        self.sample_biomarkers = {}
        self.all_biomarkers = []
        
        # 处理每个样本（带进度条）
        for idx in tqdm(sample_indices, desc=f"处理 {len(sample_indices)} 个样本", unit="个", ncols=80):
            X_sample = X_samples[idx]
            y_sample = int(y_samples[idx])
            
            try:
                # 提取并评估生物标志物
                biomarkers = self.extractor.extract_and_evaluate(X_sample, y_sample)
                
                # 筛选重要性高的生物标志物
                filtered_biomarkers = self._filter_biomarkers(biomarkers)
                
                self.sample_biomarkers[idx] = filtered_biomarkers
                self.all_biomarkers.extend(filtered_biomarkers)
                
            except Exception as e:
                print(f"样本 {idx} 处理失败: {e}")
                self.sample_biomarkers[idx] = []
        
        # 聚合所有生物标志物
        aggregated_biomarkers = self._aggregate_biomarkers()
        
        print(f"\n处理完成！")
        print(f"  - 处理的样本数: {len(self.sample_biomarkers)}")
        print(f"  - 提取的生物标志物总数: {len(self.all_biomarkers)}")
        print(f"  - 聚合后的生物标志物数: {len(aggregated_biomarkers)}")
        
        return self.sample_biomarkers
    
    def _filter_biomarkers(self, biomarkers: List[BiomarkerUnit]) -> List[BiomarkerUnit]:
        """
        筛选生物标志物（按重要性阈值和top K）
        
        Parameters:
        -----------
        biomarkers : list
            生物标志物列表
        
        Returns:
        --------
        filtered : list
            筛选后的生物标志物列表
        """
        # 按重要性绝对值排序
        biomarkers.sort(key=lambda x: abs(x.importance_score), reverse=True)
        
        # 应用重要性阈值
        threshold = self.config.evaluation_config.importance_threshold
        filtered = [b for b in biomarkers if abs(b.importance_score) >= threshold]
        
        # 选择top K
        top_k = self.config.evaluation_config.top_k_biomarkers
        filtered = filtered[:min(len(filtered), top_k)]
        
        return filtered
    
    def _aggregate_biomarkers(self) -> List[BiomarkerUnit]:
        """
        聚合所有样本的生物标志物
        
        Returns:
        --------
        aggregated : list
            聚合后的生物标志物列表（按重要性排序）
        """
        # 按重要性绝对值排序
        self.all_biomarkers.sort(key=lambda x: abs(x.importance_score), reverse=True)
        
        # 选择top K
        top_k = self.config.evaluation_config.top_k_biomarkers * 3  # 聚合时选择更多
        aggregated = self.all_biomarkers[:min(len(self.all_biomarkers), top_k)]
        
        return aggregated
    
    def get_top_biomarkers(self, top_k: Optional[int] = None) -> List[BiomarkerUnit]:
        """
        获取top K生物标志物
        
        Parameters:
        -----------
        top_k : int, optional
            Top K数量，如果为None则使用配置中的值
        
        Returns:
        --------
        biomarkers : list
            Top K生物标志物列表
        """
        if top_k is None:
            top_k = self.config.evaluation_config.top_k_biomarkers
        
        # 按重要性绝对值排序
        sorted_biomarkers = sorted(
            self.all_biomarkers,
            key=lambda x: abs(x.importance_score),
            reverse=True
        )
        
        return sorted_biomarkers[:min(len(sorted_biomarkers), top_k)]
    
    def save_results(self, output_folder: str):
        """
        保存结果到文件
        
        Parameters:
        -----------
        output_folder : str
            输出文件夹路径
        """
        os.makedirs(output_folder, exist_ok=True)
        self.results_folder = output_folder
        
        # 保存配置
        config_path = os.path.join(output_folder, 'config.json')
        self.config.save_to_json(config_path)
        
        # 保存所有生物标志物
        all_biomarkers_path = os.path.join(output_folder, 'all_biomarkers.json')
        with open(all_biomarkers_path, 'w', encoding='utf-8') as f:
            json.dump([b.to_dict() for b in self.all_biomarkers], f, indent=2, ensure_ascii=False)
        
        # 保存top K生物标志物
        top_biomarkers = self.get_top_biomarkers()
        top_biomarkers_path = os.path.join(output_folder, 'top_biomarkers.json')
        with open(top_biomarkers_path, 'w', encoding='utf-8') as f:
            json.dump([b.to_dict() for b in top_biomarkers], f, indent=2, ensure_ascii=False)
        
        # 保存每个样本的生物标志物
        sample_biomarkers_path = os.path.join(output_folder, 'sample_biomarkers.json')
        sample_dict = {
            str(idx): [b.to_dict() for b in biomarkers]
            for idx, biomarkers in self.sample_biomarkers.items()
        }
        with open(sample_biomarkers_path, 'w', encoding='utf-8') as f:
            json.dump(sample_dict, f, indent=2, ensure_ascii=False)
        
        # 保存统计信息
        stats_path = os.path.join(output_folder, 'statistics.json')
        stats = self._compute_statistics()
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {output_folder}")
        print(f"  - 配置: {config_path}")
        print(f"  - 所有生物标志物: {all_biomarkers_path}")
        print(f"  - Top K生物标志物: {top_biomarkers_path}")
        print(f"  - 样本生物标志物: {sample_biomarkers_path}")
        print(f"  - 统计信息: {stats_path}")
    
    def _compute_statistics(self) -> Dict:
        """计算统计信息"""
        if len(self.all_biomarkers) == 0:
            return {}
        
        importance_scores = [abs(b.importance_score) for b in self.all_biomarkers]
        quality_scores = [b.quality_score for b in self.all_biomarkers]
        
        # 按类型统计
        type_counts = {}
        type_importances = {}
        for b in self.all_biomarkers:
            ptype = b.primitive_type
            type_counts[ptype] = type_counts.get(ptype, 0) + 1
            if ptype not in type_importances:
                type_importances[ptype] = []
            type_importances[ptype].append(abs(b.importance_score))
        
        stats = {
            'total_biomarkers': len(self.all_biomarkers),
            'n_samples': len(self.sample_biomarkers),
            'importance_statistics': {
                'mean': float(np.mean(importance_scores)),
                'std': float(np.std(importance_scores)),
                'min': float(np.min(importance_scores)),
                'max': float(np.max(importance_scores)),
                'median': float(np.median(importance_scores))
            },
            'quality_statistics': {
                'mean': float(np.mean(quality_scores)),
                'std': float(np.std(quality_scores)),
                'min': float(np.min(quality_scores)),
                'max': float(np.max(quality_scores)),
                'median': float(np.median(quality_scores))
            },
            'type_distribution': type_counts,
            'type_importance': {
                ptype: {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'count': len(scores)
                }
                for ptype, scores in type_importances.items()
            }
        }
        
        return stats
