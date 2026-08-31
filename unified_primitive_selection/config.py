#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一基元选拔配置模块

复用先验配置，并添加重要性评估参数

Author: CMS-LIME Framework
Date: 2025
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any
import json

# 添加父目录到路径，以便导入先验配置
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from two_stage_selection.prior_knowledge_config import (
        PriorKnowledgeConfig,
        ShapeletPriors,
        MicrostatePriors,
        TimeFreqPriors,
        create_default_prior_config
    )
except ImportError:
    # 如果无法导入，尝试从当前目录导入
    try:
        from prior_knowledge_config import (
            PriorKnowledgeConfig,
            ShapeletPriors,
            MicrostatePriors,
            TimeFreqPriors,
            create_default_prior_config
        )
    except ImportError:
        # 尝试从父目录导入
        try:
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from two_stage_selection.prior_knowledge_config import (
                PriorKnowledgeConfig,
                ShapeletPriors,
                MicrostatePriors,
                TimeFreqPriors,
                create_default_prior_config
            )
        except ImportError:
            raise ImportError("无法导入先验配置模块，请确保 two_stage_selection 或 prior_knowledge_config 可用")


@dataclass
class ImportanceEvaluationConfig:
    """重要性评估配置"""
    # 扰动参数
    n_perturbations: int = 10  # 每个基元的扰动次数
    stability_iterations: int = 3  # 稳定性迭代次数（多次扰动取平均）
    
    # 因果扰动参数
    use_causal_perturbation: bool = True
    causal_graph: Optional[Any] = None
    auto_create_causal_graph: bool = True
    
    def get_or_create_causal_graph(self, n_channels: int, max_lag: int = 5) -> Any:
        """获取或创建因果图对象"""
        if self.causal_graph is not None:
            return self.causal_graph
        
        if self.auto_create_causal_graph:
            try:
                from cms_lime import CausalGraph
                self.causal_graph = CausalGraph(n_channels=n_channels, max_lag=max_lag)
                return self.causal_graph
            except ImportError:
                try:
                    from causal_perturbation import CausalGraph
                    self.causal_graph = CausalGraph(n_nodes=n_channels)
                    return self.causal_graph
                except ImportError:
                    return None
        return None
    
    # 重要性阈值
    importance_threshold: float = 0.01  # 重要性阈值（用于筛选生物标志物）
    
    # 选择参数
    top_k_biomarkers: int = 20  # 选择top K个生物标志物
    min_importance_score: float = 0.001  # 最小重要性分数
    


@dataclass
class UnifiedSelectionConfig:
    """统一基元选拔配置"""
    # 先验配置（复用）
    prior_config: PriorKnowledgeConfig = field(default_factory=create_default_prior_config)
    
    # 重要性评估配置
    evaluation_config: ImportanceEvaluationConfig = field(default_factory=ImportanceEvaluationConfig)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'prior_config': self.prior_config.to_dict(),
            'evaluation_config': {
                'n_perturbations': self.evaluation_config.n_perturbations,
                'stability_iterations': self.evaluation_config.stability_iterations,
                'use_causal_perturbation': self.evaluation_config.use_causal_perturbation,
                'auto_create_causal_graph': self.evaluation_config.auto_create_causal_graph,
                'importance_threshold': self.evaluation_config.importance_threshold,
                'top_k_biomarkers': self.evaluation_config.top_k_biomarkers,
                'min_importance_score': self.evaluation_config.min_importance_score
            }
        }
    
    def save_to_json(self, filepath: str):
        """保存配置到JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'UnifiedSelectionConfig':
        """从JSON文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        prior_config = PriorKnowledgeConfig.load_from_json_dict(data['prior_config'])
        eval_config = ImportanceEvaluationConfig(**data['evaluation_config'])
        
        return cls(prior_config=prior_config, evaluation_config=eval_config)
    
    def get_summary(self) -> str:
        """获取配置摘要"""
        summary = "=" * 80 + "\n"
        summary += "统一基元选拔配置摘要\n"
        summary += "=" * 80 + "\n\n"
        
        summary += self.prior_config.get_summary()
        summary += "\n"
        
        summary += "5. 重要性评估配置:\n"
        summary += f"   - 扰动次数: {self.evaluation_config.n_perturbations}\n"
        summary += f"   - 稳定性迭代次数: {self.evaluation_config.stability_iterations}\n"
        summary += f"   - 使用因果扰动: {self.evaluation_config.use_causal_perturbation}\n"
        summary += f"   - 重要性阈值: {self.evaluation_config.importance_threshold}\n"
        summary += f"   - Top K生物标志物: {self.evaluation_config.top_k_biomarkers}\n"
        summary += f"   - 最小重要性分数: {self.evaluation_config.min_importance_score}\n"
        summary += "=" * 80 + "\n"
        
        return summary


# 扩展 PriorKnowledgeConfig 以支持从字典加载
if not hasattr(PriorKnowledgeConfig, 'load_from_json_dict'):
    @classmethod
    def load_from_json_dict(cls, data: Dict) -> PriorKnowledgeConfig:
        """从字典加载配置（用于嵌套配置）"""
        shapelet_priors = ShapeletPriors(**data['shapelet_priors'])
        microstate_priors = MicrostatePriors(**data['microstate_priors'])
        timefreq_priors = TimeFreqPriors(
            freq_bands={k: tuple(v) for k, v in data['timefreq_priors']['freq_bands'].items()},
            time_window_size=data['timefreq_priors']['time_window_size'],
            overlap_ratio=data['timefreq_priors']['overlap_ratio'],
            method=data['timefreq_priors']['method'],
            cwt_wavelet=data['timefreq_priors']['cwt_wavelet'],
            stft_window=data['timefreq_priors']['stft_window'],
            stft_nperseg=data['timefreq_priors']['stft_nperseg'],
            sampling_rate=data['timefreq_priors']['sampling_rate'],
            min_relative_power=data['timefreq_priors']['min_relative_power'],
            min_quality_score=data['timefreq_priors']['min_quality_score']
        )
        
        return cls(
            shapelet_priors=shapelet_priors,
            microstate_priors=microstate_priors,
            timefreq_priors=timefreq_priors,
            global_quality_threshold=data.get('global_quality_threshold', 0.1),
            target_shapelet_count=data.get('target_counts', {}).get('target_shapelet_count', 100),
            target_microstate_count=data.get('target_counts', {}).get('target_microstate_count', 100),
            target_timefreq_count=data.get('target_counts', {}).get('target_timefreq_count', 100),
            use_count_mode=data.get('use_count_mode', True),
            n_channels=data['data_format']['n_channels'],
            n_timepoints=data['data_format']['n_timepoints'],
            sampling_rate=data['data_format']['sampling_rate']
        )
    
    PriorKnowledgeConfig.load_from_json_dict = load_from_json_dict


def create_default_config() -> UnifiedSelectionConfig:
    """创建默认的统一选拔配置"""
    return UnifiedSelectionConfig()


if __name__ == "__main__":
    # 测试配置
    config = create_default_config()
    print(config.get_summary())
    
    # 保存配置
    config.save_to_json("default_unified_config.json")
    print("\n配置已保存到 default_unified_config.json")
