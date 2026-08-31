#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两阶段基元选拔框架

整合第一场选拔（基元提取）和第二场选拔（因果扰动解释）

Author: CMS-LIME Framework
Date: 2025
"""

import numpy as np
from typing import List, Dict, Optional, Any, Callable
import os
import sys
import json
from datetime import datetime

# 支持相对导入和绝对导入
try:
    # 尝试相对导入（作为包的一部分）
    from .prior_knowledge_config import PriorKnowledgeConfig, create_default_prior_config
    from .primitive_selection_stage1 import PrimitiveExtractor, PrimitiveUnit
    from .primitive_selection_stage2 import (
        CausalPerturbationExplainer, 
        BiomarkerEvaluator, 
        BiomarkerCandidate
    )
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    # 添加当前目录到路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from prior_knowledge_config import PriorKnowledgeConfig, create_default_prior_config
    from primitive_selection_stage1 import PrimitiveExtractor, PrimitiveUnit
    from primitive_selection_stage2 import (
        CausalPerturbationExplainer, 
        BiomarkerEvaluator, 
        BiomarkerCandidate
    )


class TwoStagePrimitiveSelection:
    """两阶段基元选拔主框架"""
    
    def __init__(self, 
                 prior_config: Optional[PriorKnowledgeConfig] = None,
                 model: Optional[Callable] = None,
                 n_perturbations: int = 10,
                 importance_threshold: float = 0.1,
                 top_k_biomarkers: int = 20):
        """
        初始化两阶段选拔框架
        
        Parameters:
        -----------
        prior_config : PriorKnowledgeConfig, optional
            先验信息配置，如果为None则使用默认配置
        model : Callable, optional
            预测模型，接受 (n_channels, n_timepoints) 输入
        n_perturbations : int
            每个基元的扰动次数
        importance_threshold : float
            重要性阈值
        top_k_biomarkers : int
            选择top K个生物标志物
        """
        self.prior_config = prior_config or create_default_prior_config()
        self.model = model
        
        # 第一场选拔：基元提取器
        self.stage1_extractor = PrimitiveExtractor(self.prior_config)
        
        # 第二场选拔：因果扰动解释器
        self.stage2_explainer = None
        if model is not None:
            self.stage2_explainer = CausalPerturbationExplainer(
                model=model,
                n_perturbations=n_perturbations,
                importance_threshold=importance_threshold
            )
        
        # 生物标志物评估器
        self.biomarker_evaluator = BiomarkerEvaluator(
            top_k=top_k_biomarkers,
            importance_threshold=importance_threshold
        )
        
        # 存储结果
        self.candidate_primitives: List[PrimitiveUnit] = []
        self.biomarker_candidates: List[BiomarkerCandidate] = []
        self.selected_biomarkers: List[BiomarkerCandidate] = []
        
        # 结果保存路径
        self.results_folder: Optional[str] = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'TwoStagePrimitiveSelection':
        """
        第一场选拔：提取候选基元
        
        Parameters:
        -----------
        X_train : np.ndarray
            训练数据，形状为 (n_samples, n_channels, n_timepoints)
        y_train : np.ndarray
            训练标签
        
        Returns:
        --------
        self : TwoStagePrimitiveSelection
        """
        print("\n" + "=" * 80)
        print("开始第一场选拔：基元提取")
        print("=" * 80)
        
        # 提取候选基元
        self.candidate_primitives = self.stage1_extractor.extract(X_train, y_train)
        
        print(f"\n第一场选拔完成：共提取 {len(self.candidate_primitives)} 个候选基元")
        
        return self
    
    def explain(self, X_sample: np.ndarray, y_sample: int) -> List[BiomarkerCandidate]:
        """
        第二场选拔：评估基元重要性，筛选生物标志物
        
        Parameters:
        -----------
        X_sample : np.ndarray
            样本数据，形状为 (n_channels, n_timepoints) 或 (1, n_channels, n_timepoints)
        y_sample : int
            样本标签
        
        Returns:
        --------
        biomarkers : list
            生物标志物列表
        """
        if self.stage2_explainer is None:
            raise ValueError("模型未设置，无法进行第二场选拔。请先设置模型。")
        
        if len(self.candidate_primitives) == 0:
            raise ValueError("候选基元为空，请先调用fit方法进行第一场选拔。")
        
        print("\n" + "=" * 80)
        print("开始第二场选拔：因果扰动解释")
        print("=" * 80)
        
        # 确保输入格式正确
        if X_sample.ndim == 3:
            X_sample = X_sample[0]  # 取第一个样本
        
        # 评估基元重要性
        self.biomarker_candidates = self.stage2_explainer.explain(
            X_sample, 
            self.candidate_primitives,
            target_class=y_sample
        )
        
        # 选择生物标志物
        self.selected_biomarkers = self.biomarker_evaluator.select_biomarkers(
            self.biomarker_candidates
        )
        
        print(f"\n第二场选拔完成：筛选出 {len(self.selected_biomarkers)} 个生物标志物")
        
        return self.selected_biomarkers
    
    def set_model(self, model: Callable):
        """设置预测模型"""
        self.model = model
        self.stage2_explainer = CausalPerturbationExplainer(
            model=model,
            n_perturbations=self.stage2_explainer.n_perturbations if self.stage2_explainer else 10,
            importance_threshold=self.stage2_explainer.importance_threshold if self.stage2_explainer else 0.1
        )
    
    def save_results(self, output_folder: str):
        """
        保存所有结果
        
        Parameters:
        -----------
        output_folder : str
            输出文件夹路径
        """
        os.makedirs(output_folder, exist_ok=True)
        self.results_folder = output_folder
        
        # 1. 保存先验信息配置
        prior_config_path = os.path.join(output_folder, 'prior_knowledge_config.json')
        self.prior_config.save_to_json(prior_config_path)
        print(f"先验信息配置已保存: {prior_config_path}")
        
        # 2. 保存先验信息记录
        prior_record_path = os.path.join(output_folder, 'prior_knowledge_record.json')
        self.stage1_extractor.save_prior_knowledge(prior_record_path)
        print(f"先验信息记录已保存: {prior_record_path}")
        
        # 3. 保存候选基元（第一场选拔结果）
        candidate_primitives_path = os.path.join(output_folder, 'candidate_primitives.json')
        self.stage1_extractor.save_primitives(candidate_primitives_path)
        print(f"候选基元已保存: {candidate_primitives_path}")
        
        # 4. 保存生物标志物（第二场选拔结果）
        if self.selected_biomarkers:
            biomarkers_path = os.path.join(output_folder, 'biomarkers.json')
            self.biomarker_evaluator.save_biomarkers(self.selected_biomarkers, biomarkers_path)
            print(f"生物标志物已保存: {biomarkers_path}")
        
        # 5. 生成综合报告
        report_path = os.path.join(output_folder, 'selection_report.txt')
        self._generate_report(report_path)
        print(f"综合报告已保存: {report_path}")
    
    def _generate_report(self, filepath: str):
        """生成综合报告"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("两阶段基元选拔综合报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 第一场选拔摘要
            f.write(self.stage1_extractor.get_summary())
            f.write("\n")
            
            # 第二场选拔摘要
            if self.selected_biomarkers:
                f.write(self.biomarker_evaluator.get_summary_text(self.selected_biomarkers))
            else:
                f.write("第二场选拔：尚未执行\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("报告结束\n")
            f.write("=" * 80 + "\n")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取综合摘要"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'stage1_summary': {
                'n_candidate_primitives': len(self.candidate_primitives),
                'prior_knowledge_used': self.prior_config.to_dict()
            },
            'stage2_summary': {
                'n_biomarker_candidates': len(self.biomarker_candidates),
                'n_selected_biomarkers': len(self.selected_biomarkers)
            }
        }
        
        if self.selected_biomarkers:
            summary['stage2_summary']['biomarker_summary'] = \
                self.biomarker_evaluator.generate_summary(self.selected_biomarkers)
        
        return summary


def create_results_folder(base_path: str = "results/two_stage_selection_results") -> str:
    """创建结果文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{base_path}_{timestamp}"
    folder_path = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


if __name__ == "__main__":
    # 示例使用
    print("两阶段基元选拔框架")
    print("=" * 80)
    
    # 创建框架实例
    framework = TwoStagePrimitiveSelection()
    
    # 显示先验信息配置
    print(framework.prior_config.get_summary())
    
    print("\n框架已初始化，可以使用fit和explain方法进行两阶段选拔。")
