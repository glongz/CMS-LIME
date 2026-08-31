#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态阈值调整模块

根据患者历史表现、生物标志物出现情况等因素动态调整预测阈值，
以平衡敏感性和特异性
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
import json


@dataclass
class ThresholdConfig:
    """阈值配置"""
    base_threshold: float = 0.5
    min_threshold: float = 0.2
    max_threshold: float = 0.8
    adjustment_step: float = 0.05
    sensitivity_target: float = 0.8
    specificity_target: float = 0.8


class DynamicThresholdAdjuster:
    """动态阈值调整器"""
    
    def __init__(self, config: Optional[ThresholdConfig] = None):
        """
        初始化动态阈值调整器
        
        Parameters:
        -----------
        config : ThresholdConfig, optional
            阈值配置，如果为None则使用默认配置
        """
        self.config = config or ThresholdConfig()
        
        # 患者特定的阈值
        self.patient_thresholds: Dict[int, float] = {}
        
        # 患者历史性能记录
        self.patient_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # 生物标志物出现频率统计
        self.biomarker_frequency: Dict[int, Dict] = defaultdict(lambda: {
            'total_samples': 0,
            'biomarker_present': 0,
            'biomarker_present_and_preictal': 0,
            'biomarker_present_and_interictal': 0
        })
    
    def get_threshold(self, patient_id: int, 
                     biomarker_present: bool = False,
                     recent_performance: Optional[Dict] = None) -> float:
        """
        获取患者特定的动态阈值
        
        Parameters:
        -----------
        patient_id : int
            患者ID
        biomarker_present : bool
            当前样本是否有生物标志物
        recent_performance : Dict, optional
            最近性能指标，包含sensitivity和specificity
        
        Returns:
        --------
        threshold : float
            调整后的阈值
        """
        # 获取基础阈值
        base_threshold = self.patient_thresholds.get(
            patient_id, self.config.base_threshold
        )
        
        # 根据生物标志物调整
        if biomarker_present:
            # 如果有生物标志物，降低阈值（更容易预测为Preictal）
            biomarker_adjustment = -0.1
        else:
            biomarker_adjustment = 0.0
        
        # 根据最近性能调整
        performance_adjustment = 0.0
        if recent_performance:
            sensitivity = recent_performance.get('sensitivity', 0.5)
            specificity = recent_performance.get('specificity', 0.5)
            
            # 如果敏感性低，降低阈值
            if sensitivity < self.config.sensitivity_target:
                performance_adjustment -= self.config.adjustment_step * 2
            
            # 如果特异性低，提高阈值
            if specificity < self.config.specificity_target:
                performance_adjustment += self.config.adjustment_step * 2
        
        # 应用调整
        adjusted_threshold = base_threshold + biomarker_adjustment + performance_adjustment
        
        # 限制在合理范围内
        adjusted_threshold = np.clip(
            adjusted_threshold,
            self.config.min_threshold,
            self.config.max_threshold
        )
        
        return adjusted_threshold
    
    def update_threshold(self, patient_id: int,
                        current_sensitivity: float,
                        current_specificity: float,
                        target_sensitivity: Optional[float] = None,
                        target_specificity: Optional[float] = None):
        """
        根据性能更新患者阈值
        
        Parameters:
        -----------
        patient_id : int
            患者ID
        current_sensitivity : float
            当前敏感性
        current_specificity : float
            当前特异性
        target_sensitivity : float, optional
            目标敏感性，如果为None则使用配置中的值
        target_specificity : float, optional
            目标特异性，如果为None则使用配置中的值
        """
        target_sens = target_sensitivity or self.config.sensitivity_target
        target_spec = target_specificity or self.config.specificity_target
        
        # 获取当前阈值
        current_threshold = self.patient_thresholds.get(
            patient_id, self.config.base_threshold
        )
        
        # 计算需要调整的方向
        sensitivity_gap = target_sens - current_sensitivity
        specificity_gap = target_spec - current_specificity
        
        # 如果敏感性低于目标，降低阈值
        if sensitivity_gap > 0.1:
            adjustment = -self.config.adjustment_step * 2
        elif sensitivity_gap > 0.05:
            adjustment = -self.config.adjustment_step
        # 如果特异性低于目标，提高阈值
        elif specificity_gap > 0.1:
            adjustment = self.config.adjustment_step * 2
        elif specificity_gap > 0.05:
            adjustment = self.config.adjustment_step
        else:
            adjustment = 0.0
        
        # 更新阈值
        new_threshold = current_threshold + adjustment
        new_threshold = np.clip(
            new_threshold,
            self.config.min_threshold,
            self.config.max_threshold
        )
        
        self.patient_thresholds[patient_id] = new_threshold
        
        # 记录历史
        self.patient_history[patient_id].append({
            'threshold': new_threshold,
            'sensitivity': current_sensitivity,
            'specificity': current_specificity,
            'adjustment': adjustment
        })
    
    def update_biomarker_statistics(self, patient_id: int,
                                   biomarker_present: bool,
                                   true_label: int):
        """
        更新生物标志物统计信息
        
        Parameters:
        -----------
        patient_id : int
            患者ID
        biomarker_present : bool
            是否有生物标志物
        true_label : int
            真实标签（0: Interictal, 1: Preictal）
        """
        stats = self.biomarker_frequency[patient_id]
        stats['total_samples'] += 1
        
        if biomarker_present:
            stats['biomarker_present'] += 1
            if true_label == 1:
                stats['biomarker_present_and_preictal'] += 1
            else:
                stats['biomarker_present_and_interictal'] += 1
    
    def get_biomarker_adjusted_threshold(self, patient_id: int) -> float:
        """
        根据生物标志物统计信息获取调整后的阈值
        
        Parameters:
        -----------
        patient_id : int
            患者ID
        
        Returns:
        --------
        threshold : float
            调整后的阈值
        """
        stats = self.biomarker_frequency[patient_id]
        
        if stats['total_samples'] == 0:
            return self.patient_thresholds.get(patient_id, self.config.base_threshold)
        
        # 计算生物标志物预测Preictal的准确率
        biomarker_preictal_rate = 0.0
        if stats['biomarker_present'] > 0:
            biomarker_preictal_rate = (
                stats['biomarker_present_and_preictal'] / stats['biomarker_present']
            )
        
        # 如果生物标志物预测Preictal的准确率高，降低阈值
        base_threshold = self.patient_thresholds.get(patient_id, self.config.base_threshold)
        
        if biomarker_preictal_rate > 0.7:
            # 生物标志物很可靠，降低阈值
            adjustment = -0.15
        elif biomarker_preictal_rate > 0.5:
            adjustment = -0.1
        elif biomarker_preictal_rate < 0.3:
            # 生物标志物不太可靠，提高阈值
            adjustment = 0.1
        else:
            adjustment = 0.0
        
        adjusted_threshold = base_threshold + adjustment
        return np.clip(
            adjusted_threshold,
            self.config.min_threshold,
            self.config.max_threshold
        )
    
    def predict_with_dynamic_threshold(self,
                                       patient_id: int,
                                       probs: np.ndarray,
                                       biomarker_present: bool = False) -> Tuple[int, float]:
        """
        使用动态阈值进行预测
        
        Parameters:
        -----------
        patient_id : int
            患者ID
        probs : np.ndarray
            预测概率 [P(Interictal), P(Preictal)]
        biomarker_present : bool
            是否有生物标志物
        
        Returns:
        --------
        prediction : int
            预测类别（0或1）
        threshold_used : float
            使用的阈值
        """
        # 获取动态阈值
        threshold = self.get_biomarker_adjusted_threshold(patient_id)
        
        # 如果有生物标志物，使用更低的阈值
        if biomarker_present:
            threshold = max(self.config.min_threshold, threshold - 0.1)
        
        # 进行预测
        prediction = 1 if probs[1] >= threshold else 0
        
        return prediction, threshold
    
    def get_patient_statistics(self, patient_id: int) -> Dict:
        """获取患者统计信息"""
        threshold = self.patient_thresholds.get(patient_id, self.config.base_threshold)
        history = list(self.patient_history[patient_id])
        stats = self.biomarker_frequency[patient_id]
        
        biomarker_preictal_rate = 0.0
        if stats['biomarker_present'] > 0:
            biomarker_preictal_rate = (
                stats['biomarker_present_and_preictal'] / stats['biomarker_present']
            )
        
        return {
            'patient_id': patient_id,
            'current_threshold': threshold,
            'history_length': len(history),
            'biomarker_statistics': {
                'total_samples': stats['total_samples'],
                'biomarker_present_rate': (
                    stats['biomarker_present'] / stats['total_samples']
                    if stats['total_samples'] > 0 else 0.0
                ),
                'biomarker_preictal_rate': biomarker_preictal_rate
            },
            'recent_performance': (
                {
                    'sensitivity': history[-1]['sensitivity'],
                    'specificity': history[-1]['specificity']
                } if history else None
            )
        }
    
    def save_config(self, filepath: str):
        """保存配置到文件"""
        config_dict = {
            'base_threshold': self.config.base_threshold,
            'min_threshold': self.config.min_threshold,
            'max_threshold': self.config.max_threshold,
            'adjustment_step': self.config.adjustment_step,
            'sensitivity_target': self.config.sensitivity_target,
            'specificity_target': self.config.specificity_target,
            'patient_thresholds': self.patient_thresholds
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def load_config(self, filepath: str):
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        self.config = ThresholdConfig(
            base_threshold=config_dict.get('base_threshold', 0.5),
            min_threshold=config_dict.get('min_threshold', 0.2),
            max_threshold=config_dict.get('max_threshold', 0.8),
            adjustment_step=config_dict.get('adjustment_step', 0.05),
            sensitivity_target=config_dict.get('sensitivity_target', 0.8),
            specificity_target=config_dict.get('specificity_target', 0.8)
        )
        
        self.patient_thresholds = config_dict.get('patient_thresholds', {})


if __name__ == "__main__":
    print("动态阈值调整模块")
    print("=" * 80)
    
    # 示例用法
    adjuster = DynamicThresholdAdjuster()
    
    # 模拟更新阈值
    adjuster.update_threshold(
        patient_id=1,
        current_sensitivity=0.4,  # 敏感性低
        current_specificity=0.8
    )
    
    print(f"患者1的阈值: {adjuster.patient_thresholds.get(1, 0.5):.3f}")
    
    # 模拟生物标志物统计
    adjuster.update_biomarker_statistics(patient_id=1, biomarker_present=True, true_label=1)
    adjuster.update_biomarker_statistics(patient_id=1, biomarker_present=True, true_label=1)
    adjuster.update_biomarker_statistics(patient_id=1, biomarker_present=True, true_label=0)
    
    # 获取调整后的阈值
    probs = np.array([0.6, 0.4])
    prediction, threshold = adjuster.predict_with_dynamic_threshold(
        patient_id=1,
        probs=probs,
        biomarker_present=True
    )
    
    print(f"预测概率: {probs}")
    print(f"使用的阈值: {threshold:.3f}")
    print(f"预测结果: {prediction}")
    print(f"患者统计: {adjuster.get_patient_statistics(1)}")
