#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
低性能患者处理模块

针对敏感性和特异性特别低的患者，采用特殊策略来提高预测性能

核心策略：
1. 识别低性能患者（敏感性或特异性低于阈值）
2. 针对低性能患者采用更保守或更激进的阈值
3. 使用多模型集成
4. 增加生物标志物权重
5. 采用时间序列上下文信息
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class PatientPerformanceProfile:
    """患者性能档案"""
    patient_id: int
    sensitivity: float
    specificity: float
    accuracy: float
    f1_score: float
    n_samples: int
    n_preictal_samples: int
    n_interictal_samples: int
    
    def is_low_performance(self, 
                          sensitivity_threshold: float = 0.5,
                          specificity_threshold: float = 0.5) -> bool:
        """判断是否为低性能患者"""
        return (self.sensitivity < sensitivity_threshold or 
                self.specificity < specificity_threshold)
    
    def get_performance_type(self) -> str:
        """获取性能类型"""
        if self.sensitivity < 0.5 and self.specificity >= 0.5:
            return "low_sensitivity"  # 敏感性低
        elif self.specificity < 0.5 and self.sensitivity >= 0.5:
            return "low_specificity"  # 特异性低
        elif self.sensitivity < 0.5 and self.specificity < 0.5:
            return "low_both"  # 两者都低
        else:
            return "normal"  # 正常


class LowPerformancePatientHandler:
    """低性能患者处理器"""
    
    def __init__(self,
                 sensitivity_threshold: float = 0.5,
                 specificity_threshold: float = 0.5,
                 use_ensemble: bool = True,
                 use_temporal_context: bool = True):
        """
        初始化低性能患者处理器
        
        Parameters:
        -----------
        sensitivity_threshold : float
            敏感性阈值，低于此值视为低性能
        specificity_threshold : float
            特异性阈值，低于此值视为低性能
        use_ensemble : bool
            是否使用集成学习
        use_temporal_context : bool
            是否使用时间序列上下文
        """
        self.sensitivity_threshold = sensitivity_threshold
        self.specificity_threshold = specificity_threshold
        self.use_ensemble = use_ensemble
        self.use_temporal_context = use_temporal_context
        
        # 患者性能档案
        self.patient_profiles: Dict[int, PatientPerformanceProfile] = {}
        
        # 患者特定的预测策略
        self.patient_strategies: Dict[int, Dict] = {}
        
        # 时间序列上下文缓存
        self.temporal_context: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    
    def register_patient_performance(self, patient_id: int, 
                                    performance: PatientPerformanceProfile):
        """
        注册患者性能档案
        
        Parameters:
        -----------
        patient_id : int
            患者ID
        performance : PatientPerformanceProfile
            患者性能档案
        """
        self.patient_profiles[patient_id] = performance
        
        # 根据性能类型设置策略
        if performance.is_low_performance(self.sensitivity_threshold, 
                                         self.specificity_threshold):
            self._setup_patient_strategy(patient_id, performance)
    
    def _setup_patient_strategy(self, patient_id: int, 
                               performance: PatientPerformanceProfile):
        """为低性能患者设置特殊策略"""
        perf_type = performance.get_performance_type()
        
        strategy = {
            'performance_type': perf_type,
            'adjusted_threshold': 0.5,
            'biomarker_weight_multiplier': 1.0,
            'ensemble_weight': 1.0,
            'temporal_window_size': 5
        }
        
        if perf_type == "low_sensitivity":
            # 敏感性低：降低阈值，增加Preictal预测倾向
            strategy['adjusted_threshold'] = 0.3  # 更低的阈值
            strategy['biomarker_weight_multiplier'] = 1.5  # 增加生物标志物权重
            strategy['ensemble_weight'] = 1.2  # 增加集成权重
            print(f"患者 {patient_id}: 敏感性低，采用更激进的预测策略")
            
        elif perf_type == "low_specificity":
            # 特异性低：提高阈值，减少假阳性
            strategy['adjusted_threshold'] = 0.7  # 更高的阈值
            strategy['biomarker_weight_multiplier'] = 0.8  # 减少生物标志物权重
            strategy['ensemble_weight'] = 0.9  # 减少集成权重
            print(f"患者 {patient_id}: 特异性低，采用更保守的预测策略")
            
        elif perf_type == "low_both":
            # 两者都低：使用中等阈值，但增加集成学习
            strategy['adjusted_threshold'] = 0.5
            strategy['biomarker_weight_multiplier'] = 1.2
            strategy['ensemble_weight'] = 1.5  # 显著增加集成权重
            strategy['temporal_window_size'] = 10  # 增加时间窗口
            print(f"患者 {patient_id}: 敏感性和特异性都低，采用综合改进策略")
        
        self.patient_strategies[patient_id] = strategy
    
    def predict_with_strategy(self, 
                            patient_id: int,
                            base_probs: np.ndarray,
                            biomarker_boost: Optional[Dict] = None,
                            temporal_probs: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, Dict]:
        """
        使用患者特定策略进行预测
        
        Parameters:
        -----------
        patient_id : int
            患者ID
        base_probs : np.ndarray
            基础模型预测概率 [P(Interictal), P(Preictal)]
        biomarker_boost : Dict, optional
            生物标志物增强信息
        temporal_probs : List[np.ndarray], optional
            时间序列上下文中的历史预测概率
        
        Returns:
        --------
        adjusted_probs : np.ndarray
            调整后的预测概率
        strategy_info : Dict
            使用的策略信息
        """
        # 检查是否有特殊策略
        if patient_id not in self.patient_strategies:
            # 正常患者，直接返回基础预测
            return base_probs, {'strategy': 'normal', 'adjustments': {}}
        
        strategy = self.patient_strategies[patient_id]
        adjusted_probs = base_probs.copy()
        
        # 1. 应用生物标志物增强（如果有）
        if biomarker_boost is not None:
            boost_multiplier = strategy['biomarker_weight_multiplier']
            if biomarker_boost.get('boost_strength', 0) > 0:
                boost_strength = biomarker_boost['boost_strength'] * boost_multiplier
                if biomarker_boost.get('boost_direction') == 1:
                    transfer = min(boost_strength * adjusted_probs[0] * 0.5, adjusted_probs[0] * 0.3)
                    adjusted_probs[0] -= transfer
                    adjusted_probs[1] += transfer
        
        # 2. 应用时间序列上下文（如果有）
        if self.use_temporal_context and temporal_probs is not None:
            adjusted_probs = self._apply_temporal_context(
                adjusted_probs, temporal_probs, strategy['temporal_window_size']
            )
        
        # 3. 应用集成学习（如果有多个模型）
        if self.use_ensemble:
            # 这里可以集成多个模型的预测
            # 简化版本：对当前预测进行平滑
            ensemble_weight = strategy['ensemble_weight']
            if ensemble_weight > 1.0:
                # 增强预测的置信度
                adjusted_probs = self._enhance_confidence(adjusted_probs, ensemble_weight)
        
        # 确保概率归一化
        adjusted_probs = np.clip(adjusted_probs, 0.0, 1.0)
        adjusted_probs = adjusted_probs / (adjusted_probs.sum() + 1e-10)
        
        strategy_info = {
            'strategy': strategy['performance_type'],
            'adjusted_threshold': strategy['adjusted_threshold'],
            'adjustments': {
                'biomarker_multiplier': strategy['biomarker_weight_multiplier'],
                'ensemble_weight': strategy['ensemble_weight']
            }
        }
        
        return adjusted_probs, strategy_info
    
    def _apply_temporal_context(self, 
                               current_probs: np.ndarray,
                               temporal_probs: List[np.ndarray],
                               window_size: int) -> np.ndarray:
        """
        应用时间序列上下文
        
        使用历史预测来平滑当前预测
        
        Parameters:
        -----------
        current_probs : np.ndarray
            当前预测概率
        temporal_probs : List[np.ndarray]
            历史预测概率列表
        window_size : int
            时间窗口大小
        
        Returns:
        --------
        smoothed_probs : np.ndarray
            平滑后的预测概率
        """
        if not temporal_probs:
            return current_probs
        
        # 只使用最近的window_size个预测
        recent_probs = temporal_probs[-window_size:]
        
        # 计算加权平均（越近的权重越大）
        weights = np.linspace(0.5, 1.0, len(recent_probs))
        weights = weights / weights.sum()
        
        weighted_avg = np.zeros(2)
        for prob, weight in zip(recent_probs, weights):
            weighted_avg += prob * weight
        
        # 结合当前预测和历史平均（当前预测权重0.6，历史平均权重0.4）
        smoothed_probs = 0.6 * current_probs + 0.4 * weighted_avg
        
        return smoothed_probs
    
    def _enhance_confidence(self, probs: np.ndarray, weight: float) -> np.ndarray:
        """
        增强预测置信度
        
        通过增加高概率类别的概率来增强置信度
        
        Parameters:
        -----------
        probs : np.ndarray
            预测概率
        weight : float
            增强权重
        
        Returns:
        --------
        enhanced_probs : np.ndarray
            增强后的概率
        """
        # 找到最大概率的类别
        max_idx = np.argmax(probs)
        
        # 增加该类别的概率
        enhancement = (weight - 1.0) * 0.1  # 最多增加10%
        enhanced_probs = probs.copy()
        enhanced_probs[max_idx] = min(1.0, probs[max_idx] + enhancement)
        
        # 从其他类别中减少相应概率
        other_idx = 1 - max_idx
        enhanced_probs[other_idx] = max(0.0, probs[other_idx] - enhancement)
        
        return enhanced_probs
    
    def update_temporal_context(self, patient_id: int, probs: np.ndarray, label: Optional[int] = None):
        """
        更新时间序列上下文
        
        Parameters:
        -----------
        patient_id : int
            患者ID
        probs : np.ndarray
            预测概率
        label : int, optional
            真实标签（用于学习）
        """
        self.temporal_context[patient_id].append((probs[1], label if label is not None else -1))
        
        # 限制上下文长度（最多保留100个历史记录）
        if len(self.temporal_context[patient_id]) > 100:
            self.temporal_context[patient_id] = self.temporal_context[patient_id][-100:]
    
    def get_patient_strategy(self, patient_id: int) -> Optional[Dict]:
        """获取患者策略"""
        return self.patient_strategies.get(patient_id)
    
    def get_low_performance_patients(self) -> List[int]:
        """获取所有低性能患者ID列表"""
        return list(self.patient_strategies.keys())
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total_patients = len(self.patient_profiles)
        low_perf_patients = len(self.patient_strategies)
        
        perf_type_counts = defaultdict(int)
        for strategy in self.patient_strategies.values():
            perf_type_counts[strategy['performance_type']] += 1
        
        return {
            'total_patients': total_patients,
            'low_performance_patients': low_perf_patients,
            'low_performance_rate': low_perf_patients / total_patients if total_patients > 0 else 0.0,
            'performance_type_distribution': dict(perf_type_counts)
        }


def evaluate_patient_performance(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                patient_id: int) -> PatientPerformanceProfile:
    """
    评估患者性能并创建性能档案
    
    Parameters:
    -----------
    y_true : np.ndarray
        真实标签
    y_pred : np.ndarray
        预测标签
    patient_id : int
        患者ID
    
    Returns:
    --------
    profile : PatientPerformanceProfile
        患者性能档案
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
    else:
        # 处理只有一个类别的情况
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 0:
                TN, FP, FN, TP = len(y_true), 0, 0, 0
            else:
                TN, FP, FN, TP = 0, 0, 0, len(y_true)
        else:
            TN, FP, FN, TP = 0, 0, 0, 0
    
    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    n_preictal = np.sum(y_true == 1)
    n_interictal = np.sum(y_true == 0)
    
    return PatientPerformanceProfile(
        patient_id=patient_id,
        sensitivity=sensitivity,
        specificity=specificity,
        accuracy=accuracy,
        f1_score=f1,
        n_samples=len(y_true),
        n_preictal_samples=n_preictal,
        n_interictal_samples=n_interictal
    )


if __name__ == "__main__":
    print("低性能患者处理模块")
    print("=" * 80)
    
    # 示例用法
    handler = LowPerformancePatientHandler(
        sensitivity_threshold=0.5,
        specificity_threshold=0.5
    )
    
    # 模拟患者性能评估
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 1])  # 一些错误预测
    
    profile = evaluate_patient_performance(y_true, y_pred, patient_id=1)
    handler.register_patient_performance(1, profile)
    
    print(f"患者1性能: 敏感性={profile.sensitivity:.3f}, 特异性={profile.specificity:.3f}")
    print(f"是否为低性能: {profile.is_low_performance()}")
    print(f"性能类型: {profile.get_performance_type()}")
    print(f"统计信息: {handler.get_statistics()}")
