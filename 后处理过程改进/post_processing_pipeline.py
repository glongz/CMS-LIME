#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整后处理流程

整合所有后处理模块，提供端到端的后处理解决方案
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import os
from pathlib import Path

# 导入其他模块
from biomarker_enhanced_prediction import BiomarkerEnhancedPredictor, create_biomarker_enhanced_predictor
from low_performance_patient_handler import LowPerformancePatientHandler, evaluate_patient_performance
from dynamic_threshold_adjustment import DynamicThresholdAdjuster, ThresholdConfig


class PostProcessingPipeline:
    """完整后处理流程"""
    
    def __init__(self,
                 biomarker_dir: str,
                 base_model_predictor: Callable,
                 biomarker_weight: float = 0.3,
                 sensitivity_threshold: float = 0.5,
                 specificity_threshold: float = 0.5,
                 use_dynamic_threshold: bool = True,
                 use_temporal_context: bool = True):
        """
        初始化后处理流程
        
        Parameters:
        -----------
        biomarker_dir : str
            生物标志物文件目录
        base_model_predictor : Callable
            基础模型预测函数
        biomarker_weight : float
            生物标志物权重
        sensitivity_threshold : float
            敏感性阈值
        specificity_threshold : float
            特异性阈值
        use_dynamic_threshold : bool
            是否使用动态阈值
        use_temporal_context : bool
            是否使用时间序列上下文
        """
        # 1. 生物标志物增强预测器
        self.biomarker_predictor = create_biomarker_enhanced_predictor(
            biomarker_dir=biomarker_dir,
            model_predictor=base_model_predictor,
            biomarker_weight=biomarker_weight
        )
        
        # 2. 低性能患者处理器
        self.low_perf_handler = LowPerformancePatientHandler(
            sensitivity_threshold=sensitivity_threshold,
            specificity_threshold=specificity_threshold,
            use_ensemble=True,
            use_temporal_context=use_temporal_context
        )
        
        # 3. 动态阈值调整器
        self.threshold_adjuster = DynamicThresholdAdjuster() if use_dynamic_threshold else None
        
        # 统计信息
        self.stats = {
            'total_predictions': 0,
            'biomarker_enhanced': 0,
            'low_perf_adjusted': 0,
            'dynamic_threshold_used': 0
        }
    
    def predict(self,
                sample_data: np.ndarray,
                patient_id: int,
                sample_index: Optional[int] = None) -> Tuple[int, np.ndarray, Dict]:
        """
        完整的后处理预测流程
        
        Parameters:
        -----------
        sample_data : np.ndarray
            样本EEG数据，形状为 (n_channels, n_timepoints)
        patient_id : int
            患者ID
        sample_index : int, optional
            样本索引
        
        Returns:
        --------
        prediction : int
            最终预测类别（0: Interictal, 1: Preictal）
        probabilities : np.ndarray
            最终预测概率 [P(Interictal), P(Preictal)]
        info : Dict
            预测详细信息
        """
        self.stats['total_predictions'] += 1
        
        # 步骤1: 生物标志物增强预测
        enhanced_probs, biomarker_info = self.biomarker_predictor.predict(
            sample_data, sample_index=sample_index
        )
        
        if biomarker_info['biomarker_matches'] > 0:
            self.stats['biomarker_enhanced'] += 1
        
        biomarker_present = biomarker_info['biomarker_matches'] > 0
        biomarker_boost = biomarker_info['biomarker_boost']
        
        # 步骤2: 低性能患者特殊处理
        # 获取时间序列上下文（如果有）
        temporal_probs = None
        if self.low_perf_handler.use_temporal_context:
            # 这里可以从缓存中获取历史预测
            # 简化版本：暂时不使用
            pass
        
        adjusted_probs, strategy_info = self.low_perf_handler.predict_with_strategy(
            patient_id=patient_id,
            base_probs=enhanced_probs,
            biomarker_boost=biomarker_boost,
            temporal_probs=temporal_probs
        )
        
        if strategy_info['strategy'] != 'normal':
            self.stats['low_perf_adjusted'] += 1
        
        # 步骤3: 动态阈值调整
        final_probs = adjusted_probs
        threshold_used = 0.5
        
        if self.threshold_adjuster is not None:
            # 获取最近性能（如果有）
            recent_performance = None
            patient_stats = self.threshold_adjuster.get_patient_statistics(patient_id)
            if patient_stats.get('recent_performance'):
                recent_performance = patient_stats['recent_performance']
            
            # 使用动态阈值进行预测
            prediction, threshold_used = self.threshold_adjuster.predict_with_dynamic_threshold(
                patient_id=patient_id,
                probs=adjusted_probs,
                biomarker_present=biomarker_present
            )
            
            self.stats['dynamic_threshold_used'] += 1
        else:
            # 使用固定阈值
            threshold_used = 0.5
            prediction = 1 if adjusted_probs[1] >= threshold_used else 0
        
        # 更新时间序列上下文
        self.low_perf_handler.update_temporal_context(patient_id, final_probs)
        
        # 更新生物标志物统计
        if self.threshold_adjuster is not None:
            # 注意：这里需要真实标签，但在预测时可能没有
            # 可以在评估阶段更新
            pass
        
        # 组装信息
        info = {
            'base_probs': biomarker_info['base_probs'],
            'enhanced_probs': enhanced_probs,
            'adjusted_probs': adjusted_probs,
            'final_probs': final_probs,
            'prediction': prediction,
            'threshold_used': threshold_used,
            'biomarker_info': {
                'matches': biomarker_info['biomarker_matches'],
                'boost': biomarker_boost
            },
            'strategy_info': strategy_info,
            'patient_id': patient_id,
            'sample_index': sample_index,
            'biomarker_weight_used': float(self.biomarker_predictor.biomarker_weight),
            'min_biomarker_confidence_used': float(
                self.biomarker_predictor.min_biomarker_confidence
            ),
        }
        
        return prediction, final_probs, info
    
    def evaluate_and_update(self,
                           patient_id: int,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           predictions_info: List[Dict]):
        """
        评估性能并更新后处理参数
        
        Parameters:
        -----------
        patient_id : int
            患者ID
        y_true : np.ndarray
            真实标签
        y_pred : np.ndarray
            预测标签
        predictions_info : List[Dict]
            预测信息列表
        """
        from sklearn.metrics import confusion_matrix
        
        # 1. 评估患者性能
        profile = evaluate_patient_performance(y_true, y_pred, patient_id)
        self.low_perf_handler.register_patient_performance(patient_id, profile)
        
        # 2. 计算性能指标
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        else:
            sensitivity = profile.sensitivity
            specificity = profile.specificity
        
        # 3. 更新动态阈值
        if self.threshold_adjuster is not None:
            self.threshold_adjuster.update_threshold(
                patient_id=patient_id,
                current_sensitivity=sensitivity,
                current_specificity=specificity
            )
            
            # 更新生物标志物统计
            for i, info in enumerate(predictions_info):
                if i < len(y_true):
                    biomarker_present = info.get('biomarker_info', {}).get('matches', 0) > 0
                    true_label = y_true[i]
                    self.threshold_adjuster.update_biomarker_statistics(
                        patient_id=patient_id,
                        biomarker_present=biomarker_present,
                        true_label=true_label
                    )
        
        return {
            'patient_id': patient_id,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': profile.accuracy,
            'f1_score': profile.f1_score,
            'is_low_performance': profile.is_low_performance()
        }
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 添加各模块的统计信息
        stats['biomarker_stats'] = self.biomarker_predictor.get_statistics()
        stats['low_perf_stats'] = self.low_perf_handler.get_statistics()
        
        if self.threshold_adjuster is not None:
            stats['threshold_info'] = {
                'n_patients_with_custom_threshold': len(self.threshold_adjuster.patient_thresholds)
            }
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'total_predictions': 0,
            'biomarker_enhanced': 0,
            'low_perf_adjusted': 0,
            'dynamic_threshold_used': 0
        }
        self.biomarker_predictor.reset_statistics()


def create_post_processing_pipeline(biomarker_dir: str,
                                    base_model_predictor: Callable,
                                    **kwargs) -> PostProcessingPipeline:
    """
    创建后处理流程的便捷函数
    
    Parameters:
    -----------
    biomarker_dir : str
        生物标志物文件目录
    base_model_predictor : Callable
        基础模型预测函数
    **kwargs
        其他配置参数
    
    Returns:
    --------
    pipeline : PostProcessingPipeline
        后处理流程对象
    """
    return PostProcessingPipeline(
        biomarker_dir=biomarker_dir,
        base_model_predictor=base_model_predictor,
        **kwargs
    )


if __name__ == "__main__":
    print("完整后处理流程模块")
    print("=" * 80)
    
    # 示例用法
    """
    def dummy_model_predictor(sample):
        return np.array([0.7, 0.3])
    
    biomarker_dir = r"D:\2025_important_projects\data\chbmit_biomarkers\01\raw_eeg"
    
    pipeline = create_post_processing_pipeline(
        biomarker_dir=biomarker_dir,
        base_model_predictor=dummy_model_predictor,
        biomarker_weight=0.3,
        sensitivity_threshold=0.5,
        specificity_threshold=0.5
    )
    
    # 测试预测
    sample_data = np.random.randn(22, 1280)
    prediction, probs, info = pipeline.predict(sample_data, patient_id=1, sample_index=0)
    
    print(f"预测结果: {prediction}")
    print(f"预测概率: {probs}")
    print(f"统计信息: {pipeline.get_statistics()}")
    """
    
    print("模块已加载，请根据实际需求配置使用")
