#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后处理流程使用示例

展示如何使用完整的后处理流程来改进癫痫预测任务
"""

import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from post_processing_pipeline import create_post_processing_pipeline
from evaluation_metrics import (
    calculate_metrics, compare_metrics, print_metrics_comparison,
    plot_metrics_comparison, evaluate_patient_wise_performance
)


def example_model_predictor(sample_data):
    """
    示例模型预测函数
    
    在实际使用中，这里应该是你的真实模型预测函数
    例如：model.predict_proba(sample_data)
    
    Parameters:
    -----------
    sample_data : np.ndarray
        样本EEG数据，形状为 (n_channels, n_timepoints)
    
    Returns:
    --------
    probs : np.ndarray
        预测概率 [P(Interictal), P(Preictal)]
    """
    # 这里使用模拟预测
    # 实际使用时，应该调用真实的模型
    np.random.seed(hash(sample_data.tobytes()) % 2**32)
    probs = np.random.dirichlet([2, 1])
    return probs


def main_example():
    """主示例函数"""
    print("=" * 80)
    print("后处理流程使用示例")
    print("=" * 80)
    
    # 1. 配置参数
    biomarker_dir = r"D:\2025_important_projects\data\chbmit_biomarkers\01\raw_eeg"
    # 如果目录不存在，使用相对路径或提示用户
    if not os.path.exists(biomarker_dir):
        print(f"警告: 生物标志物目录不存在: {biomarker_dir}")
        print("请根据实际情况修改biomarker_dir路径")
        biomarker_dir = None  # 设置为None，后处理流程会跳过生物标志物增强
    
    # 2. 创建后处理流程
    print("\n1. 创建后处理流程...")
    pipeline = create_post_processing_pipeline(
        biomarker_dir=biomarker_dir if biomarker_dir else "",
        base_model_predictor=example_model_predictor,
        biomarker_weight=0.3,
        sensitivity_threshold=0.5,
        specificity_threshold=0.5,
        use_dynamic_threshold=True,
        use_temporal_context=True
    )
    print("✓ 后处理流程创建成功")
    
    # 3. 模拟数据
    print("\n2. 准备测试数据...")
    n_samples = 50
    n_channels = 22
    n_timepoints = 1280
    
    # 生成模拟EEG数据
    X_test = np.random.randn(n_samples, n_channels, n_timepoints) * 0.1
    y_test = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    print(f"  测试样本数: {n_samples}")
    print(f"  数据形状: {X_test.shape}")
    print(f"  标签分布: Interictal={np.sum(y_test==0)}, Preictal={np.sum(y_test==1)}")
    
    # 4. 进行预测（后处理前）
    print("\n3. 进行基础模型预测（后处理前）...")
    y_pred_before = []
    y_probs_before = []
    
    for i in range(n_samples):
        probs = example_model_predictor(X_test[i])
        pred = np.argmax(probs)
        y_pred_before.append(pred)
        y_probs_before.append(probs)
    
    y_pred_before = np.array(y_pred_before)
    y_probs_before = np.array(y_probs_before)
    
    # 5. 进行后处理预测
    print("\n4. 进行后处理预测...")
    y_pred_after = []
    y_probs_after = []
    predictions_info = []
    
    patient_id = 1  # 假设所有样本来自同一患者
    
    for i in range(n_samples):
        prediction, probs, info = pipeline.predict(
            sample_data=X_test[i],
            patient_id=patient_id,
            sample_index=i
        )
        y_pred_after.append(prediction)
        y_probs_after.append(probs)
        predictions_info.append(info)
    
    y_pred_after = np.array(y_pred_after)
    y_probs_after = np.array(y_probs_after)
    
    # 6. 评估性能
    print("\n5. 评估性能...")
    
    # 计算指标
    metrics_before = calculate_metrics(y_test, y_pred_before, y_probs_before)
    metrics_after = calculate_metrics(y_test, y_pred_after, y_probs_after)
    
    # 比较指标
    comparison = compare_metrics(metrics_before, metrics_after)
    print_metrics_comparison(comparison)
    
    # 7. 评估并更新后处理参数
    print("\n6. 评估并更新后处理参数...")
    evaluation_result = pipeline.evaluate_and_update(
        patient_id=patient_id,
        y_true=y_test,
        y_pred=y_pred_after,
        predictions_info=predictions_info
    )
    
    print(f"  患者 {patient_id} 性能:")
    print(f"    敏感性: {evaluation_result['sensitivity']:.4f}")
    print(f"    特异性: {evaluation_result['specificity']:.4f}")
    print(f"    准确率: {evaluation_result['accuracy']:.4f}")
    print(f"    F1分数: {evaluation_result['f1_score']:.4f}")
    print(f"    是否为低性能: {evaluation_result['is_low_performance']}")
    
    # 8. 显示统计信息
    print("\n7. 后处理统计信息...")
    stats = pipeline.get_statistics()
    print(f"  总预测数: {stats['total_predictions']}")
    print(f"  生物标志物增强: {stats['biomarker_enhanced']}")
    print(f"  低性能调整: {stats['low_perf_adjusted']}")
    print(f"  动态阈值使用: {stats['dynamic_threshold_used']}")
    
    if 'biomarker_stats' in stats:
        print(f"  生物标志物匹配率: {stats['biomarker_stats'].get('biomarker_match_rate', 0):.4f}")
    
    if 'low_perf_stats' in stats:
        print(f"  低性能患者数: {stats['low_perf_stats'].get('low_performance_patients', 0)}")
    
    print("\n" + "=" * 80)
    print("示例完成！")
    print("=" * 80)


def example_with_real_data():
    """
    使用真实数据的示例
    
    这个函数展示了如何在实际项目中使用后处理流程
    """
    print("\n" + "=" * 80)
    print("真实数据使用示例")
    print("=" * 80)
    
    # 这里需要根据实际情况加载数据
    # 示例代码框架：
    """
    # 1. 加载数据
    from cms_lime_chb_analysis_enhanced import load_chb_data_robust
    
    patient_id = 1
    data_root = r"D:\public_data\CHBMIT\1_data_clean\chb%02d" % patient_id
    segment_info = r"D:\public_data\CHBMIT\segment_clean\30-5-240\chb%02d\segment_info.json" % patient_id
    
    X_data, y_data = load_chb_data_robust(
        patient_id, data_root, segment_info, max_samples=100
    )
    
    # 2. 加载模型
    import torch
    from model.EEGInception_SE import EEGInception
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = r"D:\public_data\CHBMIT\weight\eeginception+se\patient1_*.pth"
    
    n_chans = X_data.shape[1]
    model = EEGInception(input_time=1280, fs=256, ncha=n_chans, n_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    def model_predictor(sample):
        with torch.no_grad():
            x_tensor = torch.from_numpy(sample).unsqueeze(0).to(device).float()
            probs = torch.softmax(model(x_tensor), dim=1).cpu().numpy()[0]
            return probs
    
    # 3. 创建后处理流程
    biomarker_dir = r"D:\2025_important_projects\data\chbmit_biomarkers\01\raw_eeg"
    pipeline = create_post_processing_pipeline(
        biomarker_dir=biomarker_dir,
        base_model_predictor=model_predictor,
        biomarker_weight=0.3
    )
    
    # 4. 进行预测
    predictions = []
    probabilities = []
    for i in range(len(X_data)):
        pred, probs, info = pipeline.predict(X_data[i], patient_id=patient_id, sample_index=i)
        predictions.append(pred)
        probabilities.append(probs)
    
    # 5. 评估
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    
    y_pred = np.array(predictions)
    accuracy = accuracy_score(y_data, y_pred)
    sensitivity = recall_score(y_data, y_pred)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"敏感性: {sensitivity:.4f}")
    """
    
    print("请根据实际情况实现此函数")


if __name__ == "__main__":
    # 运行示例
    main_example()
    
    # 如果需要使用真实数据，取消下面的注释
    # example_with_real_data()
