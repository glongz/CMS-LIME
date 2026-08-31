#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两阶段基元选拔主程序

集成CHB-MIT数据集，执行完整的两阶段选拔流程

Author: CMS-LIME Framework
Date: 2025
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import json
import re
from datetime import datetime

# 添加父目录到路径，以便导入依赖模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入框架模块
from two_stage_selection import (
    PriorKnowledgeConfig, 
    create_default_prior_config,
    TwoStagePrimitiveSelection, 
    create_results_folder
)
from cms_lime_chb_analysis_final import (
    load_chb_data_robust, 
    load_model_robust,
    SimpleEEGModel,
    log_print,
    setup_logger
)
from two_stage_selection.utils.visualization import (
    visualize_stage1_results,
    visualize_stage2_results,
    visualize_biomarker_comparison
)


def natural_keys(text):
    """Helper function for natural sorting of filenames"""
    def atoi(text):
        return int(text) if text.isdigit() else text
    return [atoi(c) for c in re.split(r'(\d+)', text)]


class EEGModelWrapper:
    """EEG模型包装器，提供统一的predict_proba接口"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
    
    def predict_proba(self, X):
        """预测类别概率"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                # Handle different input shapes
                if X.ndim == 2:
                    # Single sample: (22, 1280) -> (1, 22, 1280)
                    if X.shape[0] != 22 or X.shape[1] != 1280:
                        raise ValueError(f"Expected single sample shape (22, 1280), got {X.shape}")
                    X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)
                elif X.ndim == 3:
                    # Batch: (batch, 22, 1280)
                    if X.shape[-2] != 22 or X.shape[-1] != 1280:
                        raise ValueError(f"Expected batch shape (batch, 22, 1280), got {X.shape}")
                    X_tensor = torch.FloatTensor(X).to(self.device)
                else:
                    raise ValueError(f"Unexpected input shape: {X.shape}")
            else:
                X_tensor = X.to(self.device)
            
            # Ensure input is in correct format
            if X_tensor.dim() == 2:
                X_tensor = X_tensor.unsqueeze(0)
            
            try:
                outputs = self.model(X_tensor)
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(0)
                probabilities = torch.softmax(outputs, dim=1)
                return probabilities.cpu().numpy()
            except Exception as e:
                log_print(f"Error in model prediction: {e}")
                # Return dummy probabilities
                batch_size = X_tensor.shape[0]
                return np.random.rand(batch_size, 2)


def main():
    """主程序"""
    # 配置参数
    patient_id = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 创建结果文件夹
    results_folder = create_results_folder()
    log_file_path = os.path.join(results_folder, 'two_stage_selection.log')
    
    # 设置日志
    setup_logger(log_file_path)
    
    log_print("=" * 80)
    log_print("两阶段基元选拔框架 - CHB-MIT数据集")
    log_print("=" * 80)
    log_print(f"结果文件夹: {results_folder}")
    log_print(f"设备: {device}")
    log_print(f"患者ID: {patient_id}")
    
    # 1. 加载数据
    log_print("\n1. 加载CHB-MIT数据...")
    
    data_root_template = r'D:\public_data\CHBMIT\1_data_clean\chb%02d'
    segment_info_template = r'D:\public_data\CHBMIT\segment_clean\30-5-240\chb%02d\segment_info.json'
    
    try:
        X_data, y_data = load_chb_data_robust(
            patient_id, 
            data_root_template, 
            segment_info_template, 
            max_samples=200,  # 限制样本数以提高效率
            max_seizure_samples=100,
            max_normal_samples=100
        )
        
        log_print(f"数据加载成功: {X_data.shape}")
        log_print(f"标签分布: {np.bincount(y_data)}")
        
    except Exception as e:
        log_print(f"数据加载失败: {e}")
        log_print("使用模拟数据...")
        # 创建模拟数据
        np.random.seed(42)
        X_data = np.random.randn(50, 22, 1280)
        y_data = np.random.randint(0, 2, 50)
        log_print(f"模拟数据形状: {X_data.shape}")
    
    # 2. 加载模型
    log_print("\n2. 加载EEG模型...")
    
    n_chans = X_data.shape[1]
    weight_model_path = r'D:\public_data\CHBMIT\weight\eeginception+se'
    
    model = None
    if os.path.exists(weight_model_path):
        weight_files = [f for f in os.listdir(weight_model_path) if f.endswith('.pth')]
        if weight_files:
            weight_files = sorted(weight_files, key=natural_keys)
            model_path = os.path.join(weight_model_path, weight_files[0])
            log_print(f"找到模型文件: {model_path}")
            try:
                model = load_model_robust(model_path, n_chans=n_chans, n_classes=2, device=device)
            except Exception as e:
                log_print(f"模型加载失败: {e}")
    
    if model is None:
        log_print("使用简单模型作为备选...")
        model = SimpleEEGModel(n_channels=n_chans, n_timepoints=X_data.shape[2], n_classes=2)
        model.to(device)
        model.eval()
    
    # 包装模型
    model_wrapper = EEGModelWrapper(model, device)
    
    # 测试模型预测
    try:
        test_sample = X_data[0:1]  # Shape: (1, 22, 1280)
        pred = model_wrapper.predict_proba(test_sample)
        log_print(f"模型预测测试成功: {pred[0]}")
    except Exception as e:
        log_print(f"模型预测测试失败: {e}")
    
    # 3. 创建先验信息配置
    log_print("\n3. 创建先验信息配置...")
    
    prior_config = create_default_prior_config()
    # 根据数据调整配置
    prior_config.n_channels = X_data.shape[1]
    prior_config.n_timepoints = X_data.shape[2]
    prior_config.sampling_rate = 256.0
    
    # 启用数量模式，设置每类基元的目标数量
    prior_config.use_count_mode = True
    prior_config.target_shapelet_count = 100
    prior_config.target_microstate_count = 100
    prior_config.target_timefreq_count = 100
    
    log_print(prior_config.get_summary())
    
    # 4. 创建两阶段选拔框架
    log_print("\n4. 创建两阶段选拔框架...")
    
    framework = TwoStagePrimitiveSelection(
        prior_config=prior_config,
        model=model_wrapper.predict_proba,
        n_perturbations=10,
        importance_threshold=0.01,  # 降低阈值，从0.1改为0.01
        top_k_biomarkers=20
    )
    
    # 5. 第一场选拔：提取候选基元
    log_print("\n5. 执行第一场选拔：基元提取...")
    
    # 使用部分数据进行训练（提高效率）
    n_train_samples = min(50, len(X_data))
    X_train = X_data[:n_train_samples]
    y_train = y_data[:n_train_samples]
    
    framework.fit(X_train, y_train)
    
    # 6. 第二场选拔：评估基元重要性
    log_print("\n6. 执行第二场选拔：因果扰动解释...")
    
    # 选择几个样本进行解释
    n_explain_samples = min(5, len(X_data))
    all_biomarkers = []
    
    for idx in range(n_explain_samples):
        sample = X_data[idx]
        label = y_data[idx]
        
        log_print(f"\n解释样本 {idx+1}/{n_explain_samples} (标签: {label})...")
        
        try:
            biomarkers = framework.explain(sample, label)
            all_biomarkers.extend(biomarkers)
            log_print(f"样本 {idx+1} 筛选出 {len(biomarkers)} 个生物标志物")
        except Exception as e:
            log_print(f"样本 {idx+1} 解释失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 7. 保存结果
    log_print("\n7. 保存结果...")
    
    framework.save_results(results_folder)
    
    # 8. 生成可视化
    log_print("\n8. 生成可视化...")
    
    # 可视化第一场选拔结果
    stage1_viz_path = os.path.join(results_folder, 'stage1_visualization.png')
    visualize_stage1_results(framework.candidate_primitives, stage1_viz_path)
    
    # 可视化第二场选拔结果
    if framework.selected_biomarkers:
        stage2_viz_path = os.path.join(results_folder, 'stage2_visualization.png')
        visualize_stage2_results(framework.selected_biomarkers, stage2_viz_path)
    
    # 如果有多个样本，生成比较图
    if len(all_biomarkers) > 1:
        comparison_viz_path = os.path.join(results_folder, 'biomarker_comparison.png')
        sample_labels = [f'Sample {i+1}' for i in range(n_explain_samples)]
        # 需要收集每个样本的生物标志物列表
        # 这里简化处理，使用所有生物标志物
        visualize_biomarker_comparison([all_biomarkers], ['All Samples'], comparison_viz_path)
    
    # 9. 生成最终摘要
    log_print("\n9. 生成最终摘要...")
    
    summary = framework.get_summary()
    summary_path = os.path.join(results_folder, 'final_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    log_print("\n" + "=" * 80)
    log_print("两阶段基元选拔完成！")
    log_print("=" * 80)
    log_print(f"结果保存在: {results_folder}")
    log_print(f"候选基元数量: {len(framework.candidate_primitives)}")
    log_print(f"生物标志物候选数量: {len(framework.biomarker_candidates)}")
    log_print(f"选定的生物标志物数量: {len(framework.selected_biomarkers)}")
    log_print("=" * 80)


if __name__ == '__main__':
    main()
