#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一基元选拔主程序

对每个样本提取基元后立即评估重要性，保存高重要性基元作为生物标志物

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

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from unified_primitive_selection import (
    UnifiedPrimitiveSelection,
    UnifiedSelectionConfig,
    create_default_config,
    BiomarkerUnit
)

from cms_lime_chb_analysis_final import (
    load_chb_data_robust,
    load_model_robust,
    SimpleEEGModel,
    log_print,
    setup_logger
)


def natural_keys(text):
    """Helper function for natural sorting of filenames"""
    def atoi(text):
        return int(text) if text.isdigit() else text
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def create_results_folder(base_path: str = "results/unified_primitive_selection_results") -> str:
    """创建结果文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{base_path}_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


class EEGModelWrapper:
    """EEG模型包装器，提供统一的predict_proba接口"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率分布
        
        Parameters:
        -----------
        X : np.ndarray
            输入数据，形状为 (n_samples, n_channels, n_timepoints) 或 (n_channels, n_timepoints)
        
        Returns:
        --------
        probabilities : np.ndarray
            概率分布，形状为 (n_samples, n_classes) 或 (n_classes,)
        """
        # 确保输入格式正确
        if X.ndim == 2:
            X = X[np.newaxis, :, :]  # (1, n_channels, n_timepoints)
        
        # 转换为torch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            # 如果是logits，转换为概率
            if outputs.shape[-1] > 1:
                probabilities = torch.softmax(outputs, dim=-1)
            else:
                # 二分类情况
                probabilities = torch.sigmoid(outputs)
                probabilities = torch.cat([1 - probabilities, probabilities], dim=-1)
        
        # 转换为numpy数组
        probabilities = probabilities.cpu().numpy()
        
        # 如果输入是单个样本，返回一维数组
        if X.ndim == 2 or (X.ndim == 3 and X.shape[0] == 1):
            probabilities = probabilities[0]
        
        return probabilities


def main():
    """主程序"""
    # 配置参数
    patient_id = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 创建结果文件夹
    results_folder = create_results_folder()
    log_file_path = os.path.join(results_folder, 'unified_selection.log')
    
    # 设置日志
    setup_logger(log_file_path)
    
    log_print("=" * 80)
    log_print("统一基元选拔框架 - CHB-MIT数据集")
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
            max_samples=200,
            max_seizure_samples=100,
            max_normal_samples=100
        )
        
        log_print(f"数据加载成功: {X_data.shape}")
        log_print(f"标签分布: {np.bincount(y_data)}")
        
    except Exception as e:
        log_print(f"数据加载失败: {e}")
        log_print("使用模拟数据...")
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
        test_sample = X_data[0:1]
        pred = model_wrapper.predict_proba(test_sample)
        log_print(f"模型预测测试成功: {pred[0]}")
    except Exception as e:
        log_print(f"模型预测测试失败: {e}")
    
    # 3. 创建配置
    log_print("\n3. 创建统一选拔配置...")
    
    config = create_default_config()
    # 根据数据调整配置
    config.prior_config.n_channels = X_data.shape[1]
    config.prior_config.n_timepoints = X_data.shape[2]
    config.prior_config.sampling_rate = 256.0
    
    # 启用数量模式
    config.prior_config.use_count_mode = True
    config.prior_config.target_shapelet_count = 50  # 每个样本提取50个
    config.prior_config.target_microstate_count = 50
    config.prior_config.target_timefreq_count = 50
    
    # 设置评估参数
    config.evaluation_config.importance_threshold = 0.01
    config.evaluation_config.top_k_biomarkers = 20
    # 减少扰动次数以提高速度
    config.evaluation_config.n_perturbations = 5  # 从10减少到5
    config.evaluation_config.stability_iterations = 2  # 从3减少到2
    
    # 3.1. 创建因果图（如果启用因果扰动）
    if config.evaluation_config.use_causal_perturbation:
        log_print("\n3.1. 创建因果图...")
        
        try:
            # 方法1: 使用 GrangerCausalityAnalyzer（最准确，推荐）
            from causal_perturbation import GrangerCausalityAnalyzer
            
            max_lag = 5
            significance_level = 0.05
            
            log_print(f"使用 GrangerCausalityAnalyzer 构建因果图...")
            log_print(f"  最大滞后: {max_lag}")
            log_print(f"  显著性水平: {significance_level}")
            log_print(f"  数据形状: {X_data.shape}")
            
            # 创建因果分析器
            causal_analyzer = GrangerCausalityAnalyzer(
                max_lag=max_lag,
                significance_level=significance_level
            )
            
            # 拟合因果模型（使用所有数据）
            log_print("拟合因果模型（这可能需要一些时间）...")
            causal_analyzer.fit(X_data)
            
            # 获取因果图
            causal_graph = causal_analyzer.get_causal_graph()
            
            # 统计因果边数量
            n_edges = np.sum(causal_graph.adjacency_matrix > 0)
            log_print(f"因果图创建成功！")
            log_print(f"  节点数: {causal_graph.n_nodes}")
            log_print(f"  因果边数: {n_edges}")
            
            # 设置到配置中
            config.evaluation_config.causal_graph = causal_graph
            log_print("因果图已设置到配置中")
            
        except ImportError:
            # 方法2: 使用 cms_lime.CausalGraph（简化版本）
            log_print("无法导入 GrangerCausalityAnalyzer，使用 cms_lime.CausalGraph...")
            
            try:
                from cms_lime import CausalGraph
                
                n_channels = X_data.shape[1]
                max_lag = 5
                
                causal_graph = CausalGraph(n_channels=n_channels, max_lag=max_lag)
                
                # 添加一些基于相关性的简单因果边（示例）
                # 注意：这是简化版本，实际应用中应该使用真实的因果分析
                log_print("添加基于相关性的因果边（简化方法）...")
                edge_count = 0
                
                # 计算通道间的相关性
                if X_data.ndim == 3:
                    # 计算平均相关性
                    correlations = np.zeros((n_channels, n_channels))
                    for i in range(n_channels):
                        for j in range(n_channels):
                            if i != j:
                                # 计算两个通道的平均相关性
                                ch_i = X_data[:, i, :].flatten()
                                ch_j = X_data[:, j, :].flatten()
                                corr = np.abs(np.corrcoef(ch_i, ch_j)[0, 1])
                                correlations[i, j] = corr
                    
                    # 添加相关性较高的边
                    threshold = np.percentile(correlations[correlations > 0], 75)  # 使用75分位数作为阈值
                    for i in range(n_channels):
                        for j in range(n_channels):
                            if i != j and correlations[i, j] > threshold:
                                strength = float(correlations[i, j])
                                causal_graph.add_edge(i, j, lag=1, strength=strength)
                                edge_count += 1
                
                log_print(f"添加了 {edge_count} 条因果边（基于相关性）")
                config.evaluation_config.causal_graph = causal_graph
                
            except ImportError:
                log_print("警告: 无法导入任何因果图类，将使用普通扰动")
                config.evaluation_config.use_causal_perturbation = False
                
        except Exception as e:
            log_print(f"创建因果图时出错: {e}")
            log_print("将使用普通扰动")
            config.evaluation_config.use_causal_perturbation = False
    else:
        log_print("\n3.1. 使用普通扰动（未启用因果扰动）")
    
    log_print(config.get_summary())
    
    # 4. 创建统一选拔框架
    log_print("\n4. 创建统一选拔框架...")
    
    framework = UnifiedPrimitiveSelection(
        config=config,
        model=model_wrapper.predict_proba
    )
    
    # 5. 处理样本
    log_print("\n5. 处理样本（提取并评估生物标志物）...")
    
    # 选择部分样本进行处理（提高效率）
    n_samples_to_process = min(10, len(X_data))
    sample_indices = list(range(n_samples_to_process))
    
    sample_biomarkers = framework.process_samples(
        X_data,
        y_data,
        sample_indices=sample_indices
    )
    
    # 6. 获取top K生物标志物
    log_print("\n6. 获取Top K生物标志物...")
    
    top_biomarkers = framework.get_top_biomarkers(top_k=20)
    log_print(f"Top {len(top_biomarkers)} 个生物标志物:")
    for i, biomarker in enumerate(top_biomarkers[:10], 1):  # 只显示前10个
        log_print(f"  {i}. {biomarker.primitive_type} - "
                 f"重要性: {biomarker.importance_score:.6f}, "
                 f"质量: {biomarker.quality_score:.4f}")
    
    # 7. 保存结果
    log_print("\n7. 保存结果...")
    
    framework.save_results(results_folder)
    
    log_print("\n" + "=" * 80)
    log_print("统一基元选拔完成！")
    log_print("=" * 80)
    log_print(f"结果保存在: {results_folder}")
    log_print(f"处理的样本数: {len(sample_biomarkers)}")
    log_print(f"提取的生物标志物总数: {len(framework.all_biomarkers)}")
    log_print(f"Top K生物标志物数: {len(top_biomarkers)}")


if __name__ == "__main__":
    main()
