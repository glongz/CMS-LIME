#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMS-LIME框架完整示例和测试用例
展示如何使用CMS-LIME进行EEG信号解释
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入CMS-LIME框架
from cms_lime_explainer import CMSLimeExplainer, CMSLimeConfig

class EEGCNNModel(nn.Module):
    """示例EEG CNN模型"""
    
    def __init__(self, n_channels=32, n_classes=3, n_timepoints=1000):
        super(EEGCNNModel, self).__init__()
        
        # 时间卷积层
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=25, stride=1, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(128, 256, kernel_size=10, stride=2, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 计算卷积后的时间维度
        conv_output_size = self._get_conv_output_size(n_timepoints)
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(256 * conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        
    def _get_conv_output_size(self, input_size):
        """计算卷积层输出尺寸"""
        size = input_size
        # 第一层: kernel=25, stride=1, padding=12
        size = (size + 2*12 - 25) // 1 + 1
        # 第二层: kernel=15, stride=2, padding=7
        size = (size + 2*7 - 15) // 2 + 1
        # 第三层: kernel=10, stride=2, padding=4
        size = (size + 2*4 - 10) // 2 + 1
        return size
        
    def forward(self, x):
        # x shape: (batch_size, n_channels, n_timepoints)
        x = self.temporal_conv(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x
    
    def predict_proba(self, x):
        """为兼容性添加predict_proba方法"""
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                if x.ndim == 2:  # (n_channels, n_timepoints)
                    x = torch.FloatTensor(x).unsqueeze(0)
                elif x.ndim == 3:  # (batch_size, n_channels, n_timepoints)
                    x = torch.FloatTensor(x)
                else:
                    x = torch.FloatTensor(x.reshape(1, *x.shape[-2:]))
            
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            return probs.numpy()

def generate_synthetic_eeg_data(n_samples=200, n_channels=32, n_timepoints=1000, n_classes=3):
    """生成合成EEG数据"""
    np.random.seed(42)
    
    X = []
    y = []
    
    for class_idx in range(n_classes):
        for _ in range(n_samples // n_classes):
            # 基础噪声
            signal = np.random.randn(n_channels, n_timepoints) * 0.1
            
            # 添加类别特异性模式
            if class_idx == 0:  # 类别0: 低频振荡
                t = np.linspace(0, 4*np.pi, n_timepoints)
                for ch in range(n_channels):
                    signal[ch] += 0.5 * np.sin(2 * t + ch * 0.1)
                    
            elif class_idx == 1:  # 类别1: 中频振荡
                t = np.linspace(0, 8*np.pi, n_timepoints)
                for ch in range(n_channels):
                    signal[ch] += 0.3 * np.sin(5 * t + ch * 0.2)
                    
            else:  # 类别2: 高频振荡
                t = np.linspace(0, 16*np.pi, n_timepoints)
                for ch in range(n_channels):
                    signal[ch] += 0.2 * np.sin(10 * t + ch * 0.3)
            
            # 添加一些随机事件
            n_events = np.random.randint(1, 4)
            for _ in range(n_events):
                event_start = np.random.randint(0, n_timepoints - 100)
                event_end = event_start + np.random.randint(50, 100)
                event_channels = np.random.choice(n_channels, 
                                                size=np.random.randint(1, 8), 
                                                replace=False)
                
                for ch in event_channels:
                    signal[ch, event_start:event_end] += np.random.randn(event_end - event_start) * 0.3
            
            X.append(signal)
            y.append(class_idx)
    
    return np.array(X), np.array(y)

def train_eeg_model(X_train, y_train, X_test, y_test, epochs=50):
    """训练EEG模型"""
    n_channels, n_timepoints = X_train.shape[1], X_train.shape[2]
    n_classes = len(np.unique(y_train))
    
    # 创建模型
    model = EEGCNNModel(n_channels=n_channels, 
                       n_classes=n_classes, 
                       n_timepoints=n_timepoints)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 训练循环
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        
        # 前向传播
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_losses.append(loss.item())
        
        # 测试准确率
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_pred = torch.argmax(test_outputs, dim=1)
                test_acc = accuracy_score(y_test, test_pred.numpy())
                test_accuracies.append(test_acc)
                
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Test Acc: {test_acc:.4f}")
    
    return model, train_losses, test_accuracies

def test_cms_lime_framework():
    """测试CMS-LIME框架"""
    print("=== CMS-LIME框架测试 ===")
    
    # 1. 生成数据
    print("\n1. 生成合成EEG数据...")
    X, y = generate_synthetic_eeg_data(n_samples=300, n_channels=32, n_timepoints=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"类别分布: {np.bincount(y_train)}")
    
    # 2. 训练模型
    print("\n2. 训练EEG分类模型...")
    model, train_losses, test_accuracies = train_eeg_model(X_train, y_train, X_test, y_test, epochs=30)
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        test_probs = model.predict_proba(X_test)
        test_pred = np.argmax(test_probs, axis=1)
        final_accuracy = accuracy_score(y_test, test_pred)
        
    print(f"\n最终测试准确率: {final_accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, test_pred))
    
    # 3. 创建CMS-LIME解释器
    print("\n3. 创建CMS-LIME解释器...")
    config = CMSLimeConfig(
        # 基元构造配置
        microstate_method='kmeans',
        n_microstates=6,
        shapelet_method='information_gain',
        n_shapelets=100,
        timefreq_method='cwt',
        
        # 因果分析配置
        causal_method='granger',
        max_lag=3,
        
        # 扰动配置
        perturbation_method='mask',
        n_perturbations=200,
        
        # 局部回归配置
        kernel_type='multidim',
        regression_method='ridge',
        selection_method='dpp',
        n_features=15,
        
        verbose=True
    )
    
    explainer = CMSLimeExplainer(config)
    
    # 4. 拟合解释器
    print("\n4. 拟合CMS-LIME解释器...")
    explainer.fit(X_train, y_train, model)
    
    # 5. 生成解释
    print("\n5. 生成解释结果...")
    
    # 选择几个测试样本进行解释
    test_indices = [0, 10, 20]  # 每个类别选一个
    
    for i, test_idx in enumerate(test_indices):
        print(f"\n--- 解释测试样本 {test_idx} (真实类别: {y_test[test_idx]}) ---")
        
        test_sample = X_test[test_idx]
        
        # 生成解释
        explanation = explainer.explain_instance(
            test_sample,
            unit_types=['microstate', 'shapelet', 'timefreq']
        )
        
        # 打印解释摘要
        summary = explainer.get_explanation_summary(explanation)
        print(summary)
        
        # 可视化解释（仅第一个样本）
        if i == 0:
            print("\n生成可视化图表...")
            explainer.visualize_explanation(
                explanation, 
                test_sample, 
                save_path=f'cms_lime_explanation_sample_{test_idx}.png'
            )
    
    # 6. 框架性能评估
    print("\n6. 框架性能评估...")
    
    # 测试不同基元类型的解释效果
    unit_type_combinations = [
        ['microstate'],
        ['shapelet'],
        ['timefreq'],
        ['microstate', 'shapelet'],
        ['microstate', 'timefreq'],
        ['shapelet', 'timefreq'],
        ['microstate', 'shapelet', 'timefreq']
    ]
    
    performance_results = {}
    
    for unit_types in unit_type_combinations:
        print(f"\n测试基元组合: {unit_types}")
        
        # 对前10个测试样本生成解释
        explanations = []
        for test_idx in range(min(10, len(X_test))):
            try:
                explanation = explainer.explain_instance(
                    X_test[test_idx],
                    unit_types=unit_types
                )
                explanations.append(explanation)
            except Exception as e:
                print(f"样本 {test_idx} 解释失败: {e}")
                continue
        
        # 计算平均解释质量指标
        if explanations:
            avg_n_units = np.mean([len(exp['combined_explanation']['selected_units']) 
                                 if exp['combined_explanation'] else 0 
                                 for exp in explanations])
            
            avg_max_importance = np.mean([max(np.abs(exp['combined_explanation']['selected_importances'])) 
                                        if exp['combined_explanation'] and exp['combined_explanation']['selected_importances'] else 0
                                        for exp in explanations])
            
            performance_results[str(unit_types)] = {
                'avg_n_units': avg_n_units,
                'avg_max_importance': avg_max_importance,
                'success_rate': len(explanations) / 10
            }
            
            print(f"  平均选择基元数: {avg_n_units:.2f}")
            print(f"  平均最大重要性: {avg_max_importance:.4f}")
            print(f"  成功率: {len(explanations)/10:.2f}")
    
    # 7. 结果总结
    print("\n=== CMS-LIME框架测试完成 ===")
    print(f"模型准确率: {final_accuracy:.4f}")
    print("\n不同基元组合性能:")
    for unit_combo, metrics in performance_results.items():
        print(f"  {unit_combo}: 平均基元数={metrics['avg_n_units']:.2f}, "
              f"平均重要性={metrics['avg_max_importance']:.4f}, "
              f"成功率={metrics['success_rate']:.2f}")
    
    return explainer, model, performance_results

def demonstrate_cms_lime_features():
    """演示CMS-LIME的主要特性"""
    print("\n=== CMS-LIME特性演示 ===")
    
    # 创建简单示例数据
    np.random.seed(42)
    n_channels, n_timepoints = 16, 500
    
    # 生成具有明显模式的信号
    signal = np.random.randn(n_channels, n_timepoints) * 0.1
    
    # 添加特定模式
    # 1. 低频振荡 (前8个通道)
    t = np.linspace(0, 4*np.pi, n_timepoints)
    for ch in range(8):
        signal[ch] += 0.5 * np.sin(2 * t + ch * 0.1)
    
    # 2. 高频爆发 (后8个通道，中间时段)
    burst_start, burst_end = 200, 300
    for ch in range(8, 16):
        signal[ch, burst_start:burst_end] += 0.8 * np.sin(20 * t[burst_start:burst_end])
    
    # 创建简单的二分类数据
    X_demo = np.array([signal, signal + np.random.randn(n_channels, n_timepoints) * 0.2])
    y_demo = np.array([0, 1])
    
    # 创建简单模型
    class SimpleModel:
        def predict_proba(self, x):
            # 基于信号能量的简单分类
            if x.ndim == 2:
                x = x.reshape(1, *x.shape)
            
            probs = []
            for sample in x:
                energy = np.mean(sample**2)
                if energy > 0.1:
                    probs.append([0.2, 0.8])  # 高能量 -> 类别1
                else:
                    probs.append([0.8, 0.2])  # 低能量 -> 类别0
            return np.array(probs)
    
    simple_model = SimpleModel()
    
    # 创建解释器
    config = CMSLimeConfig(
        n_microstates=3,
        n_shapelets=100,
        n_perturbations=100,
        n_features=10,
        verbose=True
    )
    
    explainer = CMSLimeExplainer(config)
    
    print("\n拟合演示解释器...")
    explainer.fit(X_demo, y_demo, simple_model)
    
    print("\n生成演示解释...")
    explanation = explainer.explain_instance(
        signal,
        unit_types=['microstate', 'shapelet', 'timefreq']
    )
    
    print("\n演示解释结果:")
    print(explainer.get_explanation_summary(explanation))
    
    # 可视化
    explainer.visualize_explanation(
        explanation, 
        signal, 
        save_path='cms_lime_demo_explanation.png'
    )
    
    return explanation

if __name__ == "__main__":
    print("CMS-LIME框架完整示例")
    print("=" * 50)
    
    try:
        # 运行完整测试
        explainer, model, results = test_cms_lime_framework()
        
        # 运行特性演示
        demo_explanation = demonstrate_cms_lime_features()
        
        print("\n所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()