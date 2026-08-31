#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMS-LIME框架演示
快速验证CMS-LIME框架的完整功能
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from cms_lime_explainer import CMSLimeExplainer, CMSLimeConfig

# 简化的EEG CNN模型
class SimpleEEGCNN(nn.Module):
    def __init__(self, n_channels=8, n_classes=3, n_timepoints=100):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(10)
        self.fc = nn.Linear(32 * 10, n_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def generate_simple_eeg_data(n_samples=60, n_channels=8, n_timepoints=100, n_classes=3):
    """生成简化的合成EEG数据"""
    np.random.seed(42)
    
    X = []
    y = []
    
    for class_idx in range(n_classes):
        for _ in range(n_samples // n_classes):
            # 为每个类别生成不同的频率特征
            base_freq = 8 + class_idx * 5  # 8Hz, 13Hz, 18Hz
            t = np.linspace(0, 4, n_timepoints)
            
            sample = np.zeros((n_channels, n_timepoints))
            for ch in range(n_channels):
                # 主频率成分
                signal = np.sin(2 * np.pi * base_freq * t)
                # 添加噪声
                noise = np.random.normal(0, 0.3, n_timepoints)
                # 添加通道间的相关性
                if ch > 0:
                    signal += 0.3 * sample[ch-1]
                
                sample[ch] = signal + noise
            
            X.append(sample)
            y.append(class_idx)
    
    return np.array(X), np.array(y)

def train_simple_model(X_train, y_train, X_test, y_test, epochs=5):
    """训练简化模型"""
    model = SimpleEEGCNN(n_channels=X_train.shape[1], n_classes=len(np.unique(y_train)), 
                        n_timepoints=X_train.shape[2])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 转换为tensor
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                _, predicted = torch.max(test_outputs.data, 1)
                accuracy = (predicted == y_test_tensor).float().mean()
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Test Acc: {accuracy:.4f}")
            model.train()
    
    return model

def demo_cms_lime():
    """演示CMS-LIME框架"""
    print("CMS-LIME框架演示")
    print("=" * 50)
    
    # 1. 生成数据
    print("\n1. 生成演示数据...")
    X, y = generate_simple_eeg_data(n_samples=60, n_channels=8, n_timepoints=100)
    
    # 分割数据
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 2. 训练模型
    print("\n2. 训练演示模型...")
    model = train_simple_model(X_train, y_train, X_test, y_test, epochs=5)
    
    # 3. 创建CMS-LIME解释器
    print("\n3. 创建CMS-LIME解释器...")
    config = CMSLimeConfig(
        n_microstates=4,
        n_shapelets=10,
        freq_bands={'alpha': (8, 13), 'beta': (13, 30)},
        causal_method='granger',
        n_perturbations=50
    )
    
    explainer = CMSLimeExplainer(config)
    explainer.model = model
    
    # 4. 拟合解释器
    print("\n4. 拟合CMS-LIME解释器...")
    explainer.fit(X_train, y_train, model)
    
    # 5. 生成解释
    print("\n5. 生成解释结果...")
    test_sample = X_test[0]
    target_class = y_test[0]
    
    print(f"\n--- 解释测试样本 (真实类别: {target_class}) ---")
    
    try:
        explanation = explainer.explain_instance(
            test_sample, 
            target_class=target_class
        )
        
        print("\n✅ 解释生成成功！")
        print(f"解释包含 {len(explanation['selected_units'])} 个基元单位")
        
        # 显示前几个重要的基元
        print("\n🔍 重要基元单位:")
        for i, (unit, importance) in enumerate(zip(explanation['selected_units'][:3], explanation['selected_importances'][:3])):
            print(f"  {i+1}. 类型: {unit.get('type', 'unknown')}, 重要性: {importance:.4f}")
        
        return explainer, model, explanation
        
    except Exception as e:
        print(f"❌ 解释生成失败: {e}")
        import traceback
        traceback.print_exc()
        return explainer, model, None

if __name__ == "__main__":
    try:
        explainer, model, explanation = demo_cms_lime()
        
        if explanation is not None:
            print("\n🎉 CMS-LIME框架演示成功完成！")
            print("\n📊 演示总结:")
            print("- ✅ 数据生成和模型训练")
            print("- ✅ CMS-LIME解释器创建和拟合")
            print("- ✅ 基元单位构造和解释生成")
            print("- ✅ 因果一致性扰动和局部回归")
            print("\n🚀 框架已准备就绪，可用于实际EEG信号解释！")
        else:
            print("\n⚠️  演示部分完成，解释生成遇到问题")
            
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()