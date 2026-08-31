#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试导入 - 验证所有模块可以正确导入

Author: CMS-LIME Framework
Date: 2025
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """测试所有模块导入"""
    print("测试模块导入...")
    
    try:
        # 测试先验信息配置
        print("1. 测试先验信息配置...")
        from two_stage_selection import PriorKnowledgeConfig, create_default_prior_config
        config = create_default_prior_config()
        print("   [OK] 先验信息配置导入成功")
        
        # 测试第一场选拔
        print("2. 测试第一场选拔模块...")
        from two_stage_selection import PrimitiveExtractor, PrimitiveUnit
        print("   [OK] 第一场选拔模块导入成功")
        
        # 测试第二场选拔
        print("3. 测试第二场选拔模块...")
        from two_stage_selection import CausalPerturbationExplainer, BiomarkerEvaluator
        print("   [OK] 第二场选拔模块导入成功")
        
        # 测试主框架
        print("4. 测试主框架...")
        from two_stage_selection import TwoStagePrimitiveSelection, create_results_folder
        print("   [OK] 主框架导入成功")
        
        # 测试可视化工具
        print("5. 测试可视化工具...")
        from two_stage_selection.utils import (
            visualize_stage1_results,
            visualize_stage2_results
        )
        print("   [OK] 可视化工具导入成功")
        
        # 测试依赖模块（从父目录）
        print("6. 测试依赖模块...")
        try:
            from shapelet_analysis import ShapeletAnalyzer
            print("   [OK] shapelet_analysis 导入成功")
        except ImportError as e:
            print(f"   [WARN] shapelet_analysis 导入失败: {e}")
        
        try:
            from microstate_analysis import MicrostateAnalyzer
            print("   [OK] microstate_analysis 导入成功")
        except ImportError as e:
            print(f"   [WARN] microstate_analysis 导入失败: {e}")
        
        try:
            from timefreq_analysis import TimeFreqAnalyzer
            print("   [OK] timefreq_analysis 导入成功")
        except ImportError as e:
            print(f"   [WARN] timefreq_analysis 导入失败: {e}")
        
        print("\n" + "=" * 80)
        print("所有核心模块导入测试完成！")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
