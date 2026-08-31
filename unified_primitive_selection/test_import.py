#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试导入"""

import sys
import os

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from unified_primitive_selection import UnifiedPrimitiveSelection
    print("[OK] UnifiedPrimitiveSelection 导入成功")
except Exception as e:
    print(f"[ERROR] UnifiedPrimitiveSelection 导入失败: {e}")

try:
    from unified_primitive_selection import create_default_config
    print("[OK] create_default_config 导入成功")
except Exception as e:
    print(f"[ERROR] create_default_config 导入失败: {e}")

try:
    from unified_primitive_selection import BiomarkerUnit
    print("[OK] BiomarkerUnit 导入成功")
except Exception as e:
    print(f"[ERROR] BiomarkerUnit 导入失败: {e}")

print("\n所有导入测试完成！")
