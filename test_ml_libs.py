#!/usr/bin/env python3
"""
测试机器学习势能库的安装情况
"""

print("=== 测试机器学习势能库的安装情况 ===")

# 测试NequIP
try:
    import nequip
    print("✅ NequIP导入成功")
    print(f"   版本: {nequip.__version__}")
except Exception as e:
    print(f"❌ NequIP导入失败: {e}")

# 测试MACE
try:
    import mace
    print("✅ MACE导入成功")
    print(f"   版本: {mace.__version__}")
except Exception as e:
    print(f"❌ MACE导入失败: {e}")

# 测试相关依赖
try:
    import torch
    print(f"✅ PyTorch版本: {torch.__version__}")
except Exception as e:
    print(f"❌ PyTorch导入失败: {e}")

try:
    import ase
    print(f"✅ ASE版本: {ase.__version__}")
except Exception as e:
    print(f"❌ ASE导入失败: {e}")

try:
    import pymatgen
    print("✅ PyMatGen导入成功")
    print("   版本: 已安装")
except Exception as e:
    print(f"❌ PyMatGen导入失败: {e}")

try:
    import e3nn
    print(f"✅ E3NN版本: {e3nn.__version__}")
except Exception as e:
    print(f"❌ E3NN导入失败: {e}")

print("\n=== 测试完成 ===") 