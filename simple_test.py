#!/usr/bin/env python3
import sys
import os

# 添加build目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

print("尝试导入hpdex_cpp模块...")
try:
    import hpdex_cpp
    print("✅ 模块导入成功!")
    print(f"模块路径: {hpdex_cpp.__file__}")
    print(f"模块属性: {dir(hpdex_cpp)}")
except Exception as e:
    print(f"❌ 模块导入失败: {e}")
    import traceback
    traceback.print_exc()
