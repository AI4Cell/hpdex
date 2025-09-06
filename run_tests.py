#!/usr/bin/env python3
"""
Simple test runner for hpdex benchmark tests.
Run this script to execute all tests without pytest.
"""

import sys
import os

# Add the src directory to the path so we can import hpdex
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from benchmark import (
        test_rank_sum_kernel_float_consistency,
        test_rank_sum_kernel_hist_consistency,
        test_performance_benchmark,
        test_edge_cases,
        test_parallel_difference_expression_consistency
    )
    
    print("Running hpdex benchmark tests...")
    print("=" * 50)
    
    # Run all tests
    test_rank_sum_kernel_float_consistency()
    test_rank_sum_kernel_hist_consistency()
    test_performance_benchmark()
    test_edge_cases()
    test_parallel_difference_expression_consistency()
    
    print("\n" + "=" * 50)
    print("✅ All tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the required dependencies:")
    print("pip install -r requirements-test.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
