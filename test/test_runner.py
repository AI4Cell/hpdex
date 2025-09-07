#!/usr/bin/env python3
"""
Simple test runner for hpdex framework.

This script provides an easy way to run tests without requiring pytest installation.
"""

import sys
import time
import warnings
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_basic_tests():
    """Run basic functionality tests."""
    import numpy as np
    import pandas as pd
    import anndata as ad
    import tempfile
    
    # Suppress AnnData warnings
    warnings.filterwarnings("ignore", ".*Transforming to str index.*")
    
    print("🧪 Running hpdex basic tests...")
    print("=" * 50)
    
    # Import hpdex
    try:
        from hpdex import parallel_differential_expression, parallel_differential_expression_stream
        print("✓ Successfully imported hpdex")
    except ImportError as e:
        print(f"✗ Failed to import hpdex: {e}")
        return False
    
    # Generate test data
    print("\n📊 Generating test data...")
    np.random.seed(42)
    
    n_cells, n_genes = 300, 100
    X = np.random.normal(0, 1, (n_cells, n_genes))
    
    # Add differential expression pattern
    X[100:200, :20] += 1.5  # Group 1 upregulation
    X[200:300, :20] -= 1.0  # Group 2 downregulation
    X[200:300, 20:40] += 2.0  # Group 2 different pattern
    
    obs = pd.DataFrame({
        'group': ['control'] * 100 + ['treatment_A'] * 100 + ['treatment_B'] * 100,
        'batch': ['batch1'] * 150 + ['batch2'] * 150
    })
    
    var = pd.DataFrame({
        'gene_name': [f'GENE_{i:03d}' for i in range(n_genes)]
    })
    
    adata = ad.AnnData(X=X, obs=obs, var=var)
    print(f"✓ Generated data: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"  Groups: {dict(adata.obs['group'].value_counts())}")
    
    # Test 1: Basic API functionality
    print("\n🔧 Test 1: Basic API functionality")
    try:
        start_time = time.time()
        results = parallel_differential_expression(
            adata=adata,
            groupby_key="group",
            reference="control",
            num_workers=1
        )
        elapsed = time.time() - start_time
        
        print(f"✓ Basic API completed in {elapsed:.2f}s")
        print(f"  Results shape: {results.shape}")
        print(f"  Columns: {list(results.columns)}")
        print(f"  Significant genes (FDR < 0.05): {sum(results['fdr'] < 0.05)}")
        
        # Validate results
        assert len(results) > 0, "No results returned"
        assert all(col in results.columns for col in [
            'target', 'feature', 'p_value', 'fold_change', 'log2_fold_change', 'fdr'
        ]), "Missing required columns"
        assert not any(pd.isna(results['p_value'])), "NaN p-values found"
        assert all(0 <= p <= 1 for p in results['p_value']), "Invalid p-values"
        
    except Exception as e:
        print(f"✗ Basic API test failed: {e}")
        return False
    
    # Test 2: Multiprocessing
    print("\n🚀 Test 2: Multiprocessing")
    try:
        start_time = time.time()
        results_mp = parallel_differential_expression(
            adata=adata,
            groupby_key="group",
            reference="control",
            num_workers=4
        )
        elapsed = time.time() - start_time
        
        print(f"✓ Multiprocessing completed in {elapsed:.2f}s")
        
        # Compare with single-threaded results
        results_sorted = results.sort_values(['target', 'feature']).reset_index(drop=True)
        results_mp_sorted = results_mp.sort_values(['target', 'feature']).reset_index(drop=True)
        
        p_diff = np.abs(results_sorted['p_value'] - results_mp_sorted['p_value']).max()
        print(f"  Max p-value difference: {p_diff:.2e}")
        assert p_diff < 1e-10, f"Large p-value difference: {p_diff}"
        
    except Exception as e:
        print(f"✗ Multiprocessing test failed: {e}")
        return False
    
    # Test 3: Different metrics
    print("\n📈 Test 3: Different metrics")
    try:
        # Test wilcoxon-hist with integer data
        X_int = np.random.poisson(5, (200, 50))
        X_int[100:, :10] += 3  # Add differential expression
        
        obs_int = pd.DataFrame({
            'group': ['control'] * 100 + ['treatment'] * 100
        })
        
        adata_int = ad.AnnData(X=X_int, obs=obs_int)
        
        results_hist = parallel_differential_expression(
            adata=adata_int,
            groupby_key="group",
            reference="control",
            metric="wilcoxon-hist",
            num_workers=2
        )
        
        print(f"✓ Histogram algorithm: {results_hist.shape[0]} results")
        print(f"  Significant genes (FDR < 0.05): {sum(results_hist['fdr'] < 0.05)}")
        
    except Exception as e:
        print(f"✗ Different metrics test failed: {e}")
        return False
    
    # Test 4: Streaming API
    print("\n🌊 Test 4: Streaming API")
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / "test_data.h5ad"
            adata.write_h5ad(test_file)
            
            start_time = time.time()
            results_stream = parallel_differential_expression_stream(
                data_path=test_file,
                groupby_key="group",
                reference="control",
                memory_limit_gb=2.0,
                num_workers=2
            )
            elapsed = time.time() - start_time
            
            print(f"✓ Streaming API completed in {elapsed:.2f}s")
            print(f"  Results shape: {results_stream.shape}")
            print(f"  Significant genes (FDR < 0.05): {sum(results_stream['fdr'] < 0.05)}")
            
            # Compare with regular API
            results_sorted = results.sort_values(['target', 'feature']).reset_index(drop=True)
            results_stream_sorted = results_stream.sort_values(['target', 'feature']).reset_index(drop=True)
            
            p_diff = np.abs(results_sorted['p_value'] - results_stream_sorted['p_value']).max()
            print(f"  Max p-value difference vs regular API: {p_diff:.2e}")
            
    except Exception as e:
        print(f"✗ Streaming API test failed: {e}")
        return False
    
    # Test 5: Edge cases
    print("\n⚠️  Test 5: Edge cases")
    try:
        # Test with very small groups
        small_adata = adata[:20].copy()  # 20 cells total
        small_adata.obs['group'] = ['control'] * 10 + ['treatment'] * 10
        
        results_small = parallel_differential_expression(
            adata=small_adata,
            groupby_key="group",
            reference="control",
            min_samples=5
        )
        
        print(f"✓ Small groups: {results_small.shape[0]} results")
        
        # Test with sparse data
        from scipy.sparse import csr_matrix
        adata_sparse = adata.copy()
        adata_sparse.X = csr_matrix(adata_sparse.X)
        
        results_sparse = parallel_differential_expression(
            adata=adata_sparse,
            groupby_key="group",
            reference="control",
            num_workers=2
        )
        
        print(f"✓ Sparse data: {results_sparse.shape[0]} results")
        
    except Exception as e:
        print(f"✗ Edge cases test failed: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 All tests passed successfully!")
    print(f"📊 Final validation:")
    print(f"  - Total genes tested: {adata.n_vars}")
    print(f"  - Total comparisons: {len(results)}")
    print(f"  - Significant genes (FDR < 0.05): {sum(results['fdr'] < 0.05)}")
    print(f"  - Top 5 most significant genes:")
    
    top_genes = results.nsmallest(5, 'fdr')[['target', 'feature', 'p_value', 'fdr', 'log2_fold_change']]
    for _, row in top_genes.iterrows():
        print(f"    {row['feature']} ({row['target']}): FDR={row['fdr']:.2e}, log2FC={row['log2_fold_change']:.2f}")
    
    return True


def run_performance_test():
    """Run a simple performance test."""
    import numpy as np
    import pandas as pd
    import anndata as ad
    import time
    
    print("\n🏃 Performance Test")
    print("=" * 30)
    
    from hpdex import parallel_differential_expression
    
    # Larger dataset
    np.random.seed(42)
    n_cells, n_genes = 2000, 1000
    X = np.random.normal(0, 1, (n_cells, n_genes))
    
    # Add differential expression
    X[1000:, :100] += 1.0
    
    obs = pd.DataFrame({
        'group': ['control'] * 1000 + ['treatment'] * 1000
    })
    
    adata = ad.AnnData(X=X, obs=obs)
    print(f"Dataset: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Test different worker counts
    for num_workers in [1, 2, 4]:
        start_time = time.time()
        results = parallel_differential_expression(
            adata=adata,
            groupby_key="group",
            reference="control",
            num_workers=num_workers
        )
        elapsed = time.time() - start_time
        
        print(f"  {num_workers} workers: {elapsed:.2f}s ({len(results)} results)")
    
    print("✓ Performance test completed")


def main():
    """Main test runner."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--performance":
            run_performance_test()
            return
        elif sys.argv[1] == "--help":
            print("Usage: python test_runner.py [--performance] [--help]")
            print("  --performance: Run performance benchmarks")
            print("  --help: Show this help message")
            return
    
    # Run basic tests
    success = run_basic_tests()
    
    if success:
        print("\n✨ Consider running the full test suite with:")
        print("   pytest test_hpdex_comprehensive.py -v")
        print("   pytest test_hpdex_comprehensive.py --test-all")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
