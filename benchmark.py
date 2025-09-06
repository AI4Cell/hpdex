"""
Benchmark and testing module for hpdex rank sum kernels.

This module provides pytest tests to verify that hpdex rank sum kernels
produce results consistent with scipy.stats.mannwhitneyu.
"""

import pytest
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from typing import Tuple, Optional
import time
import warnings

from hpdex.backen import (
    rank_sum_chunk_kernel_float,
    rank_sum_chunk_kernel_hist,
    parallel_difference_expression
)


class ScipyWrapper:
    """Wrapper for scipy.stats.mannwhitneyu to match hpdex interface."""
    
    @staticmethod
    def rank_sum_float(
        ref_sorted: np.ndarray,
        tar_sorted: np.ndarray,
        tie_correction: bool = True,
        continuity_correction: bool = True,
        use_asymptotic: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scipy wrapper for Mann-Whitney U test using floating-point algorithm.
        
        Args:
            ref_sorted: Reference group data with shape [..., n_ref], sorted ascending
            tar_sorted: Target group data with shape [..., n_tar], sorted ascending
            tie_correction: Whether to apply tie correction (ignored in scipy)
            continuity_correction: Whether to apply continuity correction
            use_asymptotic: Force asymptotic approximation. None for auto-selection
            
        Returns:
            Tuple of (p_values, U_statistics) with same leading dimensions as input
        """
        # Ensure 2D for batch processing
        original_shape = ref_sorted.shape
        if ref_sorted.ndim == 1:
            ref_sorted = ref_sorted.reshape(1, -1)
            tar_sorted = tar_sorted.reshape(1, -1)
        
        batch_size = ref_sorted.shape[0]
        p_values = np.zeros(batch_size)
        U_statistics = np.zeros(batch_size)
        
        for i in range(batch_size):
            ref_data = ref_sorted[i]
            tar_data = tar_sorted[i]
            
            # Remove NaN values
            ref_clean = ref_data[~np.isnan(ref_data)]
            tar_clean = tar_data[~np.isnan(tar_data)]
            
            if len(ref_clean) == 0 or len(tar_clean) == 0:
                p_values[i] = np.nan
                U_statistics[i] = np.nan
                continue
            
            try:
                # Determine method based on sample size and ties
                if use_asymptotic is None:
                    # Auto-select: use exact for small samples without ties
                    n_ref, n_tar = len(ref_clean), len(tar_clean)
                    has_ties = (len(np.unique(ref_clean)) < len(ref_clean) or 
                               len(np.unique(tar_clean)) < len(tar_clean))
                    use_exact = (n_ref <= 8 and n_tar <= 8 and not has_ties)
                else:
                    use_exact = not use_asymptotic
                
                # Perform Mann-Whitney U test
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    statistic, p_value = mannwhitneyu(
                        tar_clean,  # scipy uses x, y order
                        ref_clean,  # reference group
                        alternative='two-sided',
                        use_continuity=continuity_correction,
                        method='exact' if use_exact else 'asymptotic'
                    )
                
                p_values[i] = p_value
                U_statistics[i] = statistic
                
            except Exception as e:
                p_values[i] = np.nan
                U_statistics[i] = np.nan
        
        # Reshape to original dimensions
        if len(original_shape) == 1:
            return p_values[0], U_statistics[0]
        else:
            return p_values.reshape(original_shape[:-1]), U_statistics.reshape(original_shape[:-1])
    
    @staticmethod
    def rank_sum_hist(
        ref_sorted: np.ndarray,
        tar: np.ndarray,
        tie_correction: bool = True,
        continuity_correction: bool = True,
        use_asymptotic: Optional[bool] = None,
        max_bins: int = 200_000,
        float_dtype: np.dtype = np.float64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scipy wrapper for Mann-Whitney U test using histogram algorithm.
        Falls back to float algorithm for consistency.
        """
        # For scipy, we just use the float algorithm since scipy doesn't have histogram optimization
        return ScipyWrapper.rank_sum_float(
            ref_sorted, tar, tie_correction, continuity_correction, use_asymptotic
        )


def generate_test_data(
    n_samples: int = 100,
    n_genes: int = 10,
    n_groups: int = 3,
    data_type: str = 'float',
    add_ties: bool = False,
    add_outliers: bool = False,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate test data for benchmarking.
    
    Args:
        n_samples: Number of samples per group
        n_genes: Number of genes
        n_groups: Number of groups
        data_type: 'float', 'int', or 'mixed'
        add_ties: Whether to add tied values
        add_outliers: Whether to add outliers
        seed: Random seed
        
    Returns:
        Tuple of (ref_data, tar_data, group_labels)
    """
    np.random.seed(seed)
    
    if data_type == 'float':
        base_data = np.random.normal(0, 1, (n_groups, n_genes, n_samples))
    elif data_type == 'int':
        base_data = np.random.poisson(5, (n_groups, n_genes, n_samples))
    else:  # mixed
        base_data = np.random.normal(0, 1, (n_groups, n_genes, n_samples))
        base_data = np.round(base_data).astype(int)
    
    if add_ties:
        # Add some tied values
        for i in range(n_groups):
            for j in range(n_genes):
                if np.random.random() < 0.3:  # 30% chance of ties
                    tie_value = base_data[i, j, 0]
                    tie_indices = np.random.choice(n_samples, size=min(5, n_samples), replace=False)
                    base_data[i, j, tie_indices] = tie_value
    
    if add_outliers:
        # Add outliers
        for i in range(n_groups):
            for j in range(n_genes):
                if np.random.random() < 0.1:  # 10% chance of outliers
                    outlier_indices = np.random.choice(n_samples, size=min(2, n_samples), replace=False)
                    base_data[i, j, outlier_indices] *= 10
    
    # Create reference and target groups
    ref_data = base_data[0]  # First group as reference
    tar_data = base_data[1]  # Second group as target
    
    # Create group labels
    group_labels = np.array(['ref'] * n_samples + ['tar'] * n_samples)
    
    return ref_data, tar_data, group_labels


def test_rank_sum_kernel_float_consistency():
    """Test that hpdex float kernel matches scipy results."""
    print("\n=== Testing rank_sum_chunk_kernel_float consistency ===")
    
    # Test cases with different data types and sizes
    test_cases = [
        {'n_samples': 20, 'n_genes': 5, 'data_type': 'float', 'add_ties': False},
        {'n_samples': 50, 'n_genes': 10, 'data_type': 'float', 'add_ties': True},
        {'n_samples': 100, 'n_genes': 20, 'data_type': 'int', 'add_ties': False},
        {'n_samples': 30, 'n_genes': 8, 'data_type': 'mixed', 'add_ties': True},
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\nTest case {i+1}: {case}")
        
        ref_data, tar_data, _ = generate_test_data(**case, seed=42+i)
        
        # Sort data as required by hpdex
        ref_sorted = np.sort(ref_data, axis=1)
        tar_sorted = np.sort(tar_data, axis=1)
        
        # Test different parameter combinations
        param_combinations = [
            {'tie_correction': True, 'continuity_correction': True, 'use_asymptotic': None},
            {'tie_correction': False, 'continuity_correction': False, 'use_asymptotic': True},
            {'tie_correction': True, 'continuity_correction': False, 'use_asymptotic': False},
        ]
        
        for params in param_combinations:
            print(f"  Testing params: {params}")
            
            # hpdex results
            hpdex_pvals, hpdex_stats = rank_sum_chunk_kernel_float(
                ref_sorted, tar_sorted, **params
            )
            
            # scipy results
            scipy_pvals, scipy_stats = ScipyWrapper.rank_sum_float(
                ref_sorted, tar_sorted, **params
            )
            
            # Compare results
            pval_diff = np.abs(hpdex_pvals - scipy_pvals)
            stat_diff = np.abs(hpdex_stats - scipy_stats)
            
            # Check for reasonable agreement (allowing for numerical precision)
            pval_agreement = np.all(pval_diff < 1e-10) or np.allclose(hpdex_pvals, scipy_pvals, rtol=1e-6, atol=1e-10)
            stat_agreement = np.allclose(hpdex_stats, scipy_stats, rtol=1e-10, atol=1e-10)
            
            print(f"    P-value agreement: {pval_agreement} (max diff: {np.max(pval_diff):.2e})")
            print(f"    Statistic agreement: {stat_agreement} (max diff: {np.max(stat_diff):.2e})")
            
            assert pval_agreement, f"P-values don't match: max diff = {np.max(pval_diff):.2e}"
            assert stat_agreement, f"Statistics don't match: max diff = {np.max(stat_diff):.2e}"


def test_rank_sum_kernel_hist_consistency():
    """Test that hpdex hist kernel matches scipy results for integer data."""
    print("\n=== Testing rank_sum_chunk_kernel_hist consistency ===")
    
    # Test cases with integer data
    test_cases = [
        {'n_samples': 30, 'n_genes': 8, 'data_type': 'int', 'add_ties': False},
        {'n_samples': 50, 'n_genes': 15, 'data_type': 'int', 'add_ties': True},
        {'n_samples': 100, 'n_genes': 25, 'data_type': 'int', 'add_ties': False},
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\nTest case {i+1}: {case}")
        
        ref_data, tar_data, _ = generate_test_data(**case, seed=42+i)
        
        # Ensure integer data
        ref_data = ref_data.astype(np.int32)
        tar_data = tar_data.astype(np.int32)
        
        # Sort reference data as required by hpdex
        ref_sorted = np.sort(ref_data, axis=1)
        
        # Test different parameter combinations
        param_combinations = [
            {'tie_correction': True, 'continuity_correction': True, 'use_asymptotic': None},
            {'tie_correction': False, 'continuity_correction': False, 'use_asymptotic': True},
        ]
        
        for params in param_combinations:
            print(f"  Testing params: {params}")
            
            # hpdex results
            hpdex_pvals, hpdex_stats = rank_sum_chunk_kernel_hist(
                ref_sorted, tar_data, **params
            )
            
            # scipy results
            scipy_pvals, scipy_stats = ScipyWrapper.rank_sum_hist(
                ref_sorted, tar_data, **params
            )
            
            # Compare results
            pval_diff = np.abs(hpdex_pvals - scipy_pvals)
            stat_diff = np.abs(hpdex_stats - scipy_stats)
            
            # Check for reasonable agreement
            pval_agreement = np.all(pval_diff < 1e-10) or np.allclose(hpdex_pvals, scipy_pvals, rtol=1e-6, atol=1e-10)
            stat_agreement = np.allclose(hpdex_stats, scipy_stats, rtol=1e-10, atol=1e-10)
            
            print(f"    P-value agreement: {pval_agreement} (max diff: {np.max(pval_diff):.2e})")
            print(f"    Statistic agreement: {stat_agreement} (max diff: {np.max(stat_diff):.2e})")
            
            assert pval_agreement, f"P-values don't match: max diff = {np.max(pval_diff):.2e}"
            assert stat_agreement, f"Statistics don't match: max diff = {np.max(stat_diff):.2e}"


def test_performance_benchmark():
    """Benchmark performance comparison between hpdex and scipy."""
    print("\n=== Performance Benchmark ===")
    
    # Large dataset for performance testing
    n_samples = 1000
    n_genes = 100
    n_groups = 5
    
    print(f"Testing with {n_samples} samples, {n_genes} genes, {n_groups} groups")
    
    # Generate test data
    ref_data, tar_data, group_labels = generate_test_data(
        n_samples=n_samples, n_genes=n_genes, n_groups=n_groups, 
        data_type='float', seed=42
    )
    
    # Sort data
    ref_sorted = np.sort(ref_data, axis=1)
    tar_sorted = np.sort(tar_data, axis=1)
    
    # Benchmark hpdex
    print("\nBenchmarking hpdex...")
    start_time = time.time()
    hpdex_pvals, hpdex_stats = rank_sum_chunk_kernel_float(
        ref_sorted, tar_sorted, 
        tie_correction=True, 
        continuity_correction=True
    )
    hpdex_time = time.time() - start_time
    
    # Benchmark scipy
    print("Benchmarking scipy...")
    start_time = time.time()
    scipy_pvals, scipy_stats = ScipyWrapper.rank_sum_float(
        ref_sorted, tar_sorted,
        tie_correction=True,
        continuity_correction=True
    )
    scipy_time = time.time() - start_time
    
    # Calculate speedup
    speedup = scipy_time / hpdex_time
    
    print(f"\nResults:")
    print(f"  hpdex time: {hpdex_time:.3f} seconds")
    print(f"  scipy time: {scipy_time:.3f} seconds")
    print(f"  speedup: {speedup:.2f}x")
    
    # Verify results are consistent
    pval_diff = np.abs(hpdex_pvals - scipy_pvals)
    stat_diff = np.abs(hpdex_stats - scipy_stats)
    
    print(f"  Max p-value difference: {np.max(pval_diff):.2e}")
    print(f"  Max statistic difference: {np.max(stat_diff):.2e}")
    
    # Assert reasonable performance improvement
    assert speedup > 1.0, f"hpdex should be faster than scipy, got {speedup:.2f}x speedup"
    
    # Assert results are consistent
    assert np.allclose(hpdex_pvals, scipy_pvals, rtol=1e-6, atol=1e-10), "Results should be consistent"


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")
    
    # Test with very small datasets
    print("Testing small datasets...")
    ref_small = np.array([[1, 2, 3]])
    tar_small = np.array([[4, 5, 6]])
    
    hpdex_pvals, hpdex_stats = rank_sum_chunk_kernel_float(ref_small, tar_small)
    scipy_pvals, scipy_stats = ScipyWrapper.rank_sum_float(ref_small, tar_small)
    
    assert np.allclose(hpdex_pvals, scipy_pvals, rtol=1e-6), "Small dataset results should match"
    assert np.allclose(hpdex_stats, scipy_stats, rtol=1e-6), "Small dataset statistics should match"
    
    # Test with identical values (ties)
    print("Testing identical values...")
    ref_identical = np.array([[1, 1, 1, 1]])
    tar_identical = np.array([[2, 2, 2, 2]])
    
    hpdex_pvals, hpdex_stats = rank_sum_chunk_kernel_float(ref_identical, tar_identical)
    scipy_pvals, scipy_stats = ScipyWrapper.rank_sum_float(ref_identical, tar_identical)
    
    assert np.allclose(hpdex_pvals, scipy_pvals, rtol=1e-6), "Identical values results should match"
    assert np.allclose(hpdex_stats, scipy_stats, rtol=1e-6), "Identical values statistics should match"
    
    # Test with NaN values
    print("Testing NaN handling...")
    ref_nan = np.array([[1, 2, np.nan, 4]])
    tar_nan = np.array([[5, 6, 7, np.nan]])
    
    hpdex_pvals, hpdex_stats = rank_sum_chunk_kernel_float(ref_nan, tar_nan)
    scipy_pvals, scipy_stats = ScipyWrapper.rank_sum_float(ref_nan, tar_nan)
    
    # Both should handle NaN gracefully
    assert np.isnan(hpdex_pvals[0]) == np.isnan(scipy_pvals[0]), "NaN handling should be consistent"


def test_parallel_difference_expression_consistency():
    """Test that parallel_difference_expression produces consistent results."""
    print("\n=== Testing parallel_difference_expression consistency ===")
    
    # Create a simple AnnData object for testing
    n_cells = 200
    n_genes = 50
    
    # Generate expression data
    np.random.seed(42)
    X = np.random.poisson(5, (n_cells, n_genes))
    
    # Create group labels
    group_labels = np.array(['control'] * 100 + ['treatment'] * 100)
    
    # Create AnnData object
    adata = pd.DataFrame(X, columns=[f'gene_{i}' for i in range(n_genes)])
    adata['group'] = group_labels
    
    # Convert to AnnData format (simplified)
    import anndata as ad
    adata = ad.AnnData(X=X, obs=pd.DataFrame({'group': group_labels}))
    adata.var_names = [f'gene_{i}' for i in range(n_genes)]
    
    # Test parallel_difference_expression
    print("Testing parallel_difference_expression...")
    
    try:
        results = parallel_difference_expression(
            adata,
            groupby_key="group",
            reference="control",
            num_workers=2
        )
        
        print(f"  Results shape: {results.shape}")
        print(f"  Columns: {list(results.columns)}")
        print(f"  Number of significant genes (FDR < 0.05): {len(results[results['fdr'] < 0.05])}")
        
        # Basic sanity checks
        assert len(results) > 0, "Should return some results"
        assert 'pval' in results.columns, "Should have pval column"
        assert 'fdr' in results.columns, "Should have fdr column"
        assert 'fold_change' in results.columns, "Should have fold_change column"
        
        print("  ✓ parallel_difference_expression test passed")
        
    except Exception as e:
        print(f"  ✗ parallel_difference_expression test failed: {e}")
        raise


if __name__ == "__main__":
    # Run all tests
    print("Running hpdex benchmark tests...")
    
    test_rank_sum_kernel_float_consistency()
    test_rank_sum_kernel_hist_consistency()
    test_performance_benchmark()
    test_edge_cases()
    test_parallel_difference_expression_consistency()
    
    print("\n✅ All tests passed!")
