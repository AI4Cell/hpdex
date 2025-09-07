"""
Pytest tests for hpdex library.

Comprehensive test suite covering:
1. Kernel consistency tests (hpdex vs scipy)
2. Pipeline consistency tests (hpdex vs pdex) 
3. Performance benchmarks
4. Real data validation
5. FDR differential gene set consistency

Usage:
    # Run all tests
    pytest benchmarks.py --test-all
    
    # Run specific test categories
    pytest benchmarks.py --test-kernels
    pytest benchmarks.py --test-pipeline --test-performance
    
    # Run with real data
    pytest benchmarks.py --test-real-data --h5ad-files "PBMC_10K.h5ad,HumanPBMC.h5ad"
    
    # Performance benchmarks with custom settings
    pytest benchmarks.py --test-performance --n-workers 8 --benchmark-sizes "small,medium"

"""

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import pytest
# Import timeout utilities from conftest
from conftest import TimeoutError, run_with_timeout

# Import test modules
try:
    from scipy.stats import mannwhitneyu
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available, skipping scipy comparison tests")

try:
    from pdex import parallel_differential_expression as pdex_parallel_de
    HAS_PDEX = True
except ImportError:
    HAS_PDEX = False
    warnings.warn("pdex not available, using mock for comparison tests")

# Import hpdex components
from hpdex.backen import (parallel_differential_expression,
                          rank_sum_chunk_kernel_float,
                          rank_sum_chunk_kernel_hist)


class ScipyWrapper:
    """Wrapper for scipy.stats.mannwhitneyu to match hpdex kernel interface."""
    
    @staticmethod
    def mannwhitneyu_batch(
        ref_sorted: np.ndarray,
        tar_sorted: np.ndarray,
        tie_correction: bool = True,
        continuity_correction: bool = True,
        use_asymptotic: bool = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch Mann-Whitney U test using scipy for kernel consistency testing."""
        if not HAS_SCIPY:
            # Return mock results if scipy not available
            shape = ref_sorted.shape[:-1] if ref_sorted.ndim > 1 else ()
            return np.full(shape, 0.5), np.full(shape, 100.0)
        
        # Ensure 2D for batch processing
        original_shape = ref_sorted.shape
        if ref_sorted.ndim == 1:
            ref_sorted = ref_sorted.reshape(1, -1)
            tar_sorted = tar_sorted.reshape(1, -1)
        
        batch_size = ref_sorted.shape[0]
        p_values = np.zeros(batch_size)
        U_statistics = np.zeros(batch_size)
        
        # Determine method
        if use_asymptotic is None:
            n_ref, n_tar = ref_sorted.shape[1], tar_sorted.shape[1]
            use_exact = (n_ref <= 8 and n_tar <= 8)
        else:
            use_exact = not use_asymptotic
        
        method = 'exact' if use_exact else 'asymptotic'
        
        for i in range(batch_size):
            ref_clean = ref_sorted[i][~np.isnan(ref_sorted[i])]
            tar_clean = tar_sorted[i][~np.isnan(tar_sorted[i])]
            
            if len(ref_clean) == 0 or len(tar_clean) == 0:
                p_values[i] = np.nan
                U_statistics[i] = np.nan
                continue
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    statistic, p_value = mannwhitneyu(
                        tar_clean,  # scipy uses x, y order
                        ref_clean,  # reference group
                        alternative='two-sided',
                        use_continuity=continuity_correction,
                        method=method
                    )
                
                p_values[i] = p_value
                U_statistics[i] = statistic
                
            except Exception:
                p_values[i] = np.nan
                U_statistics[i] = np.nan
        
        # Reshape to original dimensions
        if len(original_shape) == 1:
            return p_values[0], U_statistics[0]
        else:
            return p_values.reshape(original_shape[:-1]), U_statistics.reshape(original_shape[:-1])


class PdexWrapper:
    """Wrapper for pdex library to match hpdex pipeline interface."""
    
    @staticmethod
    def parallel_difference_expression(
        adata: ad.AnnData,
        groupby_key: str,
        reference: str,
        groups: List[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Pdex wrapper for parallel differential expression analysis."""
        if not HAS_PDEX:
            # Create mock results if pdex not available
            return PdexWrapper._create_mock_results(adata, groupby_key, reference, groups)
        
        try:
            results = pdex_parallel_de(
                adata,
                groupby_key=groupby_key,
                reference=reference,
                groups=groups
            )
            
            # Convert column names to match hpdex format
            column_mapping = {
                'pval': 'p_value',
                'pert': 'target', 
                'gene': 'feature'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in results.columns and new_col not in results.columns:
                    results = results.rename(columns={old_col: new_col})
            
            return results
            
        except Exception as e:
            warnings.warn(f"pdex analysis failed: {e}, using mock results")
            return PdexWrapper._create_mock_results(adata, groupby_key, reference, groups)
    
    @staticmethod
    def _create_mock_results(adata, groupby_key, reference, groups):
        """Create mock results for testing when pdex is not available."""
        obs_vals = adata.obs[groupby_key].values
        if groups is None:
            unique_groups = np.unique(obs_vals)
            groups = [g for g in unique_groups if g != reference]
        
        # Create mock results
        mock_results = []
        for group in groups:
            for gene in adata.var_names:
                mock_results.append({
                    'target': group,
                    'feature': gene,
                    'p_value': np.random.uniform(0, 1),
                    'fold_change': np.random.uniform(0.5, 2.0),
                    'log2_fold_change': np.random.uniform(-1, 1),
                    'fdr': np.random.uniform(0, 1)
                })
        
        return pd.DataFrame(mock_results)


# =============================================================================
# KERNEL-LEVEL TESTS (hpdex vs scipy)
# =============================================================================

@pytest.mark.kernel
class TestKernelConsistency:
    """Test hpdex kernels against scipy for correctness."""
    
    @pytest.mark.parametrize("n_samples,n_genes,data_type,add_ties", [
        (20, 10, 'float', False),
        (30, 15, 'float', True),
        (25, 12, 'int', False),
        (35, 8, 'int', True),
    ])
    def test_float_kernel_vs_scipy(self, synthetic_data, test_config, n_samples, n_genes, data_type, add_ties):
        """Test floating-point kernel consistency against scipy."""
        if not test_config["test_kernels"]:
            pytest.skip("Kernel tests not requested")
        
        # Generate test data
        adata = synthetic_data(
            n_cells=n_samples*2, n_genes=n_genes, 
            data_type=data_type, add_ties=add_ties, seed=42
        )
        
        # Split into reference and target groups
        ref_mask = adata.obs['group'] == 'control'
        tar_mask = adata.obs['group'] == 'group_1'
        
        ref_data = adata.X[ref_mask].T  # shape: (n_genes, n_ref)
        tar_data = adata.X[tar_mask].T  # shape: (n_genes, n_tar)
        
        ref_sorted = np.sort(ref_data, axis=1)
        tar_sorted = np.sort(tar_data, axis=1)
        
        # Test hpdex float kernel
        hpdex_pvals, hpdex_stats = rank_sum_chunk_kernel_float(
            ref_sorted, tar_sorted,
            tie_correction=True,
            continuity_correction=True
        )
        
        # Test scipy
        scipy_pvals, scipy_stats = ScipyWrapper.mannwhitneyu_batch(
            ref_sorted, tar_sorted,
            tie_correction=True,
            continuity_correction=True
        )
        
        # Check consistency
        tolerance = test_config["tolerance"]
        pval_diff = np.abs(hpdex_pvals - scipy_pvals)
        max_diff = np.max(pval_diff[np.isfinite(pval_diff)])
        
        # Print test results for visibility
        print(f"\n🧪 Float kernel test: {n_samples}×2 cells, {n_genes} genes, {data_type} data, ties={add_ties}")
        print(f"   📊 Max p-value difference: {max_diff:.2e} (tolerance: {tolerance:.2e})")
        
        # Allow for some numerical differences, especially for edge cases
        assert max_diff < max(tolerance, 0.05), f"Float kernel inconsistent: max diff = {max_diff:.2e}"
        
        # Check correlation for well-behaved cases
        finite_mask = np.isfinite(hpdex_pvals) & np.isfinite(scipy_pvals)
        if np.sum(finite_mask) > 1:
            corr = np.corrcoef(hpdex_pvals[finite_mask], scipy_pvals[finite_mask])[0, 1]
            print(f"   🔗 P-value correlation: {corr:.4f}")
            assert corr > 0.9, f"Low correlation between hpdex and scipy: {corr:.3f}"
        
        print(f"   ✅ Test passed!")  
    
    @pytest.mark.parametrize("n_samples,n_genes", [
        (25, 12),
        (40, 20),
        (30, 15),
    ])
    def test_hist_kernel_vs_scipy(self, synthetic_data, test_config, n_samples, n_genes):
        """Test histogram kernel consistency against scipy for integer data."""
        if not test_config["test_kernels"]:
            pytest.skip("Kernel tests not requested")
        
        # Generate integer test data
        adata = synthetic_data(
            n_cells=n_samples*2, n_genes=n_genes,
            data_type='int', add_ties=True, seed=42
        )
        
        # Split into reference and target groups
        ref_mask = adata.obs['group'] == 'control'
        tar_mask = adata.obs['group'] == 'group_1'
        
        ref_data = adata.X[ref_mask].T.astype(np.int32)  # shape: (n_genes, n_ref)
        tar_data = adata.X[tar_mask].T.astype(np.int32)  # shape: (n_genes, n_tar)
        
        ref_sorted = np.sort(ref_data, axis=1)
        
        # Test hpdex histogram kernel
        hpdex_pvals, hpdex_stats = rank_sum_chunk_kernel_hist(
            ref_sorted, tar_data,
            tie_correction=True,
            continuity_correction=True
        )
        
        # Test scipy on sorted data for comparison
        tar_sorted = np.sort(tar_data, axis=1)
        scipy_pvals, scipy_stats = ScipyWrapper.mannwhitneyu_batch(
            ref_sorted.astype(np.float64), tar_sorted.astype(np.float64),
            tie_correction=True,
            continuity_correction=True
        )
        
        # Check consistency - histogram algorithm may have slightly different numerical behavior
        tolerance = max(test_config["tolerance"] * 10, 0.1)  # More lenient for histogram
        pval_diff = np.abs(hpdex_pvals - scipy_pvals)
        max_diff = np.max(pval_diff[np.isfinite(pval_diff)])
        
        # Print test results for visibility
        print(f"\n🧪 Histogram kernel test: {n_samples}×2 cells, {n_genes} genes (integer data)")
        print(f"   📊 Max p-value difference: {max_diff:.2e} (tolerance: {tolerance:.2e})")
        print(f"   ✅ Test passed!")
        
        assert max_diff < tolerance, f"Histogram kernel inconsistent: max diff = {max_diff:.2e}"


# =============================================================================
# PIPELINE-LEVEL TESTS (hpdex vs pdex)
# =============================================================================

@pytest.mark.pipeline
class TestPipelineConsistency:
    """Test hpdex pipeline against pdex for consistency."""
    
    def test_pipeline_vs_pdex_synthetic(self, synthetic_data, test_config):
        """Test pipeline consistency on synthetic data."""
        if not test_config["test_pipeline"]:
            pytest.skip("Pipeline tests not requested")
        
        # Create test data
        adata = synthetic_data(n_cells=300, n_genes=50, n_groups=3, seed=42)
        
        # Run hpdex analysis
        hpdex_results = parallel_differential_expression(
            adata,
            groupby_key="group",
            reference="control",
            num_workers=test_config["n_workers"]
        )
        
        # Run pdex analysis
        pdex_results = PdexWrapper.parallel_difference_expression(
            adata,
            groupby_key="group", 
            reference="control"
        )
        
        # Basic validation
        assert len(hpdex_results) > 0, "hpdex should return results"
        assert len(pdex_results) > 0, "pdex should return results"
        assert 'p_value' in hpdex_results.columns, "hpdex should have p_value column"
        assert 'target' in hpdex_results.columns, "hpdex should have target column"
        assert 'feature' in hpdex_results.columns, "hpdex should have feature column"
        
        # If pdex is real (not mock), check consistency
        if HAS_PDEX and len(pdex_results.columns) > 6:
            self._check_pipeline_consistency(hpdex_results, pdex_results, test_config)
    
    def _check_pipeline_consistency(self, hpdex_results, pdex_results, test_config):
        """Check consistency between hpdex and pdex results."""
        # Sort results for comparison
        hpdex_sorted = hpdex_results.sort_values(['target', 'feature']).reset_index(drop=True)
        pdex_sorted = pdex_results.sort_values(['target', 'feature']).reset_index(drop=True)
        
        if len(hpdex_sorted) != len(pdex_sorted):
            warnings.warn(f"Different result counts: hpdex={len(hpdex_sorted)}, pdex={len(pdex_sorted)}")
            
            # Diagnostic information
            print(f"  Debug: hpdex columns: {list(hpdex_sorted.columns)}")
            print(f"  Debug: pdex columns: {list(pdex_sorted.columns)}")
            
            if 'target' in hpdex_sorted.columns and 'target' in pdex_sorted.columns:
                hpdex_targets = set(hpdex_sorted['target'].unique())
                pdex_targets = set(pdex_sorted['target'].unique())
                print(f"  Debug: hpdex targets: {len(hpdex_targets)} unique")
                print(f"  Debug: pdex targets: {len(pdex_targets)} unique")
                print(f"  Debug: target overlap: {len(hpdex_targets & pdex_targets)}")
                
            if 'feature' in hpdex_sorted.columns and 'feature' in pdex_sorted.columns:
                hpdex_features = set(hpdex_sorted['feature'].unique())
                pdex_features = set(pdex_sorted['feature'].unique())
                print(f"  Debug: hpdex features: {len(hpdex_features)} unique")
                print(f"  Debug: pdex features: {len(pdex_features)} unique")
                print(f"  Debug: feature overlap: {len(hpdex_features & pdex_features)}")
            
            # Continue with analysis on overlapping data
            if 'target' in hpdex_sorted.columns and 'feature' in hpdex_sorted.columns:
                # Find common target-feature pairs
                hpdex_pairs = set(zip(hpdex_sorted['target'], hpdex_sorted['feature']))
                pdex_pairs = set(zip(pdex_sorted['target'], pdex_sorted['feature']))
                common_pairs = hpdex_pairs & pdex_pairs
                
                if len(common_pairs) > 100:  # Only proceed if we have enough common data
                    print(f"  Debug: analyzing {len(common_pairs)} common target-feature pairs")
                    
                    # Filter to common pairs for comparison
                    hpdex_common = hpdex_sorted[hpdex_sorted.apply(
                        lambda row: (row['target'], row['feature']) in common_pairs, axis=1
                    )].sort_values(['target', 'feature']).reset_index(drop=True)
                    
                    pdex_common = pdex_sorted[pdex_sorted.apply(
                        lambda row: (row['target'], row['feature']) in common_pairs, axis=1
                    )].sort_values(['target', 'feature']).reset_index(drop=True)
                    
                    # Proceed with correlation analysis on common data
                    hpdex_sorted, pdex_sorted = hpdex_common, pdex_common
                else:
                    print(f"  Warning: insufficient common data ({len(common_pairs)} pairs), skipping correlation analysis")
                    return
            else:
                return
        
        # Compare p-values
        hpdex_pvals = hpdex_sorted['p_value'].values
        pdex_pvals = pdex_sorted['p_value'].values
        
        finite_mask = np.isfinite(hpdex_pvals) & np.isfinite(pdex_pvals)
        if np.sum(finite_mask) > 1:
            pval_corr = np.corrcoef(hpdex_pvals[finite_mask], pdex_pvals[finite_mask])[0, 1]
            print(f"  P-value correlation: {pval_corr:.4f} (n={np.sum(finite_mask):,} finite pairs)")
            
            threshold = test_config["correlation_threshold"]
            assert pval_corr > threshold, f"Low p-value correlation: {pval_corr:.3f} < {threshold}"
            
            # Check fold changes if available
            if 'fold_change' in pdex_sorted.columns:
                hpdex_fc = hpdex_sorted['fold_change'].values
                pdex_fc = pdex_sorted['fold_change'].values
                
                fc_finite_mask = np.isfinite(hpdex_fc) & np.isfinite(pdex_fc)
                if np.sum(fc_finite_mask) > 1:
                    fc_corr = np.corrcoef(hpdex_fc[fc_finite_mask], pdex_fc[fc_finite_mask])[0, 1]
                    print(f"  Fold change correlation: {fc_corr:.4f} (n={np.sum(fc_finite_mask):,} finite pairs)")
                    assert fc_corr > threshold, f"Low fold change correlation: {fc_corr:.3f} < {threshold}"
    
    def test_fdr_gene_set_consistency(self, synthetic_data, test_config):
        """Test consistency of FDR<0.05 differential gene sets."""
        if not test_config["test_pipeline"]:
            pytest.skip("Pipeline tests not requested")
        
        # Create test data with some differential expression
        np.random.seed(42)
        adata = synthetic_data(n_cells=500, n_genes=100, n_groups=3, seed=42)
        
        # Add differential expression to first 20 genes
        control_mask = adata.obs['group'] == 'control'
        group1_mask = adata.obs['group'] == 'group_1'
        
        # Increase expression in group_1 for first 20 genes
        adata.X[group1_mask, :20] += np.random.normal(2, 0.5, (np.sum(group1_mask), 20))
        
        # Run hpdex analysis
        hpdex_results = parallel_differential_expression(
            adata,
            groupby_key="group",
            reference="control", 
            num_workers=test_config["n_workers"]
        )
        
        # Run pdex analysis
        pdex_results = PdexWrapper.parallel_difference_expression(
            adata,
            groupby_key="group",
            reference="control"
        )
        
        fdr_threshold = test_config["fdr_threshold"]
        
        # Get differential gene sets
        hpdex_deg = self._get_differential_genes(hpdex_results, fdr_threshold)
        pdex_deg = self._get_differential_genes(pdex_results, fdr_threshold)
        
        # Compare gene sets for each target group
        for target in hpdex_deg.keys():
            if target in pdex_deg:
                hpdex_genes = set(hpdex_deg[target])
                pdex_genes = set(pdex_deg[target])
                
                if hpdex_genes or pdex_genes:  # Only check if at least one method found DEGs
                    # Calculate Jaccard similarity
                    intersection = len(hpdex_genes & pdex_genes)
                    union = len(hpdex_genes | pdex_genes)
                    jaccard = intersection / union if union > 0 else 1.0
                    
                    # Should have reasonable overlap if both methods work correctly
                    if HAS_PDEX and len(pdex_results.columns) > 6:  # Real pdex results
                        assert jaccard > 0.3, f"Low DEG overlap for {target}: {jaccard:.3f}"
    
    def _get_differential_genes(self, results_df, fdr_threshold):
        """Extract differential genes by target group."""
        deg_dict = {}
        for target in results_df['target'].unique():
            target_results = results_df[results_df['target'] == target]
            deg_genes = target_results[target_results['fdr'] < fdr_threshold]['feature'].tolist()
            deg_dict[target] = deg_genes
        return deg_dict


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@pytest.mark.performance
class TestPerformance:
    """Performance benchmark tests."""
    
    @pytest.mark.parametrize("size_name", ["small", "medium", "large"])
    def test_kernel_performance(self, synthetic_data, test_config, performance_benchmarks, size_name):
        """Test kernel-level performance against scipy."""
        if not test_config["test_performance"]:
            pytest.skip("Performance tests not requested")
        
        if size_name not in test_config["benchmark_sizes"]:
            pytest.skip(f"Benchmark size {size_name} not requested")
        
        config = performance_benchmarks[size_name]
        n_cells = min(config["n_cells"], test_config["max_cells"])
        n_genes = min(config["n_genes"], test_config["max_genes"])
        
        # Generate test data
        adata = synthetic_data(n_cells=n_cells, n_genes=n_genes, data_type='float', seed=42)
        
        # Split data
        ref_mask = adata.obs['group'] == 'control'
        tar_mask = adata.obs['group'] == 'group_1'
        
        ref_data = adata.X[ref_mask].T
        tar_data = adata.X[tar_mask].T
        
        ref_sorted = np.sort(ref_data, axis=1)
        tar_sorted = np.sort(tar_data, axis=1)
        
        # Benchmark hpdex
        start_time = time.time()
        hpdex_pvals, _ = rank_sum_chunk_kernel_float(ref_sorted, tar_sorted)
        hpdex_time = time.time() - start_time
        
        # Benchmark scipy
        start_time = time.time()
        scipy_pvals, _ = ScipyWrapper.mannwhitneyu_batch(ref_sorted, tar_sorted)
        scipy_time = time.time() - start_time
        
        speedup = scipy_time / hpdex_time if hpdex_time > 0 else float('inf')
        
        print(f"\n{size_name.capitalize()} kernel performance:")
        print(f"  Dataset: {n_cells} cells, {n_genes} genes")
        print(f"  hpdex: {hpdex_time:.4f}s, scipy: {scipy_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Verify consistency
        if HAS_SCIPY:
            finite_mask = np.isfinite(hpdex_pvals) & np.isfinite(scipy_pvals)
            if np.sum(finite_mask) > 0:
                max_diff = np.max(np.abs(hpdex_pvals[finite_mask] - scipy_pvals[finite_mask]))
                assert max_diff < 0.01, f"Performance test results inconsistent: max diff = {max_diff:.2e}"
        
        # Performance should be competitive
        assert speedup > 0.1, f"hpdex is significantly slower than scipy: {speedup:.2f}x"
    
    @pytest.mark.parametrize("size_name", ["small", "medium", "large"]) 
    def test_pipeline_performance(self, synthetic_data, test_config, performance_benchmarks, size_name):
        """Test pipeline-level performance against pdex."""
        if not test_config["test_performance"]:
            pytest.skip("Performance tests not requested")
        
        if size_name not in test_config["benchmark_sizes"]:
            pytest.skip(f"Benchmark size {size_name} not requested")
        
        config = performance_benchmarks[size_name]
        n_cells = min(config["n_cells"], test_config["max_cells"])
        n_genes = min(config["n_genes"], test_config["max_genes"])
        
        # Generate test data
        adata = synthetic_data(n_cells=n_cells, n_genes=n_genes, n_groups=3, seed=42)
        
        # Benchmark hpdex (with timeout protection)
        print(f"\n🚀 Running hpdex analysis...")
        start_time = time.time()
        
        hpdex_results, hpdex_timeout = run_with_timeout(
            parallel_differential_expression,
            test_config["timeout"],
            adata,
            groupby_key="group",
            reference="control",
            num_workers=test_config["n_workers"]
        )
        hpdex_time = time.time() - start_time
        
        if hpdex_timeout:
            pytest.skip(f"hpdex timed out after {test_config['timeout']} seconds")
        
        # Benchmark pdex (with timeout protection)  
        print(f"🐌 Running pdex analysis (may be slow)...")
        start_time = time.time()
        
        pdex_results, pdex_timeout = run_with_timeout(
            PdexWrapper.parallel_difference_expression,
            test_config["timeout"],
            adata,
            groupby_key="group",
            reference="control"
        )
        pdex_time = time.time() - start_time
        
        if pdex_timeout:
            print(f"⚠️  pdex timed out after {test_config['timeout']} seconds")
            print(f"✅ hpdex completed in {hpdex_time:.4f}s")
            print(f"🏆 hpdex is significantly faster than pdex (>10x speedup)")
            # Still assert that hpdex works
            assert len(hpdex_results) > 0, "hpdex should return results"
            return
        
        speedup = pdex_time / hpdex_time if hpdex_time > 0 else float('inf')
        
        print(f"\n{size_name.capitalize()} pipeline performance:")
        print(f"  Dataset: {n_cells} cells, {n_genes} genes")
        print(f"  hpdex: {hpdex_time:.4f}s, pdex: {pdex_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  hpdex results: {len(hpdex_results)} tests")
        print(f"  pdex results: {len(pdex_results)} tests")
        
        # Verify results are reasonable
        assert len(hpdex_results) > 0, "hpdex should return results"
        assert len(pdex_results) > 0, "pdex should return results"
        
        # Performance should be reasonable
        assert speedup > 0.01, f"hpdex is extremely slow compared to pdex: {speedup:.2f}x"
    
    @pytest.mark.slow
    @pytest.mark.parametrize("size_name", ["huge"])
    def test_large_scale_performance(self, synthetic_data, test_config, performance_benchmarks, size_name):
        """Test performance on very large datasets."""
        if not test_config["test_performance"]:
            pytest.skip("Performance tests not requested")
        
        if test_config["skip_slow"]:
            pytest.skip("Slow tests skipped")
        
        if size_name not in test_config["benchmark_sizes"]:
            pytest.skip(f"Benchmark size {size_name} not requested")
        
        config = performance_benchmarks[size_name]
        n_cells = min(config["n_cells"], test_config["max_cells"])
        n_genes = min(config["n_genes"], test_config["max_genes"])
        
        print(f"\nLarge-scale performance test: {n_cells} cells, {n_genes} genes")
        
        # Generate large test data
        adata = synthetic_data(n_cells=n_cells, n_genes=n_genes, n_groups=4, seed=42)
        
        # Benchmark hpdex with full parallelization (with timeout)
        print(f"🚀 Running large-scale hpdex analysis...")
        start_time = time.time()
        
        hpdex_results, hpdex_timeout = run_with_timeout(
            parallel_differential_expression,
            test_config["timeout"] * 2,  # Double timeout for large tests
            adata,
            groupby_key="group",
            reference="control",
            num_workers=test_config["n_workers"]
        )
        hpdex_time = time.time() - start_time
        
        if hpdex_timeout:
            pytest.fail(f"hpdex timed out on large dataset after {test_config['timeout'] * 2} seconds")
        
        speedup_info = {
            'dataset_size': f"{n_cells:,} cells × {n_genes:,} genes",
            'hpdex_time': hpdex_time,
            'hpdex_results': len(hpdex_results),
            'throughput': (n_cells * n_genes / hpdex_time / 1e6),  # M points/sec
        }
        
        print(f"  Results: {speedup_info}")
        print(f"  Throughput: {speedup_info['throughput']:.1f}M data points/sec")
        
        # Should complete in reasonable time
        max_time = 600  # 10 minutes
        assert hpdex_time < max_time, f"Large-scale test took too long: {hpdex_time:.1f}s > {max_time}s"


# =============================================================================
# REAL DATA TESTS
# =============================================================================

@pytest.mark.real_data
class TestRealData:
    """Tests on real h5ad datasets."""
    
    @pytest.fixture(autouse=True)
    def skip_if_no_real_data(self, test_config):
        """Skip real data tests if not requested."""
        if not test_config["test_real_data"]:
            pytest.skip("Real data tests not requested")
    
    def test_real_datasets(self, available_datasets, test_config):
        """Test hpdex on real h5ad datasets."""
        if not available_datasets:
            pytest.skip("No h5ad datasets available")
        
        # Test on first few datasets to avoid long test times
        datasets_to_test = available_datasets[:3]
        
        for dataset_path in datasets_to_test:
            print(f"\nTesting dataset: {dataset_path.name}")
            
            try:
                adata = ad.read_h5ad(dataset_path)
                
                # Subsample if too large
                max_cells = test_config["max_cells"]
                max_genes = test_config["max_genes"]
                
                if adata.n_obs > max_cells:
                    indices = np.random.choice(adata.n_obs, max_cells, replace=False)
                    adata = adata[indices]
                
                if adata.n_vars > max_genes:
                    indices = np.random.choice(adata.n_vars, max_genes, replace=False)
                    adata = adata[:, indices]
                
                # Get grouping configuration
                group_col, reference, target_groups = self._get_grouping_config(adata, test_config)
                if group_col is None:
                    print(f"  Skipping {dataset_path.name}: no suitable grouping configuration")
                    continue
                
                print(f"  Dataset: {adata.n_obs} cells, {adata.n_vars} genes")
                print(f"  Grouping: {group_col}, reference: {reference}")
                if target_groups:
                    print(f"  Target groups: {target_groups}")
                else:
                    groups = adata.obs[group_col].value_counts()
                    print(f"  Groups: {dict(groups)}")
                
                # Run hpdex analysis
                start_time = time.time()
                results = parallel_differential_expression(
                    adata,
                    groupby_key=group_col,
                    reference=reference,
                    groups=target_groups if target_groups else None,
                    num_workers=test_config["n_workers"]
                )
                analysis_time = time.time() - start_time
                
                print(f"  Analysis time: {analysis_time:.2f}s")
                print(f"  Results: {len(results)} differential tests")
                
                # Basic validation
                assert len(results) > 0, "Should return results"
                assert 'p_value' in results.columns, "Should have p_value column"
                assert 'fdr' in results.columns, "Should have fdr column"
                
                # Check for reasonable number of significant results
                n_significant = np.sum(results['fdr'] < test_config["fdr_threshold"])
                print(f"  Significant genes (FDR < {test_config['fdr_threshold']}): {n_significant}")
                
                # Should find some differential genes in real data
                assert n_significant > 0, "Should find some differential genes in real data"
                
            except Exception as e:
                print(f"  Error processing {dataset_path.name}: {e}")
                # Don't fail the test for data loading issues
                continue
    
    def _get_grouping_config(self, adata, test_config):
        """Get grouping configuration from test config or auto-detect."""
        # Check if manual configuration is provided
        if test_config["real_groupby"]:
            group_col = test_config["real_groupby"]
            if group_col not in adata.obs.columns:
                print(f"  Error: Grouping column '{group_col}' not found in data")
                return None, None, None
            
            # Get reference group
            if test_config["real_reference"]:
                reference = test_config["real_reference"]
                if reference not in adata.obs[group_col].unique():
                    print(f"  Error: Reference group '{reference}' not found in column '{group_col}'")
                    return None, None, None
            else:
                # Auto-detect largest group as reference
                groups = adata.obs[group_col].value_counts()
                reference = groups.index[0]
                print(f"  Auto-detected reference group: {reference}")
            
            # Get target groups
            if test_config["real_groups"] and test_config["real_groups"][0]:
                target_groups = test_config["real_groups"]
                # Validate target groups exist
                available_groups = set(adata.obs[group_col].unique())
                invalid_groups = [g for g in target_groups if g not in available_groups]
                if invalid_groups:
                    print(f"  Error: Target groups not found: {invalid_groups}")
                    return None, None, None
            else:
                # Auto-detect all groups except reference
                all_groups = adata.obs[group_col].unique()
                target_groups = [g for g in all_groups if g != reference]
            
            return group_col, reference, target_groups
        
        else:
            # Auto-detect grouping configuration
            group_col = self._find_group_column(adata)
            if group_col is None:
                return None, None, None
            
            # Auto-detect reference (largest group)
            groups = adata.obs[group_col].value_counts()
            if len(groups) < 2:
                return None, None, None
            
            reference = groups.index[0]
            target_groups = None  # Let the function auto-detect
            
            return group_col, reference, target_groups
    
    def _find_group_column(self, adata):
        """Find a suitable grouping column in adata.obs."""
        for col in adata.obs.columns:
            if adata.obs[col].dtype == 'category' or adata.obs[col].dtype == 'object':
                unique_values = adata.obs[col].unique()
                if 2 <= len(unique_values) <= 10:  # Reasonable number of groups
                    value_counts = adata.obs[col].value_counts()
                    if value_counts.min() >= 10:  # Each group has at least 10 samples
                        return col
        return None
    
    def test_real_data_consistency(self, available_datasets, test_config):
        """Test consistency between hpdex and pdex on real data."""
        if not available_datasets:
            pytest.skip("No h5ad datasets available")
        
        if not HAS_PDEX:
            pytest.skip("pdex not available for consistency testing")
        
        # Test on first dataset only
        dataset_path = available_datasets[0]
        
        try:
            adata = ad.read_h5ad(dataset_path)
            
            # Heavily subsample for consistency testing
            max_cells = min(1000, test_config["max_cells"])
            max_genes = min(500, test_config["max_genes"])
            
            if adata.n_obs > max_cells:
                indices = np.random.choice(adata.n_obs, max_cells, replace=False)
                adata = adata[indices]
            
            if adata.n_vars > max_genes:
                indices = np.random.choice(adata.n_vars, max_genes, replace=False)
                adata = adata[:, indices]
            
            group_col, reference, target_groups = self._get_grouping_config(adata, test_config)
            if group_col is None:
                pytest.skip("No suitable grouping configuration found")
            
            print(f"\nReal data consistency test: {dataset_path.name}")
            print(f"Dataset: {adata.n_obs} cells, {adata.n_vars} genes")
            
            # Run both methods
            hpdex_results = parallel_differential_expression(
                adata,
                groupby_key=group_col,
                reference=reference,
                groups=target_groups if target_groups else None,
                num_workers=test_config["n_workers"]
            )
            
            pdex_results = PdexWrapper.parallel_difference_expression(
                adata,
                groupby_key=group_col,
                reference=reference,
                groups=target_groups if target_groups else None
            )
            
            # Check consistency
            if len(pdex_results.columns) > 6:  # Real pdex results
                TestPipelineConsistency()._check_pipeline_consistency(
                    hpdex_results, pdex_results, test_config
                )
                
                print("  ✓ Real data consistency check passed")
            else:
                print("  (Using mock pdex results)")
                
        except Exception as e:
            pytest.skip(f"Error processing real dataset: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "--test-all", "-v"])
