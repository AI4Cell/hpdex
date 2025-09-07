"""
Comprehensive test suite for hpdex framework.

This test suite covers:
1. Basic API functionality
2. Kernel correctness
3. Multi-process support  
4. Streaming backend
5. Edge cases and error handling
6. Performance validation
"""

import tempfile
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats
from scipy.sparse import csr_matrix

# Import hpdex functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hpdex import parallel_differential_expression, parallel_differential_expression_stream
from hpdex.backen import (
    rank_sum_chunk_kernel_float,
    rank_sum_chunk_kernel_hist,
    _compute_fold_changes,
    _compute_log2_fold_change,
    _compute_fdr
)


class TestBasicFunctionality:
    """Test basic API functionality."""
    
    def test_basic_api_call(self, synthetic_data):
        """Test basic API call with default parameters."""
        adata = synthetic_data(n_cells=100, n_genes=20, n_groups=3)
        
        results = parallel_differential_expression(
            adata=adata,
            groupby_key="group",
            reference="control"
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert all(col in results.columns for col in [
            'target', 'feature', 'p_value', 'fold_change', 'log2_fold_change', 'fdr'
        ])
    
    def test_parameter_validation(self, synthetic_data):
        """Test parameter validation."""
        adata = synthetic_data(n_cells=100, n_genes=20, n_groups=3)
        
        # Test invalid metric
        with pytest.raises(ValueError, match="Unsupported statistical test method"):
            parallel_differential_expression(
                adata=adata,
                groupby_key="group", 
                reference="control",
                metric="invalid_metric"
            )
        
        # Test invalid reference
        with pytest.raises(ValueError, match="not found"):
            parallel_differential_expression(
                adata=adata,
                groupby_key="group",
                reference="nonexistent_group"
            )
        
        # Test invalid groupby_key
        with pytest.raises(KeyError):
            parallel_differential_expression(
                adata=adata,
                groupby_key="nonexistent_column",
                reference="control"
            )
    
    def test_metric_types(self, synthetic_data):
        """Test different metric types."""
        adata_float = synthetic_data(n_cells=100, n_genes=20, data_type='float')
        adata_int = synthetic_data(n_cells=100, n_genes=20, data_type='int')
        
        # Test wilcoxon metric
        results_wilcoxon = parallel_differential_expression(
            adata=adata_float,
            groupby_key="group",
            reference="control",
            metric="wilcoxon"
        )
        assert len(results_wilcoxon) > 0
        
        # Test wilcoxon-hist metric with integer data
        results_hist = parallel_differential_expression(
            adata=adata_int,
            groupby_key="group", 
            reference="control",
            metric="wilcoxon-hist"
        )
        assert len(results_hist) > 0
    
    def test_multiprocessing(self, synthetic_data):
        """Test multiprocessing functionality."""
        adata = synthetic_data(n_cells=200, n_genes=50, n_groups=3)
        
        # Single worker
        results_single = parallel_differential_expression(
            adata=adata,
            groupby_key="group",
            reference="control",
            num_workers=1
        )
        
        # Multiple workers
        results_multi = parallel_differential_expression(
            adata=adata,
            groupby_key="group",
            reference="control", 
            num_workers=4
        )
        
        # Results should be very similar
        assert len(results_single) == len(results_multi)
        
        # Sort by target and feature for comparison
        results_single = results_single.sort_values(['target', 'feature']).reset_index(drop=True)
        results_multi = results_multi.sort_values(['target', 'feature']).reset_index(drop=True)
        
        # P-values should be nearly identical
        np.testing.assert_allclose(
            results_single['p_value'].values,
            results_multi['p_value'].values,
            rtol=1e-10
        )


class TestKernelCorrectness:
    """Test kernel correctness against scipy."""
    
    def test_float_kernel_vs_scipy(self):
        """Test float kernel against scipy's mannwhitneyu."""
        np.random.seed(42)
        
        n_genes = 10
        n_ref = 20
        n_tar = 15
        
        # Generate test data
        ref_data = np.random.normal(0, 1, (n_genes, n_ref))
        tar_data = np.random.normal(0.5, 1, (n_genes, n_tar))
        
        # Sort data as expected by kernel
        ref_sorted = np.sort(ref_data, axis=1)
        tar_sorted = np.sort(tar_data, axis=1)
        
        # Test kernel
        p_values, u_stats = rank_sum_chunk_kernel_float(
            ref_sorted, tar_sorted,
            tie_correction=True,
            continuity_correction=True,
            use_asymptotic=True
        )
        
        # Compare with scipy for each gene
        for i in range(n_genes):
            scipy_stat, scipy_p = stats.mannwhitneyu(
                tar_data[i], ref_data[i],
                alternative='two-sided',
                use_continuity=True
            )
            
            # p-values should be close
            assert abs(p_values[i] - scipy_p) < 1e-5, f"Gene {i}: hpdex={p_values[i]}, scipy={scipy_p}"
    
    def test_histogram_kernel_integer_data(self):
        """Test histogram kernel with integer data."""
        np.random.seed(42)
        
        n_genes = 5
        n_ref = 30
        n_tar = 25
        
        # Generate integer test data
        ref_data = np.random.poisson(5, (n_genes, n_ref))
        tar_data = np.random.poisson(7, (n_genes, n_tar))
        
        # Test histogram kernel
        p_values, u_stats = rank_sum_chunk_kernel_hist(
            ref_data, tar_data,
            tie_correction=True,
            continuity_correction=True,
            use_asymptotic=True,
            max_bins=100
        )
        
        # Should produce valid p-values
        assert all(0 <= p <= 1 for p in p_values)
        assert not any(np.isnan(p_values))
        assert not any(np.isnan(u_stats))
    
    def test_tie_handling(self):
        """Test handling of tied values."""
        np.random.seed(42)
        
        # Create data with many ties
        ref_data = np.array([[1, 1, 2, 2, 3, 3]])
        tar_data = np.array([[1, 2, 2, 3, 3, 4]])
        
        # Test float kernel
        ref_sorted = np.sort(ref_data, axis=1) 
        tar_sorted = np.sort(tar_data, axis=1)
        
        p_values, u_stats = rank_sum_chunk_kernel_float(
            ref_sorted, tar_sorted,
            tie_correction=True,
            continuity_correction=True
        )
        
        assert 0 <= p_values[0] <= 1
        assert not np.isnan(p_values[0])


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_cell_groups(self, synthetic_data):
        """Test with very small groups."""
        adata = synthetic_data(n_cells=6, n_genes=10, n_groups=3)
        
        # This should work with min_samples=1
        results = parallel_differential_expression(
            adata=adata,
            groupby_key="group",
            reference="control",
            min_samples=1
        )
        
        assert len(results) > 0
    
    def test_empty_groups(self, synthetic_data):
        """Test with groups below min_samples threshold."""
        adata = synthetic_data(n_cells=50, n_genes=10, n_groups=3)
        
        # This should exclude groups with < 20 samples
        results = parallel_differential_expression(
            adata=adata,
            groupby_key="group",
            reference="control",
            min_samples=20
        )
        
        # Should have fewer or no target groups
        unique_targets = results['target'].nunique() if len(results) > 0 else 0
        assert unique_targets <= 2  # Originally 3 groups minus reference
    
    def test_sparse_data(self, synthetic_data):
        """Test with sparse data."""
        adata = synthetic_data(n_cells=100, n_genes=20, n_groups=3)
        
        # Convert to sparse
        adata.X = csr_matrix(adata.X)
        
        results = parallel_differential_expression(
            adata=adata,
            groupby_key="group",
            reference="control"
        )
        
        assert len(results) > 0
        assert all(col in results.columns for col in [
            'target', 'feature', 'p_value', 'fold_change', 'log2_fold_change', 'fdr'
        ])
    
    def test_constant_genes(self):
        """Test with genes that have constant expression."""
        n_cells = 100
        n_genes = 10
        
        # Create data with some constant genes
        X = np.random.normal(0, 1, (n_cells, n_genes))
        X[:, 0] = 5.0  # Constant gene
        X[:, 1] = 0.0  # Another constant gene
        
        obs = pd.DataFrame({
            'group': ['control'] * 50 + ['treatment'] * 50
        })
        
        adata = ad.AnnData(X=X, obs=obs)
        
        results = parallel_differential_expression(
            adata=adata,
            groupby_key="group",
            reference="control"
        )
        
        # Should handle constant genes gracefully
        assert len(results) > 0
        
        # P-values for constant genes might be NaN or 1.0
        constant_gene_results = results[results['feature'].isin(['0', '1'])]
        for _, row in constant_gene_results.iterrows():
            assert np.isnan(row['p_value']) or row['p_value'] == 1.0


class TestStreamingBackend:
    """Test streaming backend functionality."""
    
    def test_streaming_basic(self, synthetic_data, tmp_path):
        """Test basic streaming functionality."""
        adata = synthetic_data(n_cells=200, n_genes=50, n_groups=3)
        
        # Save to temporary file
        test_file = tmp_path / "test_data.h5ad"
        adata.write_h5ad(test_file)
        
        # Test streaming API
        results = parallel_differential_expression_stream(
            data_path=test_file,
            groupby_key="group",
            reference="control",
            memory_limit_gb=1.0
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert all(col in results.columns for col in [
            'target', 'feature', 'p_value', 'fold_change', 'log2_fold_change', 'fdr'
        ])
    
    def test_streaming_vs_regular(self, synthetic_data, tmp_path):
        """Test that streaming gives similar results to regular API."""
        adata = synthetic_data(n_cells=150, n_genes=30, n_groups=3)
        
        # Regular API
        results_regular = parallel_differential_expression(
            adata=adata,
            groupby_key="group",
            reference="control",
            num_workers=1  # Use single worker for deterministic results
        )
        
        # Save and test streaming API
        test_file = tmp_path / "test_data.h5ad"
        adata.write_h5ad(test_file)
        
        results_streaming = parallel_differential_expression_stream(
            data_path=test_file,
            groupby_key="group",
            reference="control",
            memory_limit_gb=2.0,
            num_workers=1
        )
        
        # Sort both results for comparison
        results_regular = results_regular.sort_values(['target', 'feature']).reset_index(drop=True)
        results_streaming = results_streaming.sort_values(['target', 'feature']).reset_index(drop=True)
        
        assert len(results_regular) == len(results_streaming)
        
        # P-values should be very similar
        np.testing.assert_allclose(
            results_regular['p_value'].values,
            results_streaming['p_value'].values,
            rtol=1e-8
        )
    
    def test_streaming_multiprocess(self, synthetic_data, tmp_path):
        """Test streaming with multiprocessing."""
        adata = synthetic_data(n_cells=200, n_genes=40, n_groups=3)
        
        test_file = tmp_path / "test_data.h5ad"
        adata.write_h5ad(test_file)
        
        # Test with multiple workers
        results = parallel_differential_expression_stream(
            data_path=test_file,
            groupby_key="group",
            reference="control",
            memory_limit_gb=2.0,
            num_workers=4
        )
        
        assert len(results) > 0
        assert not any(np.isnan(results['p_value']))


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_fold_change_computation(self):
        """Test fold change computation."""
        target_means = np.array([2.0, 0.0, 4.0, 1e-12])
        ref_means = np.array([1.0, 1.0, 2.0, 1e-12])
        
        fold_changes = _compute_fold_changes(target_means, ref_means, clip_value=20.0)
        
        expected = [2.0, 0.0, 2.0, 20.0]  # Last one clipped
        np.testing.assert_allclose(fold_changes, expected)
    
    def test_log2_fold_change(self):
        """Test log2 fold change computation."""
        fold_changes = np.array([1.0, 2.0, 4.0, 0.5, 0.0])
        
        log2_fc = _compute_log2_fold_change(fold_changes)
        
        expected = [0.0, 1.0, 2.0, -1.0, -np.inf]
        np.testing.assert_allclose(log2_fc, expected)
    
    def test_fdr_correction(self):
        """Test FDR correction."""
        p_values = np.array([0.01, 0.05, 0.1, 0.3, 0.8])
        
        fdr_values = _compute_fdr(p_values)
        
        # FDR values should be >= original p-values
        assert all(fdr >= p for fdr, p in zip(fdr_values, p_values))
        
        # Should be valid probabilities
        assert all(0 <= fdr <= 1 for fdr in fdr_values)


class TestPerformance:
    """Performance and stress tests."""
    
    @pytest.mark.slow
    def test_large_dataset_performance(self, synthetic_data):
        """Test performance with larger datasets."""
        import time
        
        adata = synthetic_data(n_cells=1000, n_genes=500, n_groups=4)
        
        start_time = time.time()
        results = parallel_differential_expression(
            adata=adata,
            groupby_key="group",
            reference="control",
            num_workers=4
        )
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time (less than 60 seconds)
        assert elapsed < 60, f"Large dataset test took {elapsed:.2f} seconds"
        assert len(results) > 0
        
        # Check that we got results for all genes and groups
        expected_rows = (adata.obs['group'].nunique() - 1) * adata.n_vars
        assert len(results) == expected_rows
    
    @pytest.mark.slow 
    def test_memory_efficiency(self, synthetic_data):
        """Test memory efficiency with streaming."""
        import psutil
        import os
        
        # Create larger dataset
        adata = synthetic_data(n_cells=2000, n_genes=1000, n_groups=3)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / "large_test.h5ad"
            adata.write_h5ad(test_file)
            
            # Monitor memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            results = parallel_differential_expression_stream(
                data_path=test_file,
                groupby_key="group",
                reference="control",
                memory_limit_gb=1.0,  # Strict memory limit
                num_workers=2
            )
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            # Should use reasonable amount of memory
            assert memory_increase < 2000, f"Memory usage increased by {memory_increase:.1f} MB"
            assert len(results) > 0


class TestRealData:
    """Tests with real datasets."""
    
    @pytest.mark.real_data
    def test_real_dataset(self, available_datasets, test_config):
        """Test with real h5ad datasets."""
        if not available_datasets:
            pytest.skip("No real datasets available")
        
        # Use the smallest dataset for testing
        test_file = available_datasets[0]
        
        try:
            adata = ad.read_h5ad(test_file)
        except Exception as e:
            pytest.skip(f"Could not load {test_file}: {e}")
        
        # Find a suitable groupby column
        groupby_key = None
        for col in adata.obs.columns:
            if adata.obs[col].dtype == 'object' or adata.obs[col].dtype.name == 'category':
                unique_vals = adata.obs[col].unique()
                if 2 <= len(unique_vals) <= 10:  # Reasonable number of groups
                    groupby_key = col
                    break
        
        if groupby_key is None:
            pytest.skip("No suitable groupby column found")
        
        # Find reference group (most common)
        reference = adata.obs[groupby_key].value_counts().index[0]
        
        # Subsample if too large
        if adata.n_obs > 5000:
            adata = adata[np.random.choice(adata.n_obs, 5000, replace=False)]
        
        if adata.n_vars > 2000:
            adata = adata[:, np.random.choice(adata.n_vars, 2000, replace=False)]
        
        # Run test
        results = parallel_differential_expression(
            adata=adata,
            groupby_key=groupby_key,
            reference=reference,
            min_samples=5,
            num_workers=2
        )
        
        assert len(results) > 0
        assert not any(np.isnan(results['p_value']))
        
        # Check for reasonable fold changes
        assert any(results['fold_change'] > 1.5)  # Some up-regulation
        assert any(results['fold_change'] < 0.67)  # Some down-regulation


if __name__ == "__main__":
    # Basic smoke test when run directly
    import sys
    
    def run_basic_test():
        """Run basic functionality test."""
        np.random.seed(42)
        
        # Generate synthetic data
        n_cells, n_genes = 200, 50
        X = np.random.normal(0, 1, (n_cells, n_genes))
        
        # Add some differential expression
        X[100:, :10] += 2.0  # Upregulate first 10 genes in treatment group
        
        obs = pd.DataFrame({
            'group': ['control'] * 100 + ['treatment'] * 100
        })
        
        adata = ad.AnnData(X=X, obs=obs)
        
        # Test regular API
        print("Testing regular API...")
        results = parallel_differential_expression(
            adata=adata,
            groupby_key="group",
            reference="control",
            num_workers=2
        )
        
        print(f"✓ Regular API: {len(results)} results")
        print(f"  Significant genes (FDR < 0.05): {sum(results['fdr'] < 0.05)}")
        
        # Test streaming API
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / "test.h5ad"
            adata.write_h5ad(test_file)
            
            print("Testing streaming API...")
            results_stream = parallel_differential_expression_stream(
                data_path=test_file,
                groupby_key="group",
                reference="control",
                memory_limit_gb=1.0,
                num_workers=2
            )
            
            print(f"✓ Streaming API: {len(results_stream)} results")
            print(f"  Significant genes (FDR < 0.05): {sum(results_stream['fdr'] < 0.05)}")
        
        print("\n🎉 All basic tests passed!")
        return True
    
    if len(sys.argv) == 1:
        run_basic_test()
