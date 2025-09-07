"""
Real Dataset Benchmark Tests for HPDEX vs pdex

This module tests HPDEX against pdex using real single-cell datasets to evaluate:
- Performance (speed and memory usage)
- Result consistency (p-value correlation)  
- Scalability with real data characteristics
"""

import time
import tempfile
import psutil
import os
import gc
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm

# Import HPDEX
try:
    import sys
    sys.path.insert(0, '../src')
    from hpdex import parallel_differential_expression
    from hpdex.stream import parallel_differential_expression as parallel_differential_expression_stream
except ImportError as e:
    raise ImportError(f"Failed to import HPDEX: {e}")

from pdex import parallel_differential_expression as pdex_de



class RealDataBenchmarkTester:
    """Test HPDEX vs pdex on real single-cell datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.datasets_dir = Path(config.get('datasets_dir', '../Datasets/CellFM'))
        self.correlation_threshold = config.get('correlation_threshold', 0.95)
        self.timeout_seconds = config.get('timeout_seconds', 1800)
        self.max_cells = config.get('max_cells', 50000)
        self.max_genes = config.get('max_genes', 10000)
        self.sample_fraction = config.get('sample_fraction', 1.0)
        
    def run_all_benchmarks(self, num_workers: int = 4) -> Tuple[bool, Dict[str, Any]]:
        """Run benchmarks on all configured datasets."""
        
        datasets = self.config.get('datasets', [])
        if not datasets:
            return False, {'error': 'No datasets configured'}
            
        if pdex_de is None:
            return False, {'error': 'pdex not available for comparison'}
            
        print(f"🧬 Testing {len(datasets)} real datasets against pdex...")
        
        results = []
        overall_success = True
        
        for dataset_config in tqdm(datasets, desc="Real data benchmarks"):
            try:
                success, result = self._benchmark_dataset(dataset_config, num_workers)
                results.append({
                    'dataset': dataset_config['name'],
                    'success': success,
                    'result': result
                })
                
                if not success:
                    overall_success = False
                    
            except Exception as e:
                overall_success = False
                results.append({
                    'dataset': dataset_config['name'],
                    'success': False,
                    'result': {'error': str(e)}
                })
                
        # Compute summary statistics
        successful_tests = [r for r in results if r['success']]
        
        summary = {
            'total_datasets': len(results),
            'successful_datasets': len(successful_tests),
            'success_rate': len(successful_tests) / len(results) if results else 0,
            'benchmark_results': results
        }
        
        if successful_tests:
            # Aggregate performance metrics
            speedups = [r['result'].get('speedup', 0) for r in successful_tests if 'speedup' in r['result']]
            correlations = [r['result'].get('p_correlation', 0) for r in successful_tests if 'p_correlation' in r['result']]
            
            if speedups:
                summary['avg_speedup'] = float(np.mean(speedups))
                summary['min_speedup'] = float(np.min(speedups))
                summary['max_speedup'] = float(np.max(speedups))
                
            if correlations:
                summary['avg_correlation'] = float(np.mean(correlations))
                summary['min_correlation'] = float(np.min(correlations))
                
        return overall_success, summary
    
    def _benchmark_dataset(self, dataset_config: Dict[str, Any], num_workers: int) -> Tuple[bool, Dict[str, Any]]:
        """Benchmark HPDEX vs pdex on a single dataset."""
        
        dataset_name = dataset_config['name']
        file_path = self.datasets_dir / dataset_config['file']
        groupby_key = dataset_config['groupby_key']
        reference_group = dataset_config.get('reference_group')
        skip_if_no_groups = dataset_config.get('skip_if_no_groups', True)
        
        print(f"  📊 Dataset: {dataset_name}")
        
        # Load and validate dataset
        try:
            adata = ad.read_h5ad(file_path)
            print(f"    Original: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
            
        except Exception as e:
            return False, {'error': f'Failed to load dataset: {e}'}
            
        # Check if groupby column exists
        if groupby_key not in adata.obs.columns:
            if skip_if_no_groups:
                return False, {'error': f'Groupby column "{groupby_key}" not found, skipping'}
            else:
                return False, {'error': f'Groupby column "{groupby_key}" not found'}
                
        # Get groups and validate
        unique_groups = adata.obs[groupby_key].unique()
        group_counts = adata.obs[groupby_key].value_counts()
        
        # Filter groups with sufficient cells (>= 10)
        valid_groups = group_counts[group_counts >= 10].index.tolist()
        
        if len(valid_groups) < 2:
            return False, {'error': f'Insufficient groups with >= 10 cells (found {len(valid_groups)})'}
            
        # Auto-detect reference group (largest valid group)
        if reference_group is None or reference_group not in valid_groups:
            reference_group = group_counts[group_counts.index.isin(valid_groups)].index[0]
            
        target_groups = [g for g in valid_groups if g != reference_group]
        
        if not target_groups:
            return False, {'error': 'No valid target groups found'}
            
        print(f"    Groups: {len(valid_groups)} valid groups, ref={reference_group}, targets={len(target_groups)}")
        
        # Subsample data if needed
        adata_processed = self._subsample_data(adata, valid_groups, groupby_key)
        
        print(f"    Processed: {adata_processed.n_obs:,} cells × {adata_processed.n_vars:,} genes")
        
        # Run benchmarks
        results = {
            'dataset_name': dataset_name,
            'original_shape': (adata.n_obs, adata.n_vars),
            'processed_shape': (adata_processed.n_obs, adata_processed.n_vars),
            'n_groups': len(valid_groups),
            'n_target_groups': len(target_groups),
            'reference_group': reference_group,
            'target_groups': target_groups[:5]  # Limit for display
        }
        
        # Benchmark HPDEX
        try:
            print("    Running HPDEX...")
            hpdex_start = time.time()
            start_memory = self._get_memory_usage()
            
            hpdex_results = parallel_differential_expression(
                adata=adata_processed,
                groupby_key=groupby_key,
                reference=reference_group,
                num_workers=num_workers
            )
            
            hpdex_time = time.time() - hpdex_start
            hpdex_memory = self._get_memory_usage() - start_memory
            
            results['hpdex'] = {
                'time': hpdex_time,
                'memory_mb': hpdex_memory,
                'n_results': len(hpdex_results)
            }
            
            print(f"      HPDEX: {hpdex_time:.1f}s, {hpdex_memory:.0f}MB, {len(hpdex_results):,} results")
            
        except Exception as e:
            return False, {'error': f'HPDEX failed: {e}'}
            
        # Benchmark pdex
        try:
            print("    Running pdex...")
            pdex_start = time.time()
            start_memory = self._get_memory_usage()
            
            pdex_results = pdex_de(
                adata=adata_processed,
                groupby_key=groupby_key,
                reference=reference_group,
                num_workers=num_workers,
                batch_size=500
            )
            
            pdex_time = time.time() - pdex_start
            pdex_memory = self._get_memory_usage() - start_memory
            
            results['pdex'] = {
                'time': pdex_time,
                'memory_mb': pdex_memory,
                'n_results': len(pdex_results)
            }
            
            print(f"      pdex: {pdex_time:.1f}s, {pdex_memory:.0f}MB, {len(pdex_results):,} results")
            
        except Exception as e:
            return False, {'error': f'pdex failed: {e}'}
            
        # Compare results
        try:
            comparison = self._compare_results(hpdex_results, pdex_results)
            results.update(comparison)
            
            # Calculate performance metrics
            speedup = pdex_time / hpdex_time
            memory_ratio = hpdex_memory / pdex_memory if pdex_memory > 0 else 1.0
            
            results['speedup'] = float(speedup)
            results['memory_ratio'] = float(memory_ratio)
            
            print(f"      Speedup: {speedup:.2f}x, Memory ratio: {memory_ratio:.2f}x")
            print(f"      P-value correlation: {results.get('p_correlation', 0):.4f}")
            
            # Success criteria
            success = (
                results.get('p_correlation', 0) >= self.correlation_threshold and
                speedup > 0.1  # HPDEX shouldn't be more than 10x slower
            )
            
        except Exception as e:
            return False, {'error': f'Result comparison failed: {e}'}
            
        # Clean up
        del adata, adata_processed, hpdex_results, pdex_results
        gc.collect()
        
        return success, results
    
    def _subsample_data(self, adata: ad.AnnData, valid_groups: List[str], groupby_key: str) -> ad.AnnData:
        """Subsample data to meet size constraints."""
        
        # Filter to valid groups only
        mask = adata.obs[groupby_key].isin(valid_groups)
        adata_filtered = adata[mask].copy()
        
        # Subsample cells if needed
        if adata_filtered.n_obs > self.max_cells:
            # Sample proportionally from each group
            sampled_indices = []
            group_counts = adata_filtered.obs[groupby_key].value_counts()
            
            for group in valid_groups:
                group_mask = adata_filtered.obs[groupby_key] == group
                group_indices = np.where(group_mask)[0]
                
                if len(group_indices) > 0:
                    # Sample proportionally but ensure at least 10 cells per group
                    target_size = max(10, int(len(group_indices) * self.max_cells / adata_filtered.n_obs))
                    target_size = min(target_size, len(group_indices))
                    
                    sampled = np.random.choice(group_indices, size=target_size, replace=False)
                    sampled_indices.extend(sampled)
                    
            adata_filtered = adata_filtered[sampled_indices]
            
        # Subsample genes if needed (select most variable genes)
        if adata_filtered.n_vars > self.max_genes:
            # Calculate variance for gene selection
            if hasattr(adata_filtered.X, 'toarray'):
                X_dense = adata_filtered.X.toarray()
            else:
                X_dense = adata_filtered.X
                
            gene_vars = np.var(X_dense, axis=0)
            top_genes = np.argsort(gene_vars)[-self.max_genes:]
            adata_filtered = adata_filtered[:, top_genes]
            
        return adata_filtered
    
    def _compare_results(self, hpdex_results: pd.DataFrame, pdex_results: pd.DataFrame) -> Dict[str, Any]:
        """Compare HPDEX and pdex results for consistency."""
        
        # Standardize column names for comparison
        hpdex_clean = hpdex_results.copy()
        pdex_clean = pdex_results.copy()
        
        # Merge on target and feature
        try:
            merged = pd.merge(
                hpdex_clean[['target', 'feature', 'p_value']],
                pdex_clean[['target', 'feature', 'p_value']],
                on=['target', 'feature'],
                suffixes=('_hpdex', '_pdex')
            )
            
            if len(merged) == 0:
                return {'p_correlation': 0.0, 'n_common': 0, 'comparison_error': 'No common results'}
                
            # Calculate correlation
            p_correlation = np.corrcoef(merged['p_value_hpdex'], merged['p_value_pdex'])[0, 1]
            
            # Additional metrics
            p_diff = np.abs(merged['p_value_hpdex'] - merged['p_value_pdex'])
            max_p_diff = np.max(p_diff)
            mean_p_diff = np.mean(p_diff)
            
            return {
                'p_correlation': float(p_correlation),
                'n_common': len(merged),
                'max_p_diff': float(max_p_diff),
                'mean_p_diff': float(mean_p_diff)
            }
            
        except Exception as e:
            return {'p_correlation': 0.0, 'n_common': 0, 'comparison_error': str(e)}
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
