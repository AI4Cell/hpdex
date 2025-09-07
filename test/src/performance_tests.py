"""
Performance Tests

Benchmarks HPDEX against scipy and pdex for performance validation.
"""

import time
import tempfile
import psutil
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import mannwhitneyu

# Import libraries
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import HPDEX
from hpdex import parallel_differential_expression, parallel_differential_expression_stream
from hpdex.kernel import rank_sum_chunk_kernel_float

# Import pdex for comparison
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pdex" / "src"))
try:
    from pdex._single_cell import parallel_differential_expression as pdex_de
except ImportError:
    pdex_de = None


class PerformanceTester:
    """Benchmarks HPDEX performance against reference implementations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.process = psutil.Process(os.getpid())
    
    def benchmark_kernels(
        self,
        ref_data: np.ndarray,
        tar_data: np.ndarray,
        warmup_runs: int = 2,
        benchmark_runs: int = 5,
        timeout_seconds: int = 300
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Benchmark HPDEX kernels against scipy.
        
        Args:
            ref_data: Reference data (n_genes, n_ref)
            tar_data: Target data (n_genes, n_tar)
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs
            timeout_seconds: Timeout for each test
            
        Returns:
            (success, results_dict)
        """
        n_genes, n_ref = ref_data.shape
        n_tar = tar_data.shape[1]
        
        print(f"    Kernel benchmark: {n_genes} genes, {n_ref}×{n_tar} samples")
        
        results = {
            'n_genes': n_genes,
            'n_ref': n_ref,
            'n_tar': n_tar,
            'warmup_runs': warmup_runs,
            'benchmark_runs': benchmark_runs
        }
        
        # Test HPDEX vectorized kernel
        try:
            print(f"      Testing HPDEX kernel...")
            hpdex_times = self._benchmark_hpdex_kernel(
                ref_data, tar_data, warmup_runs, benchmark_runs, timeout_seconds
            )
            results['hpdex'] = {
                'times': hpdex_times,
                'mean_time': np.mean(hpdex_times),
                'std_time': np.std(hpdex_times),
                'min_time': np.min(hpdex_times)
            }
        except Exception as e:
            return False, {'error': f'HPDEX kernel benchmark failed: {e}'}
        
        # Test scipy baseline (gene-by-gene)
        try:
            print(f"      Testing scipy baseline...")
            scipy_times = self._benchmark_scipy_kernel(
                ref_data, tar_data, warmup_runs, benchmark_runs, timeout_seconds
            )
            results['scipy'] = {
                'times': scipy_times,
                'mean_time': np.mean(scipy_times),
                'std_time': np.std(scipy_times),
                'min_time': np.min(scipy_times)
            }
        except Exception as e:
            return False, {'error': f'scipy kernel benchmark failed: {e}'}
        
        # Calculate speedup
        hpdex_mean = np.mean(hpdex_times)
        scipy_mean = np.mean(scipy_times)
        speedup = scipy_mean / hpdex_mean
        
        results['speedup'] = float(speedup)
        results['hpdex_faster'] = speedup > 1.0
        
        # Performance assessment
        success = speedup > 0.5  # HPDEX should be at least competitive
        
        print(f"      HPDEX: {hpdex_mean:.3f}s, scipy: {scipy_mean:.3f}s, speedup: {speedup:.2f}x")
        
        return success, results
    
    def benchmark_pipeline(
        self,
        adata: ad.AnnData,
        num_workers: int = 4,
        warmup_runs: int = 1,
        benchmark_runs: int = 3,
        timeout_seconds: int = 600
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Benchmark HPDEX pipeline against pdex.
        
        Args:
            adata: Test dataset
            num_workers: Number of workers
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs
            timeout_seconds: Timeout for each test
            
        Returns:
            (success, results_dict)
        """
        if pdex_de is None:
            return False, {'error': 'pdex not available for benchmarking'}
        
        print(f"    Pipeline benchmark: {adata.n_obs} cells × {adata.n_vars} genes")
        
        # Get groups
        unique_groups = adata.obs['group'].unique()
        reference = 'control'
        target_groups = [g for g in unique_groups if g != reference]
        
        results = {
            'n_cells': adata.n_obs,
            'n_genes': adata.n_vars,
            'n_groups': len(target_groups),
            'num_workers': num_workers,
            'warmup_runs': warmup_runs,
            'benchmark_runs': benchmark_runs
        }
        
        # Benchmark HPDEX
        try:
            print(f"      Testing HPDEX pipeline...")
            hpdex_times = self._benchmark_hpdex_pipeline(
                adata, reference, num_workers,
                warmup_runs, benchmark_runs, timeout_seconds
            )
            results['hpdex'] = {
                'times': hpdex_times,
                'mean_time': np.mean(hpdex_times),
                'std_time': np.std(hpdex_times),
                'min_time': np.min(hpdex_times)
            }
        except Exception as e:
            return False, {'error': f'HPDEX pipeline benchmark failed: {e}'}
        
        # Benchmark pdex
        try:
            print(f"      Testing pdex pipeline...")
            pdex_times = self._benchmark_pdex_pipeline(
                adata, reference, target_groups, num_workers,
                warmup_runs, benchmark_runs, timeout_seconds
            )
            results['pdex'] = {
                'times': pdex_times,
                'mean_time': np.mean(pdex_times),
                'std_time': np.std(pdex_times),
                'min_time': np.min(pdex_times)
            }
        except Exception as e:
            return False, {'error': f'pdex pipeline benchmark failed: {e}'}
        
        # Calculate speedup
        hpdex_mean = np.mean(hpdex_times)
        pdex_mean = np.mean(pdex_times)
        speedup = pdex_mean / hpdex_mean
        
        results['speedup'] = float(speedup)
        results['hpdex_faster'] = speedup > 1.0
        
        # Performance assessment
        success = speedup > 0.2  # HPDEX should be reasonably competitive
        
        print(f"      HPDEX: {hpdex_mean:.1f}s, pdex: {pdex_mean:.1f}s, speedup: {speedup:.2f}x")
        
        return success, results
    
    def benchmark_streaming(
        self,
        adata: ad.AnnData,
        memory_limit_gb: float = 4.0,
        num_workers: int = 4,
        timeout_seconds: int = 900
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Benchmark streaming vs regular HPDEX.
        
        Args:
            adata: Test dataset  
            memory_limit_gb: Memory limit for streaming
            num_workers: Number of workers
            timeout_seconds: Timeout for each test
            
        Returns:
            (success, results_dict)
        """
        print(f"    Streaming benchmark: {adata.n_obs} cells × {adata.n_vars} genes")
        print(f"      Memory limit: {memory_limit_gb} GB")
        
        # Get groups
        unique_groups = adata.obs['group'].unique()
        reference = 'control'
        target_groups = [g for g in unique_groups if g != reference]
        
        results = {
            'n_cells': adata.n_obs,
            'n_genes': adata.n_vars,
            'n_groups': len(target_groups),
            'memory_limit_gb': memory_limit_gb,
            'num_workers': num_workers
        }
        
        # Test regular HPDEX
        try:
            print(f"      Testing regular HPDEX...")
            start_memory = self._get_memory_usage()
            
            start_time = time.time()
            regular_results = parallel_differential_expression(
                adata=adata,
                groupby_key='group',
                reference=reference,
                num_workers=num_workers
            )
            regular_time = time.time() - start_time
            
            peak_memory = self._get_memory_usage()
            regular_memory = peak_memory - start_memory
            
            results['regular'] = {
                'time': regular_time,
                'memory_mb': regular_memory,
                'n_results': len(regular_results)
            }
            
        except Exception as e:
            return False, {'error': f'Regular HPDEX failed: {e}'}
        
        # Test streaming HPDEX
        try:
            print(f"      Testing streaming HPDEX...")
            
            # Save dataset to temporary file
            with tempfile.TemporaryDirectory() as tmp_dir:
                data_path = Path(tmp_dir) / "test_data.h5ad"
                adata.write_h5ad(data_path)
                
                start_memory = self._get_memory_usage()
                
                start_time = time.time()
                streaming_results = parallel_differential_expression_stream(
                    data_path=data_path,
                    groupby_key='group',
                    reference=reference,
                    memory_limit_gb=memory_limit_gb,
                    num_workers=num_workers
                )
                streaming_time = time.time() - start_time
                
                peak_memory = self._get_memory_usage()
                streaming_memory = peak_memory - start_memory
            
            results['streaming'] = {
                'time': streaming_time,
                'memory_mb': streaming_memory,
                'n_results': len(streaming_results)
            }
            
        except Exception as e:
            return False, {'error': f'Streaming HPDEX failed: {e}'}
        
        # Calculate metrics
        time_ratio = streaming_time / regular_time
        memory_ratio = streaming_memory / regular_memory if regular_memory > 0 else 1.0
        
        results['time_ratio'] = float(time_ratio)
        results['memory_ratio'] = float(memory_ratio)
        results['memory_savings'] = float(1 - memory_ratio)
        
        # Check result consistency
        try:
            # Sort both results for comparison
            regular_sorted = regular_results.sort_values(['target', 'feature']).reset_index(drop=True)
            streaming_sorted = streaming_results.sort_values(['target', 'feature']).reset_index(drop=True)
            
            # Compare p-values
            if len(regular_sorted) == len(streaming_sorted):
                p_correlation = np.corrcoef(
                    regular_sorted['p_value'], 
                    streaming_sorted['p_value']
                )[0, 1]
                results['p_correlation'] = float(p_correlation)
                results['results_consistent'] = p_correlation > 0.99
            else:
                results['results_consistent'] = False
                results['p_correlation'] = 0.0
                
        except Exception as e:
            results['results_consistent'] = False
            results['consistency_error'] = str(e)
        
        # Performance assessment
        success = (
            time_ratio < 3.0 and  # Streaming shouldn't be more than 3x slower
            (memory_ratio < 1.5 or abs(regular_memory) < 10) and  # Should be memory-efficient (allow some measurement noise)
            results.get('results_consistent', False)  # Results should be consistent
        )
        
        print(f"      Regular: {regular_time:.1f}s, {regular_memory:.0f}MB")
        print(f"      Streaming: {streaming_time:.1f}s, {streaming_memory:.0f}MB")
        print(f"      Time ratio: {time_ratio:.2f}, Memory ratio: {memory_ratio:.2f}")
        
        return success, results
    
    def _benchmark_hpdex_kernel(
        self, ref_data: np.ndarray, tar_data: np.ndarray,
        warmup_runs: int, benchmark_runs: int, timeout_seconds: int
    ) -> List[float]:
        """Benchmark HPDEX kernel performance."""
        
        # Prepare data (pre-sort for float kernel)
        ref_sorted = np.sort(ref_data, axis=1)
        tar_sorted = np.sort(tar_data, axis=1)
        
        # Warmup
        for _ in range(warmup_runs):
            _ = rank_sum_chunk_kernel_float(
                ref_sorted, tar_sorted,
                tie_correction=True,
                continuity_correction=True,
                use_asymptotic=True
            )
        
        # Benchmark
        times = []
        for _ in range(benchmark_runs):
            start_time = time.time()
            _ = rank_sum_chunk_kernel_float(
                ref_sorted, tar_sorted,
                tie_correction=True,
                continuity_correction=True,
                use_asymptotic=True
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            if elapsed > timeout_seconds:
                raise TimeoutError(f"HPDEX kernel benchmark exceeded {timeout_seconds}s")
        
        return times
    
    def _benchmark_scipy_kernel(
        self, ref_data: np.ndarray, tar_data: np.ndarray,
        warmup_runs: int, benchmark_runs: int, timeout_seconds: int
    ) -> List[float]:
        """Benchmark scipy kernel performance (gene-by-gene)."""
        
        n_genes = ref_data.shape[0]
        
        # Warmup
        for _ in range(warmup_runs):
            for i in range(min(10, n_genes)):  # Just sample for warmup
                _ = mannwhitneyu(tar_data[i], ref_data[i], alternative='two-sided')
        
        # Benchmark
        times = []
        for _ in range(benchmark_runs):
            start_time = time.time()
            
            for i in range(n_genes):
                _ = mannwhitneyu(tar_data[i], ref_data[i], alternative='two-sided')
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            if elapsed > timeout_seconds:
                raise TimeoutError(f"scipy kernel benchmark exceeded {timeout_seconds}s")
        
        return times
    
    def _benchmark_hpdex_pipeline(
        self, adata: ad.AnnData, reference: str,
        num_workers: int, warmup_runs: int, benchmark_runs: int, timeout_seconds: int
    ) -> List[float]:
        """Benchmark HPDEX pipeline performance."""
        
        # Warmup
        for _ in range(warmup_runs):
            _ = parallel_differential_expression(
                adata=adata,
                groupby_key='group',
                reference=reference,
                num_workers=num_workers
            )
        
        # Benchmark
        times = []
        for _ in range(benchmark_runs):
            start_time = time.time()
            _ = parallel_differential_expression(
                adata=adata,
                groupby_key='group',
                reference=reference,
                num_workers=num_workers
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            if elapsed > timeout_seconds:
                raise TimeoutError(f"HPDEX pipeline benchmark exceeded {timeout_seconds}s")
        
        return times
    
    def _benchmark_pdex_pipeline(
        self, adata: ad.AnnData, reference: str, target_groups: List[str],
        num_workers: int, warmup_runs: int, benchmark_runs: int, timeout_seconds: int
    ) -> List[float]:
        """Benchmark pdex pipeline performance."""
        
        # Warmup
        for _ in range(warmup_runs):
            _ = pdex_de(
                adata=adata,
                groups=target_groups,
                reference=reference,
                groupby_key='group',
                num_workers=num_workers,
                batch_size=500  # Large batch size to avoid bottleneck
            )
        
        # Benchmark
        times = []
        for _ in range(benchmark_runs):
            start_time = time.time()
            _ = pdex_de(
                adata=adata,
                groups=target_groups,
                reference=reference,
                groupby_key='group',
                num_workers=num_workers,
                batch_size=500
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            if elapsed > timeout_seconds:
                raise TimeoutError(f"pdex pipeline benchmark exceeded {timeout_seconds}s")
        
        return times
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
