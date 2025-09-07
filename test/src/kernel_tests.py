"""
Kernel Consistency Tests

Tests HPDEX kernels against scipy for correctness validation.
"""

import numpy as np
import scipy.stats as stats
from typing import Tuple, Dict, Any
from tqdm import tqdm

# Import HPDEX kernels
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hpdex.kernel import rank_sum_chunk_kernel_float, rank_sum_chunk_kernel_hist


class KernelConsistencyTester:
    """Tests HPDEX kernels against scipy for correctness."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def test_float_kernel(
        self, 
        ref_data: np.ndarray, 
        tar_data: np.ndarray, 
        tolerance: float = 1e-10
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Test float kernel against scipy.mannwhitneyu.
        
        Args:
            ref_data: Reference data (n_genes, n_ref)
            tar_data: Target data (n_genes, n_tar)
            tolerance: Tolerance for numerical comparison
            
        Returns:
            (success, results_dict)
        """
        n_genes = ref_data.shape[0]
        
        # Prepare data for HPDEX kernel (expects sorted data)
        ref_sorted = np.sort(ref_data, axis=1)
        tar_sorted = np.sort(tar_data, axis=1)
        
        # Run HPDEX kernel
        try:
            hpdex_p, hpdex_u = rank_sum_chunk_kernel_float(
                ref_sorted, tar_sorted,
                tie_correction=True,
                continuity_correction=True,
                use_asymptotic=True
            )
        except Exception as e:
            return False, {'error': f"HPDEX kernel failed: {e}"}
        
        # Compare against scipy for each gene
        scipy_p_values = []
        scipy_u_stats = []
        p_value_errors = []
        u_stat_errors = []
        
        for i in range(n_genes):
            try:
                # Scipy Mann-Whitney U test
                scipy_stat, scipy_p = stats.mannwhitneyu(
                    tar_data[i], ref_data[i],
                    alternative='two-sided',
                    use_continuity=True
                )
                
                scipy_p_values.append(scipy_p)
                scipy_u_stats.append(scipy_stat)
                
                # Calculate errors
                p_error = abs(hpdex_p[i] - scipy_p)
                u_error = abs(hpdex_u[i] - scipy_stat)
                
                p_value_errors.append(p_error)
                u_stat_errors.append(u_error)
                
            except Exception as e:
                return False, {'error': f"Scipy failed on gene {i}: {e}"}
        
        # Analyze results
        max_p_error = max(p_value_errors)
        max_u_error = max(u_stat_errors)
        mean_p_error = np.mean(p_value_errors)
        mean_u_error = np.mean(u_stat_errors)
        
        # Check if within tolerance
        p_values_match = max_p_error < tolerance
        u_stats_match = max_u_error < tolerance * 1000  # U stats have larger scale
        
        success = p_values_match and u_stats_match
        
        results = {
            'n_genes': n_genes,
            'n_ref': ref_data.shape[1],
            'n_tar': tar_data.shape[1],
            'max_p_error': max_p_error,
            'max_u_error': max_u_error,
            'mean_p_error': mean_p_error,
            'mean_u_error': mean_u_error,
            'tolerance': tolerance,
            'p_values_match': p_values_match,
            'u_stats_match': u_stats_match,
            'correlation_p': np.corrcoef(hpdex_p, scipy_p_values)[0, 1],
            'correlation_u': np.corrcoef(hpdex_u, scipy_u_stats)[0, 1]
        }
        
        # Add sample comparisons for debugging
        if n_genes >= 5:
            results['sample_comparisons'] = []
            for i in range(min(5, n_genes)):
                results['sample_comparisons'].append({
                    'gene': i,
                    'hpdex_p': float(hpdex_p[i]),
                    'scipy_p': float(scipy_p_values[i]),
                    'p_error': float(p_value_errors[i]),
                    'hpdex_u': float(hpdex_u[i]),
                    'scipy_u': float(scipy_u_stats[i]),
                    'u_error': float(u_stat_errors[i])
                })
        
        return success, results
    
    def test_hist_kernel(
        self, 
        ref_data: np.ndarray, 
        tar_data: np.ndarray, 
        tolerance: float = 1e-10,
        max_bins: int = 1000
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Test histogram kernel against scipy.mannwhitneyu.
        
        Args:
            ref_data: Reference data (n_genes, n_ref) - integer values
            tar_data: Target data (n_genes, n_tar) - integer values
            tolerance: Tolerance for numerical comparison
            max_bins: Maximum number of bins for histogram
            
        Returns:
            (success, results_dict)
        """
        n_genes = ref_data.shape[0]
        
        # Ensure integer data
        ref_data = ref_data.astype(np.int64)
        tar_data = tar_data.astype(np.int64)
        
        # Run HPDEX histogram kernel
        try:
            hpdex_p, hpdex_u = rank_sum_chunk_kernel_hist(
                ref_data, tar_data,
                tie_correction=True,
                continuity_correction=True,
                use_asymptotic=True,
                max_bins=max_bins
            )
        except Exception as e:
            return False, {'error': f"HPDEX histogram kernel failed: {e}"}
        
        # Compare against scipy for each gene
        scipy_p_values = []
        scipy_u_stats = []
        p_value_errors = []
        u_stat_errors = []
        
        for i in range(n_genes):
            try:
                # Scipy Mann-Whitney U test
                scipy_stat, scipy_p = stats.mannwhitneyu(
                    tar_data[i], ref_data[i],
                    alternative='two-sided',
                    use_continuity=True
                )
                
                scipy_p_values.append(scipy_p)
                scipy_u_stats.append(scipy_stat)
                
                # Calculate errors
                p_error = abs(hpdex_p[i] - scipy_p)
                u_error = abs(hpdex_u[i] - scipy_stat)
                
                p_value_errors.append(p_error)
                u_stat_errors.append(u_error)
                
            except Exception as e:
                return False, {'error': f"Scipy failed on gene {i}: {e}"}
        
        # Analyze results
        max_p_error = max(p_value_errors)
        max_u_error = max(u_stat_errors)
        mean_p_error = np.mean(p_value_errors)
        mean_u_error = np.mean(u_stat_errors)
        
        # Check if within tolerance (histogram kernel may have slightly different precision)
        p_values_match = max_p_error < tolerance * 10  # Slightly more lenient for hist
        u_stats_match = max_u_error < tolerance * 10000
        
        success = p_values_match and u_stats_match
        
        results = {
            'n_genes': n_genes,
            'n_ref': ref_data.shape[1],
            'n_tar': tar_data.shape[1],
            'max_bins': max_bins,
            'max_p_error': max_p_error,
            'max_u_error': max_u_error,
            'mean_p_error': mean_p_error,
            'mean_u_error': mean_u_error,
            'tolerance': tolerance,
            'p_values_match': p_values_match,
            'u_stats_match': u_stats_match,
            'correlation_p': np.corrcoef(hpdex_p, scipy_p_values)[0, 1],
            'correlation_u': np.corrcoef(hpdex_u, scipy_u_stats)[0, 1]
        }
        
        # Add sample comparisons for debugging
        if n_genes >= 5:
            results['sample_comparisons'] = []
            for i in range(min(5, n_genes)):
                results['sample_comparisons'].append({
                    'gene': i,
                    'hpdex_p': float(hpdex_p[i]),
                    'scipy_p': float(scipy_p_values[i]),
                    'p_error': float(p_value_errors[i]),
                    'hpdex_u': float(hpdex_u[i]),
                    'scipy_u': float(scipy_u_stats[i]),
                    'u_error': float(u_stat_errors[i])
                })
        
        return success, results
    
    def test_kernel_edge_cases(self) -> Tuple[bool, Dict[str, Any]]:
        """Test kernels with edge cases."""
        edge_cases = [
            {
                'name': 'identical_data',
                'ref': np.ones((5, 10)),
                'tar': np.ones((5, 10))
            },
            {
                'name': 'zero_variance',
                'ref': np.zeros((5, 10)),
                'tar': np.ones((5, 10))
            },
            {
                'name': 'single_sample',
                'ref': np.random.normal(0, 1, (5, 1)),
                'tar': np.random.normal(1, 1, (5, 1))
            }
        ]
        
        results = []
        overall_success = True
        
        for case in edge_cases:
            try:
                # Test float kernel
                ref_sorted = np.sort(case['ref'], axis=1)
                tar_sorted = np.sort(case['tar'], axis=1)
                
                hpdex_p, hpdex_u = rank_sum_chunk_kernel_float(
                    ref_sorted, tar_sorted,
                    tie_correction=True,
                    continuity_correction=True
                )
                
                # Check for valid outputs
                valid = (
                    not np.any(np.isnan(hpdex_p)) and
                    not np.any(np.isnan(hpdex_u)) and
                    np.all(hpdex_p >= 0) and
                    np.all(hpdex_p <= 1)
                )
                
                results.append({
                    'case': case['name'],
                    'success': valid,
                    'p_range': [float(np.min(hpdex_p)), float(np.max(hpdex_p))],
                    'has_nan': bool(np.any(np.isnan(hpdex_p)))
                })
                
                overall_success &= valid
                
            except Exception as e:
                results.append({
                    'case': case['name'],
                    'success': False,
                    'error': str(e)
                })
                overall_success = False
        
        return overall_success, {'edge_case_results': results}
