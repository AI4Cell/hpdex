"""
Pipeline Consistency Tests

Tests HPDEX pipeline against pdex for consistency validation.
"""

# Import libraries
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import HPDEX
from hpdex import parallel_differential_expression

# Import pdex for comparison
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pdex" / "src"))
try:
    from pdex._single_cell import parallel_differential_expression as pdex_de
except ImportError:
    # Fallback if pdex not available
    pdex_de = None


class PipelineConsistencyTester:
    """Tests HPDEX pipeline against pdex for consistency."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        if pdex_de is None:
            print("⚠️ Warning: pdex not available, pipeline consistency tests will be skipped")
    
    def test_against_pdex(
        self,
        adata: ad.AnnData,
        correlation_threshold: float = 0.95,
        pdex_batch_size: int = 500,
        num_workers: int = 4
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Test HPDEX pipeline against pdex.
        
        Args:
            adata: Test dataset
            correlation_threshold: Minimum correlation for p-values
            pdex_batch_size: Batch size for pdex (larger to avoid bottleneck)
            num_workers: Number of workers
            
        Returns:
            (success, results_dict)
        """
        if pdex_de is None:
            return False, {'error': 'pdex not available for comparison'}
        
        # Get unique groups
        unique_groups = adata.obs['group'].unique()
        reference = 'control'
        target_groups = [g for g in unique_groups if g != reference]
        
        if len(target_groups) == 0:
            return False, {'error': 'No target groups found (need groups other than control)'}
        
        print(f"  Testing {len(target_groups)} target groups vs {reference}")
        print(f"  Dataset: {adata.n_obs} cells × {adata.n_vars} genes")
        
        # Run HPDEX
        try:
            print("  Running HPDEX...")
            hpdex_results = parallel_differential_expression(
                adata=adata,
                groupby_key='group',
                reference=reference,
                num_workers=num_workers,
                metric='wilcoxon'
            )
            
            # Sort for consistent comparison
            hpdex_results = hpdex_results.sort_values(['target', 'feature']).reset_index(drop=True)
            
        except Exception as e:
            return False, {'error': f'HPDEX failed: {e}'}
        
        # Run pdex
        try:
            print("  Running pdex...")
            pdex_results = pdex_de(
                adata=adata,
                groups=target_groups,
                reference=reference,
                groupby_key='group',
                num_workers=num_workers,
                batch_size=pdex_batch_size,
                metric='wilcoxon',
                tie_correct=True
            )
            
            # Sort for consistent comparison
            pdex_results = pdex_results.sort_values(['target', 'feature']).reset_index(drop=True)
            
        except Exception as e:
            return False, {'error': f'pdex failed: {e}'}
        
        # Align results for comparison
        try:
            aligned_results = self._align_results(hpdex_results, pdex_results)
            if aligned_results is None:
                return False, {'error': 'Could not align results between HPDEX and pdex'}
            
            hpdex_aligned, pdex_aligned = aligned_results
            
        except Exception as e:
            return False, {'error': f'Result alignment failed: {e}'}
        
        # Compare results
        comparison_results = self._compare_results(
            hpdex_aligned, pdex_aligned, correlation_threshold
        )
        
        success = comparison_results['correlations']['p_value'] >= correlation_threshold
        
        return success, comparison_results
    
    def _align_results(self, hpdex_df: pd.DataFrame, pdex_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align results from HPDEX and pdex for comparison."""
        
        # Check column names and rename if necessary
        hpdex_cols = set(hpdex_df.columns)
        pdex_cols = set(pdex_df.columns)
        
        # Expected columns
        required_cols = {'target', 'feature', 'p_value'}
        
        if not required_cols.issubset(hpdex_cols):
            missing = required_cols - hpdex_cols
            raise ValueError(f"HPDEX results missing columns: {missing}")
        
        if not required_cols.issubset(pdex_cols):
            missing = required_cols - pdex_cols
            raise ValueError(f"pdex results missing columns: {missing}")
        
        # Create merge keys
        hpdex_df['merge_key'] = hpdex_df['target'] + '|' + hpdex_df['feature']
        pdex_df['merge_key'] = pdex_df['target'] + '|' + pdex_df['feature']
        
        # Find common comparisons
        common_keys = set(hpdex_df['merge_key']) & set(pdex_df['merge_key'])
        
        if len(common_keys) == 0:
            print(f"  No common comparisons found!")
            print(f"  HPDEX keys sample: {list(hpdex_df['merge_key'].head())}")
            print(f"  pdex keys sample: {list(pdex_df['merge_key'].head())}")
            return None
        
        print(f"  Found {len(common_keys)} common comparisons")
        
        # Filter to common comparisons
        hpdex_common = hpdex_df[hpdex_df['merge_key'].isin(common_keys)].copy()
        pdex_common = pdex_df[pdex_df['merge_key'].isin(common_keys)].copy()
        
        # Sort by merge key for alignment
        hpdex_common = hpdex_common.sort_values('merge_key').reset_index(drop=True)
        pdex_common = pdex_common.sort_values('merge_key').reset_index(drop=True)
        
        return hpdex_common, pdex_common
    
    def _compare_results(
        self, 
        hpdex_df: pd.DataFrame, 
        pdex_df: pd.DataFrame,
        correlation_threshold: float
    ) -> Dict[str, Any]:
        """Compare aligned results between HPDEX and pdex."""
        
        n_comparisons = len(hpdex_df)
        
        # Extract values for comparison
        comparisons = {}
        correlations = {}
        
        # Compare p-values (most important)
        hpdex_p = hpdex_df['p_value'].values
        pdex_p = pdex_df['p_value'].values
        
        # Remove any NaN or infinite values
        valid_mask = (
            np.isfinite(hpdex_p) & np.isfinite(pdex_p) &
            (hpdex_p > 0) & (pdex_p > 0) &
            (hpdex_p <= 1) & (pdex_p <= 1)
        )
        
        if np.sum(valid_mask) < len(hpdex_p) * 0.9:
            print(f"  Warning: Only {np.sum(valid_mask)}/{len(hpdex_p)} comparisons have valid p-values")
        
        hpdex_p_valid = hpdex_p[valid_mask]
        pdex_p_valid = pdex_p[valid_mask]
        
        if len(hpdex_p_valid) == 0:
            return {
                'error': 'No valid p-values for comparison',
                'n_comparisons': n_comparisons,
                'n_valid': 0
            }
        
        # Calculate correlations
        p_pearson, p_pearson_pval = pearsonr(hpdex_p_valid, pdex_p_valid)
        p_spearman, p_spearman_pval = spearmanr(hpdex_p_valid, pdex_p_valid)
        
        correlations['p_value'] = float(p_pearson)
        correlations['p_value_spearman'] = float(p_spearman)
        
        # Compare on log scale (more meaningful for p-values)
        log_hpdex_p = -np.log10(hpdex_p_valid)
        log_pdex_p = -np.log10(pdex_p_valid)
        
        log_p_pearson, _ = pearsonr(log_hpdex_p, log_pdex_p)
        correlations['log_p_value'] = float(log_p_pearson)
        
        # Compare fold changes if available
        if 'fold_change' in hpdex_df.columns and 'fold_change' in pdex_df.columns:
            hpdex_fc = hpdex_df['fold_change'].values[valid_mask]
            pdex_fc = pdex_df['fold_change'].values[valid_mask]
            
            # Remove infinite values
            fc_valid_mask = np.isfinite(hpdex_fc) & np.isfinite(pdex_fc)
            if np.sum(fc_valid_mask) > 0:
                fc_pearson, _ = pearsonr(hpdex_fc[fc_valid_mask], pdex_fc[fc_valid_mask])
                correlations['fold_change'] = float(fc_pearson)
        
        # Calculate error statistics
        p_errors = np.abs(hpdex_p_valid - pdex_p_valid)
        
        error_stats = {
            'max_p_error': float(np.max(p_errors)),
            'mean_p_error': float(np.mean(p_errors)),
            'median_p_error': float(np.median(p_errors)),
            'p95_p_error': float(np.percentile(p_errors, 95))
        }
        
        # Identify significant differences
        large_error_threshold = 0.01  # 1% p-value difference
        large_errors = p_errors > large_error_threshold
        n_large_errors = np.sum(large_errors)
        
        # Sample some comparisons for debugging
        sample_size = min(10, len(hpdex_p_valid))
        sample_indices = np.random.choice(len(hpdex_p_valid), sample_size, replace=False)
        
        sample_comparisons = []
        for i in sample_indices:
            sample_comparisons.append({
                'hpdex_p': float(hpdex_p_valid[i]),
                'pdex_p': float(pdex_p_valid[i]),
                'p_error': float(p_errors[i]),
                'log_hpdex_p': float(log_hpdex_p[i]),
                'log_pdex_p': float(log_pdex_p[i])
            })
        
        # Overall assessment
        success = correlations['p_value'] >= correlation_threshold
        
        return {
            'success': success,
            'n_comparisons': n_comparisons,
            'n_valid': len(hpdex_p_valid),
            'correlations': correlations,
            'error_stats': error_stats,
            'n_large_errors': int(n_large_errors),
            'large_error_threshold': large_error_threshold,
            'correlation_threshold': correlation_threshold,
            'sample_comparisons': sample_comparisons
        }
    
    def test_parameter_consistency(self, adata: ad.AnnData) -> Tuple[bool, Dict[str, Any]]:
        """Test consistency across different parameters."""
        
        if pdex_de is None:
            return False, {'error': 'pdex not available'}
        
        # Test different tie correction settings
        results = {}
        overall_success = True
        
        reference = 'control'
        target_groups = [g for g in adata.obs['group'].unique() if g != reference]
        
        parameter_sets = [
            {'tie_correct': True, 'metric': 'wilcoxon'},
            {'tie_correct': False, 'metric': 'wilcoxon'}
        ]
        
        for i, params in enumerate(parameter_sets):
            try:
                # HPDEX (tie correction always enabled in current implementation)
                hpdex_results = parallel_differential_expression(
                    adata=adata,
                    groupby_key='group',
                    reference=reference,
                    metric='wilcoxon'
                )
                
                # pdex
                pdex_results = pdex_de(
                    adata=adata,
                    groups=target_groups,
                    reference=reference,
                    groupby_key='group',
                    metric=params['metric'],
                    tie_correct=params['tie_correct']
                )
                
                # Compare
                aligned = self._align_results(hpdex_results, pdex_results)
                if aligned is not None:
                    hpdex_aligned, pdex_aligned = aligned
                    comparison = self._compare_results(hpdex_aligned, pdex_aligned, 0.8)
                    
                    results[f'param_set_{i}'] = {
                        'parameters': params,
                        'comparison': comparison
                    }
                else:
                    results[f'param_set_{i}'] = {'error': 'Could not align results'}
                    overall_success = False
                    
            except Exception as e:
                results[f'param_set_{i}'] = {'error': str(e)}
                overall_success = False
        
        return overall_success, results
