"""
Edge Case Tests

Tests HPDEX robustness with challenging scenarios.
"""

import numpy as np
import pandas as pd
import anndata as ad
from typing import Tuple, Dict, Any
from scipy.sparse import csr_matrix

# Import HPDEX
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hpdex import parallel_differential_expression


class EdgeCaseTester:
    """Tests HPDEX with edge cases and challenging scenarios."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def test_scenario(self, scenario: Dict[str, Any], data_generator) -> Tuple[bool, Dict[str, Any]]:
        """Test a specific edge case scenario."""
        
        scenario_name = scenario['name']
        
        if scenario_name == "tiny_groups":
            return self._test_tiny_groups(scenario, data_generator)
        elif scenario_name == "constant_genes":
            return self._test_constant_genes(scenario, data_generator)
        elif scenario_name == "sparse_data":
            return self._test_sparse_data(scenario, data_generator)
        elif scenario_name == "extreme_values":
            return self._test_extreme_values(scenario, data_generator)
        else:
            return False, {'error': f'Unknown scenario: {scenario_name}'}
    
    def _test_tiny_groups(self, scenario: Dict[str, Any], data_generator) -> Tuple[bool, Dict[str, Any]]:
        """Test with very small groups."""
        
        print(f"      Testing tiny groups...")
        
        # Generate data (remove name from scenario dict)
        scenario_params = {k: v for k, v in scenario.items() if k != 'name'}
        adata = data_generator.generate_edge_case_data("tiny_groups", **scenario_params)
        
        # Get group sizes
        group_counts = adata.obs['group'].value_counts()
        min_group_size = group_counts.min()
        
        try:
            # Run HPDEX with min_samples=1 to allow tiny groups
            results = parallel_differential_expression(
                adata=adata,
                groupby_key='group',
                reference='control',
                num_workers=1   # Use single worker for small data
            )
            
            success = len(results) > 0 and not any(pd.isna(results['p_value']))
            
            return success, {
                'n_cells': adata.n_obs,
                'n_genes': adata.n_vars,
                'group_counts': group_counts.to_dict(),
                'min_group_size': int(min_group_size),
                'n_results': len(results),
                'has_valid_pvalues': not any(pd.isna(results['p_value'])) if len(results) > 0 else False
            }
            
        except Exception as e:
            return False, {
                'error': str(e),
                'group_counts': group_counts.to_dict(),
                'min_group_size': int(min_group_size)
            }
    
    def _test_constant_genes(self, scenario: Dict[str, Any], data_generator) -> Tuple[bool, Dict[str, Any]]:
        """Test with genes that have constant expression."""
        
        print(f"      Testing constant genes...")
        
        # Generate data (remove name from scenario dict)
        scenario_params = {k: v for k, v in scenario.items() if k != 'name'}
        adata = data_generator.generate_edge_case_data("constant_genes", **scenario_params)
        
        # Identify constant genes
        gene_variances = np.var(adata.X, axis=0)
        if hasattr(adata.X, 'toarray'):
            gene_variances = np.var(adata.X.toarray(), axis=0)
        
        constant_genes = np.sum(gene_variances < 1e-10)
        
        try:
            # Run HPDEX
            results = parallel_differential_expression(
                adata=adata,
                groupby_key='group',
                reference='control',
                num_workers=1
            )
            
            # Check results for constant genes
            constant_gene_names = [f"gene_{i}" for i in np.where(gene_variances < 1e-10)[0]]
            constant_results = results[results['feature'].isin(constant_gene_names)]
            
            # For constant genes, p-values should be NaN or 1.0
            valid_constant_handling = all(
                pd.isna(p) or p >= 0.99 
                for p in constant_results['p_value']
            )
            
            success = len(results) > 0 and valid_constant_handling
            
            return success, {
                'n_cells': adata.n_obs,
                'n_genes': adata.n_vars,
                'n_constant_genes': int(constant_genes),
                'constant_fraction': float(constant_genes / adata.n_vars),
                'n_results': len(results),
                'n_constant_results': len(constant_results),
                'valid_constant_handling': valid_constant_handling
            }
            
        except Exception as e:
            return False, {
                'error': str(e),
                'n_constant_genes': int(constant_genes)
            }
    
    def _test_sparse_data(self, scenario: Dict[str, Any], data_generator) -> Tuple[bool, Dict[str, Any]]:
        """Test with highly sparse data."""
        
        print(f"      Testing sparse data...")
        
        # Generate sparse data (remove name from scenario dict)
        scenario_params = {k: v for k, v in scenario.items() if k != 'name'}
        adata = data_generator.generate_edge_case_data("sparse_data", **scenario_params)
        
        # Calculate sparsity
        if hasattr(adata.X, 'toarray'):
            data_array = adata.X.toarray()
            is_sparse = True
        else:
            data_array = adata.X
            is_sparse = False
        
        zero_fraction = np.sum(data_array == 0) / data_array.size
        
        try:
            # Run HPDEX
            results = parallel_differential_expression(
                adata=adata,
                groupby_key='group',
                reference='control',
                num_workers=1
            )
            
            success = len(results) > 0 and not any(pd.isna(results['p_value']))
            
            return success, {
                'n_cells': adata.n_obs,
                'n_genes': adata.n_vars,
                'zero_fraction': float(zero_fraction),
                'is_sparse_matrix': is_sparse,
                'n_results': len(results),
                'has_valid_pvalues': not any(pd.isna(results['p_value'])) if len(results) > 0 else False
            }
            
        except Exception as e:
            return False, {
                'error': str(e),
                'zero_fraction': float(zero_fraction),
                'is_sparse_matrix': is_sparse
            }
    
    def _test_extreme_values(self, scenario: Dict[str, Any], data_generator) -> Tuple[bool, Dict[str, Any]]:
        """Test with extreme values and outliers."""
        
        print(f"      Testing extreme values...")
        
        # Generate data with extreme values
        adata = data_generator.generate_edge_case_data("extreme_values", **scenario)
        
        # Analyze data characteristics
        if hasattr(adata.X, 'toarray'):
            data_array = adata.X.toarray()
        else:
            data_array = adata.X
        
        data_max = np.max(data_array)
        data_min = np.min(data_array)
        data_mean = np.mean(data_array)
        data_std = np.std(data_array)
        
        # Identify potential outliers (values > 3 std from mean)
        outlier_threshold = data_mean + 3 * data_std
        n_outliers = np.sum(data_array > outlier_threshold)
        
        try:
            # Run HPDEX
            results = parallel_differential_expression(
                adata=adata,
                groupby_key='group',
                reference='control',
                num_workers=1
            )
            
            # Check for reasonable results despite extreme values
            finite_p_values = np.isfinite(results['p_value'])
            valid_p_range = (results['p_value'] >= 0) & (results['p_value'] <= 1)
            
            success = (
                len(results) > 0 and 
                np.sum(finite_p_values) > len(results) * 0.8 and  # At least 80% finite
                np.sum(valid_p_range) > len(results) * 0.8       # At least 80% in valid range
            )
            
            return success, {
                'n_cells': adata.n_obs,
                'n_genes': adata.n_vars,
                'data_range': [float(data_min), float(data_max)],
                'data_mean': float(data_mean),
                'data_std': float(data_std),
                'n_outliers': int(n_outliers),
                'outlier_fraction': float(n_outliers / data_array.size),
                'n_results': len(results),
                'n_finite_pvalues': int(np.sum(finite_p_values)),
                'n_valid_pvalues': int(np.sum(valid_p_range))
            }
            
        except Exception as e:
            return False, {
                'error': str(e),
                'data_range': [float(data_min), float(data_max)],
                'n_outliers': int(n_outliers)
            }
    
    def test_parameter_edge_cases(self) -> Tuple[bool, Dict[str, Any]]:
        """Test edge cases in parameters."""
        
        print(f"      Testing parameter edge cases...")
        
        # Create simple test data
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 50))
        
        obs = pd.DataFrame({
            'group': ['control'] * 50 + ['treatment'] * 50
        })
        
        adata = ad.AnnData(X=X, obs=obs)
        
        edge_cases = []
        overall_success = True
        
        # Test 1: Single worker
        try:
            results = parallel_differential_expression(
                adata=adata,
                groupby_key='group',
                reference='control',
                num_workers=1
            )
            edge_cases.append({
                'case': 'single_worker',
                'success': len(results) > 0,
                'n_results': len(results)
            })
        except Exception as e:
            edge_cases.append({
                'case': 'single_worker',
                'success': False,
                'error': str(e)
            })
            overall_success = False
        
        # Test 2: Many workers (more than needed)
        try:
            results = parallel_differential_expression(
                adata=adata,
                groupby_key='group',
                reference='control',
                num_workers=16  # More workers than necessary
            )
            edge_cases.append({
                'case': 'many_workers',
                'success': len(results) > 0,
                'n_results': len(results)
            })
        except Exception as e:
            edge_cases.append({
                'case': 'many_workers',
                'success': False,
                'error': str(e)
            })
            overall_success = False
        
        # Test 3: Different metrics
        for metric in ['wilcoxon', 'wilcoxon-hist']:
            try:
                results = parallel_differential_expression(
                    adata=adata,
                    groupby_key='group',
                    reference='control',
                    metric=metric
                )
                edge_cases.append({
                    'case': f'metric_{metric}',
                    'success': len(results) > 0,
                    'n_results': len(results)
                })
            except Exception as e:
                edge_cases.append({
                    'case': f'metric_{metric}',
                    'success': False,
                    'error': str(e)
                })
                overall_success = False
        
        return overall_success, {'edge_cases': edge_cases}
