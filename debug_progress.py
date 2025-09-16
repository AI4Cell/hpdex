#!/usr/bin/env python3
"""Debug script for progress bar nullptr issue."""

import numpy as np
import scipy.sparse
import anndata as ad
from src.hpdex.de import parallel_differential_expression

def test_with_progress():
    """Test differential expression with progress bar enabled."""
    print("=== Creating test data ===")
    
    # Create minimal test case
    n_cells = 100
    n_genes = 10
    
    # Create sparse matrix
    np.random.seed(42)
    X = np.random.poisson(2, size=(n_cells, n_genes)).astype(np.float32)
    X[X < 1] = 0
    
    # Create anndata
    adata = ad.AnnData(X)
    adata.obs['group'] = ['control'] * 50 + ['treatment'] * 50
    adata.var_names = [f'gene_{i}' for i in range(n_genes)]
    
    print(f"Data shape: {adata.shape}")
    print(f"Groups: {adata.obs['group'].value_counts()}")
    
    # Test WITHOUT progress bar first
    print("\n=== Testing without progress bar ===")
    try:
        result_no_progress = parallel_differential_expression(
            adata,
            groupby_key='group',
            reference='control',
            groups=['treatment'],
            show_progress=False,
            threads=1
        )
        print("✅ No progress bar: SUCCESS")
        print(f"Result shape: {result_no_progress.shape}")
    except Exception as e:
        print(f"❌ No progress bar: FAILED with {e}")
        return
    
    # Test WITH progress bar
    print("\n=== Testing with progress bar ===")
    try:
        result_with_progress = parallel_differential_expression(
            adata,
            groupby_key='group',
            reference='control',
            groups=['treatment'],
            show_progress=True,
            threads=1
        )
        print("✅ With progress bar: SUCCESS")
        print(f"Result shape: {result_with_progress.shape}")
    except Exception as e:
        print(f"❌ With progress bar: FAILED with {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_progress()
