"""
Pytest configuration file for hpdex testing.

This file provides configuration and fixtures for pytest tests,
including command-line options for customizing test parameters.

"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import warnings


def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    
    # Test data configuration
    parser.addoption(
        "--test-data-dir",
        action="store",
        default="../Datasets",
        help="Directory containing test datasets (default: ../Datasets)"
    )
    
    parser.addoption(
        "--h5ad-files",
        action="store",
        default="",
        help="Comma-separated list of h5ad files to test (default: auto-detect)"
    )
    
    # Test scope configuration
    parser.addoption(
        "--test-kernels",
        action="store_true",
        default=False,
        help="Run kernel consistency tests against scipy"
    )
    
    parser.addoption(
        "--test-pipeline",
        action="store_true", 
        default=False,
        help="Run pipeline consistency tests against pdex"
    )
    
    parser.addoption(
        "--test-performance",
        action="store_true",
        default=False,
        help="Run performance benchmark tests"
    )
    
    parser.addoption(
        "--test-real-data",
        action="store_true",
        default=False,
        help="Run tests on real h5ad datasets"
    )
    
    parser.addoption(
        "--test-all",
        action="store_true",
        default=False,
        help="Run all test categories (equivalent to all --test-* flags)"
    )
    
    # Test parameters
    parser.addoption(
        "--n-workers",
        action="store",
        type=int,
        default=4,
        help="Number of workers for parallel testing (default: 4)"
    )
    
    parser.addoption(
        "--max-cells",
        action="store",
        type=int,
        default=10000,
        help="Maximum number of cells for testing large datasets (default: 10000)"
    )
    
    parser.addoption(
        "--max-genes",
        action="store",
        type=int,
        default=2000,
        help="Maximum number of genes for testing large datasets (default: 2000)"
    )
    
    parser.addoption(
        "--tolerance",
        action="store",
        type=float,
        default=1e-4,
        help="Tolerance for numerical comparisons (default: 1e-4)"
    )
    
    parser.addoption(
        "--correlation-threshold",
        action="store",
        type=float,
        default=0.8,
        help="Minimum correlation threshold for consistency tests (default: 0.8)"
    )
    
    parser.addoption(
        "--fdr-threshold",
        action="store",
        type=float,
        default=0.05,
        help="FDR threshold for differential gene analysis (default: 0.05)"
    )
    
    # Performance test configuration
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="Skip slow/large-scale performance tests"
    )
    
    parser.addoption(
        "--benchmark-sizes",
        action="store",
        default="small,medium,large",
        help="Comma-separated benchmark sizes: small,medium,large,huge (default: small,medium,large)"
    )


@pytest.fixture(scope="session")
def test_config(request):
    """Test configuration fixture."""
    config = {
        # Data paths
        "test_data_dir": Path(request.config.getoption("--test-data-dir")),
        "h5ad_files": request.config.getoption("--h5ad-files").split(",") if request.config.getoption("--h5ad-files") else [],
        
        # Test scope
        "test_kernels": request.config.getoption("--test-kernels") or request.config.getoption("--test-all"),
        "test_pipeline": request.config.getoption("--test-pipeline") or request.config.getoption("--test-all"),
        "test_performance": request.config.getoption("--test-performance") or request.config.getoption("--test-all"),
        "test_real_data": request.config.getoption("--test-real-data") or request.config.getoption("--test-all"),
        
        # Test parameters
        "n_workers": request.config.getoption("--n-workers"),
        "max_cells": request.config.getoption("--max-cells"),
        "max_genes": request.config.getoption("--max-genes"),
        "tolerance": request.config.getoption("--tolerance"),
        "correlation_threshold": request.config.getoption("--correlation-threshold"),
        "fdr_threshold": request.config.getoption("--fdr-threshold"),
        
        # Performance config
        "skip_slow": request.config.getoption("--skip-slow"),
        "benchmark_sizes": request.config.getoption("--benchmark-sizes").split(","),
    }
    
    return config


@pytest.fixture(scope="session") 
def available_datasets(test_config):
    """Discover available h5ad datasets."""
    data_dir = test_config["test_data_dir"]
    
    if not data_dir.exists():
        warnings.warn(f"Test data directory not found: {data_dir}")
        return []
    
    # Auto-discover h5ad files if not specified
    if test_config["h5ad_files"] and test_config["h5ad_files"][0]:
        h5ad_files = []
        for filename in test_config["h5ad_files"]:
            filepath = data_dir / "CellFM" / filename
            if filepath.exists():
                h5ad_files.append(filepath)
            else:
                warnings.warn(f"H5AD file not found: {filepath}")
    else:
        # Auto-discover
        cellFM_dir = data_dir / "CellFM"
        if cellFM_dir.exists():
            h5ad_files = list(cellFM_dir.glob("*.h5ad"))
            # Sort by file size (smaller first for faster testing)
            h5ad_files.sort(key=lambda x: x.stat().st_size)
        else:
            h5ad_files = []
    
    return h5ad_files


@pytest.fixture
def synthetic_data():
    """Generate synthetic test data."""
    def _generate(n_cells=200, n_genes=50, n_groups=3, data_type='float', add_ties=False, seed=42):
        np.random.seed(seed)
        
        if data_type == 'float':
            X = np.random.normal(0, 1, (n_cells, n_genes))
        elif data_type == 'int':
            X = np.random.poisson(5, (n_cells, n_genes))
        else:  # mixed
            X = np.random.normal(0, 1, (n_cells, n_genes))
            X = np.round(X).astype(int)
        
        if add_ties:
            for i in range(n_genes):
                if np.random.random() < 0.3:
                    tie_value = X[0, i]
                    tie_indices = np.random.choice(n_cells, size=min(5, n_cells), replace=False)
                    X[tie_indices, i] = tie_value
        
        # Create group labels
        cells_per_group = n_cells // n_groups
        group_labels = []
        for i in range(n_groups):
            group_name = f"group_{i}" if i > 0 else "control"
            group_labels.extend([group_name] * cells_per_group)
        
        # Pad with control if needed
        while len(group_labels) < n_cells:
            group_labels.append("control")
        
        group_labels = np.array(group_labels[:n_cells])
        
        # Create AnnData object
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame({'group': group_labels}),
            var=pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
        )
        
        return adata
    
    return _generate


@pytest.fixture
def performance_benchmarks():
    """Performance benchmark configurations."""
    return {
        "small": {"n_cells": 1000, "n_genes": 500},
        "medium": {"n_cells": 5000, "n_genes": 1000}, 
        "large": {"n_cells": 10000, "n_genes": 2000},
        "huge": {"n_cells": 50000, "n_genes": 10000},
    }


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command-line options."""
    
    # Skip tests based on command-line flags
    skip_kernels = not (config.getoption("--test-kernels") or config.getoption("--test-all"))
    skip_pipeline = not (config.getoption("--test-pipeline") or config.getoption("--test-all")) 
    skip_performance = not (config.getoption("--test-performance") or config.getoption("--test-all"))
    skip_real_data = not (config.getoption("--test-real-data") or config.getoption("--test-all"))
    skip_slow = config.getoption("--skip-slow")
    
    for item in items:
        # Skip kernel tests
        if skip_kernels and "kernel" in item.nodeid:
            item.add_marker(pytest.mark.skip(reason="Kernel tests not requested"))
        
        # Skip pipeline tests
        if skip_pipeline and "pipeline" in item.nodeid:
            item.add_marker(pytest.mark.skip(reason="Pipeline tests not requested"))
        
        # Skip performance tests  
        if skip_performance and "performance" in item.nodeid:
            item.add_marker(pytest.mark.skip(reason="Performance tests not requested"))
        
        # Skip real data tests
        if skip_real_data and "real_data" in item.nodeid:
            item.add_marker(pytest.mark.skip(reason="Real data tests not requested"))
        
        # Skip slow tests
        if skip_slow and ("slow" in item.keywords or "huge" in item.nodeid):
            item.add_marker(pytest.mark.skip(reason="Slow tests skipped"))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "kernel: marks tests as kernel-level tests")
    config.addinivalue_line("markers", "pipeline: marks tests as pipeline-level tests") 
    config.addinivalue_line("markers", "performance: marks tests as performance benchmarks")
    config.addinivalue_line("markers", "real_data: marks tests that use real datasets")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
