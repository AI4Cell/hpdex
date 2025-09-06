# hpdex

High-Performance Parallel Differential Expression Analysis for Single-Cell Perturbation Sequencing

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

hpdex is a high-performance parallel differential expression analysis tool designed specifically for single-cell perturbation sequencing data. Through shared memory parallelization and optimized algorithms, it significantly improves the analysis efficiency of large-scale single-cell data.

### Key Features

- 🚀 **High-Performance Parallelization**: Uses shared memory multiprocessing to avoid data copying, dramatically improving computational speed
- 🔬 **Algorithm Optimization**: Rewritten rank-sum test operators with pre-computed reference groups for enhanced speed
- 📊 **Compatibility**: Algorithmically consistent with scipy's mannwhitneyu operator, providing broadcasting support
- 🔄 **API Compatibility**: `parallel_difference_expression` function maintains complete consistency with pdex library
- ⚡ **Smart Optimization**: Histogram algorithm optimized for integer data
- 📈 **Statistical Analysis**: Supports FDR correction and fold change calculation

## Installation

### Install from GitHub

```bash
git clone https://github.com/AI4Cell/hpdex.git
cd hpdex
pip install -e .
```

### Install from PyPI (Recommended)

```bash
pip install hpdex
```

## Requirements

- Python >= 3.10
- numpy >= 1.23.5
- scipy >= 1.11.0
- numba >= 0.58.0
- pandas >= 1.5.0
- anndata >= 0.8.0
- tqdm >= 4.64.0

## Quick Start

### Basic Usage

```python
import anndata as ad
import pandas as pd
from hpdex import parallel_difference_expression

# Load data
adata = ad.read_h5ad("path/to/adata.h5ad")

# Perform differential expression analysis
df = parallel_difference_expression(
    adata,
    groupby_key="target_gene",
    reference="non-targeting",
)

print(df.head())
df.to_csv("result.csv", index=False)
```

### Advanced Usage

```python
# Advanced analysis with custom parameters
df = parallel_difference_expression(
    adata,
    groupby_key="treatment",
    reference="control",
    groups=["drug_A", "drug_B"],  # Specify comparison groups
    metric="wilcoxon",            # Statistical method
    tie_correction=True,          # Tie correction
    continuity_correction=True,   # Continuity correction
    min_samples=5,                # Minimum sample size
    num_workers=8,                # Number of parallel processes
    batch=1000000,                # Batch processing budget
)

# View results
print(f"Analyzed {df['pert'].nunique()} perturbation groups")
print(f"Detected {len(df[df['fdr'] < 0.05])} significantly differential genes")
```

## API Reference

### `parallel_difference_expression`

Performs high-performance parallel differential expression analysis.

**Parameters:**

- `adata` (AnnData): AnnData object containing gene expression data
- `groupby_key` (str): Column name in obs for grouping cells
- `reference` (str): Name of the reference group
- `groups` (List[str], optional): List of target groups to compare. If None, uses all groups except reference
- `metric` (str): Statistical test method. Currently supports "wilcoxon" (Mann-Whitney U test)
- `tie_correction` (bool): Whether to apply tie correction
- `continuity_correction` (bool): Whether to apply continuity correction
- `use_asymptotic` (bool, optional): Force asymptotic approximation. None for auto-selection
- `min_samples` (int): Minimum number of samples per group. Groups with fewer samples are excluded
- `max_bins` (int): Maximum number of bins for histogram algorithm
- `prefer_hist_if_int` (bool): Prefer histogram algorithm for integer data
- `num_workers` (int): Number of parallel worker processes
- `batch_budget` (int): Batch processing budget for task chunking

**Returns:**

DataFrame containing the following columns:
- `pert`: Perturbation group name
- `gene`: Gene name
- `pval`: P-value
- `fold_change`: Fold change (target_mean / reference_mean)
- `log2_fold_change`: Log2 fold change
- `fdr`: FDR-corrected p-value

## Performance Benchmarks

hpdex demonstrates significant performance advantages over traditional methods on large datasets:

- **Memory Efficiency**: Shared memory parallelization avoids data copying, reducing memory usage by 50%+
- **Computational Speed**: Multi-core parallel processing provides 3-10x speedup (depending on core count)
- **Algorithm Optimization**: Histogram algorithm for integer data provides 2-5x speedup

## Example Data

```python
# Test with example data
import scanpy as sc

# Create example data
adata = sc.datasets.pbmc3k()
adata.obs['treatment'] = ['control' if i % 2 == 0 else 'treated' for i in range(adata.n_obs)]

# Perform analysis
results = parallel_difference_expression(
    adata,
    groupby_key="treatment",
    reference="control",
    num_workers=4
)

print(f"Detected {len(results[results['fdr'] < 0.05])} significantly differential genes")
```

## Contributing

Issues and Pull Requests are welcome!

## License

MIT License

## Citation

If you use hpdex in your research, please cite:

```bibtex
@software{hpdex2024,
  title={hpdex: High-Performance Parallel Differential Expression Analysis},
  author={krkawzq},
  year={2024},
  url={https://github.com/AI4Cell/hpdex}
}
```