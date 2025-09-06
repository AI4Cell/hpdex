<div align="center">

# 🧬 hpdex: High-Performance Differential Expression Analysis

**Ultra-fast parallel differential expression analysis for single-cell perturbation sequencing**

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](#testing)
[![Performance](https://img.shields.io/badge/performance-10x%20faster-red.svg)](#performance)

[**Installation**](#installation) •
[**Quick Start**](#quick-start) •
[**Documentation**](#api-reference) •
[**Performance**](#performance) •
[**Testing**](#testing)

</div>

---

## 🌟 Overview

**hpdex** is a revolutionary high-performance parallel differential expression analysis library designed for single-cell perturbation sequencing data. It delivers **10x+ speed improvements** over traditional methods while maintaining algorithmic consistency with established statistical frameworks.


## 🚀 Installation

### Quick Install (Recommended)

```bash
pip install hpdex
```

### Development Install

```bash
git clone https://github.com/AI4Cell/hpdex.git
cd hpdex
pip install -e .
```

### Requirements

- **Python** ≥ 3.10
- **Core dependencies**: numpy, scipy, numba, pandas, anndata
- **Optional**: pdex (for comparison), scanpy (for examples)

---

## ⚡ Quick Start

### Basic Usage

```python
import anndata as ad
from hpdex import parallel_differential_expression

adata = ad.read_h5ad("your_data.h5ad")

results = parallel_differential_expression(
    adata,
    groupby_key="perturbation",
    reference="control",
    num_workers=8
)

print(f"🧬 Analyzed {results.shape[0]:,} gene-perturbation pairs")
print(f"📊 Found {(results['fdr'] < 0.05).sum():,} significant hits")

results.to_csv("differential_genes.csv", index=False)
```

---

## 📚 API Reference

### `parallel_differential_expression`

The main function for differential expression analysis.

```python
parallel_differential_expression(
    adata: AnnData,
    groupby_key: str,
    reference: str,
    groups: Optional[List[str]] = None,
    metric: str = "wilcoxon",
    tie_correction: bool = True,
    continuity_correction: bool = True,
    use_asymptotic: Optional[bool] = None,
    min_samples: int = 2,
    max_bins: int = 100_000,
    prefer_hist_if_int: bool = False,
    num_workers: int = 1,
    batch: int = 1_000_000
) -> pd.DataFrame
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `AnnData` | - | Single-cell data object |
| `groupby_key` | `str` | - | Column in `adata.obs` for grouping |
| `reference` | `str` | - | Reference group name |
| `groups` | `List[str]` | `None` | Target groups (auto-detect if None) |
| `metric` | `str` | `"wilcoxon"` | Statistical test (Mann-Whitney U) |
| `tie_correction` | `bool` | `True` | Apply tie correction |
| `continuity_correction` | `bool` | `True` | Apply continuity correction |
| `use_asymptotic` | `bool` | `None` | Force asymptotic (auto if None) |
| `min_samples` | `int` | `2` | Minimum cells per group |
| `max_bins` | `int` | `100000` | Max bins for histogram algorithm |
| `prefer_hist_if_int` | `bool` | `False` | Prefer histogram for integer data |
| `num_workers` | `int` | `1` | Number of parallel processes |
| `batch` | `int` | `1000000` | Batch size for memory management |
| `clip_value` | `float` | `20.0` | Value to clip fold change to if it is infinite or NaN |

#### Returns

| Column | Type | Description |
|--------|------|-------------|
| `target` | `str` | Target group name |
| `feature` | `str` | Gene name |
| `p_value` | `float` | Raw p-value from Mann-Whitney U test |
| `fold_change` | `float` | Fold change (target/reference mean) |
| `log2_fold_change` | `float` | Log2 fold change |
| `fdr` | `float` | FDR-corrected p-value (Benjamini-Hochberg) |

---

## 🧪 Testing

hpdex includes a comprehensive test suite ensuring correctness and performance.

### Quick Test

```bash
cd hpdex/test
python run_tests.py --all
```

### Test Categories

#### 1. Correctness Tests
```bash
# Test against scipy (gold standard)
python run_tests.py --kernels

# Test against pdex (pipeline consistency)
python run_tests.py --pipeline
```

#### 2. Performance Tests
```bash
# Benchmark performance
python run_tests.py --performance

# Large-scale tests (may take time)
python run_tests.py --performance --benchmark-sizes "large,huge"
```

#### 3. Real Data Tests
```bash
# Test on real datasets
python run_tests.py --real-data --h5ad-files "PBMC_10K.h5ad,norman.h5ad"
```

### Custom Test Configuration

```bash
# Configure test parameters
python run_tests.py --all \
    --n-workers 8 \
    --max-cells 20000 \
    --tolerance 1e-5 \
    --correlation-threshold 0.95
```

### Validation Results

Our test suite validates:
- ✅ **Statistical Accuracy**: P-values within 1e-4 tolerance of scipy
- ✅ **Pipeline Consistency**: >99% correlation with pdex results  
- ✅ **Edge Cases**: NaN handling, empty groups, tie scenarios
- ✅ **Memory Safety**: No memory leaks in long-running tests

---

## 🐛 Troubleshooting

### Common Issues

<details>
<summary><strong>Memory Error with Large Datasets</strong></summary>

```python
# Solution: Reduce batch size and subsample
results = parallel_differential_expression(
    adata,
    groupby_key="treatment",
    reference="control",
    num_workers=4,    # Reduce workers
    batch=500000      # Smaller batch size
)

# Or subsample the data
sc.pp.subsample(adata, n_obs=50000)
```

</details>

<details>
<summary><strong>Performance Issues</strong></summary>

```python
# Check your CPU cores
import multiprocessing
print(f"Available cores: {multiprocessing.cpu_count()}")

# Optimize for your hardware
results = parallel_differential_expression(
    adata,
    groupby_key="treatment", 
    reference="control",
    num_workers=min(16, multiprocessing.cpu_count()),  # Don't exceed available cores
    prefer_hist_if_int=True  # Use histogram algorithm for integer data
)
```

</details>

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **scipy** team for the foundational statistical implementations
- **pdex** developers for the API design inspiration  
- **numba** team for JIT compilation capabilities
- **scanpy** community for single-cell analysis ecosystem