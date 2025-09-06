# hpdex Testing Guide

This document describes how to run comprehensive tests for the hpdex library using the new pytest-based testing framework.

## Overview

The testing framework provides three main test categories:

1. **Kernel Tests** - Test algorithm correctness against scipy
2. **Pipeline Tests** - Test full pipeline consistency against pdex  
3. **Performance Tests** - Benchmark speed and efficiency
4. **Real Data Tests** - Validate on actual h5ad datasets

## Quick Start

### Install Test Dependencies

```bash
# Install test dependencies
pip install pytest pytest-cov pandas anndata scanpy

# Optional: Install pdex for comparison tests
pip install pdex
```

### Run All Tests

```bash
# Run all test categories
python run_tests.py --all

# Or using pytest directly
pytest benchmarks.py --test-all
```

## Test Categories

### 1. Kernel Consistency Tests

Test hpdex kernel functions against scipy for correctness:

```bash
# Test floating-point kernel
python run_tests.py --kernels

# Test with custom tolerance
python run_tests.py --kernels --tolerance 1e-5

# Using pytest markers
pytest benchmarks.py -m kernel
```

**What it tests:**
- `rank_sum_chunk_kernel_float` vs `scipy.stats.mannwhitneyu`
- `rank_sum_chunk_kernel_hist` vs scipy for integer data
- Consistency with tie correction and continuity correction
- Edge cases and boundary conditions

### 2. Pipeline Consistency Tests

Test full hpdex pipeline against pdex:

```bash
# Test pipeline consistency
python run_tests.py --pipeline

# Test with custom correlation threshold
python run_tests.py --pipeline --correlation-threshold 0.9

# Using pytest markers
pytest benchmarks.py -m pipeline
```

**What it tests:**
- `parallel_differential_expression` vs `pdex.parallel_differential_expression`
- P-value correlations between methods
- Fold change consistency
- FDR < 0.05 differential gene set overlap (Jaccard similarity)

### 3. Performance Benchmarks

Test speed and efficiency:

```bash
# Run performance benchmarks
python run_tests.py --performance

# Test specific sizes
python run_tests.py --performance --benchmark-sizes "small,medium"

# Skip slow tests
python run_tests.py --performance --skip-slow

# Multi-threaded performance
python run_tests.py --performance --n-workers 8
```

**What it tests:**
- Kernel-level performance (hpdex vs scipy)
- Pipeline-level performance (hpdex vs pdex)
- Large-scale performance tests
- Memory efficiency and throughput

### 4. Real Data Tests

Test on actual h5ad datasets:

```bash
# Test on available datasets
python run_tests.py --real-data

# Test specific datasets
python run_tests.py --real-data --h5ad-files "PBMC_10K.h5ad,HumanPBMC.h5ad"

# Limit dataset size for faster testing
python run_tests.py --real-data --max-cells 5000 --max-genes 1000
```

**What it tests:**
- Real h5ad file loading and processing
- Automatic group detection
- Consistency with real biological data
- Performance on real datasets

## Configuration Options

### Test Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--test-data-dir` | `../Datasets` | Directory containing test datasets |
| `--n-workers` | `4` | Number of parallel workers |
| `--max-cells` | `10000` | Maximum cells for large datasets |
| `--max-genes` | `2000` | Maximum genes for large datasets |
| `--tolerance` | `1e-4` | Numerical comparison tolerance |
| `--correlation-threshold` | `0.8` | Minimum correlation for consistency |
| `--fdr-threshold` | `0.05` | FDR threshold for DEG analysis |

### Performance Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--skip-slow` | `False` | Skip time-consuming tests |
| `--benchmark-sizes` | `small,medium,large` | Which benchmark sizes to run |

## Advanced Usage

### Custom Test Selection

```bash
# Run tests by marker
pytest benchmarks.py -m "kernel and not slow"

# Run tests by keyword
pytest benchmarks.py -k "consistency"

# Run specific test class
pytest benchmarks.py::TestKernelConsistency

# Run specific test method
pytest benchmarks.py::TestKernelConsistency::test_float_kernel_vs_scipy
```

### Coverage Reports

```bash
# Enable coverage
python run_tests.py --all --cov

# Generate HTML coverage report
pytest benchmarks.py --test-all --cov=hpdex --cov-report=html
```

### Parallel Testing

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
python run_tests.py --all --parallel
```

### Profiling

```bash
# Install pytest-profiling
pip install pytest-profiling

# Profile test execution
python run_tests.py --performance --profile
```

## Example Test Runs

### Quick Development Test
```bash
# Fast test for development
python run_tests.py --kernels --max-cells 1000 --max-genes 100 --skip-slow
```

### Comprehensive Validation
```bash
# Full validation before release
python run_tests.py --all --n-workers 8 --cov
```

### Performance Evaluation
```bash
# Detailed performance analysis
python run_tests.py --performance --benchmark-sizes "small,medium,large,huge" --n-workers 8
```

### Real Data Validation
```bash
# Test on specific real datasets
python run_tests.py --real-data --h5ad-files "PBMC_10K.h5ad,norman.h5ad" --max-cells 20000
```

## Interpreting Results

### Kernel Tests
- **P-value correlation > 0.9**: Excellent agreement with scipy
- **Max difference < 1e-4**: Numerical precision within tolerance
- **Failed tests**: Potential algorithm bugs

### Pipeline Tests  
- **P-value correlation > 0.8**: Good consistency with pdex
- **Fold change correlation > 0.8**: Consistent differential expression
- **Jaccard similarity > 0.3**: Reasonable DEG overlap

### Performance Tests
- **Speedup > 1.0x**: hpdex is faster
- **Speedup > 10x**: Significant performance improvement
- **Throughput**: Million data points processed per second

### Real Data Tests
- **No errors**: Data loading and processing works
- **Significant DEGs found**: Method detects biological signal
- **Consistent results**: Reproducible analysis

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install scipy pandas anndata pytest
   ```

2. **Memory Issues**
   ```bash
   # Reduce dataset size
   python run_tests.py --all --max-cells 5000 --max-genes 1000
   ```

3. **Slow Tests**
   ```bash
   # Skip slow tests
   python run_tests.py --all --skip-slow
   ```

4. **Dataset Not Found**
   ```bash
   # Check data directory
   python run_tests.py --real-data --test-data-dir /path/to/datasets
   ```

### Debug Mode

```bash
# Verbose output with full tracebacks
pytest benchmarks.py --test-all -v --tb=long -s
```

## Contributing

When adding new tests:

1. Use appropriate pytest markers (`@pytest.mark.kernel`, etc.)
2. Follow the existing test naming convention
3. Add proper documentation and assertions
4. Test both success and failure cases
5. Update this guide if adding new test categories

## Test Data Requirements

The real data tests expect h5ad files in the following structure:
```
../Datasets/CellFM/
├── PBMC_10K.h5ad
├── HumanPBMC.h5ad  
├── norman.h5ad
└── ...
```

Each h5ad file should have:
- `adata.X`: Expression matrix
- `adata.obs`: Cell metadata with grouping columns
- `adata.var`: Gene metadata