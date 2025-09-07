# HPDEX Testing Suite

This directory contains comprehensive tests for the hpdex framework.

## Quick Start

### 1. Simple Test (No Dependencies)

Run the basic test without requiring pytest:

```bash
cd test/
python test_runner.py
```

This will run a comprehensive set of basic functionality tests including:
- ✅ Basic API functionality  
- 🚀 Multiprocessing support
- 📈 Different metrics (wilcoxon, wilcoxon-hist)
- 🌊 Streaming API
- ⚠️ Edge cases (small groups, sparse data)

### 2. Performance Test

```bash
python test_runner.py --performance
```

### 3. Full Test Suite (Requires pytest)

Install pytest first:
```bash
pip install pytest pytest-xdist
```

Then run tests:

```bash
# Run all basic tests
pytest test_hpdex_comprehensive.py -v

# Run specific test categories
pytest test_hpdex_comprehensive.py::TestBasicFunctionality -v
pytest test_hpdex_comprehensive.py::TestKernelCorrectness -v
pytest test_hpdex_comprehensive.py::TestStreamingBackend -v

# Run with multiple workers for faster execution
pytest test_hpdex_comprehensive.py -n 4

# Run all test categories (including slow tests)
pytest test_hpdex_comprehensive.py --test-all -v
```

## Test Categories

### TestBasicFunctionality
- Basic API calls and parameter validation
- Different metric types (wilcoxon, wilcoxon-hist)
- Multiprocessing functionality
- Input validation and error handling

### TestKernelCorrectness  
- Comparison against scipy.stats.mannwhitneyu
- Histogram kernel validation
- Tie handling and edge cases
- Numerical accuracy tests

### TestStreamingBackend
- Streaming API functionality
- Memory efficiency validation
- Consistency between streaming and regular APIs
- Multiprocessing in streaming mode

### TestEdgeCases
- Single cell groups
- Empty groups
- Sparse data matrices
- Constant expression genes
- Boundary conditions

### TestUtilityFunctions
- Fold change computation
- Log2 fold change calculation
- FDR correction
- Statistical utility functions

### TestPerformance (Slow)
- Large dataset handling
- Memory efficiency monitoring
- Performance benchmarks
- Scaling tests

### TestRealData (Requires test data)
- Tests with real h5ad datasets
- Automatic parameter detection
- Robustness validation

## Advanced Usage

### Custom Test Configuration

```bash
# Test with specific parameters
pytest test_hpdex_comprehensive.py \
    --n-workers 8 \
    --max-cells 50000 \
    --tolerance 1e-5 \
    --timeout 120

# Test specific functionality
pytest test_hpdex_comprehensive.py \
    --test-kernels \
    --test-pipeline \
    --test-performance

# Skip slow tests
pytest test_hpdex_comprehensive.py --skip-slow

# Test with real datasets (requires data in test_data/)
pytest test_hpdex_comprehensive.py \
    --test-real-data \
    --test-data-dir ./test_data
```

### Test Data Setup

To test with real datasets, place `.h5ad` files in the `test_data/` directory:

```
test_data/
├── small_dataset.h5ad
├── medium_dataset.h5ad  
└── large_dataset.h5ad
```

The test suite will automatically discover and test with these datasets.

### Continuous Integration

For CI environments, use:

```bash
# Fast test subset
pytest test_hpdex_comprehensive.py \
    --test-kernels \
    --test-pipeline \
    --skip-slow \
    --timeout 60

# Memory-limited environment  
pytest test_hpdex_comprehensive.py \
    --max-cells 10000 \
    --max-genes 1000 \
    --n-workers 2
```

## Expected Test Results

### Successful Run Output
```
🧪 Running hpdex basic tests...
==================================================
✓ Successfully imported hpdex

📊 Generating test data...
✓ Generated data: 300 cells × 100 genes
  Groups: {'control': 100, 'treatment_A': 100, 'treatment_B': 100}

🔧 Test 1: Basic API functionality
✓ Basic API completed in 0.85s
  Results shape: (200, 6)
  Columns: ['target', 'feature', 'p_value', 'fold_change', 'log2_fold_change', 'fdr']
  Significant genes (FDR < 0.05): 42

🚀 Test 2: Multiprocessing
✓ Multiprocessing completed in 0.52s
  Max p-value difference: 1.23e-15

📈 Test 3: Different metrics
✓ Histogram algorithm: 50 results
  Significant genes (FDR < 0.05): 15

🌊 Test 4: Streaming API
✓ Streaming API completed in 0.91s
  Results shape: (200, 6)
  Significant genes (FDR < 0.05): 42
  Max p-value difference vs regular API: 2.34e-14

⚠️ Test 5: Edge cases
✓ Small groups: 20 results
✓ Sparse data: 200 results

==================================================
🎉 All tests passed successfully!
```

### Performance Benchmarks

Expected performance on modern hardware:

| Dataset Size | Workers | Time (approx) |
|-------------|---------|---------------|
| 1K cells × 500 genes | 1 | ~2s |
| 1K cells × 500 genes | 4 | ~0.8s |
| 5K cells × 1K genes | 1 | ~15s |
| 5K cells × 1K genes | 4 | ~5s |
| 10K cells × 2K genes | 4 | ~20s |

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure hpdex is properly installed or in Python path
2. **Memory errors**: Reduce `--max-cells` and `--max-genes` parameters
3. **Timeout errors**: Increase `--timeout` parameter
4. **Multiprocessing issues**: Try reducing `--n-workers`

### Debug Mode

For detailed debugging:

```bash
python test_runner.py 2>&1 | tee test_log.txt
pytest test_hpdex_comprehensive.py -v -s --tb=long
```

### Reporting Issues

When reporting test failures, please include:
- Python version and platform
- Full error traceback  
- Test configuration parameters
- Dataset characteristics (if using real data)
- Hardware specifications (CPU, RAM)

## Contributing

To add new tests:

1. Add test methods to appropriate test classes
2. Use descriptive names: `test_feature_description`
3. Include docstrings explaining the test purpose
4. Add appropriate pytest markers (`@pytest.mark.slow`, etc.)
5. Validate both correctness and performance when applicable
