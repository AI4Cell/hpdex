# Testing Guide for hpdex

This document describes how to run the benchmark and consistency tests for hpdex.

## Test Overview

The `benchmark.py` file contains comprehensive tests to verify that hpdex rank sum kernels produce results consistent with scipy.stats.mannwhitneyu.

### Test Categories

1. **Consistency Tests**: Verify that hpdex results match scipy results
2. **Performance Benchmarks**: Compare execution time between hpdex and scipy
3. **Edge Case Tests**: Test handling of edge cases (small datasets, ties, NaN values)
4. **Integration Tests**: Test the full `parallel_difference_expression` function

## Running Tests

### Method 1: Using pytest (Recommended)

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Install hpdex in development mode
pip install -e .

# Run all tests
pytest benchmark.py -v

# Run specific test
pytest benchmark.py::test_rank_sum_kernel_float_consistency -v

# Run with coverage
pytest benchmark.py --cov=hpdex --cov-report=html
```

### Method 2: Using the test runner script

```bash
# Install dependencies
pip install -r requirements-test.txt

# Run tests directly
python run_tests.py
```

### Method 3: Run individual test functions

```python
import sys
sys.path.insert(0, 'src')
from benchmark import test_rank_sum_kernel_float_consistency

# Run specific test
test_rank_sum_kernel_float_consistency()
```

## Test Details

### ScipyWrapper Class

The `ScipyWrapper` class provides a wrapper around `scipy.stats.mannwhitneyu` that matches the hpdex interface:

- `rank_sum_float()`: Wrapper for floating-point algorithm
- `rank_sum_hist()`: Wrapper for histogram algorithm (falls back to float)

### Test Functions

1. **`test_rank_sum_kernel_float_consistency()`**
   - Tests hpdex float kernel against scipy
   - Multiple data types and parameter combinations
   - Verifies p-values and U-statistics match

2. **`test_rank_sum_kernel_hist_consistency()`**
   - Tests hpdex histogram kernel against scipy
   - Integer data with ties
   - Verifies consistency for optimized integer algorithm

3. **`test_performance_benchmark()`**
   - Benchmarks performance against scipy
   - Large dataset (1000 samples, 100 genes)
   - Reports speedup and verifies consistency

4. **`test_edge_cases()`**
   - Tests small datasets, identical values, NaN handling
   - Ensures robust error handling

5. **`test_parallel_difference_expression_consistency()`**
   - Tests the main API function
   - Creates AnnData object and runs full analysis
   - Verifies output format and basic sanity checks

## Expected Results

All tests should pass with:
- P-value differences < 1e-10 (or rtol=1e-6 for numerical precision)
- U-statistic differences < 1e-10
- Performance speedup > 1x (hpdex should be faster than scipy)
- Proper handling of edge cases

## Troubleshooting

### Import Errors
If you get import errors, make sure:
1. Install dependencies: `pip install -r requirements-test.txt`
2. Install hpdex: `pip install -e .`
3. Check Python path includes src directory

### Test Failures
If tests fail:
1. Check that scipy and numpy versions are compatible
2. Verify that hpdex is properly installed
3. Check for numerical precision issues (may need to adjust tolerances)

### Performance Issues
If performance tests are slow:
1. Reduce dataset size in `test_performance_benchmark()`
2. Use fewer parameter combinations
3. Run with `pytest -m "not slow"` to skip slow tests

## Adding New Tests

To add new tests:

1. Create a new test function with `test_` prefix
2. Use `assert` statements to verify results
3. Add appropriate print statements for debugging
4. Consider adding to specific test categories with pytest markers

Example:
```python
@pytest.mark.benchmark
def test_my_new_feature():
    """Test my new feature."""
    # Test implementation
    result = my_function()
    expected = expected_result()
    assert np.allclose(result, expected, rtol=1e-6)
    print("✓ My new feature test passed")
```
