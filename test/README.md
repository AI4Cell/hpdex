# HPDEX Testing

Configuration-based testing framework for hpdex.

## Quick Start

### Basic Test
```bash
python test.py config_quick.yml
```

### Full Test Suite  
```bash
python test.py config.yml
```

### Specific Categories
```bash
python test.py config.yml --category kernel_consistency
python test.py config.yml --category pipeline_consistency
```

### List Available Tests
```bash
python test.py config.yml --list-categories
```

## Test Categories

### `kernel_consistency`
- Tests statistical kernel accuracy against scipy.stats.mannwhitneyu
- Validates both float and histogram algorithms  
- Success: p-value errors < 1e-10

### `pipeline_consistency`  
- Tests full pipeline against pdex library
- Validates p-values, fold changes, FDR correction
- Success: correlation > 0.95

### `performance_kernels`
- Benchmarks kernel performance vs scipy
- Tests different data scales and memory usage
- Success: comparable or better performance

### `performance_pipeline`
- Benchmarks full pipeline vs pdex  
- Tests multiprocessing scalability
- Success: significant speedup with consistent results

### `edge_cases`
- Tests robustness with edge cases
- Small groups, constant genes, sparse data
- Success: proper error handling and valid results

## Configuration Files

### `config_quick.yml` - Quick Tests
Small-scale tests for development and CI:
```yaml
test_categories:
  kernel_consistency: true
  pipeline_consistency: true
  edge_cases: true
  
kernel_tests:
  tolerance: 0.000001
  n_runs: 3
  
pipeline_tests:
  test_datasets:
    - name: "small"
      n_cells: 1000
      n_genes: 500
      n_groups: 3
```

### `config.yml` - Full Test Suite  
Comprehensive testing with performance benchmarks:
```yaml
test_categories:
  kernel_consistency: true
  pipeline_consistency: true 
  performance_kernels: true
  performance_pipeline: true
  edge_cases: true
  
performance_tests:
  kernel_benchmarks:
    benchmark_runs: 5
    timeout_seconds: 300
    
  pipeline_benchmarks:
    - name: "large"
      n_cells: 10000
      n_genes: 5000
      n_groups: 10
```

### Custom Configuration
Modify any section to customize tests:
```yaml
# Adjust test scale
pipeline_tests:
  test_datasets:
    - name: "custom"
      n_cells: 5000      # Custom cell count
      n_genes: 2000      # Custom gene count  
      n_groups: 4        # Custom group count

# Adjust tolerances
kernel_tests:
  tolerance: 1e-8      # Relaxed tolerance
  n_runs: 10           # More test runs
```

## Test Results

Results are saved to `test_results/` with timestamp:
```
test_results/
└── test_run_20240107_143022/
    ├── test_summary.json      # Test summary
    ├── detailed_results.json  # Detailed results
    └── test_report.md         # Human-readable report
```

### Example Results

**Kernel Consistency:**
```json
{
  "max_p_error": 1.23e-15,
  "correlation_p": 0.999999,
  "p_values_match": true
}
```

**Performance:**
```json
{
  "hpdex": {"mean_time": 2.1}, 
  "scipy": {"mean_time": 15.3},
  "speedup": 7.3
}
```

## Common Issues

**Missing pdex:**
```
Warning: pdex not available, pipeline tests will be skipped
```
Install pdex or skip pipeline tests.

**Memory issues:**
- Reduce data sizes in config
- Lower `num_workers`
- Increase `memory_limit_gb`

**Timeouts:**
- Increase `timeout_seconds`
- Reduce test scale
- Check system load

## Configuration Reference

### Key Configuration Options

```yaml
# Global settings
environment:
  num_workers: 4
  memory_limit_gb: 8.0
  random_seed: 42

# Test categories to run
test_categories:
  kernel_consistency: true
  pipeline_consistency: true
  performance_kernels: false
  performance_pipeline: false
  edge_cases: true

# Kernel test settings
kernel_tests:
  tolerance: 1e-10
  n_runs: 5
  max_cells: 10000
  max_genes: 5000

# Pipeline test datasets
pipeline_tests:
  correlation_threshold: 0.95
  test_datasets:
    - name: "small"
      n_cells: 1000
      n_genes: 500
      n_groups: 3
      effect_size: 1.5
      
# Performance test settings  
performance_tests:
  kernel_benchmarks:
    benchmark_runs: 5
    timeout_seconds: 300
```
