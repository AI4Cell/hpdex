#!/usr/bin/env python3
"""
Test runner script for hpdex library.

This script provides convenient ways to run different test suites
with various configurations for comprehensive testing of hpdex functionality.

Usage Examples:
    # Run all tests
    python run_tests.py --all
    
    # Run specific test categories
    python run_tests.py --kernels --pipeline
    python run_tests.py --performance --n-workers 8
    
    # Run with real data
    python run_tests.py --real-data --h5ad-files "PBMC_10K.h5ad,HumanPBMC.h5ad"
    
    # Performance benchmarks only
    python run_tests.py --performance --benchmark-sizes "small,medium" --skip-slow
    
    # Quick test run
    python run_tests.py --kernels --max-cells 1000 --max-genes 100
    
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_pytest(args_list):
    """Run pytest with given arguments."""
    cmd = ["python", "-m", "pytest"] + args_list
    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run hpdex tests with configurable parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Test categories
    test_group = parser.add_argument_group("Test Categories")
    test_group.add_argument("--all", action="store_true", 
                          help="Run all test categories")
    test_group.add_argument("--kernels", action="store_true", 
                          help="Run kernel consistency tests against scipy")
    test_group.add_argument("--pipeline", action="store_true", 
                          help="Run pipeline consistency tests against pdex")
    test_group.add_argument("--performance", action="store_true", 
                          help="Run performance benchmark tests")
    test_group.add_argument("--real-data", action="store_true", 
                          help="Run tests on real h5ad datasets")
    
    # Test configuration
    config_group = parser.add_argument_group("Test Configuration")
    config_group.add_argument("--test-data-dir", default="../Datasets",
                            help="Directory containing test datasets")
    config_group.add_argument("--h5ad-files", 
                            help="Comma-separated list of h5ad files to test")
    config_group.add_argument("--n-workers", type=int, default=4,
                            help="Number of workers for parallel testing")
    config_group.add_argument("--max-cells", type=int, default=10000,
                            help="Maximum number of cells for testing")
    config_group.add_argument("--max-genes", type=int, default=2000,
                            help="Maximum number of genes for testing")
    config_group.add_argument("--tolerance", type=float, default=1e-4,
                            help="Tolerance for numerical comparisons")
    config_group.add_argument("--correlation-threshold", type=float, default=0.8,
                            help="Minimum correlation threshold for consistency")
    config_group.add_argument("--fdr-threshold", type=float, default=0.05,
                            help="FDR threshold for differential gene analysis")
    
    # Performance configuration
    perf_group = parser.add_argument_group("Performance Configuration")
    perf_group.add_argument("--skip-slow", action="store_true",
                          help="Skip slow/large-scale performance tests")
    perf_group.add_argument("--benchmark-sizes", default="small,medium,large",
                          help="Comma-separated benchmark sizes")
    
    # Pytest options
    pytest_group = parser.add_argument_group("Pytest Options")
    pytest_group.add_argument("--verbose", "-v", action="store_true", 
                            help="Verbose output")
    pytest_group.add_argument("--quiet", "-q", action="store_true",
                            help="Quiet output") 
    pytest_group.add_argument("--tb", choices=["short", "long", "no"], default="short",
                            help="Traceback style")
    pytest_group.add_argument("--markers", "-m", 
                            help="Run tests matching given mark expression")
    pytest_group.add_argument("--keyword", "-k",
                            help="Run tests matching given keyword expression")
    pytest_group.add_argument("--capture", choices=["yes", "no", "sys"], default="yes",
                            help="Capture stdout/stderr")
    pytest_group.add_argument("--cov", action="store_true",
                            help="Enable coverage reporting")
    pytest_group.add_argument("--no-cov", action="store_true",
                            help="Disable coverage reporting")
    
    # Advanced options
    advanced_group = parser.add_argument_group("Advanced Options")
    advanced_group.add_argument("--test-file", default="benchmarks.py",
                               help="Test file to run")
    advanced_group.add_argument("--parallel", action="store_true",
                               help="Run tests in parallel (requires pytest-xdist)")
    advanced_group.add_argument("--profile", action="store_true",
                               help="Enable profiling (requires pytest-profiling)")
    advanced_group.add_argument("--benchmark", action="store_true",
                               help="Enable benchmarking (requires pytest-benchmark)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.all, args.kernels, args.pipeline, args.performance, args.real_data]):
        if not args.markers and not args.keyword:
            print("Error: Must specify at least one test category or use --all")
            print("Use --help for more information")
            return 1
    
    # Build pytest arguments
    pytest_args = [args.test_file]
    
    # Add verbosity
    if args.verbose:
        pytest_args.append("-v")
    elif args.quiet:
        pytest_args.append("-q")
    
    # Add traceback style
    pytest_args.extend(["--tb", args.tb])
    
    # Add capture mode
    if args.capture != "yes":
        pytest_args.extend(["-s" if args.capture == "no" else f"--capture={args.capture}"])
    
    # Add test category flags
    pytest_flags = []
    if args.all:
        pytest_flags.append("--test-all")
    else:
        if args.kernels:
            pytest_flags.append("--test-kernels")
        if args.pipeline:
            pytest_flags.append("--test-pipeline")
        if args.performance:
            pytest_flags.append("--test-performance")
        if args.real_data:
            pytest_flags.append("--test-real-data")
    
    # Add configuration parameters
    pytest_flags.extend([
        f"--test-data-dir={args.test_data_dir}",
        f"--n-workers={args.n_workers}",
        f"--max-cells={args.max_cells}",
        f"--max-genes={args.max_genes}",
        f"--tolerance={args.tolerance}",
        f"--correlation-threshold={args.correlation_threshold}",
        f"--fdr-threshold={args.fdr_threshold}",
        f"--benchmark-sizes={args.benchmark_sizes}"
    ])
    
    if args.h5ad_files:
        pytest_flags.append(f"--h5ad-files={args.h5ad_files}")
    
    if args.skip_slow:
        pytest_flags.append("--skip-slow")
    
    pytest_args.extend(pytest_flags)
    
    # Add markers and keywords
    if args.markers:
        pytest_args.extend(["-m", args.markers])
    
    if args.keyword:
        pytest_args.extend(["-k", args.keyword])
    
    # Add coverage options
    if args.no_cov:
        pytest_args.append("--no-cov")
    elif args.cov:
        pytest_args.extend(["--cov=hpdex", "--cov-report=term-missing"])
    
    # Add advanced options
    if args.parallel:
        pytest_args.extend(["-n", "auto"])
    
    if args.profile:
        pytest_args.append("--profile")
    
    if args.benchmark:
        pytest_args.append("--benchmark-only")
    
    print("hpdex Test Runner")
    print("=" * 80)
    print(f"Test categories: {', '.join([cat for cat in ['all', 'kernels', 'pipeline', 'performance', 'real-data'] if getattr(args, cat.replace('-', '_'), False)])}")
    print(f"Workers: {args.n_workers}")
    print(f"Max cells: {args.max_cells:,}, Max genes: {args.max_genes:,}")
    print(f"Data directory: {args.test_data_dir}")
    if args.h5ad_files:
        print(f"H5AD files: {args.h5ad_files}")
    print()
    
    return run_pytest(pytest_args)


if __name__ == "__main__":
    sys.exit(main())