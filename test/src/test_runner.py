"""
Test Runner for HPDEX Testing Framework

Handles execution of different test categories and manages test flow.
"""

import time
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

# Import testing modules
from .kernel_tests import KernelConsistencyTester
from .pipeline_tests import PipelineConsistencyTester  
from .performance_tests import PerformanceTester
from .edge_case_tests import EdgeCaseTester
from .real_data_tests import RealDataBenchmarkTester
from .data_generators import SyntheticDataGenerator


@dataclass
class TestResult:
    """Container for test results."""
    success: bool
    summary: str
    details: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None


class TestRunner:
    """Main test runner that coordinates all test categories."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env = config['environment']
        
        # Set random seed for reproducibility
        np.random.seed(self.env['random_seed'])
        
        # Initialize components
        self.data_generator = SyntheticDataGenerator(self.env['random_seed'])
        self.kernel_tester = KernelConsistencyTester(config.get('kernel_tests', {}))
        self.pipeline_tester = PipelineConsistencyTester(config.get('pipeline_tests', {}))
        self.performance_tester = PerformanceTester(config.get('performance_tests', {}))
        self.edge_case_tester = EdgeCaseTester(config.get('edge_cases', {}))
        self.real_data_tester = RealDataBenchmarkTester(config.get('real_data_benchmarks', {}))
        
        # Suppress warnings for cleaner output
        warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", ".*Transforming to str index.*")
    
    def show_test_plan(self, categories: Dict[str, bool]) -> None:
        """Show what tests would be run without executing them."""
        print("Test execution plan:")
        print("-" * 40)
        
        for category in categories:
            print(f"\n📂 {category}:")
            
            if category == "kernel_consistency":
                test_cases = self.config.get('kernel_tests', {}).get('test_cases', [])
                for case in test_cases:
                    print(f"  • {case['name']}: {case['n_genes']} genes, "
                          f"{case['n_ref']}×{case['n_tar']} samples ({case['data_type']})")
            
            elif category == "pipeline_consistency":
                test_datasets = self.config.get('pipeline_tests', {}).get('test_datasets', [])
                for ds in test_datasets:
                    print(f"  • {ds['name']}: {ds['n_cells']} cells, "
                          f"{ds['n_genes']} genes, {ds['n_groups']} groups")
            
            elif category == "performance_kernels":
                test_cases = self.config.get('performance_tests', {}).get('kernel_benchmarks', {}).get('test_cases', [])
                for case in test_cases:
                    print(f"  • {case['name']}: {case['n_genes']} genes, "
                          f"{case['n_ref']}×{case['n_tar']} samples")
            
            elif category == "performance_pipeline":
                test_cases = self.config.get('performance_tests', {}).get('pipeline_benchmarks', {}).get('test_cases', [])
                for case in test_cases:
                    print(f"  • {case['name']}: {case['n_cells']} cells, "
                          f"{case['n_genes']} genes, {case['n_groups']} groups")
            
            elif category == "performance_streaming":
                test_cases = self.config.get('performance_tests', {}).get('streaming_benchmarks', {}).get('test_cases', [])
                for case in test_cases:
                    print(f"  • {case['name']}: {case['n_cells']} cells, "
                          f"{case['n_genes']} genes, {case['n_groups']} groups")
            
            elif category == "edge_cases":
                test_scenarios = self.config.get('edge_cases', {}).get('test_scenarios', [])
                for scenario in test_scenarios:
                    print(f"  • {scenario['name']}: {scenario.get('n_cells', 'varies')} cells")
            
            elif category == "real_data_benchmarks":
                datasets = self.config.get('real_data_benchmarks', {}).get('datasets', [])
                print(f"  • {len(datasets)} real datasets configured for benchmarking")
                for dataset in datasets[:3]:  # Show first 3 datasets
                    print(f"    - {dataset['name']}: {dataset['file']}")
                if len(datasets) > 3:
                    print(f"    - ... and {len(datasets)-3} more")
    
    def run_category(self, category: str) -> Tuple[bool, Dict[str, Any]]:
        """Run a specific test category."""
        start_time = time.time()
        
        try:
            if category == "kernel_consistency":
                success, details = self._run_kernel_consistency()
            elif category == "pipeline_consistency":
                success, details = self._run_pipeline_consistency()
            elif category == "performance_kernels":
                success, details = self._run_performance_kernels()
            elif category == "performance_pipeline":
                success, details = self._run_performance_pipeline()
            elif category == "performance_streaming":
                success, details = self._run_performance_streaming()
            elif category == "edge_cases":
                success, details = self._run_edge_cases()
            elif category == "real_data_benchmarks":
                success, details = self._run_real_data_benchmarks()
            else:
                raise ValueError(f"Unknown test category: {category}")
            
            execution_time = time.time() - start_time
            
            # Generate summary
            summary = self._generate_summary(category, details, success)
            
            return success, {
                'success': success,
                'summary': summary,
                'details': details,
                'execution_time': execution_time
            }
        
        except Exception as e:
            execution_time = time.time() - start_time
            return False, {
                'success': False,
                'summary': f"Category failed with error: {str(e)}",
                'details': {'error': str(e)},
                'execution_time': execution_time
            }
    
    def _run_kernel_consistency(self) -> Tuple[bool, Dict[str, Any]]:
        """Run kernel consistency tests against scipy."""
        print("🔬 Testing kernel consistency against scipy...")
        
        test_cases = self.config.get('kernel_tests', {}).get('test_cases', [])
        tolerance = self.config.get('kernel_tests', {}).get('tolerance', 1e-10)
        
        results = []
        overall_success = True
        
        for case in tqdm(test_cases, desc="Kernel tests"):
            try:
                # Generate test data
                ref_data, tar_data = self.data_generator.generate_kernel_test_data(
                    n_genes=case['n_genes'],
                    n_ref=case['n_ref'],
                    n_tar=case['n_tar'],
                    data_type=case['data_type'],
                    add_ties=case.get('add_ties', False)
                )
                
                # Test both float and hist kernels
                if case['data_type'] in ['float', 'mixed']:
                    float_success, float_result = self.kernel_tester.test_float_kernel(
                        ref_data, tar_data, tolerance
                    )
                    results.append({
                        'case': case['name'] + '_float',
                        'success': float_success,
                        'result': float_result
                    })
                    overall_success &= float_success
                
                if case['data_type'] in ['int', 'mixed']:
                    hist_success, hist_result = self.kernel_tester.test_hist_kernel(
                        ref_data.astype(np.int64), tar_data.astype(np.int64), tolerance
                    )
                    results.append({
                        'case': case['name'] + '_hist',
                        'success': hist_success,
                        'result': hist_result
                    })
                    overall_success &= hist_success
                
            except Exception as e:
                results.append({
                    'case': case['name'],
                    'success': False,
                    'result': {'error': str(e)}
                })
                overall_success = False
        
        return overall_success, {'test_results': results}
    
    def _run_pipeline_consistency(self) -> Tuple[bool, Dict[str, Any]]:
        """Run pipeline consistency tests against pdex."""
        print("🔬 Testing pipeline consistency against pdex...")
        
        test_datasets = self.config.get('pipeline_tests', {}).get('test_datasets', [])
        correlation_threshold = self.config.get('pipeline_tests', {}).get('correlation_threshold', 0.95)
        pdex_batch_size = self.config.get('pipeline_tests', {}).get('pdex_batch_size', 500)
        
        results = []
        overall_success = True
        
        for dataset_config in tqdm(test_datasets, desc="Pipeline tests"):
            try:
                # Generate synthetic single-cell dataset
                adata = self.data_generator.generate_sc_dataset(
                    n_cells=dataset_config['n_cells'],
                    n_genes=dataset_config['n_genes'],
                    n_groups=dataset_config['n_groups'],
                    differential_fraction=dataset_config.get('differential_fraction', 0.2),
                    effect_size=dataset_config.get('effect_size', 1.5)
                )
                
                # Test against pdex
                success, result = self.pipeline_tester.test_against_pdex(
                    adata=adata,
                    correlation_threshold=correlation_threshold,
                    pdex_batch_size=pdex_batch_size,
                    num_workers=self.env['num_workers']
                )
                
                results.append({
                    'dataset': dataset_config['name'],
                    'success': success,
                    'result': result
                })
                overall_success &= success
                
            except Exception as e:
                results.append({
                    'dataset': dataset_config['name'],
                    'success': False,
                    'result': {'error': str(e)}
                })
                overall_success = False
        
        return overall_success, {'test_results': results}
    
    def _run_performance_kernels(self) -> Tuple[bool, Dict[str, Any]]:
        """Run kernel performance tests against scipy."""
        print("⚡ Testing kernel performance against scipy...")
        
        benchmark_config = self.config.get('performance_tests', {}).get('kernel_benchmarks', {})
        test_cases = benchmark_config.get('test_cases', [])
        
        results = []
        overall_success = True
        
        for case in tqdm(test_cases, desc="Kernel benchmarks"):
            try:
                # Generate benchmark data
                ref_data, tar_data = self.data_generator.generate_kernel_test_data(
                    n_genes=case['n_genes'],
                    n_ref=case['n_ref'],
                    n_tar=case['n_tar'],
                    data_type='float'
                )
                
                # Run performance comparison
                success, result = self.performance_tester.benchmark_kernels(
                    ref_data, tar_data,
                    warmup_runs=benchmark_config.get('warmup_runs', 2),
                    benchmark_runs=benchmark_config.get('benchmark_runs', 5),
                    timeout_seconds=benchmark_config.get('timeout_seconds', 300)
                )
                
                results.append({
                    'case': case['name'],
                    'success': success,
                    'result': result
                })
                overall_success &= success
            
            except Exception as e:
                results.append({
                    'case': case['name'],
                    'success': False,
                    'result': {'error': str(e)}
                })
                overall_success = False
        
        return overall_success, {'benchmark_results': results}
    
    def _run_performance_pipeline(self) -> Tuple[bool, Dict[str, Any]]:
        """Run pipeline performance tests against pdex."""
        print("⚡ Testing pipeline performance against pdex...")
        
        benchmark_config = self.config.get('performance_tests', {}).get('pipeline_benchmarks', {})
        test_cases = benchmark_config.get('test_cases', [])
        
        results = []
        overall_success = True
        
        for case in tqdm(test_cases, desc="Pipeline benchmarks"):
            try:
                # Generate benchmark dataset
                adata = self.data_generator.generate_sc_dataset(
                    n_cells=case['n_cells'],
                    n_genes=case['n_genes'],
                    n_groups=case['n_groups']
                )
                
                # Run performance comparison
                success, result = self.performance_tester.benchmark_pipeline(
                    adata=adata,
                    num_workers=self.env['num_workers'],
                    warmup_runs=benchmark_config.get('warmup_runs', 1),
                    benchmark_runs=benchmark_config.get('benchmark_runs', 3),
                    timeout_seconds=benchmark_config.get('timeout_seconds', 600)
                )
                
                results.append({
                    'case': case['name'],
                    'success': success,
                    'result': result
                })
                overall_success &= success
        
            except Exception as e:
                results.append({
                    'case': case['name'],
                    'success': False,
                    'result': {'error': str(e)}
                })
                overall_success = False
        
        return overall_success, {'benchmark_results': results}
    
    def _run_performance_streaming(self) -> Tuple[bool, Dict[str, Any]]:
        """Run streaming vs regular performance comparison."""
        print("⚡ Testing streaming vs regular performance...")
        
        benchmark_config = self.config.get('performance_tests', {}).get('streaming_benchmarks', {})
        test_cases = benchmark_config.get('test_cases', [])
        memory_limits = benchmark_config.get('memory_limits', [2.0, 4.0, 8.0])
        
        results = []
        overall_success = True
        
        for case in tqdm(test_cases, desc="Streaming benchmarks"):
            try:
                # Generate large dataset
                adata = self.data_generator.generate_sc_dataset(
                    n_cells=case['n_cells'],
                    n_genes=case['n_genes'],
                    n_groups=case['n_groups']
                )
                
                # Test different memory limits
                for memory_limit in memory_limits:
                    success, result = self.performance_tester.benchmark_streaming(
                        adata=adata,
                        memory_limit_gb=memory_limit,
                        num_workers=self.env['num_workers'],
                        timeout_seconds=benchmark_config.get('timeout_seconds', 900)
                    )
                    
                    results.append({
                        'case': f"{case['name']}_mem{memory_limit}GB",
                        'success': success,
                        'result': result
                    })
                    overall_success &= success
                
            except Exception as e:
                results.append({
                    'case': case['name'],
                    'success': False,
                    'result': {'error': str(e)}
                })
                overall_success = False
        
        return overall_success, {'benchmark_results': results}
    
    def _run_edge_cases(self) -> Tuple[bool, Dict[str, Any]]:
        """Run edge case and robustness tests."""
        print("⚠️ Testing edge cases and robustness...")
        
        test_scenarios = self.config.get('edge_cases', {}).get('test_scenarios', [])
        
        results = []
        overall_success = True
        
        for scenario in tqdm(test_scenarios, desc="Edge case tests"):
            try:
                success, result = self.edge_case_tester.test_scenario(
                    scenario, self.data_generator
                )
                
                results.append({
                    'scenario': scenario['name'],
                    'success': success,
                    'result': result
                })
                overall_success &= success
                
            except Exception as e:
                results.append({
                    'scenario': scenario['name'],
                    'success': False,
                    'result': {'error': str(e)}
                })
                overall_success = False
        
        return overall_success, {'test_results': results}
    
    def _run_real_data_benchmarks(self) -> Tuple[bool, Dict[str, Any]]:
        """Run benchmarks on real datasets against pdex."""
        print("🧬 Testing real datasets against pdex...")
        
        num_workers = self.env.get('num_workers', 4)
        return self.real_data_tester.run_all_benchmarks(num_workers)
    
    def _generate_summary(self, category: str, details: Dict[str, Any], success: bool) -> str:
        """Generate a human-readable summary for a test category."""
        if not success:
            return "❌ Tests failed"
        
        if category == "kernel_consistency":
            results = details.get('test_results', [])
            passed = sum(1 for r in results if r['success'])
            total = len(results)
            return f"✅ {passed}/{total} kernel tests passed"
        
        elif category == "pipeline_consistency":
            results = details.get('test_results', [])
            passed = sum(1 for r in results if r['success'])
            total = len(results)
            return f"✅ {passed}/{total} pipeline tests passed"
        
        elif category.startswith("performance"):
            results = details.get('benchmark_results', [])
            passed = sum(1 for r in results if r['success'])
            total = len(results)
            return f"✅ {passed}/{total} benchmarks completed"
        
        elif category == "edge_cases":
            results = details.get('test_results', [])
            passed = sum(1 for r in results if r['success'])
            total = len(results)
            return f"✅ {passed}/{total} edge case tests passed"
        
        elif category == "real_data_benchmarks":
            passed = details.get('successful_datasets', 0)
            total = details.get('total_datasets', 0)
            if total > 0:
                avg_speedup = details.get('avg_speedup', 0)
                avg_correlation = details.get('avg_correlation', 0)
                return f"✅ {passed}/{total} datasets passed (avg {avg_speedup:.1f}x speedup, {avg_correlation:.3f} correlation)"
        else:
                return f"✅ {passed}/{total} real datasets tested"

        return "✅ Tests completed"