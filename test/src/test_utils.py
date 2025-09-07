"""
Test Utilities

Helper functions for the HPDEX testing framework.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml


def setup_logging(verbose: bool = False) -> None:
    """Setup logging for test framework."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Suppress some noisy loggers
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('anndata').setLevel(logging.WARNING)


def create_output_dir(base_dir: str = './test_results') -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(base_dir) / f'test_run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_results(results: Dict[str, Any], output_dir: Optional[Path] = None) -> Path:
    """Save test results to files."""
    if output_dir is None:
        output_dir = create_output_dir()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save summary JSON
    summary_file = output_dir / 'test_summary.json'
    summary = {
        'timestamp': timestamp,
        'categories_tested': list(results.keys()),
        'overall_success': all(
            r.get('success', False) for r in results.values() 
            if not isinstance(r, dict) or 'error' not in r
        ),
        'summary_by_category': {}
    }
    
    for category, result in results.items():
        if isinstance(result, dict) and 'summary' in result:
            summary['summary_by_category'][category] = result['summary']
        else:
            summary['summary_by_category'][category] = str(result)
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save detailed results
    detailed_file = output_dir / 'detailed_results.json'
    with open(detailed_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create readable report
    report_file = output_dir / 'test_report.md'
    create_markdown_report(results, report_file)
    
    print(f"Results saved to: {output_dir}")
    return summary_file


def create_markdown_report(results: Dict[str, Any], output_file: Path) -> None:
    """Create a human-readable markdown report."""
    
    with open(output_file, 'w') as f:
        f.write("# HPDEX Test Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall summary
        f.write("## Overall Summary\n\n")
        total_categories = len(results)
        successful_categories = sum(
            1 for r in results.values() 
            if isinstance(r, dict) and r.get('success', False)
        )
        
        f.write(f"- **Categories tested**: {total_categories}\n")
        f.write(f"- **Successful**: {successful_categories}\n")
        f.write(f"- **Failed**: {total_categories - successful_categories}\n")
        f.write(f"- **Success rate**: {successful_categories/total_categories:.1%}\n\n")
        
        # Detailed results by category
        for category, result in results.items():
            f.write(f"## {category.title().replace('_', ' ')}\n\n")
            
            if not isinstance(result, dict):
                f.write(f"Error: {result}\n\n")
                continue
            
            if 'error' in result:
                f.write(f"❌ **Failed**: {result['error']}\n\n")
                continue
            
            success = result.get('success', False)
            status = "✅ **Passed**" if success else "❌ **Failed**"
            f.write(f"{status}\n\n")
            
            # Summary
            if 'summary' in result:
                f.write(f"**Summary**: {result['summary']}\n\n")
            
            # Execution time
            if 'execution_time' in result:
                f.write(f"**Execution time**: {result['execution_time']:.1f}s\n\n")
            
            # Category-specific details
            if category == 'kernel_consistency':
                _write_kernel_details(f, result.get('details', {}))
            elif category == 'pipeline_consistency':
                _write_pipeline_details(f, result.get('details', {}))
            elif category.startswith('performance'):
                _write_performance_details(f, result.get('details', {}))
            elif category == 'edge_cases':
                _write_edge_case_details(f, result.get('details', {}))
            
            f.write("---\n\n")


def _write_kernel_details(f, details: Dict[str, Any]) -> None:
    """Write kernel test details to markdown."""
    test_results = details.get('test_results', [])
    
    if test_results:
        f.write("### Test Results\n\n")
        f.write("| Test Case | Success | Max P Error | Correlation |\n")
        f.write("|-----------|---------|-------------|-------------|\n")
        
        for result in test_results:
            case = result['case']
            success = "✅" if result['success'] else "❌"
            
            if 'result' in result and isinstance(result['result'], dict):
                max_p_error = result['result'].get('max_p_error', 'N/A')
                correlation = result['result'].get('correlation_p', 'N/A')
                if isinstance(max_p_error, float):
                    max_p_error = f"{max_p_error:.2e}"
                if isinstance(correlation, float):
                    correlation = f"{correlation:.4f}"
            else:
                max_p_error = "Error"
                correlation = "Error"
            
            f.write(f"| {case} | {success} | {max_p_error} | {correlation} |\n")
        
        f.write("\n")


def _write_pipeline_details(f, details: Dict[str, Any]) -> None:
    """Write pipeline test details to markdown."""
    test_results = details.get('test_results', [])
    
    if test_results:
        f.write("### Test Results\n\n")
        f.write("| Dataset | Success | P-value Correlation | N Comparisons |\n")
        f.write("|---------|---------|-------------------|---------------|\n")
        
        for result in test_results:
            dataset = result['dataset']
            success = "✅" if result['success'] else "❌"
            
            if 'result' in result and isinstance(result['result'], dict):
                corr = result['result'].get('correlations', {}).get('p_value', 'N/A')
                n_comp = result['result'].get('n_comparisons', 'N/A')
                if isinstance(corr, float):
                    corr = f"{corr:.4f}"
            else:
                corr = "Error"
                n_comp = "Error"
            
            f.write(f"| {dataset} | {success} | {corr} | {n_comp} |\n")
        
        f.write("\n")


def _write_performance_details(f, details: Dict[str, Any]) -> None:
    """Write performance test details to markdown."""
    benchmark_results = details.get('benchmark_results', [])
    
    if benchmark_results:
        f.write("### Benchmark Results\n\n")
        f.write("| Test Case | Success | HPDEX Time | Baseline Time | Speedup |\n")
        f.write("|-----------|---------|------------|---------------|----------|\n")
        
        for result in benchmark_results:
            case = result['case']
            success = "✅" if result['success'] else "❌"
            
            if 'result' in result and isinstance(result['result'], dict):
                res = result['result']
                hpdex_time = res.get('hpdex', {}).get('mean_time', 'N/A')
                baseline_time = res.get('scipy', {}).get('mean_time') or res.get('pdex', {}).get('mean_time', 'N/A')
                speedup = res.get('speedup', 'N/A')
                
                if isinstance(hpdex_time, float):
                    hpdex_time = f"{hpdex_time:.3f}s"
                if isinstance(baseline_time, float):
                    baseline_time = f"{baseline_time:.3f}s"
                if isinstance(speedup, float):
                    speedup = f"{speedup:.2f}x"
            else:
                hpdex_time = "Error"
                baseline_time = "Error"
                speedup = "Error"
            
            f.write(f"| {case} | {success} | {hpdex_time} | {baseline_time} | {speedup} |\n")
        
        f.write("\n")


def _write_edge_case_details(f, details: Dict[str, Any]) -> None:
    """Write edge case test details to markdown."""
    test_results = details.get('test_results', [])
    
    if test_results:
        f.write("### Edge Case Results\n\n")
        f.write("| Scenario | Success | Details |\n")
        f.write("|----------|---------|----------|\n")
        
        for result in test_results:
            scenario = result['scenario']
            success = "✅" if result['success'] else "❌"
            
            if 'result' in result and isinstance(result['result'], dict):
                res = result['result']
                if 'error' in res:
                    details_str = f"Error: {res['error']}"
                else:
                    details_str = f"{res.get('n_results', 0)} results"
            else:
                details_str = "Error"
            
            f.write(f"| {scenario} | {success} | {details_str} |\n")
        
        f.write("\n")


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_size(bytes_size: float) -> str:
    """Format byte size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"{self.description} completed in {format_time(elapsed)}")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
