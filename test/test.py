#!/usr/bin/env python3
"""
HPDEX Comprehensive Testing Framework

Usage:
    python test.py config.yml
    python test.py config.yml --category kernel_consistency
    python test.py config.yml --list-categories
    python test.py config.yml --dry-run
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.test_runner import TestRunner
from src.test_utils import setup_logging, save_results, create_output_dir


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['environment', 'test_categories']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section in config: {section}")
    
    return config


def list_categories(config: Dict[str, Any]) -> None:
    """List available test categories."""
    print("Available test categories:")
    print("=" * 50)
    
    categories = config['test_categories']
    for category, enabled in categories.items():
        status = "✓ enabled" if enabled else "✗ disabled"
        print(f"  {category:<25} {status}")
    
    print("\nUse --category <name> to run specific category")


def main():
    parser = argparse.ArgumentParser(
        description="HPDEX Comprehensive Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py config.yml                          # Run all enabled tests
  python test.py config.yml --category kernel_tests  # Run specific category
  python test.py config.yml --list-categories        # List available categories
  python test.py config.yml --dry-run               # Show what would be tested
        """
    )
    
    parser.add_argument('config', type=Path, help='Path to configuration YAML file')
    parser.add_argument('--category', type=str, help='Run specific test category only')
    parser.add_argument('--list-categories', action='store_true', help='List available categories')
    parser.add_argument('--dry-run', action='store_true', help='Show test plan without running')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return 1
    
    # Override verbose setting
    if args.verbose:
        config['environment']['verbose'] = True
    
    # List categories if requested
    if args.list_categories:
        list_categories(config)
        return 0
    
    # Setup environment
    setup_logging(config['environment'].get('verbose', False))
    
    if config['output'].get('save_results', True) and not args.no_save:
        results_dir = create_output_dir(config['output'].get('results_dir', './test_results'))
        config['_results_dir'] = results_dir
    
    # Create test runner
    runner = TestRunner(config)
    
    print("🧪 HPDEX Testing Framework")
    print("=" * 50)
    print(f"Configuration: {args.config}")
    print(f"Random seed: {config['environment']['random_seed']}")
    print(f"Workers: {config['environment']['num_workers']}")
    print(f"Memory limit: {config['environment']['memory_limit_gb']} GB")
    
    # Determine which categories to run
    if args.category:
        if args.category not in config['test_categories']:
            print(f"❌ Unknown category: {args.category}")
            print("\nAvailable categories:")
            list_categories(config)
            return 1
        
        categories_to_run = {args.category: True}
        print(f"Running category: {args.category}")
    else:
        categories_to_run = {
            cat: enabled for cat, enabled in config['test_categories'].items() 
            if enabled
        }
        print(f"Running {len(categories_to_run)} enabled categories")
    
    if not categories_to_run:
        print("⚠️ No test categories enabled!")
        return 0
    
    # Show test plan
    if args.dry_run:
        print("\n📋 Test Plan (dry run):")
        runner.show_test_plan(categories_to_run)
        return 0
    
    print(f"\nCategories: {', '.join(categories_to_run.keys())}")
    print()
    
    # Run tests
    start_time = time.time()
    overall_success = True
    results = {}
    
    try:
        for category in categories_to_run:
            print(f"\n{'='*60}")
            print(f"🎯 Running: {category}")
            print(f"{'='*60}")
            
            try:
                success, category_results = runner.run_category(category)
                results[category] = category_results
                
                if success:
                    print(f"✅ {category} completed successfully")
                else:
                    print(f"❌ {category} had failures")
                    overall_success = False
                    
                    if not config.get('error_handling', {}).get('continue_on_error', True):
                        print("❌ Stopping due to error (continue_on_error=false)")
                        break
                        
            except Exception as e:
                print(f"💥 {category} crashed: {e}")
                if config['environment'].get('verbose', False):
                    traceback.print_exc()
                
                overall_success = False
                results[category] = {'error': str(e), 'traceback': traceback.format_exc()}
                
                if not config.get('error_handling', {}).get('continue_on_error', True):
                    break
    
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
        overall_success = False
    
    # Final summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Categories run: {len(results)}")
    
    for category, result in results.items():
        if 'error' in result:
            print(f"  💥 {category}: ERROR")
        else:
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            print(f"  {status} {category}: {result.get('summary', 'No summary')}")
    
    # Save results
    if config['output'].get('save_results', True) and not args.no_save:
        results_file = save_results(results, config.get('_results_dir'))
        print(f"\n💾 Results saved to: {results_file}")
    
    # Return appropriate exit code
    if overall_success:
        print(f"\n🎉 All tests completed successfully!")
        return 0
    else:
        print(f"\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
