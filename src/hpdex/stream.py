"""
Streaming backend for hpdex to handle extremely large datasets (100GB+).

This module provides memory-efficient differential expression analysis for datasets
that cannot fit entirely in memory by using streaming data loading and processing.
"""

import gc
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import core functions from main backend
from .backend.de_ import (_auto_schedule_chunk_size, _compute_fdr,
                          _compute_fold_changes, _compute_log2_fold_change,
                          _execute_multiprocess_computation,
                          _execute_singlethread_computation)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Memory usage monitor for streaming operations."""
    
    def __init__(self, memory_limit_gb: float):
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        self.current_usage = 0
        
    def estimate_group_memory(self, n_cells: int, n_genes: int, dtype=np.float64) -> int:
        """Estimate memory usage for a single group's dense data."""
        bytes_per_element = np.dtype(dtype).itemsize
        # Add small overhead for array metadata and alignment
        base_memory = n_cells * n_genes * bytes_per_element
        overhead = max(1024, int(base_memory * 0.05))  # 5% overhead or 1KB minimum
        return base_memory + overhead
        
    def can_load_group(self, n_cells: int, n_genes: int) -> bool:
        """Check if we can load a group without exceeding memory limit."""
        estimated_usage = self.estimate_group_memory(n_cells, n_genes)
        return (self.current_usage + estimated_usage) < self.memory_limit_bytes
        
    def update_usage(self, delta_bytes: int):
        """Update current memory usage."""
        self.current_usage += delta_bytes
        
    def get_usage_gb(self) -> float:
        """Get current usage in GB."""
        return self.current_usage / (1024 * 1024 * 1024)

def _stream_load_groups_batch(
    data_path: Union[str, Path],
    group_indices_dict: Dict[str, np.ndarray],
    preprocessing_pipeline: Optional[Callable] = None
) -> Dict[str, np.ndarray]:
    """
    Stream load data for multiple groups in one operation.
    
    Args:
        data_path: Path to the AnnData file
        group_indices_dict: Dict mapping group names to their cell indices
        preprocessing_pipeline: Optional preprocessing function
        
    Returns:
        Dict mapping group names to their dense numpy arrays
    """
    if not group_indices_dict:
        raise ValueError("No groups provided for loading")
        
    total_cells = sum(len(indices) for indices in group_indices_dict.values())
    logger.debug(f"Streaming {total_cells} cells for {len(group_indices_dict)} groups from {data_path}")
    
    # Open file in backed mode for streaming access
    adata = ad.read_h5ad(data_path, backed='r')
    
    try:
        n_genes = adata.shape[1]
        
        # Combine all indices for efficient loading
        all_indices = []
        group_slices = {}
        current_pos = 0
        
        for group_name, indices in group_indices_dict.items():
            if len(indices) == 0:
                logger.warning(f"Group '{group_name}' has no cells, skipping")
                continue
            group_size = len(indices)
            all_indices.extend(indices.tolist())  # Ensure it's a list
            group_slices[group_name] = (current_pos, current_pos + group_size)
            current_pos += group_size
            
        if not all_indices:
            raise ValueError("No valid cells found in any group")
        
        # Load all required cells at once
        logger.debug(f"Loading {len(all_indices)} cells across {len(group_indices_dict)} groups")
        
        # Load data for all groups
        all_adata = adata[all_indices, :].to_memory()
        
        # Apply preprocessing if provided
        if preprocessing_pipeline is not None:
            logger.debug(f"Applying preprocessing pipeline to {all_adata.shape[0]} cells")
            all_adata = preprocessing_pipeline(all_adata)
        
        # Convert to dense with proper error handling
        try:
            if hasattr(all_adata.X, 'toarray'):
                all_data_dense = all_adata.X.toarray().astype(np.float64)
            else:
                all_data_dense = np.asarray(all_adata.X, dtype=np.float64)
        except MemoryError:
            raise MemoryError(f"Failed to convert {all_adata.shape} data to dense format. "
                            f"Consider reducing batch size or increasing memory limit.")
        
        # Split data back into groups
        result_dict = {}
        for group_name, (start_idx, end_idx) in group_slices.items():
            group_data = all_data_dense[start_idx:end_idx, :].copy()  # Ensure independent copy
            result_dict[group_name] = group_data
            logger.debug(f"Extracted {group_name}: {group_data.shape}")
        
        # Clean up
        del all_adata, all_data_dense
        gc.collect()
        
        return result_dict
        
    finally:
        if hasattr(adata, 'file') and adata.file is not None:
            adata.file.close()

def _stream_load_single_group(
    data_path: Union[str, Path],
    group_indices: np.ndarray,
    preprocessing_pipeline: Optional[Callable] = None
) -> np.ndarray:
    """
    Stream load data for a single group.
    
    Args:
        data_path: Path to the AnnData file
        group_indices: Indices of cells in this group
        preprocessing_pipeline: Optional preprocessing function
        
    Returns:
        Dense numpy array for the group
    """
    
    result_dict = _stream_load_groups_batch(
        data_path, {'group': group_indices}, preprocessing_pipeline
    )
    return result_dict['group']

def _plan_group_batches(
    target_groups: List[str],
    group_counts: Dict[str, int],
    n_vars: int,
    memory_monitor: MemoryMonitor,
    ref_memory_used: int
) -> List[List[str]]:
    """
    Plan how to batch target groups based on memory constraints.
    
    Args:
        target_groups: List of target group names
        group_counts: Dict mapping group names to cell counts
        n_vars: Number of genes
        memory_monitor: Memory monitoring object
        ref_memory_used: Memory already used by reference data
        
    Returns:
        List of batches, where each batch is a list of group names
        
    Raises:
        MemoryError: If any group cannot fit in available memory by itself
    """
    
    available_memory = memory_monitor.memory_limit_bytes - ref_memory_used
    available_gb = available_memory / (1024**3)
    logger.debug(f"Available memory for target groups: {available_gb:.2f} GB")
    
    # STEP 1: Pre-validate all groups can fit individually
    logger.debug("   Validating all groups can fit in available memory...")
    invalid_groups = []
    
    for group in target_groups:
        group_memory = memory_monitor.estimate_group_memory(group_counts[group], n_vars)
        group_gb = group_memory / (1024**3)
        
        if group_memory > available_memory:
            invalid_groups.append((group, group_gb))
    
    # Report all invalid groups at once
    if invalid_groups:
        error_msg = f"The following groups cannot fit in available memory ({available_gb:.1f}GB):\n"
        for group, group_gb in invalid_groups:
            error_msg += f"  - '{group}': requires {group_gb:.1f}GB ({group_counts[group]:,} cells)\n"
        error_msg += f"\nConsider increasing memory_limit_gb or reducing group sizes."
        raise MemoryError(error_msg.strip())
    
    # STEP 2: Plan batches using greedy bin-packing algorithm
    logger.debug("   All groups validated. Planning optimal batching...")
    
    # Sort groups by size (largest first) for better packing efficiency
    sorted_groups = sorted(target_groups, key=lambda g: group_counts[g], reverse=True)
    
    batches = []
    current_batch = []
    current_batch_memory = 0
    
    for group in sorted_groups:
        group_memory = memory_monitor.estimate_group_memory(group_counts[group], n_vars)
        
        # Check if this group can fit in current batch
        if current_batch_memory + group_memory <= available_memory:
            current_batch.append(group)
            current_batch_memory += group_memory
        else:
            # Start new batch (current batch must not be empty since we pre-validated)
            if current_batch:
                batches.append(current_batch)
            
            # Start new batch with current group
            current_batch = [group]
            current_batch_memory = group_memory
    
    # Add final batch if not empty
    if current_batch:
        batches.append(current_batch)
    
    # STEP 3: Report batching plan
    logger.info(f"   Planned {len(batches)} batches for {len(target_groups)} target groups:")
    total_cells = 0
    total_memory = 0
    
    for i, batch in enumerate(batches):
        batch_cells = sum(group_counts[g] for g in batch)
        batch_memory = sum(memory_monitor.estimate_group_memory(group_counts[g], n_vars) for g in batch)
        batch_memory_gb = batch_memory / (1024**3)
        utilization = (batch_memory / available_memory) * 100
        
        total_cells += batch_cells
        total_memory += batch_memory
        
        logger.info(f"     Batch {i+1}: {len(batch)} groups, {batch_cells:,} cells, "
                   f"{batch_memory_gb:.2f}GB ({utilization:.1f}% utilization)")
        logger.debug(f"       Groups: {batch}")
    
    total_memory_gb = total_memory / (1024**3)
    avg_utilization = (total_memory / (available_memory * len(batches))) * 100
    
    logger.info(f"   Batching summary: {total_cells:,} total cells, {total_memory_gb:.2f}GB total, "
               f"{avg_utilization:.1f}% average utilization")
    
    return batches

def _get_group_indices(
    data_path: Union[str, Path],
    groupby_key: str,
    reference: str,
    target_groups: List[str]
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    """
    Get group indices for reference and target groups in one file read.
    
    Args:
        data_path: Path to the AnnData file
        groupby_key: Column name for grouping
        reference: Reference group name
        target_groups: List of target group names
        
    Returns:
        Tuple of (ref_indices, target_indices_dict, gene_names)
        
    Raises:
        ValueError: If reference group contains no cells
    """
    
    adata = ad.read_h5ad(data_path, backed='r')
    try:
        group_obs = adata.obs[groupby_key]
        
        # Get reference indices
        ref_indices = np.where(group_obs == reference)[0]
        if len(ref_indices) == 0:
            raise ValueError(f"Reference group '{reference}' contains no cells")
        
        # Get target group indices
        target_indices_dict = {
            group: np.where(group_obs == group)[0]
            for group in target_groups
        }
        
        # Get gene names
        gene_names = adata.var_names.tolist()
        
        return ref_indices, target_indices_dict, gene_names
        
    finally:
        if hasattr(adata, 'file') and adata.file is not None:
            adata.file.close()


def _get_dataset_info(data_path: Union[str, Path], groupby_key: str) -> Dict:
    """
    Get basic information about the dataset without loading it entirely.
    
    Args:
        data_path: Path to the AnnData file
        groupby_key: Column name for grouping
        
    Returns:
        Dictionary with dataset information
        
    Raises:
        KeyError: If groupby_key not found in obs
        ValueError: If dataset is invalid
    """
    
    adata = ad.read_h5ad(data_path, backed='r')
    try:
        n_obs, n_vars = adata.shape
        
        if n_obs == 0 or n_vars == 0:
            raise ValueError(f"Invalid dataset dimensions: {n_obs} cells × {n_vars} genes")
        
        # Validate groupby_key exists
        if groupby_key not in adata.obs.columns:
            available_keys = list(adata.obs.columns)
            raise KeyError(f"Grouping key '{groupby_key}' not found. Available keys: {available_keys}")
        
        # Load only the grouping information
        group_info = adata.obs[groupby_key]
        unique_groups = group_info.unique()
        group_counts = group_info.value_counts().to_dict()
        
        # Validate groups
        if len(unique_groups) < 2:
            raise ValueError(f"Dataset must have at least 2 groups, found: {len(unique_groups)}")
        
        # Estimate sparsity if sparse
        sparsity = None
        if hasattr(adata.X, 'nnz'):
            sparsity = adata.X.nnz / (n_obs * n_vars)
            
        return {
            'n_obs': n_obs,
            'n_vars': n_vars,
            'groups': unique_groups,
            'group_counts': group_counts,
            'sparsity': sparsity,
            'dtype': adata.X.dtype
        }
    finally:
        if hasattr(adata, 'file') and adata.file is not None:
            adata.file.close()

def parallel_differential_expression(
    data_path: Union[str, Path],
    groupby_key: str,
    reference: str,
    groups: Optional[List[str]] = None,
    preprocessing_pipeline: Optional[Callable] = None,
    memory_limit_gb: float = 16.0,
    metric: str = "wilcoxon",
    tie_correction: bool = True,
    continuity_correction: bool = True,
    use_asymptotic: Optional[bool] = None,
    min_samples: int = 2,
    max_bins: int = 100_000,
    prefer_hist_if_int: bool = False,
    num_workers: int = 1,
    clip_value: float = 20.0,
) -> pd.DataFrame:
    """Stream-based parallel differential expression analysis for extremely large datasets.
    
    This function handles datasets that cannot fit entirely in memory by streaming
    data loading and processing groups in memory-efficient batches.
    
    Args:
        data_path: Path to the AnnData file (.h5ad format)
        groupby_key: Column name in obs for grouping cells
        reference: Name of the reference group
        groups: List of target groups to compare. If None, uses all groups except reference
        preprocessing_pipeline: Optional function to preprocess each loaded chunk
        memory_limit_gb: Maximum memory usage in GB (default: 16GB)
        metric: Statistical test method. Currently supports "wilcoxon" and "wilcoxon-hist"
        tie_correction: Whether to apply tie correction
        continuity_correction: Whether to apply continuity correction
        use_asymptotic: Force asymptotic approximation. None for auto-selection
        min_samples: Minimum number of samples per group
        max_bins: Maximum number of bins for histogram algorithm
        prefer_hist_if_int: Prefer histogram algorithm for integer data
        num_workers: Number of parallel worker processes (for within-chunk processing)
        clip_value: Value to clip fold change if infinite or NaN
        
    Returns:
        DataFrame containing results with columns:
            - 'target': Treatment group name
            - 'feature': Gene name
            - 'p_value': P-value from statistical test
            - 'fold_change': Fold change (target_mean / reference_mean)
            - 'log2_fold_change': Log2 fold change
            - 'fdr': FDR-corrected p-value
            
    Raises:
        MemoryError: If even a single group cannot fit in memory
        ValueError: If reference group not found or invalid parameters
        TypeError: If unsupported data types are used
    """
    
    logger.info(f"🌊 Starting streaming differential expression analysis")
    logger.info(f"   Data path: {data_path}")
    logger.info(f"   Memory limit: {memory_limit_gb:.1f} GB")
    logger.info(f"   Workers: {num_workers}, Metric: {metric}")
    
    # Validate inputs
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    # Get dataset information without loading all data
    logger.info("📊 Analyzing dataset structure...")
    dataset_info = _get_dataset_info(data_path, groupby_key)
    
    n_obs = dataset_info['n_obs']
    n_vars = dataset_info['n_vars']
    all_groups = dataset_info['groups']
    group_counts = dataset_info['group_counts']
    
    logger.info(f"   Dataset: {n_obs:,} cells × {n_vars:,} genes")
    logger.info(f"   Groups: {list(all_groups)}")
    logger.info(f"   Group sizes: {group_counts}")
    
    # Validate reference group
    if reference not in all_groups:
        raise ValueError(f"Reference group '{reference}' not found in {groupby_key}")
        
    # Determine target groups
    if groups is None:
        target_groups = [g for g in all_groups if g != reference and group_counts[g] >= min_samples]
    else:
        target_groups = [g for g in groups if g != reference and g in all_groups and group_counts[g] >= min_samples]
        
    if not target_groups:
        raise ValueError("No valid target groups found")
        
    logger.info(f"   Target groups: {target_groups}")
    logger.info(f"   Reference: {reference} ({group_counts[reference]:,} cells)")
    
    # Validate metric parameter
    if metric not in ["wilcoxon", "wilcoxon-hist"]:
        raise ValueError(f"Unsupported statistical test method: {metric}. Available: ['wilcoxon', 'wilcoxon-hist']")
    
    # Initialize memory monitor
    memory_monitor = MemoryMonitor(memory_limit_gb)
    
    # Check if reference group can fit in memory
    ref_memory_needed = memory_monitor.estimate_group_memory(group_counts[reference], n_vars)
    if ref_memory_needed > memory_monitor.memory_limit_bytes:
        ref_gb = ref_memory_needed / (1024**3)
        raise MemoryError(f"Reference group requires {ref_gb:.1f}GB but limit is {memory_limit_gb:.1f}GB")
    
    # Step 1: Load and pre-sort reference data
    logger.info("📋 Loading and pre-sorting reference data...")
    
    # Get group indices efficiently - single file read
    logger.debug("   Getting group indices...")
    ref_indices, target_indices_dict, gene_names = _get_group_indices(
        data_path, groupby_key, reference, target_groups
    )
    
    # Stream load reference data
    ref_data_dense = _stream_load_single_group(
        data_path, ref_indices, preprocessing_pipeline
    )
    
    # Update memory usage
    memory_monitor.update_usage(ref_data_dense.nbytes)
    logger.info(f"   Reference loaded: {memory_monitor.get_usage_gb():.2f}GB used")
    
    # Pre-sort reference data for optimal performance
    logger.info("   Pre-sorting reference data...")
    ref_data_sorted = np.sort(ref_data_dense, axis=0)  # Sort along cell axis for each gene
    
    # Step 2: Plan batch processing strategy
    logger.info(f"🔄 Planning batch processing for {len(target_groups)} target groups...")
    
    group_batches = _plan_group_batches(
        target_groups, group_counts, n_vars, memory_monitor, ref_data_dense.nbytes
    )
    
    # Prepare results storage
    all_results = []
    
    # Pre-calculate reference means for fold change calculation (avoid recalculation)
    ref_means = np.mean(ref_data_sorted, axis=0)
    
    # target_indices_dict already loaded above to avoid duplicate file reads
    
    # Step 3: Process batches
    with tqdm(total=len(target_groups), desc="🔄 Processing groups") as pbar:
        for batch_idx, batch_groups in enumerate(group_batches):
            logger.info(f"\n🔄 Processing batch {batch_idx+1}/{len(group_batches)}: {batch_groups}")
            
            # Prepare indices for this batch
            batch_indices_dict = {
                group: target_indices_dict[group] 
                for group in batch_groups
            }
            
            # Load all groups in this batch
            with tqdm(total=len(batch_groups), desc=f"   📥 Loading batch {batch_idx+1}") as load_pbar:
                logger.info(f"   Loading {len(batch_groups)} groups...")
                batch_data_dict = _stream_load_groups_batch(
                    data_path, batch_indices_dict, preprocessing_pipeline
                )
                load_pbar.update(len(batch_groups))
            
            # Process batch efficiently - combine all groups in the batch for better performance
            if len(batch_groups) == 1:
                # Single group - use existing logic
                group = batch_groups[0]
                target_data_dense = batch_data_dict[group]
                
                logger.info(f"   📊 Computing differential expression for {group}: {target_data_dense.shape}")
                
                # Use auto-calculated chunk size based on data size
                data_size_mb = (ref_data_sorted.nbytes + target_data_dense.nbytes) / (1024 * 1024)
                chunk_size = _auto_schedule_chunk_size(
                    n_vars, 1, num_workers, data_size_mb
                )
                logger.debug(f"Using chunk size {chunk_size} for {data_size_mb:.1f}MB data")
                
                # Determine algorithm
                use_hist = (metric == "wilcoxon-hist") or (
                    prefer_hist_if_int and metric == "wilcoxon" and
                    np.issubdtype(target_data_dense.dtype, np.integer)
                )
                
                # Execute computation (single-thread or multi-process) for this group
                group_indices = [np.arange(len(target_data_dense))]  # Local indices for this group's data
                group_names_list = [group]
                
                if num_workers > 1:
                    logger.debug(f"     Using multiprocessing with {num_workers} workers for {group}")
                    group_results_list = _execute_multiprocess_computation(
                        target_data_dense, ref_data_sorted, group_indices, group_names_list, n_vars,
                        chunk_size, metric, tie_correction, continuity_correction, 
                        use_asymptotic, max_bins, gene_names
                    )
                else:
                    logger.debug(f"     Using single-threaded processing for {group}")
                    group_results_list = _execute_singlethread_computation(
                        target_data_dense, ref_data_sorted, group_indices, group_names_list, n_vars,
                        chunk_size, metric, tie_correction, continuity_correction,
                        use_asymptotic, max_bins, gene_names, use_hist
                    )
                
                # Get the result for this group and add fold changes
                group_results = group_results_list[0]  # Should only be one result
                
                # Calculate fold changes efficiently
                target_means = np.mean(target_data_dense, axis=0)
                
                # Calculate fold changes using unified function
                fold_changes = _compute_fold_changes(target_means, ref_means, clip_value)
                log2_fold_changes = _compute_log2_fold_change(fold_changes)
                
                # Add fold change columns
                group_results['fold_change'] = fold_changes
                group_results['log2_fold_change'] = log2_fold_changes
                
                all_results.append(group_results)
                
                logger.info(f"   ✅ Group {group} completed")
                pbar.update(1)
            
            else:
                # Multiple groups - batch process for better efficiency
                logger.info(f"   📊 Batch processing {len(batch_groups)} groups together")
                
                # Combine all target data for the batch
                batch_target_data = []
                batch_group_indices = []
                batch_group_names = []
                current_start = 0
                
                for group in batch_groups:
                    target_data = batch_data_dict[group]
                    batch_target_data.append(target_data)
                    
                    # Create group indices for this group in the combined matrix
                    group_size = len(target_data)
                    group_indices_range = np.arange(current_start, current_start + group_size)
                    batch_group_indices.append(group_indices_range)
                    batch_group_names.append(group)
                    current_start += group_size
                
                # Combine all target data into one matrix
                combined_target_data = np.vstack(batch_target_data)
                
                # Use auto-calculated chunk size based on combined data size
                data_size_mb = (ref_data_sorted.nbytes + combined_target_data.nbytes) / (1024 * 1024)
                chunk_size = _auto_schedule_chunk_size(
                    n_vars, len(batch_groups), num_workers, data_size_mb
                )
                logger.debug(f"Batch processing: chunk_size={chunk_size} for {data_size_mb:.1f}MB combined data")
                
                # Determine algorithm
                use_hist = (metric == "wilcoxon-hist") or (
                    prefer_hist_if_int and metric == "wilcoxon" and
                    np.issubdtype(combined_target_data.dtype, np.integer)
                )
                
                # Execute computation for all groups in the batch
                if num_workers > 1:
                    logger.debug(f"     Using multiprocessing with {num_workers} workers for batch")
                    batch_results_list = _execute_multiprocess_computation(
                        combined_target_data, ref_data_sorted, batch_group_indices, batch_group_names, n_vars,
                        chunk_size, metric, tie_correction, continuity_correction, 
                        use_asymptotic, max_bins, gene_names
                    )
                else:
                    logger.debug(f"     Using single-threaded processing for batch")
                    batch_results_list = _execute_singlethread_computation(
                        combined_target_data, ref_data_sorted, batch_group_indices, batch_group_names, n_vars,
                        chunk_size, metric, tie_correction, continuity_correction,
                        use_asymptotic, max_bins, gene_names, use_hist
                    )
                
                # Process results and add fold changes for each group
                
                for i, group_results in enumerate(batch_results_list):
                    group = batch_group_names[i]
                    target_data_for_group = batch_target_data[i]
                    
                    # Calculate fold changes for this group
                    target_means = np.mean(target_data_for_group, axis=0)
                    fold_changes = _compute_fold_changes(target_means, ref_means, clip_value)
                    log2_fold_changes = _compute_log2_fold_change(fold_changes)
                    
                    # Add fold change columns
                    group_results['fold_change'] = fold_changes
                    group_results['log2_fold_change'] = log2_fold_changes
                    
                    all_results.append(group_results)
                    
                    logger.info(f"   ✅ Group {group} completed")
                    pbar.update(1)
            
            # Clean up batch data to free memory
            del batch_data_dict
            gc.collect()
            
            logger.info(f"   🔄 Batch {batch_idx+1} completed, memory freed")
    
    # Step 3: Combine results and apply FDR correction
    logger.info("📊 Combining results and applying FDR correction...")
    
    if not all_results:
        raise ValueError("No results generated")
        
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Apply FDR correction across all tests
    logger.info("   Applying FDR correction...")
    combined_results['fdr'] = _compute_fdr(combined_results['p_value'].values)
    
    # Clean up reference data
    del ref_data_dense, ref_data_sorted
    gc.collect()
    
    logger.info(f"✅ Streaming analysis completed!")
    logger.info(f"   Total results: {len(combined_results):,}")
    logger.info(f"   Groups analyzed: {len(target_groups)}")
    logger.info(f"   Genes analyzed: {n_vars:,}")
    
    return combined_results
