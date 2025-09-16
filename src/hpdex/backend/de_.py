"""
High-Performance Parallel Differential Expression Analysis for single-cell data.

A high-performance parallel differential expression analysis tool for single-cell data.
Uses shared memory multiprocessing to compute differential expression genes with 
algorithmic alignment to pdex library while providing superior computational performance.

Key Features:
- Shared memory parallelization to avoid data copying
- Optimized histogram algorithm for integer data
- Efficient implementation of Mann-Whitney U test
- FDR correction and fold change calculation
- Progress tracking and comprehensive error handling

"""

import logging
import multiprocessing as mp
import multiprocessing.shared_memory as shm
from typing import List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from tqdm import tqdm

from .pykernel_ import rank_sum_chunk_kernel_float, rank_sum_chunk_kernel_hist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_METRICS = ["wilcoxon"]


# -- Worker Functions for Multiprocessing --
def _chunk_worker(args) -> None:
    """
    Chunk worker for multiprocessing support.
    
    Args:
        args: Tuple containing worker specifications and task parameters
    """
    (data_spec, ref_spec, result_spec, task, metric, tie_correction, 
     continuity_correction, use_asymptotic, max_bins, use_hist) = args
    
    
    data_shm = None
    ref_shm = None  
    result_shm = None
    
    try:
        # Open shared memory
        data_shm = shm.SharedMemory(name=data_spec['name'])
        ref_shm = shm.SharedMemory(name=ref_spec['name'])
        result_shm = shm.SharedMemory(name=result_spec['name'])
        
        # Create numpy arrays
        X_dense = np.ndarray(data_spec['shape'], dtype=data_spec['dtype'], buffer=data_shm.buf)
        ref_data_sorted = np.ndarray(ref_spec['shape'], dtype=ref_spec['dtype'], buffer=ref_shm.buf)
        result_array = np.ndarray(result_spec['shape'], dtype=result_spec['dtype'], buffer=result_shm.buf)
        
        group_idx, gene_start, gene_end, group_indices = task
        n_genes_total = data_spec['shape'][1]
        
        # Extract target data for this chunk
        target_data_dense = X_dense[group_indices]  # (n_target_cells, n_genes_total)
        n_genes_chunk = gene_end - gene_start
        
        # Prepare data based on metric type and compute
        if metric == "wilcoxon":
            # Float kernel expects pre-sorted data, shape (n_genes, n_cells)
            ref_chunk = ref_data_sorted[:, gene_start:gene_end].T  # (n_genes_chunk, n_ref_cells)
            if use_hist:
                tar_chunk = target_data_dense[:, gene_start:gene_end].T  # (n_genes_chunk, n_target_cells)
                
                chunk_p, chunk_u = rank_sum_chunk_kernel_hist(
                    ref_chunk, tar_chunk.astype(np.int64),
                    tie_correction=tie_correction,
                    continuity_correction=continuity_correction,
                    use_asymptotic=use_asymptotic,
                    max_bins=max_bins
                )
            else:
                tar_chunk = np.sort(target_data_dense[:, gene_start:gene_end].T, axis=1)  # (n_genes_chunk, n_target_cells)
                
                chunk_p, chunk_u = rank_sum_chunk_kernel_float(
                    ref_chunk, tar_chunk,
                    tie_correction=tie_correction,
                    continuity_correction=continuity_correction,
                    use_asymptotic=use_asymptotic
                )
        
        # Store results in shared memory
        p_start = group_idx * n_genes_total + gene_start
        u_start = len(result_spec['group_counts']) * n_genes_total + group_idx * n_genes_total + gene_start
        
        result_array[p_start:p_start + n_genes_chunk] = chunk_p
        result_array[u_start:u_start + n_genes_chunk] = chunk_u
        
    except Exception as e:
        logger.warning(f"Worker failed on task {task}: {e}")
    finally:
        # Clean up shared memory handles
        for memory in [data_shm, ref_shm, result_shm]:
            if memory is not None:
                try:
                    memory.close()
                except:
                    pass


def _create_tasks(n_genes: int, group_rows: List[np.ndarray], chunk_size: int) -> List[Tuple]:
    """Create simple task list for multiprocessing."""
    tasks = []
    
    for group_idx, group_indices in enumerate(group_rows):
        gene_start = 0
        while gene_start < n_genes:
            gene_end = min(n_genes, gene_start + chunk_size)
            tasks.append((group_idx, gene_start, gene_end, group_indices))
            gene_start = gene_end
    
    return tasks


def _create_shared_memory(X_dense: np.ndarray, ref_data_sorted: np.ndarray, 
                                group_rows: List[np.ndarray], n_genes: int) -> Tuple:
    """Create shared memory for simple multiprocessing."""
    from multiprocessing.shared_memory import SharedMemory

    # Create shared memory for main data
    data_shm = SharedMemory(create=True, size=X_dense.nbytes)
    data_array = np.ndarray(X_dense.shape, dtype=X_dense.dtype, buffer=data_shm.buf)
    data_array[:] = X_dense
    data_spec = {'name': data_shm.name, 'shape': X_dense.shape, 'dtype': X_dense.dtype}
    
    # Create shared memory for pre-sorted reference
    ref_shm = SharedMemory(create=True, size=ref_data_sorted.nbytes)
    ref_array = np.ndarray(ref_data_sorted.shape, dtype=ref_data_sorted.dtype, buffer=ref_shm.buf)
    ref_array[:] = ref_data_sorted
    ref_spec = {'name': ref_shm.name, 'shape': ref_data_sorted.shape, 'dtype': ref_data_sorted.dtype}
    
    # Create shared memory for results (p-values + u-statistics)
    total_results = len(group_rows) * n_genes * 2  # p-values + u-statistics
    result_shm = SharedMemory(create=True, size=total_results * 8)  # float64
    result_array = np.ndarray((total_results,), dtype=np.float64, buffer=result_shm.buf)
    result_array.fill(np.nan)
    result_spec = {
        'name': result_shm.name, 
        'shape': (total_results,), 
        'dtype': np.float64,
        'group_counts': [len(rows) for rows in group_rows]
    }
    
    return (data_shm, ref_shm, result_shm), (data_spec, ref_spec, result_spec)


def _create_presorted_ref_data(adata: ad.AnnData, ref_rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create pre-sorted reference group data and dense matrix (optimized version).
    
    Args:
        adata: AnnData object containing gene expression data
        ref_rows: Row indices for reference group
        
    Returns:
        Tuple of (X_dense, ref_data_sorted) where:
            - X_dense: Dense matrix with shape (n_cells, n_genes)
            - ref_data_sorted: Pre-sorted reference group data with shape (n_ref_samples, n_genes)
    """
    n_genes = adata.n_vars
    n_ref = len(ref_rows)
    
    logger.info(f"Converting to dense matrix and sorting reference group data ({n_genes} genes x {n_ref} samples)...")
    
    # Convert to dense matrix
    if hasattr(adata.X, 'toarray'):  # Sparse matrix
        X_dense = adata.X.toarray()
    else:  # Dense matrix
        X_dense = adata.X.copy()
    
    # Extract reference data and sort
    ref_data = X_dense[ref_rows, :]  # shape: (n_ref, n_genes)
    ref_data_sorted = np.sort(ref_data, axis=0)  # Sort along gene axis
    
    return X_dense, ref_data_sorted


def _compute_fold_changes(target_means: np.ndarray, ref_means: np.ndarray, 
                              clip_value: float = 20.0) -> np.ndarray:
    """
    Safely compute fold changes with proper handling of edge cases.
    
    This implementation follows pdex logic:
    - Reference mean ≈ 0: set to clip_value (infinite fold change, clipped)
    - Target mean ≈ 0: set to 1/clip_value (zero fold change, clipped)  
    - Normal case: target_mean / ref_mean
    
    Args:
        target_means: Mean expression in target group(s), shape (..., n_genes)
        ref_means: Mean expression in reference group, shape (n_genes,) or (..., n_genes)
        clip_value: Value to use for clipping infinite/zero fold changes
        
    Returns:
        Fold changes array with same shape as target_means
    """
    # Ensure inputs are numpy arrays
    target_means = np.asarray(target_means)
    ref_means = np.asarray(ref_means)
    
    # Handle broadcasting
    if ref_means.ndim == 1 and target_means.ndim > 1:
        ref_means = np.broadcast_to(ref_means, target_means.shape)
    
    # Use tolerance for zero comparison to handle floating point precision
    ref_zero_mask = (ref_means < 1e-10)
    target_zero_mask = (target_means < 1e-10)
    
    # Calculate fold changes with proper error handling
    with np.errstate(divide="ignore", invalid="ignore"):
        fc = target_means / ref_means
        
        # Handle special cases following pdex logic:
        # Note: Order matters! ref_mean = 0 takes precedence over target_mean = 0
        # 1. If target_mean ≈ 0: set to 1/clip_value (zero fold change -> clipped)
        # 2. If ref_mean ≈ 0: set to clip_value (infinite fold change -> clipped)
        fc = np.where(target_zero_mask, 1.0/clip_value, fc)
        fc = np.where(ref_zero_mask, clip_value, fc)
    
    return fc


def _compute_log2_fold_change(fc_values: np.ndarray) -> np.ndarray:
    """
    Compute log2 fold changes from pre-processed fold change values.
    
    Since fold changes are already properly handled by _compute_fold_changes_safe 
    (including clipping and edge case handling), this function can be simple.
    
    Args:
        fc_values: Fold change values that have been pre-processed
        
    Returns:
        Log2 fold changes with same shape as input
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(fc_values)


def _compute_fdr(p_matrix: np.ndarray, method: str = 'fdr_bh') -> np.ndarray:
    """Compute FDR correction using scipy (optimized version).
    
    Performs Benjamini-Hochberg FDR correction on p-values with optimized memory usage.
    
    Args:
        p_matrix: Array of p-values
        method: FDR correction method (currently only 'fdr_bh' supported)
        
    Returns:
        FDR-corrected p-values with same shape as input
    """
    original_shape = p_matrix.shape
    
    # Optimization: Use ravel() instead of flatten() for potential view return
    p_flat = p_matrix.ravel()
    
    # Optimization: Use faster isfinite to check both NaN and inf simultaneously
    valid_mask = np.isfinite(p_flat)
    
    if not np.any(valid_mask):
        return np.full(original_shape, np.nan, dtype=p_matrix.dtype)
    
    p_valid = p_flat[valid_mask]
    
    # Use scipy's false_discovery_control for BH correction
    try:
        fdr_valid = false_discovery_control(p_valid, method='bh')
        
        # Optimization: Pre-allocate result array
        fdr_flat = np.full_like(p_flat, np.nan, dtype=np.float64)
        fdr_flat[valid_mask] = fdr_valid
        
        # Optimization: Use reshape instead of creating new array
        return fdr_flat.reshape(original_shape)
        
    except Exception as e:
        logging.warning(f"FDR correction failed, using original p-values: {e}")
        return p_matrix.copy()


def _auto_schedule_chunk_size(
    n_genes: int,
    n_groups: int,
    num_workers: int,
    data_size_mb: float
) -> int:
    """
    Automatically determine optimal chunk size for gene-level processing.
    
    Optimized strategy:
    - Single worker: process all genes at once (no chunking needed)
    - Multi-worker: balance between parallelization and overhead
    - Focus on gene-level chunking, not group-level
    
    Args:
        n_genes: Number of genes
        n_groups: Number of treatment groups (excluding reference)
        num_workers: Number of worker processes
        data_size_mb: Approximate data size in MB
        
    Returns:
        Optimal chunk size (number of genes per chunk)
    """
    
    if num_workers == 1:
        # Single worker: no chunking needed, process all genes at once
        chunk_size = n_genes
        logger.info(f"Auto-scheduling: Single worker mode, chunk_size={chunk_size} (all genes)")
        
    else:
        # Multi-worker: minimize Python loop overhead while ensuring load balancing
        # Use minimal chunks for load balancing (1-2 chunks per worker)
        target_chunks_per_worker = 1  # Minimize Python loop overhead
        
        if data_size_mb > 1024:  # Very large dataset
            target_chunks_per_worker = 2  # Slightly more chunks for better load balancing
            
        target_total_chunks = num_workers * target_chunks_per_worker
        chunk_size = max(4096, n_genes // target_total_chunks)  # Large chunks to minimize Python overhead
        
        logger.info(f"Auto-scheduling: {num_workers} workers, target {target_chunks_per_worker} chunks/worker, "
                   f"chunk_size={chunk_size} genes/chunk (minimizing Python loop overhead)")
    
    return chunk_size

def _estimate_data_size(adata: ad.AnnData) -> float:
    """
    Estimate data size in MB for auto-scheduling decisions.
    
    Args:
        adata: Input data
        
    Returns:
        Estimated size in MB
    """
    
    n_cells, n_genes = adata.shape
    
    # Estimate based on data type and sparsity
    if hasattr(adata.X, 'nnz'):  # Sparse matrix
        nnz = adata.X.nnz
        sparsity = nnz / (n_cells * n_genes)
        # Sparse: data + indices + indptr
        sparse_size_mb = (nnz * 8 + nnz * 4 + (n_cells + 1) * 8) / 1024 / 1024
        # Dense equivalent for comparison
        dense_size_mb = n_cells * n_genes * 8 / 1024 / 1024
        
        logger.debug(f"Data analysis: {n_cells}×{n_genes}, sparsity={sparsity:.3f}, "
                    f"sparse={sparse_size_mb:.1f}MB, dense={dense_size_mb:.1f}MB")
        
        # Use dense size for scheduling decisions (since we convert to dense)
        return dense_size_mb
    else:
        # Dense matrix
        size_mb = n_cells * n_genes * 8 / 1024 / 1024  # Assuming float64
        logger.debug(f"Data analysis: {n_cells}×{n_genes}, dense={size_mb:.1f}MB")
        return size_mb


def _execute_singlethread_computation(
    X_dense: np.ndarray,
    ref_data_sorted: np.ndarray,
    group_rows: List[np.ndarray],
    group_names: List[str],
    n_genes: int,
    chunk_size: int,
    metric: str,
    tie_correction: bool,
    continuity_correction: bool,
    use_asymptotic: Optional[bool],
    max_bins: int,
    gene_names: np.ndarray,
    use_hist: bool
) -> List[pd.DataFrame]:
    """Execute single-threaded computation for all groups."""
    all_results = []
    
    for group_name, group_indices in tqdm(zip(group_names, group_rows), 
                                                               total=len(group_names),
                                                               desc="Processing groups"):
        target_data_dense = X_dense[group_indices]  # (n_target_cells, n_genes)
        
        # Prepare data for the kernel based on metric type (single-threaded: no chunking needed)
        if metric == "wilcoxon":
            # Float kernel expects pre-sorted data, shape (n_genes, n_cells)
            ref_chunk = ref_data_sorted.T  # (n_genes, n_ref_cells)
            if use_hist:
                tar_chunk = target_data_dense.T  # (n_genes, n_target_cells)
                
                p_values, u_stats = rank_sum_chunk_kernel_hist(
                    ref_chunk, tar_chunk.astype(np.int64),
                    tie_correction=tie_correction,
                    continuity_correction=continuity_correction,
                    use_asymptotic=use_asymptotic,
                    max_bins=max_bins
                )
            else:
                tar_chunk = np.sort(target_data_dense.T, axis=1)  # (n_genes, n_target_cells)
                
                p_values, u_stats = rank_sum_chunk_kernel_float(
                    ref_chunk, tar_chunk,
                    tie_correction=tie_correction,
                    continuity_correction=continuity_correction,
                    use_asymptotic=use_asymptotic
                )
        else:
            raise ValueError(f"Unsupported metric: {metric}; supported metrics: {SUPPORTED_METRICS}")
        
        # Store results for this group
        group_results = pd.DataFrame({
            'target': group_name,
            'feature': gene_names,
            'p_value': p_values,
            'u_statistic': u_stats
        })
        all_results.append(group_results)
    
    return all_results


def _execute_multiprocess_computation(
    X_dense: np.ndarray,
    ref_data_sorted: np.ndarray,
    group_rows: List[np.ndarray],
    group_names: List[str],
    n_genes: int,
    chunk_size: int,
    metric: str,
    tie_correction: bool,
    continuity_correction: bool,
    use_asymptotic: Optional[bool],
    max_bins: int,
    gene_names: np.ndarray,
    use_hist: bool
) -> List[pd.DataFrame]:
    """Execute multiprocess computation for all groups."""
    
    # Create shared memory and tasks
    shared_memories, (data_spec, ref_spec, result_spec) = _create_shared_memory(
        X_dense, ref_data_sorted, group_rows, n_genes
    )
    
    # Create tasks for parallel processing
    tasks = _create_tasks(n_genes, group_rows, chunk_size)
    logger.info(f"📋 Created {len(tasks)} tasks for parallel processing")
    
    try:
        # Determine number of processes
        num_processes = min(len(tasks), mp.cpu_count())
        
        # Prepare worker arguments
        worker_args = []
        for task in tasks:
            args = (data_spec, ref_spec, result_spec, task, metric,
                   tie_correction, continuity_correction, use_asymptotic, max_bins, use_hist)
            worker_args.append(args)
        
        # Execute tasks in parallel
        with mp.Pool(processes=num_processes) as pool:
            list(tqdm(
                pool.imap_unordered(_chunk_worker, worker_args),
                total=len(worker_args),
                desc=f"Parallel processing"
            ))
        
        # Extract results from shared memory
        result_shm = shared_memories[2]
        result_array = np.ndarray(result_spec['shape'], dtype=result_spec['dtype'], buffer=result_shm.buf)
        
        total_genes = len(group_rows) * n_genes
        p_values_flat = result_array[:total_genes].reshape(len(group_rows), n_genes)
        u_stats_flat = result_array[total_genes:].reshape(len(group_rows), n_genes)
        
        # Build results DataFrame from parallel results
        all_results = []
        for group_idx, group_name in enumerate(group_names):
            group_results = pd.DataFrame({
                'target': group_name,
                'feature': gene_names,
                'p_value': p_values_flat[group_idx],
                'u_statistic': u_stats_flat[group_idx]
            })
            all_results.append(group_results)
        
        return all_results
        
    finally:
        # Clean up shared memory
        for shm in shared_memories:
            try:
                shm.close()
                shm.unlink()
            except:
                pass


# -- Public API
def parallel_differential_expression(
    adata: ad.AnnData,
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
    clip_value: float = 20.0,
) -> pd.DataFrame:
    """High-performance parallel differential expression analysis.
    
    Performs differential expression analysis using optimized shared memory parallelization
    for efficient analysis of large-scale single-cell data. Features automatic chunk size
    optimization and intelligent memory management based on data characteristics.
    
    Key Features:
    - Automatic task scheduling optimized for your data and hardware
    - Zero-copy data access using NumPy views for memory efficiency
    - Pre-sorted reference data for maximum single-thread performance
    - Optimized shared memory for excellent multi-thread scaling
    - Histogram algorithm for integer data when beneficial
    
    Args:
        adata: AnnData object containing gene expression data
        groupby_key: Column name in obs for grouping cells
        reference: Name of the reference group
        groups: List of target groups to compare. If None, uses all groups except reference
        metric: Statistical test method. Currently supports "wilcoxon" (Mann-Whitney U test)
        tie_correction: Whether to apply tie correction in statistical tests
        continuity_correction: Whether to apply continuity correction
        use_asymptotic: Force asymptotic approximation. None for automatic selection
        min_samples: Minimum number of samples per group. Groups with fewer samples are excluded
        max_bins: Maximum number of bins for histogram algorithm (for integer data)
        prefer_hist_if_int: Prefer histogram algorithm for integer data types
        num_workers: Number of parallel worker processes (default: 1)
        clip_value: Value to clip fold change if infinite or NaN. None to disable clipping
        
    Returns:
        DataFrame containing results with columns:
            - 'target': Treatment group name
            - 'feature': Gene name  
            - 'p_value': P-value from statistical test
            - 'fold_change': Fold change (target_mean / reference_mean)
            - 'log2_fold_change': Log2 fold change
            - 'fdr': FDR-corrected p-value (Benjamini-Hochberg)
            
    Raises:
        ValueError: If reference group not found or no valid target groups
        TypeError: If unsupported data types are used
    """
    # Data preparation (optimized version)
    obs_vals = adata.obs[groupby_key].values
    uniq = np.unique(obs_vals)
    if reference not in uniq:
        raise ValueError(f"reference `{reference}` not found in `{groupby_key}`")
    
    # Optimization: Use set operations for faster filtering
    if groups is None:
        unique_set = set(uniq)
        unique_set.discard(reference)
        groups = list(unique_set)
    else:
        groups = [g for g in groups if g != reference and g in uniq]
    
    # Optimization: Batch row index retrieval
    row_idx_ref = np.where(obs_vals == reference)[0].astype(np.int64)
    
    group_rows = []
    group_names = []
    for g in groups:
        idx = np.where(obs_vals == g)[0].astype(np.int64)
        if len(idx) >= min_samples:
            group_rows.append(idx)
            group_names.append(g)
    
    if not group_rows:
        raise ValueError("No target groups meet minimum sample size requirement")
    
    group_sizes = [len(x) for x in group_rows]
    n_genes = adata.n_vars
    
    # Validate metric parameter
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported statistical test method: {metric}. Available: {SUPPORTED_METRICS}")
    
    # Determine whether to use histogram algorithm
    use_hist = (prefer_hist_if_int and np.issubdtype(adata.X.dtype, np.integer) and num_workers > 1)

    # Pre-sort reference group data and convert to dense matrix
    logger.info("Creating pre-sorted reference data...")
    X_dense, ref_data_sorted = _create_presorted_ref_data(adata, row_idx_ref)

    # Automatic task scheduling based on data characteristics
    logger.info("🤖 Optimizing task scheduling...")
    data_size_mb = _estimate_data_size(adata)
    logger.info(f"   Data: {adata.n_obs} cells × {n_genes} genes, {len(group_rows)} treatment groups")
    logger.info(f"   Size: {data_size_mb:.1f} MB")
    logger.info(f"   Workers: {num_workers}")
    
    # Use auto-calculated chunk size
    chunk_size = _auto_schedule_chunk_size(n_genes, len(group_rows), num_workers, data_size_mb)
    
    # Execute computation (single-thread or multi-process)
    all_results = []
    
    if num_workers > 1:
        logger.info(f"🚀 Using multiprocessing with {num_workers} workers")
        all_results = _execute_multiprocess_computation(
            X_dense, ref_data_sorted, group_rows, group_names, n_genes,
            chunk_size, metric, tie_correction, continuity_correction, 
            use_asymptotic, max_bins, adata.var.index.values, use_hist
        )
    else:
        logger.info("🔧 Using single-threaded processing")
        all_results = _execute_singlethread_computation(
            X_dense, ref_data_sorted, group_rows, group_names, n_genes,
            chunk_size, metric, tie_correction, continuity_correction,
            use_asymptotic, max_bins, adata.var.index.values, use_hist
        )

    # Combine all results into a single DataFrame
    logger.info("📊 Combining results...")
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Calculate fold changes using unified function
    logger.info("📈 Computing fold changes...")
    ref_means = np.mean(X_dense[row_idx_ref], axis=0)  # (n_genes,)
    
    for idx, (group_name, group_indices) in enumerate(zip(group_names, group_rows)):
        target_means = np.mean(X_dense[group_indices], axis=0)  # (n_genes,)
        
        # Get fold changes for this group using unified functions
        fold_changes = _compute_fold_changes(target_means, ref_means, clip_value)
        log2_fold_changes = _compute_log2_fold_change(fold_changes)
        
        # Update DataFrame for this group
        group_mask = combined_results['target'] == group_name
        combined_results.loc[group_mask, 'fold_change'] = fold_changes
        combined_results.loc[group_mask, 'log2_fold_change'] = log2_fold_changes
    
    # Apply FDR correction
    logger.info("📊 Applying FDR correction...")
    combined_results['fdr'] = _compute_fdr(combined_results['p_value'].values)
    
    logger.info("✅ Analysis completed!")
    return combined_results