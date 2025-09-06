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

Author: krkawzq
"""

import logging
import math
import multiprocessing as mp
import os
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, Iterable, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from numba import get_num_threads, get_thread_id, njit, prange
from scipy.sparse import csc_matrix, csr_matrix
from scipy.stats import false_discovery_control
from tqdm import tqdm

tools_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@njit(cache=True, fastmath=True, parallel=True, boundscheck=False)
def _merge_many_sorted_numba(ref2: np.ndarray, tar2: np.ndarray) -> np.ndarray:
    """Batch merge and scan for Mann-Whitney U test computation.
    
    Performs efficient batch merging of sorted arrays for statistical computation.
    Optimized version with disabled bounds checking and optimized memory access patterns.
    
    Args:
        ref2: Reference group data with shape [B, n_ref], any numeric dtype, C-contiguous
        tar2: Target group data with shape [B, n_tar], any numeric dtype, C-contiguous
        
    Returns:
        out: Array with shape [B, 3] (float64) containing:
            - [:, 0]: U1 statistics
            - [:, 1]: tie_sum for tie correction
            - [:, 2]: has_ties flag (0/1 stored as float64)
    """
    B = ref2.shape[0]
    out = np.empty((B, 3), dtype=np.float64)

    for b in prange(B):
        r = ref2[b]
        t = tar2[b]
        n2 = r.shape[0]
        n1 = t.shape[0]
        
        if n1 == 0 or n2 == 0:
            out[b, 0] = 0.0
            out[b, 1] = 0.0
            out[b, 2] = 0.0
            continue
        
        i = 0
        k = 0
        running = 1.0
        rank_sum_t = 0.0
        tie_sum = 0.0
        has_ties = 0.0

        while i < n2 or k < n1:
            if k >= n1:
                v = r[i]
            elif i >= n2:
                v = t[k]
            elif t[k] <= r[i]:
                v = t[k]
            else:
                v = r[i]

            cr = 0
            ct = 0
            
            while i < n2 and r[i] == v:
                cr += 1
                i += 1
            while k < n1 and t[k] == v:
                ct += 1
                k += 1

            c = cr + ct
            if c > 1:
                has_ties = 1.0
                tie_sum += c * (c * c - 1)

            if ct > 0:
                rank_sum_t += ct * (running + 0.5 * (c - 1))
            running += c

        U1 = rank_sum_t - 0.5 * n1 * (n1 + 1.0)
        out[b, 0] = U1
        out[b, 1] = tie_sum
        out[b, 2] = has_ties

    return out


@njit(cache=True, fastmath=True, parallel=True, boundscheck=False)
def _p_asymptotic_batch_numba(
    U1: np.ndarray,
    tie_sum: np.ndarray,
    n1: int, 
    n2: int,
    tie_correction: int,
    continuity_correction: int
) -> np.ndarray:
    """Compute asymptotic p-values for Mann-Whitney U test.
    
    Calculates p-values using normal approximation with optional tie and continuity corrections.
    Optimized version with disabled bounds checking and precomputed constants.
    
    Args:
        U1: U1 statistics array with shape [B], float64
        tie_sum: Tie sum array for correction with shape [B], float64
        n1: Sample size of target group
        n2: Sample size of reference group
        tie_correction: Whether to apply tie correction (0 or 1)
        continuity_correction: Whether to apply continuity correction (0 or 1)
        
    Returns:
        p: P-values array with shape [B], float64, clipped to [0, 1]
    """
    B = U1.shape[0]
    p = np.empty(B, dtype=np.float64)

    N = n1 + n2
    if N <= 1 or n1 == 0 or n2 == 0:
        p.fill(1.0)
        return p
    
    mu = 0.5 * n1 * n2
    base = (n1 * n2) * (N + 1.0) / 12.0
    use_tie = (tie_correction == 1) and (N > 1)
    k_tie = (n1 * n2) / (12.0 * N * (N - 1.0)) if use_tie else 0.0
    use_cc = (continuity_correction == 1)
    inv_sqrt2 = 0.7071067811865475  # 1.0 / math.sqrt(2.0) precomputed

    for i in prange(B):
        sigma2 = base - k_tie * tie_sum[i] if use_tie else base
        
        if sigma2 <= 0.0 or not np.isfinite(sigma2):
            p[i] = 1.0
            continue

        num = U1[i] - mu
        if use_cc and num != 0.0:
            # avoid branch prediction
            num = num - 0.5 if num > 0.0 else num + 0.5

        # often used operation, calculate square root first
        sqrt_sigma2 = math.sqrt(sigma2)
        zabs = abs(num) / sqrt_sigma2
        pj = math.erfc(zabs * inv_sqrt2)   # two-sided
        
        if not np.isfinite(pj) or pj > 1.0:
            pj = 1.0
        elif pj < 0.0:
            pj = 0.0
            
        p[i] = pj
    return p


@njit(cache=True)
def _exact_tail_table(n1: int, n2: int) -> np.ndarray:
    """Compute exact tail probabilities for Mann-Whitney U test.
    
    Calculates survival function sf[k] = P(U >= k) for exact p-value computation.
    Uses dynamic programming with pure ndarray operations for robustness.
    
    Args:
        n1: Sample size of target group
        n2: Sample size of reference group
        
    Returns:
        sf: Survival function array with length Ucap+1, where Ucap = n1 * n2
    """
    m, n = n1, n2
    Ucap = m * n

    f_prev = np.zeros((n + 1, Ucap + 1), dtype=np.int64)
    f_curr = np.zeros_like(f_prev)

    for j in range(n + 1):
        f_prev[j, 0] = 1

    for i in range(1, m + 1):
        # 清空
        for j in range(n + 1):
            for k in range(Ucap + 1):
                f_curr[j, k] = 0
        f_curr[0, 0] = 1

        cap = i * n
        for j in range(1, n + 1):
            for k in range(j):
                f_curr[j, k] = f_curr[j - 1, k]
            for k in range(j, cap + 1):
                f_curr[j, k] = f_curr[j - 1, k] + f_prev[j, k - j]

        f_prev, f_curr = f_curr, f_prev

    counts = f_prev[n] # [Ucap+1]
    total = 0
    for k in range(Ucap + 1):
        total += counts[k]

    sf = np.empty(Ucap + 1, dtype=np.float64)
    acc = 0
    for k in range(Ucap, -1, -1):
        acc += counts[k]
        sf[k] = acc / float(total)
    return sf


def _assert_supported_dtype(arr: np.ndarray) -> None:
    """Validate array dtype for numerical computation.
    
    Explicitly rejects float16 for Numba compatibility and numerical stability.
    Allows common numeric dtypes: int16, int32, float32, float64.
    
    Args:
        arr: Input array to validate
        
    Raises:
        TypeError: If dtype is not supported
    """
    if arr.dtype == np.float16:
        raise TypeError("float16 is not supported (Numba & numerical stability).")
    if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        raise TypeError(f"Unsupported dtype {arr.dtype}.")


# -- comman rank sum kernel
def rank_sum_chunk_kernel_float(
    ref_sorted: np.ndarray,
    tar_sorted: np.ndarray,
    tie_correction: bool = True,
    continuity_correction: bool = True,
    use_asymptotic: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Mann-Whitney U test using floating-point algorithm.
    
    Memory-optimized version that preserves input dtype while using float64 for 
    intermediate computations. Supports int16/int32/float32/float64 input dtypes.
    
    Args:
        ref_sorted: Reference group data with shape [..., n_ref], sorted ascending
        tar_sorted: Target group data with shape [..., n_tar], sorted ascending
        tie_correction: Whether to apply tie correction
        continuity_correction: Whether to apply continuity correction
        use_asymptotic: Force asymptotic approximation. None for auto-selection
        
    Returns:
        Tuple of (p_values, U_statistics) with same leading dimensions as input
        
    Raises:
        ValueError: If leading dimensions don't match
        TypeError: If unsupported dtype is used
    """
    _assert_supported_dtype(ref_sorted)
    _assert_supported_dtype(tar_sorted)

    raw_ref_shape = ref_sorted.shape
    raw_tar_shape = tar_sorted.shape
    if raw_ref_shape[:-1] != raw_tar_shape[:-1]:
        raise ValueError("Leading shapes must match for ref_sorted and tar_sorted")

    # ensure 2D + C-contiguous
    ref2 = np.ascontiguousarray(ref_sorted.reshape(-1, raw_ref_shape[-1]))
    tar2 = np.ascontiguousarray(tar_sorted.reshape(-1, raw_tar_shape[-1]))

    # -- 1. batch merge (compute result vector using float64)
    out = _merge_many_sorted_numba(ref2, tar2)  # [B,3] float64
    U1 = out[:, 0]
    tie_sum = out[:, 1]
    has_ties = out[:, 2].astype(np.int64)

    n_ref = ref2.shape[1]
    n_tar = tar2.shape[1]

    # -- 2. method selection
    if use_asymptotic is None:
        use_asym = (np.any(has_ties != 0)) or (min(n_ref, n_tar) > 8)
    else:
        use_asym = bool(use_asymptotic)

    # -- 3. compute p (float64 vector)
    if use_asym:
        p = _p_asymptotic_batch_numba(
            U1, tie_sum, n_tar, n_ref,
            1 if tie_correction else 0,
            1 if continuity_correction else 0,
        )
    else:
        total_U = int(n_ref * n_tar)
        U2 = total_U - U1
        # with method "exact", U should be integer; use rint→int to avoid potential overflow
        Umax_idx = np.rint(np.maximum(U1, U2)).astype(np.int64)
        sf = _exact_tail_table(n_tar, n_ref)  # float64
        p = 2.0 * sf[Umax_idx]
        np.clip(p, 0.0, 1.0, out=p)

    out_shape = raw_ref_shape[:-1]
    return p.reshape(out_shape), U1.reshape(out_shape)


def _assert_integer_dtype(arr: np.ndarray) -> None:
    if not np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_:
        raise TypeError(f"hist kernel expects integer dtype, got {arr.dtype}")
    if arr.dtype == np.int8 or arr.dtype == np.uint8:
        # Small count types may overflow/truncate on large samples, recommend at least int16
        pass  # Allow but could be upgraded


# -- histogram kernel
@njit(cache=True, fastmath=True, parallel=True, boundscheck=False)
def _hist_merge_and_stats_kernel(
    ref2: np.ndarray,
    tar2: np.ndarray,
    vmin: int,
    Kp1: int,
    pool_cnt: np.ndarray, 
    pool_cnt_t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Histogram-based Mann-Whitney U test computation kernel.
    
    Efficiently computes U statistics and tie information using histogram binning
    for integer data. Optimized version with disabled bounds checking and 
    optimized memory access patterns.
    
    Args:
        ref2: Reference group data with shape [B, n_ref], integer dtype
        tar2: Target group data with shape [B, n_tar], integer dtype
        vmin: Global minimum value (used for offset)
        Kp1: Global number of bins (vmax - vmin + 1)
        pool_cnt: Thread-private buffer for counts with shape [nthreads, Kp1], int64
        pool_cnt_t: Thread-private buffer for target counts with shape [nthreads, Kp1], int64
        
    Returns:
        Tuple of (U1, tie_sum, has_ties) where:
            - U1: U statistics array with shape [B], float64
            - tie_sum: Tie sum array for correction with shape [B], float64
            - has_ties: Tie flag array with shape [B], int64 (0/1)
    """
    B = ref2.shape[0]
    n_ref = ref2.shape[1]
    n_tar = tar2.shape[1]

    U = np.empty(B, dtype=np.float64)
    tie = np.empty(B, dtype=np.float64)
    has = np.zeros(B, dtype=np.int64)

    for b in prange(B):
        tid = get_thread_id()
        cnt = pool_cnt[tid]
        cnt_t = pool_cnt_t[tid]

        # Optimization: Use larger initial value and smaller initial value
        min_idx = Kp1
        max_idx = -1

        # Process tar data first, updating both counters simultaneously
        for i in range(n_tar):
            idx = tar2[b, i] - vmin  # Direct integer arithmetic
            cnt[idx] += 1
            cnt_t[idx] += 1
            # Optimization: min/max updates
            if idx < min_idx: 
                min_idx = idx
            if idx > max_idx: 
                max_idx = idx

        # Process ref data
        for i in range(n_ref):
            idx = ref2[b, i] - vmin
            cnt[idx] += 1
            if idx < min_idx: 
                min_idx = idx
            if idx > max_idx: 
                max_idx = idx

        # Early exit check
        if max_idx < min_idx or n_tar == 0 or n_ref == 0:
            U[b] = 0.0
            tie[b] = 0.0
            has[b] = 0
            continue

        running = 1
        rank_sum_t = 0.0
        tie_sum = 0.0
        has_tie_flag = 0

        # Optimized statistical computation loop
        for v in range(min_idx, max_idx + 1):
            c = cnt[v]
            if c > 0:
                # Optimization: Direct average rank calculation
                avg_rank = running + (c - 1) * 0.5
                rank_sum_t += cnt_t[v] * avg_rank
                
                # Optimization: tie calculation
                if c > 1:
                    tie_sum += c * (c - 1) * (c + 1)
                    has_tie_flag = 1
                    
                running += c

        # Final calculation
        U1 = rank_sum_t - 0.5 * n_tar * (n_tar + 1)
        U[b] = U1
        tie[b] = tie_sum
        has[b] = has_tie_flag

        # Fast cleanup of touched bins - memset would be better but numba doesn't support it
        for v in range(min_idx, max_idx + 1):
            cnt[v] = 0
            cnt_t[v] = 0

    return U, tie, has


# -- Histogram Kernel
def rank_sum_chunk_kernel_hist(
    ref_sorted: np.ndarray,
    tar: np.ndarray,
    tie_correction: bool = True,
    continuity_correction: bool = True,
    use_asymptotic: Optional[bool] = None,
    max_bins: int = 200_000,
    float_dtype: np.dtype = np.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Mann-Whitney U test using histogram algorithm for integer data.
    
    Efficient histogram-based computation for integer data that automatically
    falls back to floating-point algorithm when value range is too large.
    
    Args:
        ref_sorted: Reference group data with shape [..., n_ref], integer dtype, sorted
        tar: Target group data with shape [..., n_tar], integer dtype, may not be sorted
        tie_correction: Whether to apply tie correction
        continuity_correction: Whether to apply continuity correction
        use_asymptotic: Force asymptotic approximation. None for auto-selection
        max_bins: Maximum number of bins before falling back to float algorithm
        float_dtype: Data type for fallback to float algorithm
        
    Returns:
        Tuple of (p_values, U_statistics) with same leading dimensions as input
        
    Raises:
        ValueError: If leading dimensions don't match
        TypeError: If unsupported dtype is used
    """
    _assert_integer_dtype(ref_sorted)
    _assert_integer_dtype(tar)

    raw_ref_shape = ref_sorted.shape
    raw_tar_shape = tar.shape
    if raw_ref_shape[:-1] != raw_tar_shape[:-1]:
        raise ValueError("Leading shapes must match for ref_sorted and tar")

    # Ensure 2D + C-contiguous, don't change dtype
    ref2 = np.ascontiguousarray(ref_sorted.reshape(-1, raw_ref_shape[-1]))
    tar2 = np.ascontiguousarray(tar.reshape(-1, raw_tar_shape[-1]))

    B = ref2.shape[0]
    n_ref = ref2.shape[1]
    n_tar = tar2.shape[1]

    # Calculate global value range
    vmin_ref = int(np.min(ref2)) if n_ref > 0 else 0
    vmax_ref = int(np.max(ref2)) if n_ref > 0 else 0
    vmin_tar = int(np.min(tar2)) if n_tar > 0 else 0
    vmax_tar = int(np.max(tar2)) if n_tar > 0 else 0

    vmin = vmin_ref if vmin_ref < vmin_tar else vmin_tar
    vmax = vmax_ref if vmax_ref > vmax_tar else vmax_tar
    Kp1 = int(vmax - vmin + 1) if (n_ref > 0 and n_tar > 0) else 1

    # Value range too large: fallback to "float merge kernel" - needs sorting (ref already sorted, tar needs sorting)
    if Kp1 > max_bins:
        ref_f = np.ascontiguousarray(ref2, dtype=float_dtype) # Already sorted, no need to sort
        tar_f = np.ascontiguousarray(tar2, dtype=float_dtype)
        tar_f = np.sort(tar_f, axis=1) # Only sort tar

        p, U = rank_sum_chunk_kernel_float(
            ref_f, tar_f,
            tie_correction=tie_correction,
            continuity_correction=continuity_correction,
            use_asymptotic=True, # Fallback scenarios usually have large samples/many ties, force approximation for stability
        )
        out_shape = raw_ref_shape[:-1]
        return p.reshape(out_shape), U.reshape(out_shape)

    # -- Normal histogram path
    nthreads = get_num_threads()
    pool_cnt = np.zeros((nthreads, Kp1), dtype=np.int64)
    pool_cnt_t = np.zeros((nthreads, Kp1), dtype=np.int64)

    U1, tie_sum, has_ties = _hist_merge_and_stats_kernel(
        ref2, tar2, vmin, Kp1, pool_cnt, pool_cnt_t
    )

    if use_asymptotic is None:
        use_asym = (np.any(has_ties != 0)) or (min(n_ref, n_tar) > 8)
    else:
        use_asym = bool(use_asymptotic)

    if use_asym:
        p = _p_asymptotic_batch_numba(
            U1, tie_sum, n_tar, n_ref,
            1 if tie_correction else 0,
            1 if continuity_correction else 0
        )
    else:
        total_U = int(n_ref * n_tar)
        U2 = total_U - U1
        Umax_idx = np.rint(np.maximum(U1, U2)).astype(np.int64)
        sf = _exact_tail_table(n_tar, n_ref)
        p = 2.0 * sf[Umax_idx]
        np.clip(p, 0.0, 1.0, out=p)

    out_shape = raw_ref_shape[:-1]
    return p.reshape(out_shape), U1.reshape(out_shape)


def _to_csc(X: Union[csc_matrix, csr_matrix, np.ndarray]) -> csc_matrix:
    """Convert input to CSC matrix format.
    
    Args:
        X: Input matrix (CSC, CSR, or dense array)
        
    Returns:
        CSC matrix
    """
    if isinstance(X, csc_matrix):
        return X
    if isinstance(X, csr_matrix):
        return X.tocsc()
    return csc_matrix(np.asarray(X))


def _create_shared_csc(X: Union[csc_matrix, csr_matrix, np.ndarray]) -> Tuple[Dict, List[SharedMemory]]:
    """Create shared CSC matrix (optimized version).
    
    Args:
        X: Input matrix to convert to shared CSC format
        
    Returns:
        Tuple of (spec, shms) containing shared memory specifications and handles
    """
    Xc = _to_csc(X)
    
    # Optimization: Only copy data when necessary
    data = Xc.data if Xc.data.flags.c_contiguous else np.ascontiguousarray(Xc.data)
    indices = Xc.indices if Xc.indices.flags.c_contiguous else np.ascontiguousarray(Xc.indices)
    indptr = Xc.indptr if Xc.indptr.flags.c_contiguous else np.ascontiguousarray(Xc.indptr)

    # Pre-compute sizes and element sizes
    data_itemsize = data.dtype.itemsize
    indices_itemsize = indices.dtype.itemsize
    indptr_itemsize = indptr.dtype.itemsize
    
    shm_data = SharedMemory(create=True, size=data.nbytes)
    shm_idx  = SharedMemory(create=True, size=indices.nbytes)
    shm_ptr  = SharedMemory(create=True, size=indptr.nbytes)

    # Optimization: Direct memory view for block copying
    shared_data = np.ndarray(data.shape, data.dtype, buffer=shm_data.buf)
    shared_indices = np.ndarray(indices.shape, indices.dtype, buffer=shm_idx.buf)
    shared_indptr = np.ndarray(indptr.shape, indptr.dtype, buffer=shm_ptr.buf)
    
    shared_data[:] = data
    shared_indices[:] = indices
    shared_indptr[:] = indptr

    spec = {
        'shape': Xc.shape, 'dtype': Xc.dtype,
        'data_shm': shm_data.name, 'indices_shm': shm_idx.name, 'indptr_shm': shm_ptr.name,
        'data_itemsize': data_itemsize, 'indices_itemsize': indices_itemsize, 'indptr_itemsize': indptr_itemsize,
        'data_len': len(data), 'indices_len': len(indices), 'indptr_len': len(indptr)
    }
    return spec, [shm_data, shm_idx, shm_ptr]


def _open_shared_csc(spec: Dict) -> Tuple[csc_matrix, List[SharedMemory]]:
    """Open shared CSC matrix (optimized version).
    
    Args:
        spec: Shared memory specification dictionary
        
    Returns:
        Tuple of (csc_matrix, shared_memory_handles)
    """
    shm_data = SharedMemory(name=spec['data_shm'])
    shm_idx  = SharedMemory(name=spec['indices_shm'])
    shm_ptr  = SharedMemory(name=spec['indptr_shm'])
    
    # Optimization: Use pre-stored lengths and element sizes to avoid repeated calculations
    data = np.ndarray((spec['data_len'],), dtype=spec['dtype'], buffer=shm_data.buf)
    indices = np.ndarray((spec['indices_len'],), dtype=np.int32, buffer=shm_idx.buf)
    indptr  = np.ndarray((spec['indptr_len'],), dtype=np.int32, buffer=shm_ptr.buf)
    
    # copy=False to avoid unnecessary data copying
    Xc = csc_matrix((data, indices, indptr), shape=spec['shape'], copy=False)
    return Xc, [shm_data, shm_idx, shm_ptr]


def _create_shared_ref_sorted(ref_data_sorted: np.ndarray) -> Tuple[Dict, List[SharedMemory]]:
    """Create shared memory for pre-sorted reference group data.
    
    Args:
        ref_data_sorted: Pre-sorted reference group data with shape (n_genes, n_ref_samples)
        
    Returns:
        Tuple of (spec, shms) where:
            - spec: Dictionary containing shared memory specifications
            - shms: List of SharedMemory objects for cleanup
    """
    # Ensure data is C-contiguous
    if not ref_data_sorted.flags.c_contiguous:
        ref_data_sorted = np.ascontiguousarray(ref_data_sorted)
    
    # Create shared memory
    nbytes = ref_data_sorted.nbytes
    shm_data = SharedMemory(create=True, size=nbytes)
    
    # Copy data to shared memory
    shared_array = np.ndarray(ref_data_sorted.shape, dtype=ref_data_sorted.dtype, buffer=shm_data.buf)
    shared_array[:] = ref_data_sorted
    
    spec = {
        'data_shm': shm_data.name,
        'shape': ref_data_sorted.shape,
        'dtype': ref_data_sorted.dtype,
        'nbytes': nbytes
    }
    
    return spec, [shm_data]


def _open_shared_ref_sorted(spec: Dict) -> Tuple[np.ndarray, List[SharedMemory]]:
    """Open shared pre-sorted reference group data.
    
    Args:
        spec: Shared memory specification dictionary
        
    Returns:
        Tuple of (ref_data_sorted, shared_memory_handles)
    """
    shm_data = SharedMemory(name=spec['data_shm'])
    
    ref_data_sorted = np.ndarray(spec['shape'], dtype=spec['dtype'], buffer=shm_data.buf)
    
    return ref_data_sorted, [shm_data]


def _close_shared_memory(shms: Iterable[SharedMemory]) -> None:
    """Safely close and release shared memory.
    
    Args:
        shms: Iterable of SharedMemory objects to close
    """
    for s in shms:
        try:
            s.close()
            s.unlink()
        except (FileNotFoundError, OSError):
            pass

def _create_presorted_ref_data(adata: ad.AnnData, ref_rows: np.ndarray) -> np.ndarray:
    """Create pre-sorted reference group data (optimized version).
    
    Args:
        adata: AnnData object containing gene expression data
        ref_rows: Row indices for reference group
        
    Returns:
        Pre-sorted reference group data with shape (n_genes, n_ref_samples)
    """
    n_genes = adata.n_vars
    n_ref = len(ref_rows)
    
    logger.info(f"Sorting reference group data ({n_genes} genes x {n_ref} samples)...")
    
    if hasattr(adata.X, 'toarray'):  # Sparse matrix
        # Extract all reference group data at once, then transpose
        ref_data = adata.X[ref_rows, :].toarray().T  # shape: (n_genes, n_ref)
    else:  # Dense matrix
        ref_data = adata.X[ref_rows, :].T.copy()  # Transpose and copy
    
    # Allocate output memory based on data type
    ref_data_sorted = np.empty((n_genes, n_ref), dtype=adata.X.dtype)
    
    # Batch sort all genes - much faster than gene-by-gene sorting
    for i in tqdm(range(n_genes), desc="Sorting reference group genes"):
        ref_data_sorted[i, :] = np.sort(ref_data[i, :])
    
    return ref_data_sorted


def _compute_fold_change(adata: ad.AnnData, ref_rows: np.ndarray, target_rows_list: List[np.ndarray]) -> np.ndarray:
    """Compute fold change matrix aligned with pdex calculation method.
    
    Optimized version for computing fold changes between target groups and reference group.
    
    Args:
        adata: AnnData object containing expression data
        ref_rows: Row indices for reference group
        target_rows_list: List of row indices for each target group
        
    Returns:
        Fold change matrix with shape (n_groups, n_genes)
    """
    n_groups = len(target_rows_list)
    n_genes = adata.n_vars
    fc_matrix = np.empty((n_groups, n_genes), dtype=np.float64)
    
    # Calculate reference group mean expression once
    if hasattr(adata.X, 'toarray'):  # Sparse matrix
        ref_mean = np.asarray(adata.X[ref_rows, :].mean(axis=0)).ravel()
    else:  # Dense matrix
        ref_mean = adata.X[ref_rows, :].mean(axis=0)
        if ref_mean.ndim > 1:
            ref_mean = ref_mean.ravel()
    
    # Pre-compute reference group zero positions to avoid repeated checks in loop
    ref_zero_mask = (ref_mean == 0)
    
    # Batch calculate mean expression for all target groups
    if hasattr(adata.X, 'toarray'):  # Sparse matrix
        for i, target_rows in enumerate(target_rows_list):
            target_mean = np.asarray(adata.X[target_rows, :].mean(axis=0)).ravel()
            
            # Optimized fold change calculation
            with np.errstate(divide="ignore", invalid="ignore"):
                fc = target_mean / ref_mean
                # Use pre-computed mask for faster processing
                fc = np.where(ref_zero_mask, np.nan, fc)
                fc = np.where(target_mean == 0, 0.0, fc)
            
            fc_matrix[i, :] = fc
    else:  # Dense matrix
        for i, target_rows in enumerate(target_rows_list):
            target_mean = adata.X[target_rows, :].mean(axis=0)
            if target_mean.ndim > 1:
                target_mean = target_mean.ravel()
            
            # Optimized fold change calculation
            with np.errstate(divide="ignore", invalid="ignore"):
                fc = target_mean / ref_mean
                fc = np.where(ref_zero_mask, np.nan, fc)
                fc = np.where(target_mean == 0, 0.0, fc)
            
            fc_matrix[i, :] = fc
    
    return fc_matrix


def _compute_log2_fold_change(fc_matrix: np.ndarray) -> np.ndarray:
    """Compute log2 fold change matrix.
    
    Args:
        fc_matrix: Fold change matrix
        
    Returns:
        Log2 fold change matrix
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_fc = np.log2(fc_matrix)
        # Handle special cases
        log2_fc = np.where(fc_matrix == 0, -np.inf, log2_fc)  # log2(0) = -∞
        log2_fc = np.where(np.isnan(fc_matrix), np.nan, log2_fc)  # Preserve NaN
        log2_fc = np.where(np.isinf(fc_matrix) & (fc_matrix > 0), np.inf, log2_fc)  # log2(∞) = ∞
    
    return log2_fc


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
    

def _create_shared_results(total_len: int) -> Tuple[Dict, List[SharedMemory]]:
    """Create shared memory for result vectors.
    
    Args:
        total_len: Total length of result vectors
        
    Returns:
        Tuple of (spec, shms) containing shared memory specifications and handles
    """
    shm_p = SharedMemory(create=True, size=total_len * 8)  # float64
    shm_u = SharedMemory(create=True, size=total_len * 8)  # float64
    
    np.ndarray((total_len,), np.float64, buffer=shm_p.buf).fill(np.nan)
    np.ndarray((total_len,), np.float64, buffer=shm_u.buf).fill(0.0)
    
    spec = {
        'length': total_len,
        'pval_shm': shm_p.name, 'stat_shm': shm_u.name,
        'pval_nbytes': total_len*8, 'stat_nbytes': total_len*8
    }
    return spec, [shm_p, shm_u]


def _open_shared_results(spec: Dict) -> Tuple[np.ndarray, np.ndarray, List[SharedMemory]]:
    """Open shared result vectors.
    
    Args:
        spec: Shared memory specification dictionary
        
    Returns:
        Tuple of (p_values, u_statistics, shared_memory_handles)
    """
    shm_p = SharedMemory(name=spec['pval_shm'])
    shm_u = SharedMemory(name=spec['stat_shm'])
    p = np.ndarray((spec['length'],), np.float64, buffer=shm_p.buf)
    u = np.ndarray((spec['length'],), np.float64, buffer=shm_u.buf)
    return p, u, [shm_p, shm_u]


def _create_shared_hist_pool(num_workers: int, max_bins: int) -> Tuple[Dict, List[SharedMemory]]:
    """Create shared memory for histogram bin pools.
    
    Args:
        num_workers: Number of worker processes
        max_bins: Maximum number of histogram bins
        
    Returns:
        Tuple of (spec, shms) containing shared memory specifications and handles
    """
    pool_size = num_workers * max_bins * 8  # int64
    temp_pool_size = num_workers * max_bins * 8  # int64
    
    shm_pool = SharedMemory(create=True, size=pool_size)
    shm_temp = SharedMemory(create=True, size=temp_pool_size)
    
    np.ndarray((num_workers, max_bins), dtype=np.int64, buffer=shm_pool.buf).fill(0)
    np.ndarray((num_workers, max_bins), dtype=np.int64, buffer=shm_temp.buf).fill(0)
    
    spec = {
        'num_workers': num_workers, 'max_bins': max_bins,
        'pool_shm': shm_pool.name, 'pool_temp_shm': shm_temp.name,
        'pool_nbytes': pool_size, 'pool_temp_nbytes': temp_pool_size
    }
    return spec, [shm_pool, shm_temp]


def _open_shared_hist_pool(spec: Dict) -> Tuple[np.ndarray, np.ndarray, List[SharedMemory]]:
    """Open shared histogram bin pools.
    
    Args:
        spec: Shared memory specification dictionary
        
    Returns:
        Tuple of (pool, temp_pool, shared_memory_handles)
    """
    shm_pool = SharedMemory(name=spec['pool_shm'])
    shm_temp = SharedMemory(name=spec['pool_temp_shm'])
    
    pool = np.ndarray((spec['num_workers'], spec['max_bins']), dtype=np.int64, buffer=shm_pool.buf)
    temp_pool = np.ndarray((spec['num_workers'], spec['max_bins']), dtype=np.int64, buffer=shm_temp.buf)
    
    return pool, temp_pool, [shm_pool, shm_temp]


def _make_chunks(n_genes: int, group_sizes: List[int], batch_budget: int, 
                 use_hist: bool = False, min_cols: int = 16, max_cols: int = 1024) -> List[Tuple[int, int, int]]:
    """Generate task list for parallel processing (optimized version).
    
    Args:
        n_genes: Total number of genes
        group_sizes: List of group sizes
        batch_budget: Memory budget for batch processing
        use_hist: Whether using histogram algorithm
        min_cols: Minimum columns per chunk
        max_cols: Maximum columns per chunk
        
    Returns:
        List of (group_idx, col_start, col_end) tuples
    """
    tasks = []
    
    # Optimization: Pre-compute chunking strategy for all groups
    chunk_strategies = []
    for n_samp in group_sizes:
        if use_hist:
            # Histogram algorithm requires less memory
            base_cols = max(1, batch_budget // max(1, n_samp // 2))
        else:
            # Floating-point algorithm needs more memory for sorting
            base_cols = max(1, batch_budget // max(1, n_samp))
        
        cols_per_chunk = max(min_cols, min(base_cols, max_cols))
        chunk_strategies.append(cols_per_chunk)
    
    # Optimization: Batch generate tasks based on strategy
    for g, cols_per_chunk in enumerate(chunk_strategies):
        start = 0
        while start < n_genes:
            end = min(n_genes, start + cols_per_chunk)
            tasks.append((g, start, end))
            start = end
    
    return tasks


def _dense_block_from_csc(Xc: csc_matrix, row_idx: np.ndarray, c0: int, c1: int, dtype=np.float64) -> np.ndarray:
    """Extract dense block from CSC matrix (optimized version).
    
    Returns dense block with shape (len(row_idx), c1-c0).
    Optimization: Column slice first, then row slice to reduce intermediate sparse matrix creation.
    
    Args:
        Xc: CSC sparse matrix
        row_idx: Row indices to extract
        c0: Start column index
        c1: End column index
        dtype: Output data type
        
    Returns:
        Dense block array
    """
    # Optimization: Column slice first, then row slice, finally convert to dense
    col_slice = Xc[:, c0:c1]          # Column slice, still CSC
    row_col_slice = col_slice[row_idx, :]  # Row slice, keep sparse
    
    # Convert to dense once and specify type
    if dtype == row_col_slice.dtype:
        return row_col_slice.toarray()
    else:
        return row_col_slice.toarray().astype(dtype, copy=False)


def _worker_run_chunk(args) -> None:
    """Worker function: Process single data chunk.
    
    Args:
        args: Tuple containing all necessary arguments for processing
    """
    (csc_spec, results_spec, hist_pool_spec, ref_sorted_spec, target_rows_by_group,
     n_genes, task, worker_id, metric, max_bins,
     tie_correction, continuity_correction, use_asymptotic) = args

    shared_handles = []
    try:
        # Connect to shared memory
        Xc, shm_csc = _open_shared_csc(csc_spec)
        shared_handles.extend(shm_csc)
        
        # Connect to pre-sorted reference group shared memory
        ref_data_sorted, shm_ref_sorted = _open_shared_ref_sorted(ref_sorted_spec)
        shared_handles.extend(shm_ref_sorted)
        
        pvec, uvec, shm_res = _open_shared_results(results_spec)
        shared_handles.extend(shm_res)
        
        hist_pool = hist_temp_pool = None
        if hist_pool_spec:
            hist_pool, hist_temp_pool, shm_hist = _open_shared_hist_pool(hist_pool_spec)
            shared_handles.extend(shm_hist)

        g_idx, c0, c1 = task
        cols = c1 - c0
        base = g_idx * n_genes + c0

        # Use pre-sorted reference group data - direct slicing, no need to re-extract and sort
        ref_sorted_block = ref_data_sorted[c0:c1, :].T  # shape: (n_ref, cols) - transpose to match tar_block
        
        # Extract only target group data block
        tar_block = _dense_block_from_csc(Xc, target_rows_by_group[g_idx], c0, c1)  # shape: (n_tar, cols)

        # Compute statistical test - note: ref_sorted_block is already sorted
        p, U = _compute_statistical_test(
            ref_sorted_block, tar_block, metric=metric, ref_already_sorted=True,
            worker_id=worker_id, hist_pool=hist_pool, hist_temp_pool=hist_temp_pool,
            tie_correction=tie_correction, continuity_correction=continuity_correction,
            use_asymptotic=use_asymptotic, max_bins=max_bins
        )
        
        pvec[base:base+cols] = p
        uvec[base:base+cols] = U
        
    except Exception as e:
        logging.warning(f"Worker {worker_id} failed on task {task}: {e}")
        if 'pvec' in locals() and 'uvec' in locals():
            try:
                g_idx, c0, c1 = task
                base = g_idx * n_genes + c0
                pvec[base:base+(c1-c0)] = np.nan
                uvec[base:base+(c1-c0)] = np.nan
            except:
                pass
        
    finally:
        for s in shared_handles:
            try: s.close()
            except: pass


def _compute_statistical_test(ref_block: np.ndarray, tar_block: np.ndarray, metric: str = "wilcoxon", 
                             ref_already_sorted: bool = False, worker_id: int = 0, 
                             hist_pool: Optional[np.ndarray] = None, hist_temp_pool: Optional[np.ndarray] = None,
                             tie_correction: bool = True, continuity_correction: bool = True, 
                             use_asymptotic: Optional[bool] = None, max_bins: int = 200_000) -> Tuple[np.ndarray, np.ndarray]:
    """Compute statistical test p-values and statistics.
    
    Args:
        ref_block: Reference group data block [n_ref, n_cols]
        tar_block: Target group data block [n_tar, n_cols]
        metric: Statistical test method, currently supports "wilcoxon"
        ref_already_sorted: Whether reference data is already sorted
        worker_id: Worker ID (for shared pool access)
        hist_pool: Shared histogram bin pool
        hist_temp_pool: Shared temporary histogram bin pool
        tie_correction: Whether to apply tie correction
        continuity_correction: Whether to apply continuity correction
        use_asymptotic: Whether to use asymptotic approximation
        max_bins: Maximum number of bins
        
    Returns:
        Tuple of (p_values, statistics) arrays
        
    Raises:
        ValueError: If unsupported metric is used
    """
    if metric != "wilcoxon":
        raise ValueError(f"Unsupported statistical test method: {metric}")
    
    # Determine whether to use histogram algorithm
    use_hist = (hist_pool is not None and 
                np.issubdtype(ref_block.dtype, np.integer))
    
    if use_hist:
        vmin = int(min(ref_block.min(), tar_block.min()))
        vmax = int(max(ref_block.max(), tar_block.max()))
        
        if vmax - vmin + 1 <= max_bins:
            # For histogram algorithm, we need unsorted data to build histogram
            if ref_already_sorted:
                # If reference group is already sorted, need to pay attention to data layout
                ref_data = ref_block.T.astype(np.int64)  # shape: (n_genes, n_ref) 
            else:
                ref_data = ref_block.T.astype(np.int64)
                
            return rank_sum_chunk_kernel_hist(
                ref_data, tar_block.T.astype(np.int64),
                tie_correction=tie_correction,
                continuity_correction=continuity_correction,
                use_asymptotic=use_asymptotic,
                max_bins=vmax - vmin + 1
            )
    
    # Use floating-point algorithm
    if ref_already_sorted:
        # Reference group is already sorted, data layout is (n_ref, cols)
        # ref_block: (n_ref, cols), tar_block: (n_tar, cols)
        # Need to transpose to shape suitable for rank_sum_chunk_kernel_float: (cols, n_samples)
        ref_sorted = ref_block.T  # (cols, n_ref)
        tar_sorted = np.sort(tar_block.T, axis=1)  # Transpose then sort: (cols, n_tar)
    else:
        # Both need sorting
        ref_sorted = np.sort(ref_block.T, axis=1)  # (cols, n_ref)
        tar_sorted = np.sort(tar_block.T, axis=1)  # (cols, n_tar)
        
    return rank_sum_chunk_kernel_float(
        ref_sorted, tar_sorted,
        tie_correction, continuity_correction, use_asymptotic
    )


# -- Public API
def parallel_difference_expression(
    adata: ad.AnnData,
    groupby_key: str,
    reference: str,
    groups: Optional[List[str]] = None,
    metric: str = "wilcoxon",
    tie_correction: bool = True,
    continuity_correction: bool = True,
    use_asymptotic: Optional[bool] = None,
    min_samples: int = 2,
    max_bins: int = 100_000, # 1e5
    prefer_hist_if_int: bool = False,
    num_workers: int = 1,
    batch: int = 5000 * 200,
) -> pd.DataFrame:
    """High-performance parallel differential expression analysis.
    
    Performs differential expression analysis using shared memory parallelization
    for efficient analysis of large-scale single-cell data. Algorithmically aligned
    with pdex library while providing superior computational performance.
    
    Args:
        adata: AnnData object containing gene expression data
        groupby_key: Column name in obs for grouping cells
        reference: Name of the reference group
        groups: List of target groups to compare. If None, uses all groups except reference
        metric: Statistical test method. Currently supports "wilcoxon" (Mann-Whitney U test)
        tie_correction: Whether to apply tie correction
        continuity_correction: Whether to apply continuity correction
        use_asymptotic: Force asymptotic approximation. None for auto-selection
        min_samples: Minimum number of samples per group. Groups with fewer samples are excluded
        max_bins: Maximum number of bins for histogram algorithm
        prefer_hist_if_int: Prefer histogram algorithm for integer data
        num_workers: Number of parallel worker processes
        batch_budget: Batch processing budget for task chunking
        
    Returns:
        DataFrame containing results with columns:
            - 'pert': Perturbation group name
            - 'gene': Gene name
            - 'pval': P-value
            - 'fold_change': Fold change (target_mean / reference_mean)
            - 'log2_fold_change': Log2 fold change
            - 'fdr': FDR-corrected p-value
            
    Raises:
        ValueError: If reference group not found or no valid target groups
        TypeError: If unsupported data types are used
        
    Examples:
        >>> import anndata as ad
        >>> adata = ad.read_h5ad("data.h5ad")
        >>> results_df = parallel_difference_expression(
        ...     adata, 
        ...     groupby_key="treatment",
        ...     reference="control",
        ...     num_workers=4
        ... )
        >>> print(f"Analyzed {results_df['pert'].nunique()} groups")
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
    if metric != "wilcoxon":
        raise ValueError(f"Unsupported statistical test method: {metric}")
    
    # Determine whether to use histogram algorithm
    use_hist = (prefer_hist_if_int and np.issubdtype(adata.X.dtype, np.integer) and num_workers > 1)

    # Pre-sort reference group data
    logging.info("Preprocessing reference group data...")
    ref_data_sorted = _create_presorted_ref_data(adata, row_idx_ref)

    # Create shared memory
    shared_resources = []
    try:
        csc_spec, shm_csc = _create_shared_csc(adata.X)
        shared_resources.extend(shm_csc)

        # Create shared memory for pre-sorted reference group
        ref_sorted_spec, shm_ref_sorted = _create_shared_ref_sorted(ref_data_sorted)
        shared_resources.extend(shm_ref_sorted)

        total_len = len(group_rows) * n_genes
        res_spec, shm_res = _create_shared_results(total_len)
        shared_resources.extend(shm_res)

        hist_pool_spec = None
        if use_hist:
            hist_pool_spec, shm_hist = _create_shared_hist_pool(num_workers, max_bins)
            shared_resources.extend(shm_hist)

        # Task chunking
        tasks = _make_chunks(n_genes, group_sizes, batch, use_hist)

        # Parallel execution
        if num_workers <= 1:
            # Single process execution
            iterator = tqdm(tasks, desc="Processing gene chunks")
            for task in iterator:
                args = (csc_spec, res_spec, hist_pool_spec, ref_sorted_spec, group_rows,
                       n_genes, task, 0, metric, max_bins,
                       tie_correction, continuity_correction, use_asymptotic)
                _worker_run_chunk(args)
        else:
            # Multi-process execution
            args_list = []
            for i, task in enumerate(tasks):
                worker_id = i % num_workers
                args = (csc_spec, res_spec, hist_pool_spec, ref_sorted_spec, group_rows,
                       n_genes, task, worker_id, metric, max_bins,
                       tie_correction, continuity_correction, use_asymptotic)
                args_list.append(args)
            
            with mp.Pool(processes=num_workers) as pool:
                list(tqdm(pool.imap_unordered(_worker_run_chunk, args_list, chunksize=1), 
                             total=len(tasks), desc="Parallel processing chunks"))

        # Aggregate results
        pvec, uvec, views_res = _open_shared_results(res_spec)
        try:
            P = pvec.reshape(len(group_rows), n_genes).copy()
            U = uvec.reshape(len(group_rows), n_genes).copy()

            n_failed = np.sum(np.isnan(P))
            if n_failed > 0:
                logging.warning(f"{n_failed} genes failed to compute")

            # Calculate fold change
            logging.info("Calculating fold change...")
            fc_matrix = _compute_fold_change(adata, row_idx_ref, group_rows)

            # Calculate log2 fold change
            logging.info("Calculating log2 fold change...")
            log2_fc_matrix = _compute_log2_fold_change(fc_matrix)

            # Calculate FDR
            logging.info("Calculating FDR correction...")
            fdr_matrix = _compute_fdr(P)

            data = []
            gene_names = adata.var_names.tolist()
            
            for i, pert_name in enumerate(group_names):
                for j, gene_name in enumerate(gene_names):
                    data.append({
                        "target": pert_name,
                        "feature": gene_name,
                        "p_value": P[i, j],
                        "fold_change": fc_matrix[i, j],
                        "log2_fold_change": log2_fc_matrix[i, j],
                        "fdr": fdr_matrix[i, j]
                    })
            
            return pd.DataFrame(data)
            
        finally:
            for s in views_res: 
                try: s.close()
                except: pass
                
    finally:
        _close_shared_memory(shared_resources)

