import math
from typing import Optional, Tuple

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange


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