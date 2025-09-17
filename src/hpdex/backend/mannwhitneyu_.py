from typing import Literal

import numpy as np
import scipy.sparse

from .bin import kernel

__all__ = ["mannwhitneyu", "group_mean"]

# ---------------- 参数映射 ----------------
_zero_handling_map = {
    "none": 0,
    "min": 1,
    "max": 2,
    "mix": 3,
}
_alternative_map = {
    "less": 0,
    "greater": 1,
    "two-sided": 2,
}
_method_map = {
    "exact": 1,
    "asymptotic": 2,
}

# ---------------- 封装：MWU ----------------
def mannwhitneyu(
    sparse_matrix: scipy.sparse.csr_matrix | scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_targets: int,
    ref_sorted: bool = False,
    tar_sorted: bool = False,
    tie_correction: bool = True,
    use_continuity: bool = False,
    fast_norm: bool = False,
    zero_handling: Literal["none", "min", "max", "mix"] = "mix",
    alternative: Literal["less", "greater", "two-sided"] = "two-sided",
    method: Literal["exact", "asymptotic"] = "asymptotic",
    threads: int = -1,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Mann-Whitney U test on sparse CSR/CSC matrix (multi-group)."""
    if not isinstance(sparse_matrix, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        raise ValueError("sparse_matrix must be CSR or CSC")

    # For CSC: group_id corresponds to rows; For CSR: group_id corresponds to columns
    if isinstance(sparse_matrix, scipy.sparse.csc_matrix):
        R = sparse_matrix.shape[0]  # number of rows (samples)
    else:  # CSR
        R = sparse_matrix.shape[1]  # number of columns (samples)
    
    if group_id.ndim != 1 or len(group_id) != R:
        raise ValueError(
            f"group_id must be 1D with length {R} (rows of CSC / cols of CSR)"
        )

    # prepare arrays
    group_id = group_id.astype(np.int32, copy=False)
    data = sparse_matrix.data
    indices = sparse_matrix.indices.astype(np.int64, copy=False)
    indptr = sparse_matrix.indptr.astype(np.int64, copy=False)

    # map parameters
    zero_handling = _zero_handling_map[zero_handling]
    alternative = _alternative_map[alternative]
    method = _method_map[method]

    return kernel.mannwhitneyu(
        data=data,
        indices=indices,
        indptr=indptr,
        group_id=group_id,
        n_targets=n_targets,
        ref_sorted=ref_sorted,
        tar_sorted=tar_sorted,
        tie_correction=tie_correction,
        use_continuity=use_continuity,
        fast_norm=fast_norm,
        zero_handling=zero_handling,
        alternative=alternative,
        method=method,
        threads=threads,
        layout="csc" if isinstance(sparse_matrix, scipy.sparse.csc_matrix) else "csr",
        show_progress=show_progress,
    )

# ---------------- 封装：GroupMean ----------------
def group_mean(
    sparse_matrix: scipy.sparse.csr_matrix | scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_groups: int,
    include_zeros: bool = True,
    use_kahan: bool = False,
    threads: int = -1,
) -> np.ndarray:
    """Compute group-wise mean for each feature (sparse CSR/CSC)."""
    if not isinstance(sparse_matrix, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        raise ValueError("sparse_matrix must be CSR or CSC")

    # For CSC: group_id corresponds to rows; For CSR: group_id corresponds to columns
    if isinstance(sparse_matrix, scipy.sparse.csc_matrix):
        R = sparse_matrix.shape[0]  # number of rows (samples)
    else:  # CSR
        R = sparse_matrix.shape[1]  # number of columns (samples)
    
    if group_id.ndim != 1 or len(group_id) != R:
        raise ValueError(
            f"group_id must be 1D with length {R} (rows of CSC / cols of CSR)"
        )

    group_id = group_id.astype(np.int32, copy=False)
    data = sparse_matrix.data
    indices = sparse_matrix.indices.astype(np.int64, copy=False)
    indptr = sparse_matrix.indptr.astype(np.int64, copy=False)

    return kernel.group_mean(
        data=data,
        indices=indices,
        indptr=indptr,
        group_id=group_id,
        n_groups=n_groups,
        include_zeros=include_zeros,
        threads=threads,
        use_kahan=use_kahan,
        layout="csc" if isinstance(sparse_matrix, scipy.sparse.csc_matrix) else "csr",
    )
