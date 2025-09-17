"""
hpdex C++ extension module

This module provides high-performance implementations of:
- Mann-Whitney U test (MWU) for differential expression
- Group-wise mean computation over sparse matrices

Backend: implemented in C++ with OpenMP + SIMD (Highway), exposed via pybind11.
"""

from typing import Literal, Tuple

import numpy as np

__all__ = ["mannwhitneyu", "group_mean"]

def mannwhitneyu(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    group_id: np.ndarray,
    n_targets: int,
    ref_sorted: bool = False,
    tar_sorted: bool = False,
    tie_correction: bool = True,
    use_continuity: bool = False,
    fast_norm: bool = True,
    zero_handling: Literal[0, 1, 2, 3] = 0,
    alternative: Literal[0, 1, 2] = 2,
    method: Literal[1, 2] = 2,
    threads: int = -1,
    layout: Literal["csc", "csr"] = "csc",
    show_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Mann-Whitney U test for sparse matrix columns grouped by `group_id`.

    Parameters
    ----------
    data : ndarray
        Non-zero values of the sparse matrix (float32/float64/int/uint).
    indices : ndarray
        Row indices (for CSC) or column indices (for CSR).
    indptr : ndarray
        Column pointer (CSC) or row pointer (CSR).
    group_id : ndarray[int32]
        Length = R (rows). Assigns each row to a group ID in [0, n_targets].
        Group 0 is treated as the reference group.
    n_targets : int
        Number of target groups (total groups = n_targets + 1).
    ref_sorted, tar_sorted : bool
        Assume reference/target samples are pre-sorted (optimization).
    tie_correction : bool
        Apply tie correction for identical values.
    use_continuity : bool
        Apply continuity correction for normal approximation.
    fast_norm : bool
        Use fast normal approximation for large samples.
    zero_handling : {0,1,2,3}
        Strategy for handling zeros:
        0 = ignore, 1 = assign to ref, 2 = assign to targets, 3 = mixed.
    alternative : {0,1,2}
        Hypothesis type:
        0 = "less", 1 = "greater", 2 = "two-sided".
    method : {1,2}
        1 = exact (small samples), 2 = asymptotic (large samples).
    threads : int
        Number of threads (-1 = use all available).
    layout : {"csc", "csr"}
        Input sparse layout.
    show_progress : bool
        Show progress bar.

    Returns
    -------
    U1 : ndarray[float64]
        U statistics, shape = (C * n_targets,)
    P : ndarray[float64]
        p-values, shape = (C * n_targets,)
    """


def group_mean(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    group_id: np.ndarray,
    n_groups: int,
    include_zeros: bool = True,
    threads: int = -1,
    use_kahan: bool = False,
    layout: Literal["csc", "csr"] = "csc",
) -> np.ndarray:
    """
    Compute group-wise means for sparse matrix columns.

    Parameters
    ----------
    data : ndarray
        Non-zero values of the sparse matrix (any numeric type).
    indices : ndarray
        Row indices (for CSC) or column indices (for CSR).
    indptr : ndarray
        Column pointer (CSC) or row pointer (CSR).
    group_id : ndarray[int32]
        Length = R (rows). Maps each row to a group ID [0, n_groups).
    n_groups : int
        Total number of groups.
    include_zeros : bool, default=True
        Whether to treat missing entries as explicit zeros.
    threads : int
        Number of threads (-1 = use all available).
    use_kahan : bool, default=False
        Whether to use Kahan summation.
    layout : {"csc", "csr"}
        Input sparse layout.

    Returns
    -------
    ndarray[float64]
        Mean values per group per column, shape = (C * n_groups,)
        Stored in column-major order: result[j * n_groups + g]
    """
