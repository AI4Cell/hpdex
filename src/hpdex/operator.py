import numpy as np
from typing import Optional, Tuple

from .kernel import (
	rank_sum_chunk_kernel_float,
	rank_sum_chunk_kernel_hist,
)

# Optional Cython sparse kernels (if built/installed as a Python extension)
try:
	# Prefer local module name if available on PYTHONPATH
	import sparserank as _sparserank  # type: ignore
except Exception:  # pragma: no cover - optional dependency
	_sparserank = None  # type: ignore


# -------------------- Dense operators --------------------

def wilcoxon_dense(
	ref_sorted: np.ndarray,
	tar_sorted: np.ndarray,
	tie_correction: bool = True,
	continuity_correction: bool = True,
	use_asymptotic: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Wilcoxon rank-sum for dense sorted inputs.
	
	Args:
		ref_sorted: [..., n_ref] sorted ascending
		tar_sorted: [..., n_tar] sorted ascending
	Returns:
		(p_values, U_statistics)
	"""
	return rank_sum_chunk_kernel_float(
		ref_sorted,
		tar_sorted,
		tie_correction=tie_correction,
		continuity_correction=continuity_correction,
		use_asymptotic=use_asymptotic,
	)


def wilcoxon_hist_dense(
	ref_sorted_int: np.ndarray,
	tar_int: np.ndarray,
	tie_correction: bool = True,
	continuity_correction: bool = True,
	use_asymptotic: Optional[bool] = None,
	max_bins: int = 200_000,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Wilcoxon rank-sum using histogram kernel for integer dense data.
	
	Args:
		ref_sorted_int: [..., n_ref] integers, sorted ascending
		tar_int: [..., n_tar] integers (not necessarily sorted)
	Returns:
		(p_values, U_statistics)
	"""
	return rank_sum_chunk_kernel_hist(
		ref_sorted_int,
		tar_int,
		tie_correction=tie_correction,
		continuity_correction=continuity_correction,
		use_asymptotic=use_asymptotic,
		max_bins=max_bins,
	)


# -------------------- Sparse CSC operators (Cython-backed) --------------------
# We expose three interfaces aligned with frame.md:
# 1) name: generic CSC interface (both groups unsorted in CSC via masks)
# 2) name_sorted: both groups provided as concatenated sorted arrays + ptr
# 3) name_ref_sorted: ref provided as sorted+ptr, tar from CSC mask (sorted per-gene internally)


def _require_cython() -> None:
	if _sparserank is None:
		raise ImportError(
			"Cython sparse operator 'sparserank' is not available. "
			"Build the Cython extension or install the wheel in build/hpdex/dist to enable CSC operators."
		)


def wilcoxon_all_csc(
	indptr: np.ndarray,
	indices: np.ndarray,
	data: np.ndarray,
	ref_mask: np.ndarray,
	tar_mask: np.ndarray,
	track_ties: bool = True,
) -> np.ndarray:
	"""Generic CSC interface: both groups from CSC via masks.
	
	Returns array of shape (n_genes, 3): [U1, tie_sum, has_ties].
	"""
	_require_cython()
	return _sparserank.wilcoxon_all_csc(
		indptr.astype(np.int64, copy=False),
		indices.astype(np.int32, copy=False),
		data.astype(np.float64, copy=False),
		ref_mask.astype(np.uint8, copy=False),
		tar_mask.astype(np.uint8, copy=False),
		track_ties,
	)


def wilcoxon_sorted_ptr(
	r_data: np.ndarray,
	r_ptr: np.ndarray,
	t_data: np.ndarray,
	t_ptr: np.ndarray,
	track_ties: bool = True,
) -> np.ndarray:
	"""Both groups pre-sorted and packed as concatenated arrays with ptr.
	
	Returns array of shape (B, 3): [U1, tie_sum, has_ties].
	"""
	_require_cython()
	return _sparserank.wilcoxon_sorted_ptr(
		r_data.astype(np.float64, copy=False),
		r_ptr.astype(np.int64, copy=False),
		t_data.astype(np.float64, copy=False),
		t_ptr.astype(np.int64, copy=False),
		track_ties,
	)


def wilcoxon_ref_sorted_csc(
	ref_data: np.ndarray,
	ref_ptr: np.ndarray,
	indptr: np.ndarray,
	indices: np.ndarray,
	data: np.ndarray,
	tar_mask: np.ndarray,
	track_ties: bool = True,
) -> np.ndarray:
	"""Ref sorted+packed; tar from CSC via mask (sorted per-gene internally).
	
	Returns array of shape (n_genes, 3): [U1, tie_sum, has_ties].
	"""
	_require_cython()
	return _sparserank.wilcoxon_ref_sorted_csc(
		ref_data.astype(np.float64, copy=False),
		ref_ptr.astype(np.int64, copy=False),
		indptr.astype(np.int64, copy=False),
		indices.astype(np.int32, copy=False),
		data.astype(np.float64, copy=False),
		tar_mask.astype(np.uint8, copy=False),
		track_ties,
	)
