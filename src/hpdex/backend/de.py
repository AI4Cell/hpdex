import anndata as ad
import numpy as np
import pandas as pd
import scipy
from scipy.stats import false_discovery_control

from .mannwhitneyu_ import mannwhitneyu, group_mean

supported_metrics = ["wilcoxon"]

def parallel_differential_expression(
    adata: ad.AnnData,
    groupby_key: str,
    reference: str | None,   # None means pd.NA
    groups: list[str] | None = None,
    metric: str = "wilcoxon",
    tie_correction: bool = True,
    use_continuity: bool = True,
    min_samples: int = 2,
    threads: int = -1,
    clip_value: float = 20.0,
) -> pd.DataFrame:
    """Parallel differential expression analysis with Mann-Whitney U."""

    if metric not in supported_metrics:
        raise ValueError(f"Unsupported metric: {metric}; supported: {supported_metrics}")
    if groupby_key not in adata.obs.columns:
        raise ValueError(f"Groupby key `{groupby_key}` not found in adata.obs")

    obs = adata.obs.copy()

    # reference=None -> treat NA as reference
    if reference is None:
        obs[groupby_key] = obs[groupby_key].cat.add_categories("non-targeting")
        obs[groupby_key] = obs[groupby_key].fillna("non-targeting")
        reference = "non-targeting"

    uniq = obs[groupby_key].unique().tolist()
    if groups is None:
        groups = [g for g in uniq if g != reference]
    else:
        groups = [g for g in groups if g in uniq and g != reference]
    if reference not in uniq:
        raise ValueError(f"Reference `{reference}` not found in `{groupby_key}`")
    if not groups:
        raise ValueError("No valid target groups found")

    # Map groups → IDs
    group_map = {reference: 0}
    for i, g in enumerate(groups):
        group_map[g] = i + 1

    group_id = np.full(len(obs), -1, dtype=np.int32)
    for g, idx in group_map.items():
        group_id[np.asarray(obs[groupby_key] == g)] = idx

    n_targets = len(groups)

    # Sparse matrix preparation
    if isinstance(adata.X, np.ndarray):
        matrix = scipy.sparse.csc_matrix(adata.X)
    elif isinstance(adata.X, scipy.sparse.csr_matrix):
        matrix = adata.X.tocsc()
    else:
        matrix = adata.X  # already csc

    # Compute group means (for fold change)
    means = group_mean(
        matrix,
        group_id,
        n_targets + 1,   # total groups = reference + targets
        include_zeros=True,
        threads=threads,
    ).reshape(n_targets + 1, adata.n_vars)  # shape: (G, C)

    ref_means = means[0]
    tar_means = means[1:]  # (n_targets, C)

    # Compute fold change
    with np.errstate(divide="ignore", invalid="ignore"):
        fold_changes = tar_means / ref_means
        fold_changes = np.where(ref_means < 1e-10, clip_value, fold_changes)
        fold_changes = np.where(tar_means < 1e-10, 1.0/clip_value, fold_changes)
        log2_fold_changes = np.log2(fold_changes)

    # Mann-Whitney U test
    U1, P = mannwhitneyu(
        matrix,
        group_id,
        n_targets,
        ref_sorted=False,
        tar_sorted=False,
        use_continuity=use_continuity,
        tie_correction=tie_correction,
        zero_handling="mix",
        threads=threads,
    )
    U1 = np.asarray(U1).reshape(-1)
    P = np.asarray(P).reshape(-1)

    # -------- 向量化拼接 --------
    n_genes = adata.n_vars
    features = np.asarray(adata.var_names, dtype=object)

    targets_flat = np.repeat(np.asarray(groups, dtype=object), n_genes)
    features_flat = np.tile(features, n_targets)

    fold_changes_flat = fold_changes.reshape(-1)
    log2_fold_changes_flat = log2_fold_changes.reshape(-1)

    # -------- FDR correction --------
    try:
        fdr_values = false_discovery_control(P, method="bh")
    except Exception:
        fdr_values = np.full_like(P, np.nan, dtype=float)

    # -------- Assemble DataFrame --------
    result = pd.DataFrame({
        "target": targets_flat,
        "feature": features_flat,
        "p_value": P,
        "u_statistic": U1,
        "fold_change": fold_changes_flat,
        "log2_fold_change": log2_fold_changes_flat,
        "fdr": fdr_values,
    })

    return result
