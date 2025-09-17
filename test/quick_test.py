import numpy as np
import scipy.sparse
from scipy import stats
from hpdex.backend import mannwhitneyu, group_mean


def test_mannwhitneyu():
    R, C, n_targets = 10, 4, 1
    group_id = np.array([0] * R + [1] * R, dtype=np.int32)

    ref = np.random.randn(R, C).astype(np.float32)
    tar = np.random.randn(R, C).astype(np.float32)
    data = np.vstack([ref, tar])
    matrix = scipy.sparse.csc_matrix(data)

    U1_cpp, P_cpp = mannwhitneyu(
        matrix,
        group_id,
        n_targets,
        zero_handling="min",
        use_continuity=True,
        tie_correction=True,
        threads=1,
    )
    U1_cpp, P_cpp = np.array(U1_cpp).ravel(), np.array(P_cpp).ravel()

    # scipy baseline
    U1_sp, P_sp = [], []
    for j in range(C):
        r = stats.mannwhitneyu(
            ref[:, j], tar[:, j],
            alternative="two-sided",
            method="asymptotic"
        )
        U1_sp.append(r.statistic)
        P_sp.append(r.pvalue)
    U1_sp, P_sp = np.array(U1_sp), np.array(P_sp)

    du = np.max(np.abs(U1_cpp - U1_sp))
    dp = np.max(np.abs(P_cpp - P_sp))
    print(f"mannwhitneyu dU={du:.2e}, dP={dp:.2e}")
    assert du < 1e-8
    assert dp < 1e-6


def test_group_mean():
    R, C = 10, 4
    group_id = np.array([0] * R + [1] * R, dtype=np.int32)

    ref = np.random.randn(R, C).astype(np.float32)
    tar = np.random.randn(R, C).astype(np.float32)
    data = np.vstack([ref, tar])
    matrix = scipy.sparse.csc_matrix(data)

    # hpdex: flatten to (C, G)
    means = group_mean(matrix, group_id, 2, include_zeros=True, threads=10)
    means = means.reshape(C, 2).T  # (G, C)

    # baseline
    full = np.vstack([ref, tar])
    ref_mean = full[group_id == 0].mean(axis=0)
    tar_mean = full[group_id == 1].mean(axis=0)

    dref = np.max(np.abs(means[0] - ref_mean))
    dtar = np.max(np.abs(means[1] - tar_mean))
    print(f"group_mean dref={dref:.2e}, dtar={dtar:.2e}")
    assert dref < 1e-6
    assert dtar < 1e-6


if __name__ == "__main__":
    test_mannwhitneyu()
    test_group_mean()
    print("✅ quick test passed")
