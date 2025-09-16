import numpy as np
from scipy.sparse import csc_matrix
from scipy import stats
import hpdex_cpp

def main():
    R, C = 400, 100  # 略放大一点，避免样本过小导致“exact”更合适
    rng = np.random.default_rng(0)

    # 造一些带负/零/正的稠密数据（零会在转换为CSC时变成隐式稀疏，C++侧用 mix 语义补回）
    dense = rng.normal(loc=0.0, scale=1.0, size=(R, C))
    dense[rng.random((R, C)) < 0.55] = 0.0  # 随机置零

    # 分两组：前半 ref=0，后半 tar=1
    group_id = np.zeros(R, dtype=np.int32)
    group_id[R // 2 :] = 1

    # ---------- C++ 扩展（CSC） ----------
    A_csc = csc_matrix(dense)
    U1_cpp, P_cpp = hpdex_cpp.mannwhitneyu_csc(
        A_csc.data.astype(np.float64, copy=False),
        A_csc.indices.astype(np.int64, copy=False),
        A_csc.indptr.astype(np.int64, copy=False),
        group_id,
        n_groups=2,               # 2组：0(ref), 1(target)
        ref_sorted=False,
        tar_sorted=False,
        tie_correction=True,      # 和 SciPy 渐近法一致（考虑 ties）
        use_continuity=False,     # 关闭连续性校正，与 SciPy 对齐
        zero_handling=3,          # mix：负 | 0块 | 正（把隐式0当作真实0参与）
        alternative=2,            # 2=two_sided
        method=2,                 # 2=asymptotic
        threads=1,
        fast_norm=False
    )
    U1_cpp = np.asarray(U1_cpp).reshape(-1)  # Npairs = C * n_targets = C
    P_cpp  = np.asarray(P_cpp).reshape(-1)

    # ---------- SciPy（逐列） ----------
    P_sp   = np.empty(C, dtype=float)
    U1_sp  = np.empty(C, dtype=float)  # SciPy 始终返回参考组的 U1

    for j in range(C):
        x = dense[group_id == 0, j]
        y = dense[group_id == 1, j]

        # 为避免“样本量<2”与C++侧抛错不一致，这里直接断言
        assert x.size >= 2 and y.size >= 2

        # SciPy：two-sided + asymptotic + 关闭连续性校正
        r = stats.mannwhitneyu(
            x, y,
            alternative="two-sided",
            method="asymptotic",
            use_continuity=False
        )
        # r.statistic 始终是参考组的 U1（与 alternative 无关）
        P_sp[j]    = r.pvalue
        U1_sp[j]   = r.statistic

    # ---------- 对齐/对比 ----------
    # SciPy 和我们的 C++ 都始终返回参考组的 U1，应该直接比较
    n1 = (group_id == 0).sum()
    n2 = (group_id == 1).sum()

    # 误差
    p_abs_err  = np.abs(P_cpp - P_sp)
    u_abs_err  = np.abs(U1_cpp - U1_sp)

    print("=== Compare C++(CSC mix) vs SciPy(dense) ===")
    print("n1, n2:", n1, n2)
    print("P  max abs diff:", p_abs_err.max(), "  mean abs diff:", p_abs_err.mean())
    print("U  max abs diff:", u_abs_err.max(), "  mean abs diff:", u_abs_err.mean())

    # 展示前几列
    k = min(10, C)
    print("\nFirst", k, "columns:")
    for j in range(k):
        print(f"[col {j}]  P_cpp={P_cpp[j]:.6g}  P_sp={P_sp[j]:.6g}  |Δ|={p_abs_err[j]:.3g}   "
              f"U1_cpp={U1_cpp[j]:.6g}  U1_sp={U1_sp[j]:.6g}  |Δ|={u_abs_err[j]:.3g}")

if __name__ == "__main__":
    main()
