import argparse
import time
import numpy as np
from scipy.sparse import csc_matrix
from scipy import stats
from hpdex.backend import kernel


def run_once(n_targets: int, rows_per_group: int, C: int, density: float, seed: int,
             fast_norm: bool, zero_handling: int, alternative: int, method: int,
             threads: int, check_with_scipy: bool):
    rng = np.random.default_rng(seed)
    G = n_targets + 1
    R_total = rows_per_group * G

    # 为每个组生成数据后按行拼接；密度控制非零比例
    groups = []
    for g in range(G):
        mat = rng.normal(loc=0.0, scale=1.0, size=(rows_per_group, C))
        mask = rng.random((rows_per_group, C)) >= density
        mat[mask] = 0.0
        groups.append(mat)
    dense = np.vstack(groups)

    # group_id: 0..n_targets，每组 rows_per_group 行
    group_id = np.repeat(np.arange(G, dtype=np.int32), rows_per_group)

    A = csc_matrix(dense)

    t0 = time.perf_counter()
    U1_cpp, P_cpp = kernel.mannwhitneyu(
        A.data.astype(np.float64, copy=False),
        A.indices.astype(np.int64, copy=False),
        A.indptr.astype(np.int64, copy=False),
        group_id,
        n_targets=n_targets,
        ref_sorted=False,
        tar_sorted=False,
        tie_correction=True,
        use_continuity=False,
        fast_norm=fast_norm,
        zero_handling=zero_handling,
        alternative=alternative,
        method=method,
        threads=threads,
        layout="csc",
    )
    t1 = time.perf_counter()

    elapsed = t1 - t0
    Npairs = C * n_targets

    stats_out = {
        "elapsed_s": elapsed,
        "pairs": int(Npairs),
        "pairs_per_s": Npairs / elapsed if elapsed > 0 else float("inf"),
    }

    if check_with_scipy:
        U1_cpp = np.asarray(U1_cpp).reshape(-1)
        P_cpp = np.asarray(P_cpp).reshape(-1)
        P_sp = np.empty(Npairs, dtype=float)
        U1_sp = np.empty(Npairs, dtype=float)
        t_sc0 = time.perf_counter()
        for j in range(C):
            x = dense[0*rows_per_group:1*rows_per_group, j]
            for tg in range(1, G):
                y = dense[tg*rows_per_group:(tg+1)*rows_per_group, j]
                r = stats.mannwhitneyu(
                    x, y, alternative={0:"less",1:"greater",2:"two-sided"}[alternative],
                    method="asymptotic" if method==2 else "exact",
                    use_continuity=False,
                )
                idx = j * n_targets + (tg - 1)
                P_sp[idx] = r.pvalue
                U1_sp[idx] = r.statistic
        t_sc1 = time.perf_counter()
        scipy_elapsed = t_sc1 - t_sc0
        stats_out.update({
            "p_max_abs_err": float(np.max(np.abs(P_cpp - P_sp))),
            "p_mean_abs_err": float(np.mean(np.abs(P_cpp - P_sp))),
            "u_max_abs_err": float(np.max(np.abs(U1_cpp - U1_sp))),
            "scipy_elapsed_s": scipy_elapsed,
            "scipy_pairs_per_s": C / scipy_elapsed if scipy_elapsed > 0 else float("inf"),
            "speedup_vs_scipy": (C / elapsed) / (C / scipy_elapsed) if scipy_elapsed > 0 and elapsed > 0 else float("inf"),
        })

    return stats_out


def main():
    ap = argparse.ArgumentParser(description="Benchmark hpdex_cpp Mann-Whitney U")
    ap.add_argument("--n_targets", type=int, default=1, help="#target groups")
    ap.add_argument("--rows", type=int, default=2000, help="rows per group")
    ap.add_argument("--C", type=int, default=200, help="cols (features)")
    ap.add_argument("--density", type=float, default=0.45, help="non-zero density in [0,1]")
    ap.add_argument("--repeats", type=int, default=3, help="repeat count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--fast", action="store_true", help="use fast norm path")
    ap.add_argument("--precise", action="store_true", help="use precise norm path (override fast)")
    ap.add_argument("--zero", type=int, default=3, choices=[0,1,2,3], help="zero handling: 0 none,1 min,2 max,3 mix")
    ap.add_argument("--alt", type=int, default=2, choices=[0,1,2], help="alternative: 0 less,1 greater,2 two_sided")
    ap.add_argument("--method", type=int, default=2, choices=[1,2], help="method: 1 exact, 2 asymptotic")
    ap.add_argument("--check", action="store_true", help="compare against SciPy")
    args = ap.parse_args()

    if args.precise:
        fast_norm = False
    else:
        fast_norm = True if args.fast else True  # default fast

    times = []
    for k in range(args.repeats):
        stats_out = run_once(
            n_targets=args.n_targets, rows_per_group=args.rows, C=args.C,
            density=args.density, seed=args.seed + k,
            fast_norm=fast_norm, zero_handling=args.zero, alternative=args.alt,
            method=args.method, threads=args.threads, check_with_scipy=args.check,
        )
        times.append(stats_out["elapsed_s"])
        if args.check:
            # 简洁输出：Δp、Δu、双方用时与加速比
            dp = stats_out.get("p_max_abs_err", 0.0)
            du = stats_out.get("u_max_abs_err", 0.0)
            t_cpp = stats_out["elapsed_s"]
            t_sp = stats_out.get("scipy_elapsed_s", float('nan'))
            spd = stats_out.get("speedup_vs_scipy", float('nan'))
            print(f"run {k}: Δp={dp:.2e} Δu={du} ours={t_cpp:.6f}s scipy={t_sp:.6f}s speedup={spd:.2f}x")
        else:
            # 无校验时打印双方用时不可用，仅打印我方用时
            print(f"run {k}: ours={stats_out['elapsed_s']:.6f}s pairs={stats_out['pairs']}")

    times = np.array(times)
    print("summary:", {
        "median_s": float(np.median(times)),
        "mean_s": float(np.mean(times)),
        "min_s": float(np.min(times)),
        "max_s": float(np.max(times)),
    })


if __name__ == "__main__":
    main()


