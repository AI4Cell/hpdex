# bench_mwu.py
import argparse
import time

import numpy as np
import scipy
from scipy import stats

from hpdex.backend import mannwhitneyu

# ================== 预设 Config ==================
CONFIGS = {
    "mix": {
        "data": dict(value_range=(-100, 100), density=1.0),
        "hpdex": dict(zero_handling="mix"),
        "scipy": dict(),
    },
    "min": {
        "data": dict(value_range=(0, 200), density=1.0),
        "hpdex": dict(zero_handling="min"),
        "scipy": dict(),
    },
    # 你可以继续加，比如:
    "sparse": {
        "data": dict(value_range=(-50, 50), density=0.1),
        "hpdex": dict(zero_handling="mix"),
        "scipy": dict(),
    },
}

DEFAULT_HPDEX_CONFIG = {
    "ref_sorted": False,
    "tar_sorted": False,
    "tie_correction": True,
    "use_continuity": True,
    "fast_norm": False,
    "zero_handling": "mix",
    "alternative": "two-sided",
    "method": "asymptotic",
    "threads": -1,
}
DEFAULT_SCIPY_CONFIG = {
    "alternative": "two-sided",
    "method": "asymptotic",
    "use_continuity": True,
}

# ================== 数据生成器 ==================
def get_data(R, C, n_targets, density=1.0, value_range=(-100, 100), random_state=None, dtype=np.float32):
    if random_state is not None:
        np.random.seed(random_state)
    G = n_targets + 1
    group_id = np.concatenate([np.full((R,), i, dtype=np.int32) for i in range(G)], axis=0)

    datas = []
    low, high = value_range
    for _ in range(G):
        block = np.random.uniform(low, high, size=(R, C)).astype(dtype)
        if density < 1.0:
            mask = np.random.rand(R, C) < density
            block = block * mask
        datas.append(block)
    return group_id, datas


# ================== HPDEX Wrapper ==================
def prepare_mwu(group_id, datas, args):
    data = np.concatenate(datas, axis=0) # [G * R, C]
    data = scipy.sparse.csc_matrix(data)
    n_targets = len(datas) - 1
    def mwu_test(times):
        t0 = time.time()
        U1_list, P_list = [], []
        for _ in range(times):
            U1, P = mannwhitneyu(data, group_id, n_targets, **args)
            U1_list.append(U1)
            P_list.append(P)
        t1 = time.time()
        cost = t1 - t0
        return np.concatenate(U1_list), np.concatenate(P_list), cost
    return mwu_test


# ================== SciPy Wrapper ==================
def prepare_scipy(datas, args):
    ref_data = datas[0]
    tar_datas = datas[1:]
    R, C = ref_data.shape
    n_targets = len(tar_datas)

    def scipy_test(times):
        t0 = time.time()
        U1_all, P_all = [], []
        for _ in range(times):
            U1, P = [], []
            for j in range(C):                # 先遍历特征(列)
                for tar in tar_datas:         # 再遍历目标组
                    x = ref_data[:, j]
                    y = tar[:, j]
                    r = stats.mannwhitneyu(x, y, **args)
                    U1.append(r.statistic)
                    P.append(r.pvalue)
            U1_all.append(U1)
            P_all.append(P)
        t1 = time.time()
        return np.array(U1_all).ravel(), np.array(P_all).ravel(), (t1 - t0)
    return scipy_test


# ================== Main ==================
def main():
    ap = argparse.ArgumentParser(description="Benchmark hpdex Mann-Whitney U")
    ap.add_argument("--config", choices=CONFIGS.keys(), default="mix",
                    help="choose a preset config")
    ap.add_argument("--n_targets", type=int, default=1)
    ap.add_argument("--R", type=int, default=2000)
    ap.add_argument("--C", type=int, default=1000)
    ap.add_argument("--value_range", type=int, nargs=2, default=None, help="override value range")
    ap.add_argument("--density", type=float, default=None, help="override density")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--use_continuity", action="store_true", help="use continuity correction")
    ap.add_argument("--fast_norm", action="store_true", help="use fast normal approximation")
    ap.add_argument("--alternative", choices=["less", "greater", "two-sided"], default="two-sided", help="alternative hypothesis")
    ap.add_argument("--method", choices=["exact", "asymptotic"], default="asymptotic", help="method")
    ap.add_argument("--zero_handling", choices=["none", "min", "max", "mix"], default="mix", help="zero handling")
    ap.add_argument("--check", action="store_true", help="compare against SciPy")
    args = ap.parse_args()

    # 1) 合并配置
    preset = CONFIGS[args.config]
    data_cfg = {**preset["data"]}
    hpdex_cfg = {**DEFAULT_HPDEX_CONFIG, **preset["hpdex"]}
    scipy_cfg = {**DEFAULT_SCIPY_CONFIG, **preset["scipy"]}
    for k, v in vars(args).items():
        if k in hpdex_cfg:
            hpdex_cfg[k] = v
    for k, v in vars(args).items():
        if k in scipy_cfg:
            scipy_cfg[k] = v
    for k, v in vars(args).items():
        if k in data_cfg:
            data_cfg[k] = v

    # 2) 跑实验
    times = []
    for k in range(args.repeats):
        group_id, datas = get_data(args.R, args.C, args.n_targets,
                                   random_state=args.seed + k, **data_cfg)

        mwu_test = prepare_mwu(group_id, datas, hpdex_cfg)
        U1_cpp, P_cpp, t_cpp = mwu_test(1)

        print(f"run {k}: ours={t_cpp:.6f}s pairs={len(U1_cpp)}")
        if args.check:
            scipy_test = prepare_scipy(datas, scipy_cfg)
            U1_sp, P_sp, t_sp = scipy_test(1)
            dp = np.max(np.abs(P_cpp - P_sp))
            du = np.max(np.abs(U1_cpp - U1_sp))
            print(f"   Δp={dp:.2e}, Δu={du:.2e}, scipy={t_sp:.6f}s, speedup={t_sp/t_cpp:.2f}x")

        times.append(t_cpp)

    print("summary:", {
        "median": float(np.median(times)),
        "mean": float(np.mean(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    })


if __name__ == "__main__":
    main()
