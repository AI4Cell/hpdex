#include "mannwhitneyu.hpp"
#include "common.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <limits>

namespace hpdexc {

// ========================= Utilities =========================
template<class T>
static FORCE_INLINE bool is_valid_value(const T& v) {
    return !is_nan(v) && !is_inf(v);
}

template<class T>
static FORCE_INLINE typename MannWhitneyuOption<T>::Method
choose_method(size_t n1, size_t n2, bool has_tie) {
    return (n1 > 8 && n2 > 8) || has_tie
        ? MannWhitneyuOption<T>::Method::asymptotic
        : MannWhitneyuOption<T>::Method::exact;
}

// ---------- (Optional) fast erfc ----------
static FORCE_INLINE double fast_erfc(double x) {
#ifdef HPDEXC_USE_FAST_ERFC
    // Abramowitz & Stegun 7.1.26-like approximation
    // Max rel. error ~1e-7; plenty for p-values
    const double ax = std::fabs(x);
    const double t  = 1.0 / (1.0 + 0.5 * ax);
    const double tau = t * std::exp(
        -ax*ax
        - 1.26551223
        + t * ( 1.00002368
        + t * ( 0.37409196
        + t * ( 0.09678418
        + t * (-0.18628806
        + t * ( 0.27886807
        + t * (-1.13520398
        + t * ( 1.48851587
        + t * (-0.82215223
        + t * ( 0.17087277 ))))))))))
    );
    double r = (x >= 0.0) ? tau : 2.0 - tau;
    // Clamp to [0,2]
    if (r < 0.0) r = 0.0;
    if (r > 2.0) r = 2.0;
    return r;
#else
    return std::erfc(x);
#endif
}

static FORCE_INLINE double normal_sf(double z) {
    // 上尾概率：P(Z >= z)
    // 使用可选 fast_erfc 包装，避免在热路径里多次触发 libc 调用开销
    return 0.5 * fast_erfc(z / std::sqrt(2.0));
}

// 预计算：给定 (n1, n2, tie_sum) 返回 mu 与 1/sd（sd=0 则 inv_sd=0）
static FORCE_INLINE void precompute_mu_inv_sd(
    size_t n1, size_t n2, double tie_sum,
    double& mu, double& inv_sd
) {
    const double dn1 = static_cast<double>(n1);
    const double dn2 = static_cast<double>(n2);
    const double N   = dn1 + dn2;

    mu = 0.5 * dn1 * dn2;

    const double denom = N * (N - 1.0);
    const double base  = dn1 * dn2 / 12.0;
    const double var   = (denom > 0.0)
        ? base * (N + 1.0 - tie_sum / denom)
        : dn1 * dn2 * (N + 1.0) / 12.0;

    inv_sd = (var <= 0.0) ? 0.0 : (1.0 / std::sqrt(var));
}

// ====== 正态近似 p 值（按 alternative 拆分，内部仅做 O(1) 运算） ======
template<class T>
static FORCE_INLINE double p_asymptotic_two_sided_fast(
    double U1, size_t n1, size_t n2,
    double tie_sum, double cc,     // 预先计算的 continuity correction（0 或 0.5）
    double& mu_out, double& invsd_out
) {
    precompute_mu_inv_sd(n1, n2, tie_sum, mu_out, invsd_out);
    if (invsd_out == 0.0) return 1.0;
    const double z = (std::fabs(U1 - mu_out) - cc) * invsd_out;
    return 2.0 * normal_sf(z);
}

template<class T>
static FORCE_INLINE double p_asymptotic_greater_fast(
    double U1, size_t n1, size_t n2,
    double tie_sum, double cc,
    double& mu_out, double& invsd_out
) {
    precompute_mu_inv_sd(n1, n2, tie_sum, mu_out, invsd_out);
    if (invsd_out == 0.0) return 1.0;
    const double z = (U1 - mu_out - cc) * invsd_out;
    return normal_sf(z);
}

template<class T>
static FORCE_INLINE double p_asymptotic_less_fast(
    double U1, size_t n1, size_t n2,
    double tie_sum, double cc,
    double& mu_out, double& invsd_out
) {
    precompute_mu_inv_sd(n1, n2, tie_sum, mu_out, invsd_out);
    if (invsd_out == 0.0) return 1.0;
    const double z = (U1 - mu_out + cc) * invsd_out;
    return 1.0 - normal_sf(z);
}

// ====== exact p calculation (no ties), SciPy-aligned SF; ultra-optimized ======
template<class T>
static FORCE_INLINE double p_exact(double U, size_t n1, size_t n2) {
    using U64 = unsigned long long;

    const size_t Umax = n1 * n2;
    if UNLIKELY(Umax == 0) return 1.0;

    // clamp & floor like SciPy
    const double U_clip = std::max(0.0, std::min(static_cast<double>(Umax), U));
    const size_t u_stat = static_cast<size_t>(std::floor(U_clip));

    // DP buffers: only write 0..up each iteration; avoid O(SZ) clears
    const size_t SZ = Umax + 1;
    hpdexc::AlignedVector<U64, 64> dp(SZ);   // payload 未初始化没关系，我们只在 0..up 范围写
    hpdexc::AlignedVector<U64, 64> ndp(SZ);

    U64* RESTRICT dp_cur = dp.aligned_data();
    U64* RESTRICT dp_nxt = ndp.aligned_data();

    dp_cur[0] = 1ULL;

    for (size_t i = 1; i <= n1; ++i) {
        const size_t prev_up = (i - 1) * n2;
        const size_t up      =  i      * n2;

        if (up + HPDEXC_PREFETCH_DIST < SZ)
            PREFETCH_W(dp_nxt + up + HPDEXC_PREFETCH_DIST, 1);

        U64 win = 0ULL;

        size_t u = 0;
        const size_t bound1 = prev_up < up ? prev_up : up;
        for (; u <= bound1; ++u) {
            if (u + HPDEXC_PREFETCH_DIST <= prev_up)
                PREFETCH_R(dp_cur + u + HPDEXC_PREFETCH_DIST, 1);
            win += dp_cur[u];
            if (u >= n2 + 1) win -= dp_cur[u - (n2 + 1)];
            dp_nxt[u] = win;
        }

        for (; u <= up; ++u) {
            if (u >= n2 + 1) win -= dp_cur[u - (n2 + 1)];
            dp_nxt[u] = win;
        }

        U64* tmp = dp_cur; dp_cur = dp_nxt; dp_nxt = tmp;
    }

    double total = 0.0;
    for (size_t u = 0; u <= Umax; ++u) {
        if (u + HPDEXC_PREFETCH_DIST <= Umax)
            PREFETCH_R(dp_cur + u + HPDEXC_PREFETCH_DIST, 1);
        total += static_cast<double>(dp_cur[u]);
    }

    const size_t kc    = Umax - u_stat;
    const size_t small = (u_stat < kc ? u_stat : kc);

    double cdf_small = 0.0;
    for (size_t u = 0; u <= small; ++u) {
        if (u + HPDEXC_PREFETCH_DIST <= small)
            PREFETCH_R(dp_cur + u + HPDEXC_PREFETCH_DIST, 1);
        cdf_small += static_cast<double>(dp_cur[u]);
    }
    const double pmf_small = static_cast<double>(dp_cur[small]);

    double sf_ge;
    if (u_stat <= kc) {
        sf_ge = 1.0 - cdf_small / total + pmf_small / total;
    } else {
        sf_ge = cdf_small / total;
    }
    return sf_ge;
}

// ========================= Ranking cores =========================

// strict min/max 的稀疏块贡献（分支最小化 + 掩码）
static FORCE_INLINE void
sparse_strict_minor_major_core(
    const bool   position_head,
    const size_t ref_sp_cnt,
    const size_t tar_sp_cnt,
    const size_t N,
    double&      R1g,
    double&      tie_sum_g,
    bool&        has_tie_g,
    size_t&      grank_g
){
    const size_t sp_tie = ref_sp_cnt + tar_sp_cnt;
    const double tt     = static_cast<double>(sp_tie);
    const double mask   = (sp_tie > 1) ? 1.0 : 0.0;
    tie_sum_g += (tt*tt*tt - tt) * mask;
    has_tie_g  = has_tie_g || (sp_tie > 1);

    const double m   = position_head ? 1.0 : 0.0;
    const double nm  = 1.0 - m;
    const double ldN = (double)N;
    const double ldS = (double)sp_tie;
    const double start = m*1.0 + nm*(ldN - ldS + 1.0);
    const double end   = m*ldS  + nm*(ldN);
    const double avg   = 0.5 * (start + end);

    R1g     += (double)ref_sp_cnt * avg;
    grank_g += sp_tie;
}

// tar-only 归并策略（高密/稀疏两套，减少分支预测失败）
struct TarMergeDense {
    template<class T>
    FORCE_INLINE void operator()(
        const T& val, size_t count, bool& have_run, T& run_val, size_t& run_len,
        double& tie_sum_g, bool& has_tie_g, size_t& grank_g
    ) const {
        if UNLIKELY(!count) return;
        if LIKELY(have_run && !(run_val < val) && !(val < run_val)) {
            run_len += count;
        } else {
            if (run_len > 1) {
                const double tt = (double)run_len;
                tie_sum_g += (tt*tt*tt - tt);
                has_tie_g  = true;
            }
            run_val  = val;
            run_len  = count;
            have_run = true;
        }
        grank_g += count;
    }
};
struct TarMergeSparse {
    template<class T>
    FORCE_INLINE void operator()(
        const T& val, size_t count, bool& have_run, T& run_val, size_t& run_len,
        double& tie_sum_g, bool& has_tie_g, size_t& grank_g
    ) const {
        if UNLIKELY(!count) return;
        if UNLIKELY(have_run && !(run_val < val) && !(val < run_val)) {
            run_len += count;
        } else {
            if (run_len > 1) {
                const double tt = (double)run_len;
                tie_sum_g += (tt*tt*tt - tt);
                has_tie_g  = true;
            }
            run_val  = val;
            run_len  = count;
            have_run = true;
        }
        grank_g += count;
    }
};

// 稀疏值处于中间：模板化实现（根据列密度选择 TarMergeDense/ Sparse）
template<class T, class TarMerge>
static FORCE_INLINE void sparse_medium_core_impl(
    const T* RESTRICT col_val,
    const size_t* RESTRICT off,
    const size_t* RESTRICT gnnz,
    const size_t* RESTRICT sparse_value_cnt,
    const size_t  G,
    const T* RESTRICT refv,
    const size_t nref_exp,
    const T      sparse_value,
    size_t* RESTRICT tar_ptrs_local,
    size_t* RESTRICT grank,
    size_t* RESTRICT tar_eq,
    size_t* RESTRICT sp_left,
    bool*   RESTRICT have_run,
    T*      RESTRICT run_val,
    size_t* RESTRICT run_len,
    double* RESTRICT R1,
    double* RESTRICT tie_sum,
    bool*   RESTRICT has_tie,
    const TarMerge& merge
){
    col_val = ASSUME_ALIGNED(col_val, HPDEXC_ALIGN_SIZE);
    off     = ASSUME_ALIGNED(off,     HPDEXC_ALIGN_SIZE);
    gnnz    = ASSUME_ALIGNED(gnnz,    HPDEXC_ALIGN_SIZE);
    refv    = ASSUME_ALIGNED(refv,    HPDEXC_ALIGN_SIZE);

    auto flush_run = [&](size_t g){
        if (run_len[g] > 1) {
            const double tt = (double)run_len[g];
            tie_sum[g] += (tt*tt*tt - tt);
            has_tie[g]  = true;
        }
        run_len[g]  = 0;
        have_run[g] = false;
    };

    size_t i = 0;
    while (i < nref_exp || sp_left[0] > 0) {
        if (i + HPDEXC_PREFETCH_DIST < nref_exp)
            PREFETCH_R(refv + i + HPDEXC_PREFETCH_DIST, 1);

        T      vref;
        size_t ref_tie = 0;

        if (sp_left[0] > 0 && (i >= nref_exp || !(refv[i] < sparse_value))) {
            vref = sparse_value;
            size_t k_exp = 0;
            while (i + k_exp < nref_exp &&
                   !(refv[i + k_exp] < vref) && !(vref < refv[i + k_exp])) ++k_exp;
            ref_tie   = sp_left[0] + k_exp;
            sp_left[0]= 0;
            i        += k_exp;
        } else {
            vref = (i < nref_exp) ? refv[i] : sparse_value;
            const size_t ref_start = i;
            while (i + 1 < nref_exp &&
                   !(refv[i + 1] < vref) && !(vref < refv[i + 1])) ++i;
            ref_tie = i - ref_start + 1;
            ++i;
        }

        for (size_t g = 1; g < G; ++g) {
            size_t& tp        = tar_ptrs_local[g];
            const size_t gend = off[g] + gnnz[g];

            if (tp + HPDEXC_PREFETCH_DIST < gend)
                PREFETCH_R(col_val + tp + HPDEXC_PREFETCH_DIST, 1);

            while (true) {
                const bool has_exp = (tp < gend) && (col_val[tp] < vref);
                const bool has_sp  = (sp_left[g] > 0) && (sparse_value < vref);
                if (!has_exp && !has_sp) break;

                if (has_exp && has_sp) {
                    const T ev = col_val[tp];
                    if (ev < sparse_value) {
                        size_t j = tp + 1;
                        while (j < gend &&
                               !(col_val[j] < ev) && !(ev < col_val[j])) ++j;
                        const size_t blk = j - tp;
                        merge(ev, blk, have_run[g], run_val[g], run_len[g],
                              tie_sum[g], has_tie[g], grank[g]);
                        tp = j;
                    } else if (sparse_value < ev) {
                        const size_t blk = sp_left[g];
                        merge(sparse_value, blk, have_run[g], run_val[g], run_len[g],
                              tie_sum[g], has_tie[g], grank[g]);
                        sp_left[g] = 0;
                    } else {
                        size_t j = tp + 1;
                        while (j < gend &&
                               !(col_val[j] < sparse_value) && !(sparse_value < col_val[j])) ++j;
                        const size_t blk_exp = j - tp;
                        merge(sparse_value, blk_exp, have_run[g], run_val[g], run_len[g],
                              tie_sum[g], has_tie[g], grank[g]);
                        tp = j;
                        if (sp_left[g] > 0) {
                            const size_t blk_sp = sp_left[g];
                            merge(sparse_value, blk_sp, have_run[g], run_val[g], run_len[g],
                                  tie_sum[g], has_tie[g], grank[g]);
                            sp_left[g] = 0;
                        }
                    }
                } else if (has_exp) {
                    const T ev = col_val[tp];
                    size_t j = tp + 1;
                    while (j < gend &&
                           !(col_val[j] < ev) && !(ev < col_val[j])) ++j;
                    const size_t blk = j - tp;
                    merge(ev, blk, have_run[g], run_val[g], run_len[g],
                          tie_sum[g], has_tie[g], grank[g]);
                    tp = j;
                } else {
                    const size_t blk = sp_left[g];
                    merge(sparse_value, blk, have_run[g], run_val[g], run_len[g],
                          tie_sum[g], has_tie[g], grank[g]);
                    sp_left[g] = 0;
                }
            }
            flush_run(g);

            size_t eq = 0;
            while (tp < gend && !(col_val[tp] < vref) && !(vref < col_val[tp])) { ++tp; ++eq; }
            if (sp_left[g] > 0 && !(sparse_value < vref) && !(vref < sparse_value)) {
                eq += sp_left[g];
                sp_left[g] = 0;
            }
            tar_eq[g] = eq;
        }

        for (size_t g = 1; g < G; ++g) {
            const double rrcur    = (double)grank[g];
            const size_t t        = ref_tie + tar_eq[g];
            const double rrnext   = rrcur + (double)t;
            const double avg_rank = 0.5 * (rrcur + rrnext + 1.0);

            R1[g]    += (double)ref_tie * avg_rank;
            grank[g]  = (size_t)rrnext;

            if UNLIKELY(t > 1) {
                const double tt = (double)t;
                tie_sum[g] += (tt*tt*tt - tt);
                has_tie[g]  = true;
            }
            tar_eq[g] = 0;
        }
    }

    for (size_t g = 1; g < G; ++g) {
        size_t& tp  = tar_ptrs_local[g];
        const size_t gend = off[g] + gnnz[g];

        have_run[g] = false;
        run_len[g]  = 0;

        while (tp < gend || sp_left[g] > 0) {
            const bool has_exp = (tp < gend);
            const bool has_sp  = (sp_left[g] > 0);

            T cand; bool take_sp = false;
            if (!has_exp) { cand = sparse_value; take_sp = true; }
            else if (!has_sp) { cand = col_val[tp]; }
            else {
                const T ev = col_val[tp];
                if (sparse_value < ev) { cand = sparse_value; take_sp = true; }
                else                   { cand = ev; }
            }

            if (take_sp) {
                merge(cand, sp_left[g], have_run[g], run_val[g], run_len[g],
                      tie_sum[g], has_tie[g], grank[g]);
                sp_left[g] = 0;
            } else {
                size_t j = tp + 1;
                while (j < gend && !(col_val[j] < cand) && !(cand < col_val[j])) ++j;
                const size_t blk = j - tp;
                merge(cand, blk, have_run[g], run_val[g], run_len[g],
                      tie_sum[g], has_tie[g], grank[g]);
                tp = j;
            }
        }
        if (run_len[g] > 1) {
            const double tt = (double)run_len[g];
            tie_sum[g] += (tt*tt*tt - tt);
            has_tie[g]  = true;
        }
        run_len[g]  = 0;
        have_run[g] = false;
    }
}

template<class T>
static FORCE_INLINE void sparse_medium_core_dense(
    const T* RESTRICT col_val,
    const size_t* RESTRICT off,
    const size_t* RESTRICT gnnz,
    const size_t* RESTRICT sparse_value_cnt,
    const size_t  G,
    const T* RESTRICT refv,
    const size_t nref_exp,
    const T      sparse_value,
    size_t* RESTRICT tar_ptrs_local,
    size_t* RESTRICT grank,
    size_t* RESTRICT tar_eq,
    size_t* RESTRICT sp_left,
    bool*   RESTRICT have_run,
    T*      RESTRICT run_val,
    size_t* RESTRICT run_len,
    double* RESTRICT R1,
    double* RESTRICT tie_sum,
    bool*   RESTRICT has_tie
){
    TarMergeDense merger{};
    sparse_medium_core_impl<T>(col_val, off, gnnz, sparse_value_cnt, G, refv, nref_exp,
                               sparse_value, tar_ptrs_local, grank, tar_eq, sp_left,
                               have_run, run_val, run_len, R1, tie_sum, has_tie, merger);
}
template<class T>
static FORCE_INLINE void sparse_medium_core_sparse(
    const T* RESTRICT col_val,
    const size_t* RESTRICT off,
    const size_t* RESTRICT gnnz,
    const size_t* RESTRICT sparse_value_cnt,
    const size_t  G,
    const T* RESTRICT refv,
    const size_t nref_exp,
    const T      sparse_value,
    size_t* RESTRICT tar_ptrs_local,
    size_t* RESTRICT grank,
    size_t* RESTRICT tar_eq,
    size_t* RESTRICT sp_left,
    bool*   RESTRICT have_run,
    T*      RESTRICT run_val,
    size_t* RESTRICT run_len,
    double* RESTRICT R1,
    double* RESTRICT tie_sum,
    bool*   RESTRICT has_tie
){
    TarMergeSparse merger{};
    sparse_medium_core_impl<T>(col_val, off, gnnz, sparse_value_cnt, G, refv, nref_exp,
                               sparse_value, tar_ptrs_local, grank, tar_eq, sp_left,
                               have_run, run_val, run_len, R1, tie_sum, has_tie, merger);
}

// 非稀疏：带 tie 修正
template<class T>
static FORCE_INLINE void sparse_none_core(
    const T*       RESTRICT col_val,
    const size_t*  RESTRICT off,
    const size_t*  RESTRICT gnnz,
    const size_t            G,
    const T*       RESTRICT refv,
    const size_t            nrefcol,
    size_t*        RESTRICT tar_ptrs_local,
    size_t*        RESTRICT grank,
    size_t*        RESTRICT tie_cnt,
    size_t*        RESTRICT tar_eq,
    double*        RESTRICT R1,
    double*        RESTRICT tie_sum,
    bool*          RESTRICT has_tie
){
    col_val = ASSUME_ALIGNED(col_val, HPDEXC_ALIGN_SIZE);
    off     = ASSUME_ALIGNED(off,     HPDEXC_ALIGN_SIZE);
    gnnz    = ASSUME_ALIGNED(gnnz,    HPDEXC_ALIGN_SIZE);
    refv    = ASSUME_ALIGNED(refv,    HPDEXC_ALIGN_SIZE);

    auto flush_tar_run = [&](size_t g){
        if (tie_cnt[g] > 0) {
            const double t = (double)(tie_cnt[g] + 1);
            tie_sum[g] += t*t*t - t;
            tie_cnt[g]  = 0;
            has_tie[g]  = true;
        }
    };

    size_t i = 0;
    while (i < nrefcol) {
        if (i + HPDEXC_PREFETCH_DIST < nrefcol)
            PREFETCH_R(refv + i + HPDEXC_PREFETCH_DIST, 1);

        const T vref = refv[i];

        for (size_t g = 1; g < G; ++g) {
            size_t& tp        = tar_ptrs_local[g];
            const size_t gbeg = off[g];
            const size_t gend = off[g] + gnnz[g];

            if (tp + HPDEXC_PREFETCH_DIST < gend)
                PREFETCH_R(col_val + tp + HPDEXC_PREFETCH_DIST, 1);

            while (tp < gend && col_val[tp] < vref) {
                if (tp > gbeg) {
                    if (!(col_val[tp] < col_val[tp-1]) && !(col_val[tp-1] < col_val[tp])) {
                        ++tie_cnt[g];
                    } else {
                        flush_tar_run(g);
                    }
                }
                ++tp;
                ++grank[g];
            }
            flush_tar_run(g);

            size_t eq = 0;
            while (tp < gend && !(col_val[tp] < vref) && !(vref < col_val[tp])) { ++tp; ++eq; }
            tar_eq[g] = eq;
        }

        const size_t ref_start = i;
        while ((i + 1) < nrefcol && !(refv[i + 1] < vref) && !(vref < refv[i + 1])) ++i;
        const size_t ref_tie = i - ref_start + 1;

        for (size_t g = 1; g < G; ++g) {
            const double rrcur    = (double)grank[g];
            const size_t t        = ref_tie + tar_eq[g];
            const double rrnext   = rrcur + (double)t;
            const double avg_rank = 0.5 * (rrcur + rrnext + 1.0);

            R1[g]     += (double)ref_tie * avg_rank;
            grank[g]   = (size_t)rrnext;

            if UNLIKELY(t > 1) {
                const double tt = (double)t;
                tie_sum[g] += (tt*tt*tt - tt);
                has_tie[g]  = true;
            }
            tie_cnt[g] = 0;
            tar_eq[g]  = 0;
        }
        ++i;
    }

    for (size_t g = 1; g < G; ++g) {
        size_t& tp  = tar_ptrs_local[g];
        const size_t gend = off[g] + gnnz[g];
        while (tp < gend) {
            size_t j = tp + 1;
            while (j < gend && !(col_val[j] < col_val[tp]) && !(col_val[tp] < col_val[j])) ++j;
            const size_t block_len = j - tp;
            if (block_len > 1) {
                const double tt = (double)block_len;
                tie_sum[g] += (tt*tt*tt - tt);
                has_tie[g]  = true;
            }
            tp = j;
        }
    }
}

// 非稀疏：无 tie 修正（最薄热路径）
template<class T>
static FORCE_INLINE void sparse_core_without_tie(
    const T*      RESTRICT col_val,
    const size_t* RESTRICT off,
    const size_t* RESTRICT gnnz,
    const size_t           G,
    const T*      RESTRICT refv,
    const size_t           nrefcol,
    size_t*       RESTRICT tar_ptrs_local,
    size_t*       RESTRICT grank,
    size_t*       RESTRICT tar_eq,
    double*       RESTRICT R1
){
    col_val = ASSUME_ALIGNED(col_val, HPDEXC_ALIGN_SIZE);
    off     = ASSUME_ALIGNED(off,     HPDEXC_ALIGN_SIZE);
    gnnz    = ASSUME_ALIGNED(gnnz,    HPDEXC_ALIGN_SIZE);
    refv    = ASSUME_ALIGNED(refv,    HPDEXC_ALIGN_SIZE);

    size_t i = 0;
    while (i < nrefcol) {
        if (i + HPDEXC_PREFETCH_DIST < nrefcol)
            PREFETCH_R(refv + i + HPDEXC_PREFETCH_DIST, 1);

        const T vref = refv[i];

        for (size_t g = 1; g < G; ++g) {
            size_t& tp        = tar_ptrs_local[g];
            const size_t gend = off[g] + gnnz[g];

            if (tp + HPDEXC_PREFETCH_DIST < gend)
                PREFETCH_R(col_val + tp + HPDEXC_PREFETCH_DIST, 1);

            while (tp < gend && col_val[tp] < vref) { ++tp; ++grank[g]; }

            size_t eq = 0;
            while (tp < gend && !(col_val[tp] < vref) && !(vref < col_val[tp])) { ++tp; ++eq; }
            tar_eq[g] = eq;
        }

        const size_t ref_start = i;
        while ((i + 1) < nrefcol && !(refv[i + 1] < vref) && !(vref < refv[i + 1])) ++i;
        const size_t ref_tie = i - ref_start + 1;

        for (size_t g = 1; g < G; ++g) {
            const double rrcur    = (double)grank[g];
            const size_t t        = ref_tie + tar_eq[g];
            const double rrnext   = rrcur + (double)t;
            const double avg_rank = 0.5 * (rrcur + rrnext + 1.0);

            R1[g]    += (double)ref_tie * avg_rank;
            grank[g]  = (size_t)rrnext;
            tar_eq[g] = 0;
        }
        ++i;
    }
}

// ========================= 主算子 =========================

template<class T, class Idx>
MannWhitneyResult
mannwhitneyu(
    const tensor::Csc<T, Idx>& A,
    const MannWhitneyuOption<T>& opt,
    const tensor::Vector<int32_t>& group_id,
    size_t n_targets,
    int threads,
    size_t* RESTRICT progress_ptr
) {
    if UNLIKELY(group_id.size() != A.rows()) {
        throw std::invalid_argument("[mannwhitney] group_id must be length = A.rows");
    }
    if UNLIKELY(n_targets == 0) {
        return {
            tensor::Ndarray<double>::zeros({A.cols(), 0}),
            tensor::Ndarray<double>::zeros({A.cols(), 0})
        };
    }

    const bool use_sparse_value =
        opt.sparse_type != MannWhitneyuOption<T>::SparseValueMinmax::none;

    // 进度
    AlignedVector<size_t> progress_dummy;
    const int n_threads_runtime = (threads < 0 ? MAX_THREADS() : threads);
    progress_dummy.resize((size_t)std::max(1, n_threads_runtime), 0);
    size_t* RESTRICT progress_safe = progress_ptr ? progress_ptr : progress_dummy.data();

    const size_t R = A.rows();
    const size_t C = A.cols();

    const T*        RESTRICT vals    = ASSUME_ALIGNED(A.data(),         HPDEXC_ALIGN_SIZE);
    const Idx*      RESTRICT indptr  = ASSUME_ALIGNED(A.indptr(),       HPDEXC_ALIGN_SIZE);
    const Idx*      RESTRICT indices = ASSUME_ALIGNED(A.indices(),      HPDEXC_ALIGN_SIZE);
    const int32_t*  RESTRICT gid     = ASSUME_ALIGNED(group_id.data(),  HPDEXC_ALIGN_SIZE);

    const size_t G = n_targets + 1;
    AlignedVector<size_t> gcount(G, 0);
    for (size_t r = 0; r < R; ++r) {
        if (r + HPDEXC_PREFETCH_DIST < R)
            PREFETCH_R(gid + r + HPDEXC_PREFETCH_DIST, 1);
        const int32_t g = gid[r];
        if (g >= 0 && (size_t)g < G) ++gcount[(size_t)g];
    }

    AlignedVector<size_t> off(G + 1, 0);
    for (size_t g = 1; g <= G; ++g) off[g] = off[g - 1] + gcount[g - 1];
    const size_t total_cap = off[G];

    auto U1_out = tensor::Ndarray<double>::zeros({C, n_targets});
    auto P_out  = tensor::Ndarray<double>::zeros({C, n_targets});
    double* RESTRICT U1_buf = U1_out.data();
    double* RESTRICT P_buf  = P_out.data();

    const double cc = opt.use_continuity ? 0.5 : 0.0; // 预先计算 continuity correction

    PARALLEL_REGION(threads)
    {
        const size_t tid = (size_t)THREAD_ID();

        // 线程持久化缓冲：避免频繁分配/清零，列内仅覆写“有效前缀”
        AlignedVector<T>        col_val(total_cap);
        AlignedVector<size_t>   gnnz(G);
        AlignedVector<size_t>   invalid_cnt(G);
        AlignedVector<size_t>   sparse_cnt(G);

        AlignedVector<double>   R1(G);
        AlignedVector<double>   tie_sum(G);
        AlignedVector<bool>     has_tie(G);
        AlignedVector<size_t>   grank(G);
        AlignedVector<size_t>   tar_ptrs(G);
        AlignedVector<size_t>   tar_eq(G);
        AlignedVector<size_t>   tie_cnt(G);
        AlignedVector<size_t>   n2_eff(G);
        AlignedVector<double>   U1_tmp(G);

        // 列循环
        PARALLEL_FOR(dynamic, {
            for (std::ptrdiff_t cc_i = 0; cc_i < (std::ptrdiff_t)C; ++cc_i) {
                const size_t c = (size_t)cc_i;

                // 仅小数组 O(G) 的覆盖写而非 fill(0)
                for (size_t g = 0; g < G; ++g) {
                    gnnz[g] = 0;
                    invalid_cnt[g] = 0;
                }

                const Idx p0 = indptr[c];
                const Idx p1 = indptr[c + 1];

                // 1) 扫列，按组收集显式值
                for (Idx p = p0; p < p1; ++p) {
                    if (p + HPDEXC_PREFETCH_DIST < p1) {
                        PREFETCH_R(indices + p + HPDEXC_PREFETCH_DIST, 1);
                        const Idx r_next = indices[p + HPDEXC_PREFETCH_DIST];
                        if ((size_t)r_next < R) PREFETCH_R(gid + (size_t)r_next, 1);
                    }

                    const Idx rIdx = indices[p];
                    if UNLIKELY(rIdx < 0 || (size_t)rIdx >= R) continue;

                    const size_t r = (size_t)rIdx;
                    const int32_t g = gid[r];
                    if UNLIKELY(g < 0 || (size_t)g >= G) continue;

                    const T v = vals[p];
                    if UNLIKELY(!is_valid_value(v)) {
                        ++invalid_cnt[(size_t)g];
                        continue;
                    }

                    const size_t gi  = (size_t)g;
                    const size_t dst = off[gi] + gnnz[gi]++;
                    col_val[dst] = v;
                }

                // 2) 组内排序（条件排序以避免不必要的开销）
                if (gnnz[0] > 1 && !opt.ref_sorted)
                    std::sort(col_val.data() + off[0], col_val.data() + off[0] + gnnz[0]);
                for (size_t g = 1; g < G; ++g) {
                    if (gnnz[g] > 1 && !opt.tar_sorted)
                        std::sort(col_val.data() + off[g], col_val.data() + off[g] + gnnz[g]);
                }

                // 3) 稀疏值计数
                if (use_sparse_value) {
                    for (size_t g = 0; g < G; ++g) {
                        const size_t tot = gcount[g];
                        const size_t bad = invalid_cnt[g];
                        sparse_cnt[g] = (tot > gnnz[g] + bad) ? (tot - gnnz[g] - bad) : 0;
                    }
                } else {
                    for (size_t g = 0; g < G; ++g) sparse_cnt[g] = 0;
                }

                // 4) 选择合并核：R1/tie_sum/has_tie
                for (size_t g = 0; g < G; ++g) {
                    R1[g] = 0.0;
                    tie_sum[g] = 0.0;
                    has_tie[g] = false;
                    grank[g] = 0;
                    tar_eq[g] = 0;
                    tie_cnt[g] = 0;
                    tar_ptrs[g] = off[g]; // memcpy 比 std::copy 更快：但这里可直接写
                }

                const T* RESTRICT refv  = col_val.data() + off[0];
                const size_t nref_exp   = gnnz[0];

                if (!use_sparse_value) {
                    if (!opt.tie_correction) {
                        sparse_core_without_tie<T>(
                            col_val.data(), off.data(), gnnz.data(), G,
                            refv, nref_exp,
                            tar_ptrs.data(), grank.data(),
                            tar_eq.data(), R1.data()
                        );
                    } else {
                        sparse_none_core<T>(
                            col_val.data(), off.data(), gnnz.data(), G,
                            refv, nref_exp,
                            tar_ptrs.data(), grank.data(),
                            tie_cnt.data(), tar_eq.data(),
                            R1.data(), tie_sum.data(), has_tie.data()
                        );
                    }
                } else {
                    const auto st = opt.sparse_type;
                    if ((opt.tie_correction || opt.method != MannWhitneyuOption<T>::Method::exact) &&
                        st == MannWhitneyuOption<T>::SparseValueMinmax::strict_minor)
                    {
                        for (size_t g = 1; g < G; ++g) {
                            const size_t ref_sp = sparse_cnt[0];
                            const size_t tar_sp = sparse_cnt[g];
                            const size_t sp_tie = ref_sp + tar_sp;
                            if (sp_tie > 1) {
                                const double tt = (double)sp_tie;
                                tie_sum[g] += (tt*tt*tt - tt);
                                has_tie[g]  = true;
                            }
                            grank[g] = sp_tie;
                            if (ref_sp > 0) {
                                const double avg = 0.5 * (1.0 + (double)sp_tie);
                                R1[g] = (double)ref_sp * avg;
                            }
                        }
                        sparse_none_core<T>(
                            col_val.data(), off.data(), gnnz.data(), G,
                            refv, nref_exp,
                            tar_ptrs.data(), grank.data(),
                            tie_cnt.data(), tar_eq.data(),
                            R1.data(), tie_sum.data(), has_tie.data()
                        );
                    }
                    else if ((opt.tie_correction || opt.method != MannWhitneyuOption<T>::Method::exact) &&
                             st == MannWhitneyuOption<T>::SparseValueMinmax::strict_major)
                    {
                        sparse_none_core<T>(
                            col_val.data(), off.data(), gnnz.data(), G,
                            refv, nref_exp,
                            tar_ptrs.data(), grank.data(),
                            tie_cnt.data(), tar_eq.data(),
                            R1.data(), tie_sum.data(), has_tie.data()
                        );
                        for (size_t g = 1; g < G; ++g) {
                            const size_t ref_sp = sparse_cnt[0];
                            const size_t tar_sp = sparse_cnt[g];
                            const size_t sp_tie = ref_sp + tar_sp;
                            if (sp_tie > 1) {
                                const double tt = (double)sp_tie;
                                tie_sum[g] += (tt*tt*tt - tt);
                                has_tie[g]  = true;
                            }
                            grank[g] += sp_tie;
                            if (ref_sp > 0) {
                                const size_t N = gnnz[0] + sparse_cnt[0] + gnnz[g] + sparse_cnt[g];
                                const size_t start = N - sp_tie + 1;
                                const size_t end   = N;
                                const double avg = 0.5 * ((double)start + (double)end);
                                R1[g] += (double)ref_sp * avg;
                            }
                        }
                    }
                    else {
                        AlignedVector<bool>   have_run(G, false);
                        AlignedVector<T>      run_val(G);
                        AlignedVector<size_t> run_len(G, 0);

                        size_t col_exp_total = 0;
                        for (size_t g = 0; g < G; ++g) col_exp_total += gnnz[g];
                        const bool very_sparse = (col_exp_total * 2 < total_cap);

                        if (very_sparse) {
                            sparse_medium_core_sparse<T>(
                                col_val.data(), off.data(), gnnz.data(), sparse_cnt.data(), G,
                                refv, nref_exp, opt.sparse_value,
                                tar_ptrs.data(), grank.data(), tar_eq.data(),
                                sparse_cnt.data(), // sp_left
                                have_run.data(), run_val.data(), run_len.data(),
                                R1.data(), tie_sum.data(), has_tie.data()
                            );
                        } else {
                            sparse_medium_core_dense<T>(
                                col_val.data(), off.data(), gnnz.data(), sparse_cnt.data(), G,
                                refv, nref_exp, opt.sparse_value,
                                tar_ptrs.data(), grank.data(), tar_eq.data(),
                                sparse_cnt.data(), // sp_left
                                have_run.data(), run_val.data(), run_len.data(),
                                R1.data(), tie_sum.data(), has_tie.data()
                            );
                        }
                    }
                }

                // 5) 有效样本量 + R1->U1
                size_t n1_eff = gnnz[0];
                if (use_sparse_value) {
                    const size_t spr =
                        (gcount[0] > gnnz[0] + invalid_cnt[0]) ? (gcount[0] - gnnz[0] - invalid_cnt[0]) : 0;
                    n1_eff += spr;
                    for (size_t g = 1; g < G; ++g) {
                        const size_t bad = invalid_cnt[g];
                        const size_t sp  = (gcount[g] > gnnz[g] + bad) ? (gcount[g] - gnnz[g] - bad) : 0;
                        n2_eff[g] = gnnz[g] + sp;
                    }
                } else {
                    for (size_t g = 1; g < G; ++g) n2_eff[g] = gnnz[g];
                }

                if UNLIKELY(n1_eff < 2) {
                    throw std::runtime_error("Sample too small for reference at column " + std::to_string(c));
                }
                for (size_t g = 1; g < G; ++g) {
                    if UNLIKELY(n2_eff[g] < 2) {
                        throw std::runtime_error("Sample too small for group " + std::to_string(g) +
                                                 " at column " + std::to_string(c));
                    }
                }

                const double base = (double)n1_eff * ((double)n1_eff + 1.0) * 0.5;
                for (size_t g = 1; g < G; ++g) {
                    U1_tmp[g] = R1[g] - base;
                }

                // 6) 写 U1（输出）
                for (size_t g = 1; g < G; ++g) {
                    U1_buf[c * n_targets + (g - 1)] = U1_tmp[g];
                }

                // 7) 计算 P —— 尽量将分支挪到循环外，热路径最少分支
                if UNLIKELY(opt.method == MannWhitneyuOption<T>::Method::exact) {
                    // exact 时如存在并列，按 SciPy 行为回退到渐近；先快速探测
                    bool any_tie = false;
                    if (!use_sparse_value) {
                        auto has_neighbor_equal = [&](size_t g)->bool {
                            const size_t n = gnnz[g];
                            const T* a = col_val.data() + off[g];
                            for (size_t i = 1; i < n; ++i) {
                                if (!(a[i-1] < a[i]) && !(a[i] < a[i-1])) return true;
                            }
                            return false;
                        };
                        any_tie = has_neighbor_equal(0);
                        for (size_t g = 1; g < G && !any_tie; ++g) any_tie |= has_neighbor_equal(g);
                    } else {
                        for (size_t g = 1; g < G; ++g) if UNLIKELY(has_tie[g]) { any_tie = true; break; }
                    }

                    // exact 分三种 alternative，拆三段循环避免分支
                    if LIKELY(opt.alternative == MannWhitneyuOption<T>::Alternative::two_sided) {
                        for (size_t g = 1; g < G; ++g) {
                            const double U1v = U1_tmp[g];
                            double p;
                            if UNLIKELY(any_tie || has_tie[g]) {
                                double mu, invsd;
                                p = p_asymptotic_two_sided_fast<T>(U1v, n1_eff, n2_eff[g], tie_sum[g], cc, mu, invsd);
                            } else {
                                const double U2 = (double)n1_eff * (double)n2_eff[g] - U1v;
                                const double pr = p_exact<T>(std::max(U1v, U2), n1_eff, n2_eff[g]);
                                p = 2.0 * pr;
                            }
                            if (p < 0.0) p = 0.0;
                            if (p > 1.0) p = 1.0;
                            P_buf[c * n_targets + (g - 1)] = p;
                        }
                    } else if UNLIKELY(opt.alternative == MannWhitneyuOption<T>::Alternative::less) {
                        for (size_t g = 1; g < G; ++g) {
                            const double U1v = U1_tmp[g];
                            double p;
                            if UNLIKELY(any_tie || has_tie[g]) {
                                double mu, invsd;
                                p = p_asymptotic_less_fast<T>(U1v, n1_eff, n2_eff[g], tie_sum[g], cc, mu, invsd);
                            } else {
                                const double U2 = (double)n1_eff * (double)n2_eff[g] - U1v;
                                p = p_exact<T>(U2, n1_eff, n2_eff[g]);
                            }
                            if (p < 0.0) p = 0.0;
                            if (p > 1.0) p = 1.0;
                            P_buf[c * n_targets + (g - 1)] = p;
                        }
                    } else { // greater
                        for (size_t g = 1; g < G; ++g) {
                            const double U1v = U1_tmp[g];
                            double p;
                            if UNLIKELY(any_tie || has_tie[g]) {
                                double mu, invsd;
                                p = p_asymptotic_greater_fast<T>(U1v, n1_eff, n2_eff[g], tie_sum[g], cc, mu, invsd);
                            } else {
                                p = p_exact<T>(U1v, n1_eff, n2_eff[g]);
                            }
                            if (p < 0.0) p = 0.0;
                            if (p > 1.0) p = 1.0;
                            P_buf[c * n_targets + (g - 1)] = p;
                        }
                    }
                }
                else if UNLIKELY(opt.method == MannWhitneyuOption<T>::Method::asymptotic) {
                    if LIKELY(opt.alternative == MannWhitneyuOption<T>::Alternative::two_sided) {
                        for (size_t g = 1; g < G; ++g) {
                            double mu, invsd;
                            double p = p_asymptotic_two_sided_fast<T>(U1_tmp[g], n1_eff, n2_eff[g], tie_sum[g], cc, mu, invsd);
                            if (p < 0.0) p = 0.0; if (p > 1.0) p = 1.0;
                            P_buf[c * n_targets + (g - 1)] = p;
                        }
                    } else if UNLIKELY(opt.alternative == MannWhitneyuOption<T>::Alternative::less) {
                        for (size_t g = 1; g < G; ++g) {
                            double mu, invsd;
                            double p = p_asymptotic_less_fast<T>(U1_tmp[g], n1_eff, n2_eff[g], tie_sum[g], cc, mu, invsd);
                            if (p < 0.0) p = 0.0; if (p > 1.0) p = 1.0;
                            P_buf[c * n_targets + (g - 1)] = p;
                        }
                    } else { // greater
                        for (size_t g = 1; g < G; ++g) {
                            double mu, invsd;
                            double p = p_asymptotic_greater_fast<T>(U1_tmp[g], n1_eff, n2_eff[g], tie_sum[g], cc, mu, invsd);
                            if (p < 0.0) p = 0.0; if (p > 1.0) p = 1.0;
                            P_buf[c * n_targets + (g - 1)] = p;
                        }
                    }
                }
                else { // automatic
                    // 先判断 asym/exact（一次分支），再按 alternative 拆循环
                    if LIKELY(opt.alternative == MannWhitneyuOption<T>::Alternative::two_sided) {
                        for (size_t g = 1; g < G; ++g) {
                            const bool asym =
                                (choose_method<T>(n1_eff, n2_eff[g], has_tie[g])
                                 == MannWhitneyuOption<T>::Method::asymptotic);
                            double p;
                            if LIKELY(asym) {
                                double mu, invsd;
                                p = p_asymptotic_two_sided_fast<T>(U1_tmp[g], n1_eff, n2_eff[g], tie_sum[g], cc, mu, invsd);
                            } else {
                                const double U1v = U1_tmp[g];
                                const double U2  = (double)n1_eff * (double)n2_eff[g] - U1v;
                                const double pr  = p_exact<T>(std::max(U1v, U2), n1_eff, n2_eff[g]);
                                p = 2.0 * pr;
                            }
                            if (p < 0.0) p = 0.0; if (p > 1.0) p = 1.0;
                            P_buf[c * n_targets + (g - 1)] = p;
                        }
                    } else if UNLIKELY(opt.alternative == MannWhitneyuOption<T>::Alternative::less) {
                        for (size_t g = 1; g < G; ++g) {
                            const bool asym =
                                (choose_method<T>(n1_eff, n2_eff[g], has_tie[g])
                                 == MannWhitneyuOption<T>::Method::asymptotic);
                            double p;
                            if LIKELY(asym) {
                                double mu, invsd;
                                p = p_asymptotic_less_fast<T>(U1_tmp[g], n1_eff, n2_eff[g], tie_sum[g], cc, mu, invsd);
                            } else {
                                const double U1v = U1_tmp[g];
                                const double U2  = (double)n1_eff * (double)n2_eff[g] - U1v;
                                p = p_exact<T>(U2, n1_eff, n2_eff[g]);
                            }
                            if (p < 0.0) p = 0.0; if (p > 1.0) p = 1.0;
                            P_buf[c * n_targets + (g - 1)] = p;
                        }
                    } else { // greater
                        for (size_t g = 1; g < G; ++g) {
                            const bool asym =
                                (choose_method<T>(n1_eff, n2_eff[g], has_tie[g])
                                 == MannWhitneyuOption<T>::Method::asymptotic);
                            double p;
                            if LIKELY(asym) {
                                double mu, invsd;
                                p = p_asymptotic_greater_fast<T>(U1_tmp[g], n1_eff, n2_eff[g], tie_sum[g], cc, mu, invsd);
                            } else {
                                p = p_exact<T>(U1_tmp[g], n1_eff, n2_eff[g]);
                            }
                            if (p < 0.0) p = 0.0; if (p > 1.0) p = 1.0;
                            P_buf[c * n_targets + (g - 1)] = p;
                        }
                    }
                }

                // 进度
                ++progress_safe[tid];
            }
        }) // PARALLEL_FOR
    } // PARALLEL_REGION

    return { U1_out, P_out };
}

// ========================= group_mean =========================

template<class T, class Idx>
tensor::Ndarray<double> group_mean(
    const tensor::Csc<T,Idx>& A,
    const tensor::Vector<int32_t>& group_id,
    size_t n_groups,
    int threads,
    size_t* RESTRICT progress_ptr
) {
    const size_t R = A.rows();
    const size_t C = A.cols();

    if (group_id.size() != R) {
        throw std::invalid_argument("[group_mean] group_id length mismatch with matrix rows");
    }

    auto means = tensor::Ndarray<double>::zeros({n_groups, C});
    double* mean_buf = means.data();

    AlignedVector<size_t> group_size(n_groups, 0);
    for (size_t r = 0; r < R; ++r) {
        int g = group_id.data()[r];
        if (g >= 0 && (size_t)g < n_groups) ++group_size[g];
    }

    size_t dummy = 0;
    size_t* RESTRICT prog = progress_ptr ? progress_ptr : &dummy;

    const T*   RESTRICT vals   = A.data();
    const Idx* RESTRICT indptr = A.indptr();
    const Idx* RESTRICT ridx   = A.indices();

    PARALLEL_REGION(threads)
    {
        AlignedVector<double> sum_local(n_groups);
        AlignedVector<size_t> count_local(n_groups);

        PARALLEL_FOR(dynamic, for (std::ptrdiff_t cc_i = 0; cc_i < (std::ptrdiff_t)C; ++cc_i) {
            const size_t c = (size_t)cc_i;

            // 覆盖写（而非 fill）
            for (size_t g = 0; g < n_groups; ++g) {
                sum_local[g] = 0.0;
                count_local[g] = 0;
            }

            const Idx p0 = indptr[c];
            const Idx p1 = indptr[c+1];
            for (Idx p = p0; p < p1; ++p) {
                const size_t r = (size_t)ridx[p];
                const int g = group_id.data()[r];
                if (g < 0 || (size_t)g >= n_groups) continue;
                sum_local[g]   += (double)vals[p];
                count_local[g] += 1;
            }

            for (size_t g = 0; g < n_groups; ++g) {
                const size_t valid = count_local[g];
                mean_buf[g * C + c] =
                    (valid == 0) ? std::numeric_limits<double>::quiet_NaN()
                                 : (sum_local[g] / (double)valid);
            }

            const size_t tid = (size_t)THREAD_ID();
            prog[tid] ++;
        })
    }
    return means;
}

// ========================= 显式实例化 =========================

#define INST_MANNWHITNEYU(T, Idx) \
    template MannWhitneyResult mannwhitneyu<T, Idx>( \
        const tensor::Csc<T, Idx>& A, \
        const MannWhitneyuOption<T>& opt, \
        const tensor::Vector<int32_t>& group_id, \
        size_t n_targets, \
        int threads, \
        size_t* RESTRICT progress_ptr);

#define DO_IDX(T) \
    INST_MANNWHITNEYU(T, int32_t) \
    INST_MANNWHITNEYU(T, int64_t)

DTYPE_DISPATCH(DO_IDX);
#undef DO_IDX
#undef INST_MANNWHITNEYU

#define INST_GROUP_MEAN(T, Idx) \
    template tensor::Ndarray<double> group_mean<T, Idx>( \
        const tensor::Csc<T, Idx>& A, \
        const tensor::Vector<int32_t>& group_id, \
        size_t n_groups, \
        int threads, \
        size_t* RESTRICT progress_ptr);

#define DO_IDX2(T) \
    INST_GROUP_MEAN(T, int32_t) \
    INST_GROUP_MEAN(T, int64_t)

DTYPE_DISPATCH(DO_IDX2);
#undef DO_IDX2
#undef INST_GROUP_MEAN

} // namespace hpdexc
