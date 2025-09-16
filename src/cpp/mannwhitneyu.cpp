// mannwhitneyu.cpp
// mwu cpu 算子实现
#include "mannwhitneyu.hpp"
#include "common.hpp"
#include "macro.hpp"
#include "simd.hpp"
#include "sparse.hpp"
#include <atomic>
#include <cmath>

namespace hpdex {

// -------- precise normal tail helpers (erfc-based) --------
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244008443621048490392848359376887
#endif

force_inline_ double norm_sf_precise(double z) {
    return 0.5 * std::erfc(z * M_SQRT1_2);
}

force_inline_ void compute_mu_sigma_precise(
    double n1, double n2, double tie_sum, double& mu, double& sigma
) {
    const double N  = n1 + n2;
    const double nm = n1 * n2;
    mu = 0.5 * nm;

    double tie_term = 0.0;
    const double denom = N * (N - 1.0);
    if (denom > 0.0) tie_term = tie_sum / denom;

    const double var = nm * ((N + 1.0) - tie_term) / 12.0;
    sigma = (var > 0.0) ? std::sqrt(var) : 0.0;
}

static inline void array_p_asymptotic_greater_precise(
    const double* U1, const double* n1, const double* n2,
    const double* tie_sum, const double* cc,
    double* out, size_t len
){
    for (size_t i = 0; i < len; ++i) {
        double mu, sigma;
        compute_mu_sigma_precise(n1[i], n2[i], tie_sum[i], mu, sigma);
        double p = 1.0;
        if (sigma > 0.0) {
            const double z = (U1[i] - mu - cc[i]) / sigma;
            p = norm_sf_precise(z);
        }
        out[i] = (p < 0.0 ? 0.0 : (p > 1.0 ? 1.0 : p));
    }
}

static inline void array_p_asymptotic_less_precise(
    const double* U1, const double* n1, const double* n2,
    const double* tie_sum, const double* cc,
    double* out, size_t len
){
    for (size_t i = 0; i < len; ++i) {
        const double nm = n1[i] * n2[i];
        const double U2 = nm - U1[i];
        double mu, sigma;
        compute_mu_sigma_precise(n1[i], n2[i], tie_sum[i], mu, sigma);
        double p = 1.0;
        if (sigma > 0.0) {
            const double z = (U2 - mu - cc[i]) / sigma;
            p = norm_sf_precise(z);
        }
        out[i] = (p < 0.0 ? 0.0 : (p > 1.0 ? 1.0 : p));
    }
}

static inline void array_p_asymptotic_two_sided_precise(
    const double* U1, const double* n1, const double* n2,
    const double* tie_sum, const double* cc,
    double* out, size_t len
){
    for (size_t i = 0; i < len; ++i) {
        const double nm  = n1[i] * n2[i];
        const double U2  = nm - U1[i];
        const double U   = (U1[i] > U2 ? U1[i] : U2);
        double mu, sigma;
        compute_mu_sigma_precise(n1[i], n2[i], tie_sum[i], mu, sigma);

        double p = 1.0;
        if (sigma > 0.0) {
            const double z = (U - mu - cc[i]) / sigma;
            p = 2.0 * norm_sf_precise(z);
            if (p > 1.0) p = 1.0;
            if (p < 0.0) p = 0.0;
        }
        out[i] = p;
    }
}

force_inline_ double p_exact(double U, size_t n1, size_t n2) {
    using U64 = unsigned long long;

    const size_t Umax = n1 * n2;
    if unlikely_(Umax == 0) return 1.0;

    const double U_clip = std::max(0.0, std::min(static_cast<double>(Umax), U));
    const size_t u_stat = static_cast<size_t>(std::floor(U_clip));

    const size_t SZ = Umax + 1;
    const size_t STACK_LIMIT = 1 << 12; // 4096 元素以内走栈

    U64* dp     = nullptr;
    U64* ndp    = nullptr;
    bool on_heap = false;

    alignas(64) U64 dp_stack[STACK_LIMIT];
    alignas(64) U64 ndp_stack[STACK_LIMIT];

    if (SZ <= STACK_LIMIT) {
        dp  = dp_stack;
        ndp = ndp_stack;
    } else {
        dp  = static_cast<U64*>(::operator new[](SZ * sizeof(U64), std::align_val_t(64)));
        ndp = static_cast<U64*>(::operator new[](SZ * sizeof(U64), std::align_val_t(64)));
        on_heap = true;
    }

    // 初始化
    dp[0] = 1ULL;
    for (size_t i = 1; i < SZ; ++i) dp[i] = 0ULL;
    for (size_t i = 0; i < SZ; ++i) ndp[i] = 0ULL;

    U64* dp_cur = dp;
    U64* dp_nxt = ndp;

    for (size_t i = 1; i <= n1; ++i) {
        const size_t prev_up = (i - 1) * n2;
        const size_t up      = i * n2;

        U64 win = 0ULL;

        size_t u = 0;
        const size_t bound1 = prev_up < up ? prev_up : up;
        for (; u <= bound1; ++u) {
            win += dp_cur[u];
            if (u >= n2 + 1) win -= dp_cur[u - (n2 + 1)];
            dp_nxt[u] = win;
        }

        for (; u <= up; ++u) {
            if (u >= n2 + 1) win -= dp_cur[u - (n2 + 1)];
            dp_nxt[u] = win;
        }

        for (size_t j = 0; j <= up; ++j) dp_cur[j] = 0ULL;

        U64* tmp = dp_cur;
        dp_cur = dp_nxt;
        dp_nxt = tmp;
    }

    double total = 0.0;
    for (size_t u = 0; u <= Umax; ++u) {
        total += static_cast<double>(dp_cur[u]);
    }

    const size_t kc    = Umax - u_stat;
    const size_t small = (u_stat < kc ? u_stat : kc);

    double cdf_small = 0.0;
    for (size_t u = 0; u <= small; ++u) {
        cdf_small += static_cast<double>(dp_cur[u]);
    }
    const double pmf_small = static_cast<double>(dp_cur[small]);

    double sf_ge;
    if likely_(u_stat <= kc) {
        sf_ge = 1.0 - cdf_small / total + pmf_small / total;
    } else {
        sf_ge = cdf_small / total;
    }

    // 释放堆内存
    if (on_heap) {
        ::operator delete[](dp,  std::align_val_t(64));
        ::operator delete[](ndp, std::align_val_t(64));
    }

    return sf_ge;
}


force_inline_ void p_asymptotic_parallel(
    const double* U1, const double* n1, const double* n2,
    const double* tie_sum, const double* cc,
    double* out, size_t N,
    MannWhitneyuOption::Alternative alt,
    bool fast_norm,
    int threads
) {
    const int maxth = omp_get_max_threads();
    if (threads < 0) threads = maxth;
    if (threads > maxth) threads = maxth;

    using AsympFn = void(*)(const double*, const double*, const double*, const double*, const double*, double*, size_t);

    // fast 路径（原 SIMD 实现）
    AsympFn fn_fast = nullptr;
    switch (alt) {
    case MannWhitneyuOption::Alternative::two_sided: fn_fast = array_p_asymptotic_two_sided; break;
    case MannWhitneyuOption::Alternative::greater:   fn_fast = array_p_asymptotic_greater;   break;
    case MannWhitneyuOption::Alternative::less:      fn_fast = array_p_asymptotic_less;      break;
    }

    // precise 路径（erfc-based）
    AsympFn fn_precise = nullptr;
    switch (alt) {
    case MannWhitneyuOption::Alternative::two_sided: fn_precise = array_p_asymptotic_two_sided_precise; break;
    case MannWhitneyuOption::Alternative::greater:   fn_precise = array_p_asymptotic_greater_precise;   break;
    case MannWhitneyuOption::Alternative::less:      fn_precise = array_p_asymptotic_less_precise;      break;
    }

    AsympFn fn = fast_norm ? fn_fast : fn_precise;

    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        const int nth = omp_get_num_threads();

        const size_t chunk = (N + nth - 1) / nth;
        const size_t begin = tid * chunk;
        const size_t end   = std::min(N, begin + chunk);
        
        if (begin < end) {
            fn(U1 + begin, n1 + begin, n2 + begin, tie_sum + begin, cc + begin,
               out + begin, end - begin);
        }
    }
}


force_inline_ void p_exact_parallel(
    const double* U1,          // len = N
    const double* n1,          // len = N
    const double* n2,          // len = N
    double*       out,         // len = N
    size_t        N,
    MannWhitneyuOption::Alternative alt,
    int           threads
) {
    int maxth = omp_get_max_threads();
    if (threads < 0) threads = maxth;
    if (threads > maxth) threads = maxth;

    #pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)N; ++i) {
        const size_t a = static_cast<size_t>(n1[i]);
        const size_t b = static_cast<size_t>(n2[i]);

        const double U1d   = U1[i];
        const double Nprod = static_cast<double>(a) * static_cast<double>(b);

        double p = 1.0;
        switch (alt) {
            case MannWhitneyuOption::two_sided: {
                const double U2   = Nprod - U1d;
                const double Umax = (U1d > U2 ? U1d : U2);
                p = 2.0 * p_exact(Umax, a, b);
            } break;
            case MannWhitneyuOption::greater:
                p = p_exact(U1d, a, b);
                break;
            case MannWhitneyuOption::less: {
                const double U2 = Nprod - U1d;
                p = p_exact(U2, a, b);
            } break;
        }
        if (p < 0.0) p = 0.0;
        if (p > 1.0) p = 1.0;
        out[i] = p;
    }
}

// ========================= Ranking cores =========================

// === 新增：零值位置语义 ===
enum class ZeroDomain : uint8_t { Pos /*0在头*/, Neg /*0在尾*/ };
force_inline_ bool zero_at_head(ZeroDomain d) { return d == ZeroDomain::Pos; }

// ---------------------------------------------------------------------
// 1) 零值极端位置（pos/neg；替代原 sparse_strict_min_max_core）
// ---------------------------------------------------------------------
// 将 ref/tar 的“隐式零块（稀疏0）”整体放到序列头或尾；
// 支持与显式零并列（只要把显式零也计入 ref_sp_cnt/tar_sp_cnt 即可）。
force_inline_
void sparse_zero_extreme_core( // 原 sparse_strict_min_max_core 的替代
    ZeroDomain  domain,        // Pos: 0在头；Neg: 0在尾
    size_t      ref_zero_cnt,  // 参考组隐式(稀疏+显式)零的总个数
    size_t      tar_zero_cnt,  // 目标组隐式(稀疏+显式)零的总个数
    size_t      N,             // 本列总体元素个数（含零）
    double&     R1g,           // 累加：参考组rank和
    double&     tie_sum_g,     // 累加：tie 校正 ∑(t^3 - t)
    bool&       has_tie_g,     // 标记：是否存在tie
    size_t&     grank_g        // 进度：已消耗的全局秩计数
){
    const size_t z_tie = ref_zero_cnt + tar_zero_cnt;  // 该零块的并列长度
    if (unlikely_(z_tie == 0)) return;

    const double tt   = static_cast<double>(z_tie);
    tie_sum_g        += (tt*tt*tt - tt);
    has_tie_g         = true;

    const bool head   = zero_at_head(domain);
    const double m    = head ? 1.0 : 0.0;
    const double nm   = 1.0 - m;
    const double ldN  = static_cast<double>(N);
    const double ldS  = static_cast<double>(z_tie);

    // 放在头部 -> rank区间 [1, z_tie]
    // 放在尾部 -> rank区间 [N - z_tie + 1, N]
    const double start = m*1.0 + nm*(ldN - ldS + 1.0);
    const double end   = m*ldS + nm*(ldN);
    const double avg   = 0.5 * (start + end);

    R1g     += static_cast<double>(ref_zero_cnt) * avg;
    grank_g += z_tie;
}

// ---------------------------------------------------------------------
// 2) 零值位于中间（mixed）归并核心：恒以 0 作为稀疏值；可开关 use_zero
//    ——保留 Dense / Sparse 的两套 TarMerge（分支倾向不同）
// ---------------------------------------------------------------------
struct TarMergeDense {
    template<class T>
    force_inline_
    void operator()(
        const T& val, size_t count, bool& have_run, T& run_val, size_t& run_len,
        double& tie_sum_g, bool& has_tie_g, size_t& grank_g
    ) const {
        if (unlikely_(!count)) return;
        if (likely_(have_run && !(run_val < val) && !(val < run_val))) {
            run_len += count;
        } else {
            if (unlikely_(run_len > 1)) {
                const double tt = static_cast<double>(run_len);
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
    force_inline_
    void operator()(
        const T& val, size_t count, bool& have_run, T& run_val, size_t& run_len,
        double& tie_sum_g, bool& has_tie_g, size_t& grank_g
    ) const {
        if (unlikely_(!count)) return;
        if (unlikely_(have_run && !(run_val < val) && !(val < run_val))) {
            run_len += count;
        } else {
            if (unlikely_(run_len > 1)) {
                const double tt = static_cast<double>(run_len);
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

// --- 中位（0 可能落在“中间”）实现：恒用 zero=0，对稀疏零通过 use_zero 控制参与 ---
template<class T, class TarMerge>
force_inline_
void sparse_zero_medium_core_impl(
    const T*      col_val,            // 排序好的显式值（各组拼接）
    const size_t* off,                // 各组起点
    const size_t* gnnz,               // 各组显式非零个数
    const size_t* /*sparse_value_cnt*/,// 兼容旧签名：不使用
    const size_t  G,
    const T*      refv,               // 参考组显式值（升序）
    const size_t  nref_exp,
    /*const T sparse_value*/           // 固定为 0
    size_t*       tar_ptrs_local,
    size_t*       grank,
    size_t*       tar_eq,
    size_t*       sp_left,            // 各组“隐式零”剩余个数（含ref放在 sp_left[0]）
    bool*         have_run,
    T*            run_val,
    size_t*       run_len,
    double*       R1,
    double*       tie_sum,
    bool*         has_tie,
    const TarMerge& merge,
    const bool    use_zero            // 新增：是否让“隐式 0”参与归并
){
    const T zero = T(0);

    auto flush_run = [&](size_t g){
        if (unlikely_(run_len[g] > 1)) {
            const double tt = static_cast<double>(run_len[g]);
            tie_sum[g] += (tt*tt*tt - tt);
            has_tie[g]  = true;
        }
        run_len[g]  = 0;
        have_run[g] = false;
    };

    size_t i = 0;
    while (i < nref_exp || (use_zero && sp_left[0] > 0)) {
        T      vref;
        size_t ref_tie = 0;

        // 参考侧选择下一批（显式或隐式0），并把显式0与隐式0做并列归并
        const size_t ref_sp_now = use_zero ? sp_left[0] : 0;
        if (ref_sp_now > 0 && (i >= nref_exp || !(refv[i] < zero))) {
            // 以 0 为参考值，整合显式0的并列
            vref = zero;
            size_t k_exp = 0;
            while (i + k_exp < nref_exp && !(refv[i + k_exp] < vref) && !(vref < refv[i + k_exp])) {
                ++k_exp; // 累显式0
            }
            ref_tie    = ref_sp_now + k_exp;
            sp_left[0] = 0;  // 隐式0吃掉
            i         += k_exp;
        } else {
            // 以显式值为参考（可能是负/正/零）
            vref = (i < nref_exp) ? refv[i] : zero; // 不会落到这里的 zero 分支，留作安全
            const size_t ref_start = i;
            while ((i + 1) < nref_exp && !(refv[i + 1] < vref) && !(vref < refv[i + 1])) ++i;
            ref_tie = i - ref_start + 1;
            ++i;
        }

        // 目标侧：把 <vref 的元素（含隐式0）尽可能批量吐出，并行累计 tie
        for (size_t g = 1; g < G; ++g) {
            size_t& tp        = tar_ptrs_local[g];
            const size_t gend = off[g] + gnnz[g];

            while (true) {
                const bool has_exp = (tp < gend) && (col_val[tp] < vref);
                const bool has_sp  = use_zero && (sp_left[g] > 0) && (zero < vref);
                if (!(has_exp || has_sp)) break;

                if (has_exp && has_sp) {
                    const T ev = col_val[tp];
                    if (ev < zero) {
                        // 批量输出显式值 run
                        size_t j = tp + 1;
                        while (j < gend && !(col_val[j] < ev) && !(ev < col_val[j])) ++j;
                        const size_t blk = j - tp;
                        merge(ev, blk, have_run[g], run_val[g], run_len[g],
                              tie_sum[g], has_tie[g], grank[g]);
                        tp = j;
                    } else if (zero < ev) {
                        // 输出隐式0
                        const size_t blk = sp_left[g];
                        merge(zero, blk, have_run[g], run_val[g], run_len[g],
                              tie_sum[g], has_tie[g], grank[g]);
                        sp_left[g] = 0;
                    } else {
                        // ev == 0，与隐式0并列
                        size_t j = tp + 1;
                        while (j < gend && !(col_val[j] < zero) && !(zero < col_val[j])) ++j;
                        const size_t blk_exp = j - tp;
                        merge(zero, blk_exp, have_run[g], run_val[g], run_len[g],
                              tie_sum[g], has_tie[g], grank[g]);
                        tp = j;
                        if (sp_left[g] > 0) {
                            const size_t blk_sp = sp_left[g];
                            merge(zero, blk_sp, have_run[g], run_val[g], run_len[g],
                                  tie_sum[g], has_tie[g], grank[g]);
                            sp_left[g] = 0;
                        }
                    }
                } else if (has_exp) {
                    const T ev = col_val[tp];
                    size_t j = tp + 1;
                    while (j < gend && !(col_val[j] < ev) && !(ev < col_val[j])) ++j;
                    const size_t blk = j - tp;
                    merge(ev, blk, have_run[g], run_val[g], run_len[g],
                          tie_sum[g], has_tie[g], grank[g]);
                    tp = j;
                } else { // 仅隐式0
                    const size_t blk = sp_left[g];
                    merge(zero, blk, have_run[g], run_val[g], run_len[g],
                          tie_sum[g], has_tie[g], grank[g]);
                    sp_left[g] = 0;
                }
            }
            // 刷掉 <vref 的run
            flush_run(g);

            // 处理 == vref 的显式并列；若 vref==0 且 use_zero，则再并入隐式0
            size_t eq = 0;
            while (tp < gend && !(col_val[tp] < vref) && !(vref < col_val[tp])) { ++tp; ++eq; }
            if (use_zero && sp_left[g] > 0 && !(zero < vref) && !(vref < zero)) {
                eq += sp_left[g];
                sp_left[g] = 0;
            }
            tar_eq[g] = eq;
        }

        // 根据 ref_tie 与 tar_eq[g] 累加秩与 tie 修正
        for (size_t g = 1; g < G; ++g) {
            const double rrcur    = static_cast<double>(grank[g]);
            const size_t t        = ref_tie + tar_eq[g];
            const double rrnext   = rrcur + static_cast<double>(t);
            const double avg_rank = 0.5 * (rrcur + rrnext + 1.0);

            R1[g]   += static_cast<double>(ref_tie) * avg_rank;
            grank[g] = static_cast<size_t>(rrnext);

            if (unlikely_(t > 1)) {
                const double tt = static_cast<double>(t);
                tie_sum[g] += (tt*tt*tt - tt);
                has_tie[g]  = true;
            }
            tar_eq[g] = 0;
        }
    }

    // 余下目标侧尾处理：把剩余显式（以及可选的隐式0）按块吐出、累计 tie
    for (size_t g = 1; g < G; ++g) {
        size_t& tp   = tar_ptrs_local[g];
        const size_t gend = off[g] + gnnz[g];

        have_run[g] = false;
        run_len[g]  = 0;

        while (tp < gend || (use_zero && sp_left[g] > 0)) {
            const bool has_exp = (tp < gend);
            const bool has_sp  = use_zero && (sp_left[g] > 0);

            T cand; bool take_sp = false;
            if (!has_exp) { cand = zero; take_sp = true; }
            else if (!has_sp) { cand = col_val[tp]; }
            else {
                const T ev = col_val[tp];
                if (zero < ev) { cand = zero; take_sp = true; }
                else            { cand = ev; }
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
        if (unlikely_(run_len[g] > 1)) {
            const double tt = static_cast<double>(run_len[g]);
            tie_sum[g] += (tt*tt*tt - tt);
            has_tie[g]  = true;
        }
        run_len[g]  = 0;
        have_run[g] = false;
    }
}

// 包装：密/稀两套分支倾向不变；新增 use_zero 入参
template<class T>
force_inline_
void sparse_zero_medium_core_dense(
    const T* col_val, const size_t* off, const size_t* gnnz,
    const size_t* sparse_value_cnt, const size_t G,
    const T* refv, const size_t nref_exp,
    size_t* tar_ptrs_local, size_t* grank, size_t* tar_eq, size_t* sp_left,
    bool* have_run, T* run_val, size_t* run_len,
    double* R1, double* tie_sum, bool* has_tie,
    bool use_zero
){
    TarMergeDense merger{};
    sparse_zero_medium_core_impl<T>(col_val, off, gnnz, sparse_value_cnt, G, refv, nref_exp,
        tar_ptrs_local, grank, tar_eq, sp_left,
        have_run, run_val, run_len, R1, tie_sum, has_tie, merger, use_zero);
}

template<class T>
force_inline_
void sparse_zero_medium_core_sparse(
    const T* col_val, const size_t* off, const size_t* gnnz,
    const size_t* sparse_value_cnt, const size_t G,
    const T* refv, const size_t nref_exp,
    size_t* tar_ptrs_local, size_t* grank, size_t* tar_eq, size_t* sp_left,
    bool* have_run, T* run_val, size_t* run_len,
    double* R1, double* tie_sum, bool* has_tie,
    bool use_zero
){
    TarMergeSparse merger{};
    sparse_zero_medium_core_impl<T>(col_val, off, gnnz, sparse_value_cnt, G, refv, nref_exp,
        tar_ptrs_local, grank, tar_eq, sp_left,
        have_run, run_val, run_len, R1, tie_sum, has_tie, merger, use_zero);
}

// ===== 小工具 =====
template<class T>
static inline bool is_valid_value(T x) {
    if constexpr (std::is_floating_point_v<T>) return std::isfinite(x) && !std::isnan(x);
    else return true;
}

// 辅助：按段归并（不含 0 的纯负或纯正段），返回更新后的 rank（1-based）
static inline size_t merge_and_rank_segment(
    const double* a, size_t na,
    const double* b, size_t nb,
    size_t rank,                // 进入该段前的当前秩（1-based）
    double& R1, double& tie_sum, bool& has_tie
) {
    size_t i = 0, j = 0;
    while (i < na || j < nb) {
        double v; bool take_a = false, take_b = false;

        if (i < na && j < nb) {
            if (a[i] < b[j]) { v = a[i]; take_a = true; }
            else if (b[j] < a[i]) { v = b[j]; take_b = true; }
            else { v = a[i]; take_a = take_b = true; }
        } else if (i < na) {
            v = a[i]; take_a = true;
        } else {
            v = b[j]; take_b = true;
        }

        size_t eq_a = 0, eq_b = 0;
        if (take_a) { while (i + eq_a < na && !(a[i + eq_a] < v) && !(v < a[i + eq_a])) ++eq_a; }
        if (take_b) { while (j + eq_b < nb && !(b[j + eq_b] < v) && !(v < b[j + eq_b])) ++eq_b; }

        const size_t t = eq_a + eq_b;
        const double start = static_cast<double>(rank);
        const double end   = static_cast<double>(rank + t - 1);
        const double avg   = 0.5 * (start + end);

        R1 += static_cast<double>(eq_a) * avg;

        if (t > 1) {
            const double tt = static_cast<double>(t);
            tie_sum += (tt * tt * tt - tt);
            has_tie = true;
        }

        rank += t;
        i += eq_a; j += eq_b;
    }
    return rank;
}

// 显式 + 隐式 0 全量归并（中位语义）：负段 -> 零段(显式0+隐式0) -> 正段
static inline void merge_rank_sum_with_tie_include_zeros(
    const double* a, size_t n1_exp,         // 参考组显式值（升序）
    const double* b, size_t n2_exp,         // 目标组显式值（升序）
    size_t a_zero_implicit,                 // 参考组该列隐式(缺省)0 的数量
    size_t b_zero_implicit,                 // 目标组该列隐式(缺省)0 的数量
    double& R1, double& tie_sum, bool& has_tie
) {
    R1 = 0.0; tie_sum = 0.0; has_tie = false;

    // 划分 a 的负/零/正三段
    size_t ai_neg_end = 0; while (ai_neg_end < n1_exp && a[ai_neg_end] < 0.0) ++ai_neg_end;
    size_t ai_zero_end = ai_neg_end;
    while (ai_zero_end < n1_exp && !(0.0 < a[ai_zero_end]) && !(a[ai_zero_end] < 0.0)) ++ai_zero_end;

    const size_t a_neg_n   = ai_neg_end;
    const size_t a_zero_exp= ai_zero_end - ai_neg_end;
    const size_t a_pos_n   = n1_exp - ai_zero_end;

    // 划分 b 的负/零/正三段
    size_t bi_neg_end = 0; while (bi_neg_end < n2_exp && b[bi_neg_end] < 0.0) ++bi_neg_end;
    size_t bi_zero_end = bi_neg_end;
    while (bi_zero_end < n2_exp && !(0.0 < b[bi_zero_end]) && !(b[bi_zero_end] < 0.0)) ++bi_zero_end;

    const size_t b_neg_n   = bi_neg_end;
    const size_t b_zero_exp= bi_zero_end - bi_neg_end;
    const size_t b_pos_n   = n2_exp - bi_zero_end;

    // 总零个数（显式0 + 隐式0）
    const size_t A_zero_total = a_zero_exp + a_zero_implicit;
    const size_t B_zero_total = b_zero_exp + b_zero_implicit;

    size_t rank = 1; // 全局 1-based 秩

    // 负值段归并
    rank = merge_and_rank_segment(
        a, a_neg_n,
        b, b_neg_n,
        rank, R1, tie_sum, has_tie
    );

    // 零值段（若存在）
    const size_t t_zero = A_zero_total + B_zero_total;
    if (t_zero > 0) {
        const double start = static_cast<double>(rank);
        const double end   = static_cast<double>(rank + t_zero - 1);
        const double avg   = 0.5 * (start + end);

        R1 += static_cast<double>(A_zero_total) * avg;

        if (t_zero > 1) {
            const double tz = static_cast<double>(t_zero);
            tie_sum += (tz * tz * tz - tz);
            has_tie = true;
        }
        rank += t_zero;
    }

    // 正值段归并
    rank = merge_and_rank_segment(
        a + ai_zero_end, a_pos_n,
        b + bi_zero_end, b_pos_n,
        rank, R1, tie_sum, has_tie
    );
}

// 极端语义：0 在头/尾（全局最小或全局最大）
static inline void merge_rank_sum_with_tie_zero_extreme(
    const double* a, size_t n1_exp,         // 参考组显式值（已升序）
    const double* b, size_t n2_exp,         // 目标组显式值（已升序）
    size_t a_zero_implicit,                 // 参考组该列隐式(缺省)0 的数量
    size_t b_zero_implicit,                 // 目标组该列隐式(缺省)0 的数量
    bool zero_at_head,                      // true=0在头(min), false=0在尾(max)
    double& R1, double& tie_sum, bool& has_tie
) {
    R1 = 0.0; tie_sum = 0.0; has_tie = false;

    // a 的负/零/正段
    size_t ai_neg_end = 0; while (ai_neg_end < n1_exp && a[ai_neg_end] < 0.0) ++ai_neg_end;
    size_t ai_zero_end = ai_neg_end;
    while (ai_zero_end < n1_exp && !(0.0 < a[ai_zero_end]) && !(a[ai_zero_end] < 0.0)) ++ai_zero_end;

    const size_t a_neg_n    = ai_neg_end;
    const size_t a_zero_exp = ai_zero_end - ai_neg_end;
    const size_t a_pos_n    = n1_exp - ai_zero_end;

    // b 的负/零/正段
    size_t bi_neg_end = 0; while (bi_neg_end < n2_exp && b[bi_neg_end] < 0.0) ++bi_neg_end;
    size_t bi_zero_end = bi_neg_end;
    while (bi_zero_end < n2_exp && !(0.0 < b[bi_zero_end]) && !(b[bi_zero_end] < 0.0)) ++bi_zero_end;

    const size_t b_neg_n    = bi_neg_end;
    const size_t b_zero_exp = bi_zero_end - bi_neg_end;
    const size_t b_pos_n    = n2_exp - bi_zero_end;

    const size_t A_zero_total = a_zero_exp + a_zero_implicit;
    const size_t B_zero_total = b_zero_exp + b_zero_implicit;
    const size_t t_zero = A_zero_total + B_zero_total;

    size_t rank = 1; // 全局 1-based 秩

    auto emit_zero_block = [&](size_t& rk){
        if (t_zero > 0) {
            const double start = static_cast<double>(rk);
            const double end   = static_cast<double>(rk + t_zero - 1);
            const double avg   = 0.5 * (start + end);
            R1 += static_cast<double>(A_zero_total) * avg;
            if (t_zero > 1) {
                const double tz = static_cast<double>(t_zero);
                tie_sum += (tz * tz * tz - tz);
                has_tie = true;
            }
            rk += t_zero;
        }
    };

    if (zero_at_head) emit_zero_block(rank);

    // 非零部分：先负值段，再正值段（负<正，无跨段并列）
    rank = merge_and_rank_segment(a,               a_neg_n,
                                  b,               b_neg_n,
                                  rank, R1, tie_sum, has_tie);
    rank = merge_and_rank_segment(a + ai_zero_end, a_pos_n,
                                  b + bi_zero_end, b_pos_n,
                                  rank, R1, tie_sum, has_tie);

    if (!zero_at_head) emit_zero_block(rank);
}

// 合并两有序数组，返回参考组秩和 R1 与并列修正量 tie_sum，has_tie 标志
static inline void merge_rank_sum_with_tie(
    const double* a, size_t n1,
    const double* b, size_t n2,
    double& R1, double& tie_sum, bool& has_tie
) {
    size_t i = 0, j = 0, rank = 1;   // 1-based
    R1 = 0.0; tie_sum = 0.0; has_tie = false;

    while (i < n1 || j < n2) {
        double v; bool take_a = false, take_b = false;

        // 选出下一个最小值（可能相等并列）
        if (i < n1 && j < n2) {
            if (a[i] < b[j]) { v = a[i]; take_a = true; }
            else if (b[j] < a[i]) { v = b[j]; take_b = true; }
            else { v = a[i]; take_a = take_b = true; }
        } else if (i < n1) {
            v = a[i]; take_a = true;
        } else {
            v = b[j]; take_b = true;
        }

        // 找并列块大小
        size_t eq_a = 0, eq_b = 0;
        if (take_a) {
            while (i + eq_a < n1 && !(a[i + eq_a] < v) && !(v < a[i + eq_a])) ++eq_a;
        }
        if (take_b) {
            while (j + eq_b < n2 && !(b[j + eq_b] < v) && !(v < b[j + eq_b])) ++eq_b;
        }

        const size_t t = eq_a + eq_b;
        const double start = static_cast<double>(rank);
        const double end   = static_cast<double>(rank + t - 1);
        const double avg   = 0.5 * (start + end);

        // R1 只加上参考组（a）的秩和
        R1 += static_cast<double>(eq_a) * avg;

        // 并列修正
        if (t > 1) {
            const double tt = static_cast<double>(t);
            tie_sum += (tt * tt * tt - tt);
            has_tie = true;
        }

        rank += t;
        i += eq_a; j += eq_b;
    }
}

// =============== 主核：并行列 -> 收集/排序 -> 计算 U1/tie/n1/n2，退出后并行算 p ===============
template<class T>
inline std::pair<std::vector<double>, std::vector<double>>
mannWhitneyu_core(
    const T* data, const int64_t* indices, const int64_t* indptr,
    const size_t& R, const size_t& C, const size_t& nnz,
    const int32_t* group_id, size_t n_targets,
    const MannWhitneyuOption& opt, int threads
){
    if (n_targets == 0) return { {}, {} };

    const size_t G = n_targets + 1;
    const size_t Npairs = C * n_targets;

    std::vector<double> U1_out(Npairs, 0.0);
    std::vector<double> P_out (Npairs, 1.0);
    std::vector<double> n1_arr(Npairs, 0.0);
    std::vector<double> n2_arr(Npairs, 0.0);
    std::vector<double> tie_arr(Npairs, 0.0);
    std::vector<double> cc_arr (Npairs, opt.use_continuity ? 0.5 : 0.0);

    // 组行数（用于 sparse 的总样本量）
    std::vector<size_t> group_rows(G, 0);
    for (size_t r = 0; r < R; ++r) {
        int g = group_id[r];
        if (g >= 0 && size_t(g) < G) ++group_rows[size_t(g)];
    }

    if (threads < 0) threads = omp_get_max_threads();
    if (threads > omp_get_max_threads()) threads = omp_get_max_threads();

    // 错误标记
    std::atomic<bool> has_error{false};
    std::string error_message;

    // ---- 排序派发（把"要不要排"提前成函数指针；列内只调用，不再判分支）
    using SortFn = void(*)(double*, size_t);
    auto sort_impl = +[](double* ptr, size_t n){
        if (n > 1) hwy::HWY_NAMESPACE::VQSortStatic(ptr, n, hwy::SortAscending());
    };
    auto sort_noop = +[](double*, size_t){};
    SortFn sort_ref = opt.ref_sorted ? sort_noop : sort_impl;
    SortFn sort_tar = opt.tar_sorted ? sort_noop : sort_impl;

    // ---- tie 校正派发（避免每列 if）
    const double tie_mask = opt.tie_correction ? 1.0 : 0.0;

    // ---- 一个小工具：并行跑指定列处理体
    auto run_cols = [&](auto&& body) {
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (std::ptrdiff_t cc = 0; cc < (std::ptrdiff_t)C; ++cc) {
            const size_t c = (size_t)cc;
            body(c);
        }
    };

    // ---- 共享的"收集+排序"封装：每列建 gvals，然后按派发的 sort_ref/sort_tar 排序
    auto build_and_sort_gvals = [&](size_t c, std::vector<std::vector<double>>& gvals){
        gvals.assign(G, {});
        for (size_t g = 0; g < G; ++g) gvals[g].reserve(group_rows[g]);

        const int64_t p0 = indptr[c], p1 = indptr[c + 1];
        for (int64_t p = p0; p < p1; ++p) {
            const int64_t r = indices[p];
            if (r < 0 || size_t(r) >= R) continue;
            const int g = group_id[size_t(r)];
            if (g < 0 || size_t(g) >= (int)G) continue;
            const double v = static_cast<double>(data[p]);
            if (!is_valid_value(v)) continue;
            gvals[size_t(g)].push_back(v);
        }
        // 排序
        sort_ref(gvals[0].data(), gvals[0].size());
        for (size_t g = 1; g < G; ++g)
            sort_tar(gvals[g].data(), gvals[g].size());
    };

    // =========================
    // zero_handling 派发（分支外移），每个分支**内部**并行
    // =========================
    switch (opt.zero_handling) {
    case MannWhitneyuOption::none: {
        run_cols([&](size_t c){
            std::vector<std::vector<double>> gvals;
            build_and_sort_gvals(c, gvals);

            const size_t n1_total = gvals[0].size();
            if (n1_total < 2) {
                if (!has_error.exchange(true))
                    error_message = "Sample too small for reference at column " + std::to_string(c);
                return;
            }

            for (size_t g = 1; g < G; ++g) {
                const size_t n2_total = gvals[g].size();
                if (n2_total < 2) {
                    if (!has_error.exchange(true))
                        error_message = "Sample too small for group " + std::to_string(g) +
                                        " at column " + std::to_string(c);
                    continue;
                }

                double R1 = 0.0, tie_sum = 0.0; bool has_tie = false;
                merge_rank_sum_with_tie(
                    gvals[0].data(), gvals[0].size(),
                    gvals[g].data(), gvals[g].size(),
                    R1, tie_sum, has_tie
                );

                const double n1d = static_cast<double>(n1_total);
                const double U1  = R1 - n1d * (n1d + 1.0) * 0.5;

                const size_t idx = c * n_targets + (g - 1);
                U1_out[idx]  = U1;
                n1_arr[idx]  = n1d;
                n2_arr[idx]  = static_cast<double>(n2_total);
                tie_arr[idx] = tie_sum * tie_mask;  // 避免 if
            }
        });
    } break;

    case MannWhitneyuOption::min:  // 0 在头
    case MannWhitneyuOption::max:  // 0 在尾
    {
        const bool zero_at_head = (opt.zero_handling == MannWhitneyuOption::min);
        run_cols([&](size_t c){
            std::vector<std::vector<double>> gvals;
            build_and_sort_gvals(c, gvals);

            const size_t n1_total = group_rows[0];
            if (n1_total < 2) {
                if (!has_error.exchange(true))
                    error_message = "Sample too small for reference at column " + std::to_string(c);
                return;
            }

            const size_t a_zero_imp = group_rows[0] - gvals[0].size();

            for (size_t g = 1; g < G; ++g) {
                const size_t n2_total = group_rows[g];
                if (n2_total < 2) {
                    if (!has_error.exchange(true))
                        error_message = "Sample too small for group " + std::to_string(g) +
                                        " at column " + std::to_string(c);
                    continue;
                }
                const size_t b_zero_imp = group_rows[g] - gvals[g].size();

                double R1 = 0.0, tie_sum = 0.0; bool has_tie = false;
                merge_rank_sum_with_tie_zero_extreme(
                    gvals[0].data(), gvals[0].size(),
                    gvals[g].data(), gvals[g].size(),
                    a_zero_imp, b_zero_imp,
                    /*zero_at_head=*/zero_at_head,
                    R1, tie_sum, has_tie
                );

                const double n1d = static_cast<double>(n1_total);
                const double U1  = R1 - n1d * (n1d + 1.0) * 0.5;

                const size_t idx = c * n_targets + (g - 1);
                U1_out[idx]  = U1;
                n1_arr[idx]  = n1d;
                n2_arr[idx]  = static_cast<double>(n2_total);
                tie_arr[idx] = tie_sum * tie_mask;
            }
        });
    } break;

    case MannWhitneyuOption::mix: { // 负段 | 零段 | 正段
        run_cols([&](size_t c){
            std::vector<std::vector<double>> gvals;
            build_and_sort_gvals(c, gvals);

            const size_t n1_total = group_rows[0];
            if (n1_total < 2) {
                if (!has_error.exchange(true))
                    error_message = "Sample too small for reference at column " + std::to_string(c);
                return;
            }
            const size_t a_zero_imp = group_rows[0] - gvals[0].size();

            for (size_t g = 1; g < G; ++g) {
                const size_t n2_total = group_rows[g];
                if (n2_total < 2) {
                    if (!has_error.exchange(true))
                        error_message = "Sample too small for group " + std::to_string(g) +
                                        " at column " + std::to_string(c);
                    continue;
                }
                const size_t b_zero_imp = group_rows[g] - gvals[g].size();

                double R1 = 0.0, tie_sum = 0.0; bool has_tie = false;
                merge_rank_sum_with_tie_include_zeros(
                    gvals[0].data(), gvals[0].size(),
                    gvals[g].data(), gvals[g].size(),
                    a_zero_imp, b_zero_imp,
                    R1, tie_sum, has_tie
                );

                const double n1d = static_cast<double>(n1_total);
                const double U1  = R1 - n1d * (n1d + 1.0) * 0.5;

                const size_t idx = c * n_targets + (g - 1);
                U1_out[idx]  = U1;
                n1_arr[idx]  = n1d;
                n2_arr[idx]  = static_cast<double>(n2_total);
                tie_arr[idx] = tie_sum * tie_mask;
            }
        });
    } break;
    }

    if (has_error.load()) throw std::runtime_error(error_message);

    // ---- p 值：这里已无列循环内分支，方法分支也已经在函数粒度（成本低）
    if (opt.method == MannWhitneyuOption::asymptotic) {
        p_asymptotic_parallel(
            U1_out.data(), n1_arr.data(), n2_arr.data(),
            tie_arr.data(), cc_arr.data(),
            P_out.data(), Npairs,
            opt.alternative,
            opt.fast_norm,
            threads
        );
    } else {
        p_exact_parallel(
            U1_out.data(), n1_arr.data(), n2_arr.data(),
            P_out.data(), Npairs,
            opt.alternative, threads
        );
    }

    return { std::move(U1_out), std::move(P_out) };
}

template<class T>
MWUResult mannwhitneyu(
    const view::CscView<T>& A,
    const int32_t* group_id,   // group_id 指针
    const size_t&  n_targets,   // 目标组的数量（值域大小）
    const MannWhitneyuOption& option,
    const int      threads,
    size_t*        /*progress_ptr*/
) {
    // ===== 取出三元数组 =====
    const T* data = A.data();
    const int64_t* indices = A.indices();
    const int64_t* indptr  = A.indptr();
    size_t C = A.cols();
    size_t R = A.rows();
    size_t nnz = A.nnz();

    if (n_targets < 1) {
        throw std::runtime_error("[mannwhitney] n_targets must be >= 1 (>=1 target groups)");
    }

    // ===== 调用核心 =====
    auto ret = mannWhitneyu_core<T>(
        data,
        indices,
        indptr,
        R, C, nnz,
        group_id,      // 直接传指针
        n_targets,
        option,
        threads
    );

    MWUResult result;
    result.U1 = std::move(ret.first);
    result.P  = std::move(ret.second);
    return std::move(result);
}

// ========================= group_mean =========================
// 计算每列、每组的均值：返回 size = C * G 的扁平数组，布局 [c * G + g]。
// include_zeros=true 时：分母 = group_rows[g] - invalid_cnt[g]
// （隐式0计入；显式无效值剔除）；false 时：分母 = valid_cnt[g]（仅显式有效计数）。
// 显式无效值(!is_valid_value)不计入分子/分母。

#include <algorithm> // std::fill

template<class T>
static force_inline_ std::vector<double>
group_mean_core(
    const T*           data,        // nnz
    const int64_t*     indices,     // nnz
    const int64_t*     indptr,      // C+1
    const size_t&      R,
    const size_t&      C,
    const size_t&      /*nnz*/,
    const int32_t*     group_id,    // size = R
    const size_t&      n_groups,
    bool               include_zeros,
    int                threads
){
    const size_t G = n_groups;
    if (G == 0 || C == 0) return {};

    // 每组总行数：用于 include_zeros 分母基数
    std::vector<size_t> group_rows(G, 0);
    for (size_t r = 0; r < R; ++r) {
        int g = group_id[r];
        if (g >= 0 && size_t(g) < G) ++group_rows[size_t(g)];
    }

    if (threads < 0) threads = omp_get_max_threads();
    if (threads > omp_get_max_threads()) threads = omp_get_max_threads();

    std::vector<double> mean_out(C * G, 0.0);

    // 列并行（动态调度）：线程内复用列缓冲
    #pragma omp parallel num_threads(threads)
{
    std::vector<double> sum(G);
    std::vector<size_t> valid_cnt(G);
    std::vector<size_t> invalid_cnt(G);

    #pragma omp for schedule(dynamic)
    for (std::ptrdiff_t cc = 0; cc < (std::ptrdiff_t)C; ++cc) {
        // 使用 memset 快速清零
        std::memset(sum.data(),        0, G * sizeof(double));
        std::memset(valid_cnt.data(),  0, G * sizeof(size_t));
        std::memset(invalid_cnt.data(),0, G * sizeof(size_t));

        const size_t c   = static_cast<size_t>(cc);
        const int64_t p0 = indptr[c];
        const int64_t p1 = indptr[c + 1];

        for (int64_t p = p0; p < p1; ++p) {
            const int64_t r = indices[p];
            if (r < 0 || size_t(r) >= R) continue;

            const int gi = group_id[size_t(r)];
            if (gi < 0 || size_t(gi) >= G) continue;

            const double v = static_cast<double>(data[p]);
            if (is_valid_value(v)) {
                sum[size_t(gi)] += v;
                ++valid_cnt[size_t(gi)];
            } else {
                ++invalid_cnt[size_t(gi)];
            }
        }

        double* out_col = mean_out.data() + c * G;
        for (size_t g = 0; g < G; ++g) {
            size_t denom = include_zeros
                ? (group_rows[g] - invalid_cnt[g])   // 隐式0计入，显式无效剔除
                :  valid_cnt[g];                      // 仅显式有效
            out_col[g] = (denom > 0) ? (sum[g] / static_cast<double>(denom)) : 0.0;
        }
    }
}

    return std::move(mean_out);
}

// 对外包装：支持 CSC/CSR（CSR 视作“转置的 CSC”）
template<class T>
std::vector<double> group_mean(
    const view::CscView<T>& A,
    const int32_t* group_id,
    const size_t&  n_groups,
    bool           include_zeros,   // fold-change 场景建议传 true
    int            threads
){
    const T* data = A.data();
    const int64_t* indices = A.indices();
    const int64_t* indptr  = A.indptr();
    size_t C = A.cols();
    size_t R = A.rows();
    size_t nnz = A.nnz();

    return std::move(group_mean_core<T>(
        data, indices, indptr,
        R, C, nnz,
        group_id, n_groups,
        include_zeros, threads
    ));
}

// 显式实例化
#define MWU_INSTANTIATE(T) \
    template MWUResult mannwhitneyu<T>( \
        const view::CscView<T>&, \
        const int32_t*, \
        const size_t&, \
        const MannWhitneyuOption&, \
        const int, \
        size_t* \
    );

TYPE_DISPATCH(MWU_INSTANTIATE);
#undef MWU_INSTANTIATE

#define GROUP_MEAN_INSTANTIATE(T) \
    template std::vector<double> group_mean<T>( \
        const view::CscView<T>&, \
        const int32_t*, \
        const size_t&, \
        bool, \
        int \
    );

TYPE_DISPATCH(GROUP_MEAN_INSTANTIATE);
#undef GROUP_MEAN_INSTANTIATE

}