// mwu cpu 算子实现
#include "mannwhitneyu.hpp"
#include "common.hpp"
#include "config.hpp"
#include "macro.hpp"
#include "simd.hpp"

namespace hpdex {

template<class T>
void p_asymptotic_parallel(
    const T* U1, const T* n1, const T* n2,
    const T* tie_sum, const T* cc,
    T* out, size_t N,
    MannWhitneyuOption::Alternative alt,
    int threads
) {
    threads = threads < 0 ? omp_get_max_threads(): threads > omp_get_max_threads() ? omp_get_max_threads() : threads;
    using D = HWY_FULL(T);
    D d;
    const size_t step = Lanes(d);

    // -------- chunk 大小选择 --------
    const size_t chunk_align = 64 / sizeof(T); // 64B 对齐需要多少元素
    const size_t chunk_size  = std::max(step, chunk_align) * 128; // 手动放大，避免太小

    // -------- 并行划分 --------
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        const int nth = threads;

        size_t chunk_per_thread = (N + nth - 1) / nth;
        // 对齐到 step，保证前面的 chunk 是 step 的倍数
        size_t begin = tid * chunk_per_thread;
        size_t end   = std::min(N, begin + chunk_per_thread);

        // 对齐 begin 到 step 的倍数（便于 SIMD）
        if (tid > 0) {
            size_t misalign = begin % step;
            if (misalign != 0) {
                begin += (step - misalign);
                if (begin > end) begin = end;
            }
        }

        // -------- 执行对应版本 --------
        switch (alt) {
            case MannWhitneyuOption::Alternative::two_sided:
                array_p_asymptotic_two_sided<T>(
                    U1 + begin, n1 + begin, n2 + begin,
                    tie_sum + begin, cc + begin,
                    out + begin, end - begin
                );
                break;
            case MannWhitneyuOption::Alternative::greater:
                array_p_asymptotic_greater<T>(
                    U1 + begin, n1 + begin, n2 + begin,
                    tie_sum + begin, cc + begin,
                    out + begin, end - begin
                );
                break;
            case MannWhitneyuOption::Alternative::less:
                array_p_asymptotic_less<T>(
                    U1 + begin, n1 + begin, n2 + begin,
                    tie_sum + begin, cc + begin,
                    out + begin, end - begin
                );
                break;
        }
    }
}

template<class T>
force_inline_ double p_exact(double U, size_t n1, size_t n2) {
    using U64 = unsigned long long;

    const size_t Umax = n1 * n2;
    if unlikely_(Umax == 0) return 1.0;

    // clamp & floor like SciPy
    const double U_clip = std::max(0.0, std::min(static_cast<double>(Umax), U));
    const size_t u_stat = static_cast<size_t>(std::floor(U_clip));

    // DP buffers: 只写 0..up 区间，每次避免 O(SZ) 清零
    const size_t SZ = Umax + 1;
    alignas_(64) U64 dp[SZ];
    alignas_(64) U64 ndp[SZ];

    // 初始化
    dp[0] = 1ULL;
    for (size_t i = 1; i < SZ; ++i) dp[i] = 0ULL;
    for (size_t i = 0; i < SZ; ++i) ndp[i] = 0ULL;

    U64* dp_cur = dp;
    U64* dp_nxt = ndp;

    for (size_t i = 1; i <= n1; ++i) {
        const size_t prev_up = (i - 1) * n2;
        const size_t up      =  i      * n2;

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

        // 清理下一轮 buffer
        for (size_t j = 0; j <= up; ++j) dp_cur[j] = 0ULL;

        U64* tmp = dp_cur; dp_cur = dp_nxt; dp_nxt = tmp;
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
    return sf_ge;
}

// 需要你已有的 p_exact 声明：
// template<class T>
// force_inline_ double p_exact(double U, size_t n1, size_t n2);

template<class T>
void p_exact_parallel(
    const T* U1,          // len = N
    const T* n1,          // len = N
    const T* n2,          // len = N
    T*       out,         // len = N
    size_t   N,
    MannWhitneyuOption::Alternative alt,
    int      threads
) {
    // 线程数规范化
    int maxth = omp_get_max_threads();
    if (threads < 0) threads = maxth;
    if (threads > maxth) threads = maxth;

    #pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)N; ++i) {
        const size_t a = static_cast<size_t>(n1[i]);
        const size_t b = static_cast<size_t>(n2[i]);

        const double U1d = static_cast<double>(U1[i]);
        const double Nprod = static_cast<double>(a) * static_cast<double>(b);

        double p = 1.0;
        switch (alt) {
            case MannWhitneyuOption::two_sided: {
                const double U2   = Nprod - U1d;
                const double Umax = (U1d > U2 ? U1d : U2);
                p = 2.0 * p_exact<double>(Umax, a, b);
            } break;
            case MannWhitneyuOption::greater:
                p = p_exact<double>(U1d, a, b);
                break;
            case MannWhitneyuOption::less: {
                const double U2 = Nprod - U1d;
                p = p_exact<double>(U2, a, b);
            } break;
        }
        if (p < 0.0) p = 0.0;
        if (p > 1.0) p = 1.0;
        out[i] = static_cast<T>(p);
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
    if constexpr (std::is_floating_point_v<T>) return std::isfinite(x);
    else return true;
}

// 合并两有序数组，返回参考组秩和 R1 与并列修正量 tie_sum，has_tie 标志
template<class T>
static inline void merge_rank_sum_with_tie(
    const T* a, size_t n1,
    const T* b, size_t n2,
    double& R1, double& tie_sum, bool& has_tie
) {
    size_t i = 0, j = 0, rank = 1;   // 1-based
    R1 = 0.0; tie_sum = 0.0; has_tie = false;

    while (i < n1 || j < n2) {
        T v; bool take_a = false, take_b = false;
        if (i < n1 && j < n2) {
            if (a[i] < b[j]) { v = a[i]; take_a = true; }
            else if (b[j] < a[i]) { v = b[j]; take_b = true; }
            else { v = a[i]; take_a = take_b = true; }
        } else if (i < n1) { v = a[i]; take_a = true; }
        else { v = b[j]; take_b = true; }

        size_t eq_a = 0, eq_b = 0;
        if (take_a) while (i + eq_a < n1 && !(a[i + eq_a] < v) && !(v < a[i + eq_a])) ++eq_a;
        if (take_b) while (j + eq_b < n2 && !(b[j + eq_b] < v) && !(v < b[j + eq_b])) ++eq_b;

        const size_t t = eq_a + eq_b;
        const double start = double(rank);
        const double end   = double(rank + t - 1);
        const double avg   = 0.5 * (start + end);

        R1 += double(eq_a) * avg;
        if (t > 1) { tie_sum += (double(t) * double(t) * double(t) - double(t)); has_tie = true; }

        rank += t;
        i += eq_a; j += eq_b;
    }
}

// =============== 主核：并行列 -> 收集/排序 -> 计算 U1/tie/n1/n2，退出后并行算 p ===============
template<class T>
inline std::pair<std::vector<T>, std::vector<T>>
mannWhitneyu_core(
    const T*           data,        // nnz
    const int64_t*     indices,     // nnz
    const int64_t*     indptr,      // C+1
    const size_t&      R,
    const size_t&      C,
    const size_t&      nnz,
    std::vector<int>&  group_id,    // size=R, 值域 {0..n_targets}
    size_t             n_targets,
    const MannWhitneyuOption& opt,
    int                threads
) {
    if (group_id.size() != R)
        throw std::invalid_argument("[mannwhitney] group_id length must equal R");
    if (n_targets == 0)
        return { std::vector<T>(), std::vector<T>() };

    const size_t G = n_targets + 1; // 0: ref
    const size_t Npairs = C * n_targets;

    // ------- 输出与中间数组（线性布局：idx = c * n_targets + (g-1)）-------
    std::vector<T> U1_out(Npairs, T(0));
    std::vector<T> P_out (Npairs, T(1));
    // 为 p 并行准备的输入数组
    std::vector<T> n1_arr(Npairs, T(0));
    std::vector<T> n2_arr(Npairs, T(0));
    std::vector<T> tie_arr(Npairs, T(0));
    std::vector<T> cc_arr (Npairs, opt.use_continuity ? T(0.5) : T(0.0));

    // ------- 预估各组行数（仅用于 reserve，避免列内频繁扩容） -------
    std::vector<size_t> gcount(G, 0);
    for (size_t r = 0; r < R; ++r) {
        int g = group_id[r];
        if (g >= 0 && size_t(g) < G) ++gcount[size_t(g)];
    }

    // ------- 列并行：收集显式值 -> 条件排序 -> 计算 U1 / tie / n1 / n2 -------
    if (threads < 0) threads = omp_get_max_threads();
    if (threads > omp_get_max_threads()) threads = omp_get_max_threads();

    #pragma omp parallel for schedule(static) num_threads(threads)
    for (std::ptrdiff_t cc = 0; cc < (std::ptrdiff_t)C; ++cc) {
        const size_t c = (size_t)cc;

        // 每个线程/列的临时容器（避免共享写冲突）
        std::vector<std::vector<T>> gvals(G);
        for (size_t g = 0; g < G; ++g) gvals[g].reserve(gcount[g]);

        const int64_t p0 = indptr[c];
        const int64_t p1 = indptr[c + 1];

        // 收集显式值到各组
        for (int64_t p = p0; p < p1; ++p) {
            const int64_t r = indices[p];
            if (r < 0 || size_t(r) >= R) continue;
            const int g = group_id[size_t(r)];
            if (g < 0 || size_t(g) >= G) continue;
            const T v = data[p];
            if (!is_valid_value(v)) continue;
            gvals[size_t(g)].push_back(v);
        }

        // 条件排序
        if (!opt.ref_sorted && gvals[0].size() > 1)
            std::sort(gvals[0].begin(), gvals[0].end());
        if (!opt.tar_sorted) {
            for (size_t g = 1; g < G; ++g)
                if (gvals[g].size() > 1)
                    std::sort(gvals[g].begin(), gvals[g].end());
        }

        const size_t n1 = gvals[0].size();
        if (n1 < 2) {
            // 统一报错语义（和你主算子一致）
            // 注意：OpenMP 中抛异常会终止并不易收敛；可改为标记再统一抛。
            // 这里直接抛，和原逻辑一致。
            throw std::runtime_error("Sample too small for reference at column " + std::to_string(c));
        }

        // 逐目标组计算
        for (size_t g = 1; g < G; ++g) {
            const size_t n2 = gvals[g].size();
            if (n2 < 2) {
                throw std::runtime_error("Sample too small for group " + std::to_string(g) +
                                         " at column " + std::to_string(c));
            }
            double R1 = 0.0, tie_sum = 0.0; bool has_tie = false;
            merge_rank_sum_with_tie(
                gvals[0].data(), gvals[0].size(),
                gvals[g].data(), gvals[g].size(),
                R1, tie_sum, has_tie
            );
            if (!opt.tie_correction) tie_sum = 0.0;

            const double base = double(n1) * (double(n1) + 1.0) * 0.5;
            const double U1   = R1 - base;

            const size_t idx = c * n_targets + (g - 1);
            U1_out[idx]  = static_cast<T>(U1);
            n1_arr[idx]  = static_cast<T>(n1);
            n2_arr[idx]  = static_cast<T>(n2);
            tie_arr[idx] = static_cast<T>(tie_sum);
            // cc_arr[idx] 已在初始化时统一写好 0.5/0.0
        }
    } // end parallel for

    // ------- 退出并行后：用你的并行 p 计算器统一算 p -------
    if (opt.method == MannWhitneyuOption::asymptotic) {
        p_asymptotic_parallel<T>(
            U1_out.data(), n1_arr.data(), n2_arr.data(),
            tie_arr.data(), cc_arr.data(),
            P_out.data(), Npairs,
            opt.alternative, threads
        );
    } else { // exact
        // 注意：调用者需保证无并列(tie)；否则请改用 asymptotic
        p_exact_parallel<T>(
            U1_out.data(), n1_arr.data(), n2_arr.data(),
            P_out.data(), Npairs,
            opt.alternative, threads
        );
    }

    return { std::move(U1_out), std::move(P_out) };
}

}