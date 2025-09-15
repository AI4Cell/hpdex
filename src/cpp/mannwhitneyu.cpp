// mannwhitneyu.cpp - Mann-Whitney U test (parallel and cuda speeding)
#include "mannwhitneyu.hpp"
#include "common.hpp"
#include "config.hpp"
#include "sparse.hpp"
#include "macro.hpp"
#include "simd.hpp"
#include <cstdint>
#include <sys/types.h>
#include "cuda_tools.hpp"

PROJECT_BEGIN

// get [G, C] nnz for each group and column
force_inline_ std::vector<uint32_t> group_nnz(
    const int64_t* group_id,     // [R]
    const int64_t* indptr,       // [C + 1]
    const size_t& R, // R
    const size_t& C, // C
    const size_t& n_groups,       // G
    const uint8_t& threads
) {
    const size_t G = n_groups;
    const int old_threads = N_THREADS();

    std::vector<uint32_t> bucket(G * C, 0);

PARALLEL_REGION(threads)

    PARALLEL_FOR(static,, for (size_t c = 0; c < C; c++) {
        const int64_t c_off  = indptr[c];
        const int64_t r_len  = indptr[c + 1] - c_off;
        const int64_t* g_id  = group_id + c_off;

        for (size_t r = 0; r < r_len; r++) {
            const int64_t g = g_id[r];
            if likely_(g >= 0 && g < (int64_t)G) {
                bucket[g*C + c] += 1; // [G, C]
            }
        }
    })

PARALLEL_END(old_threads);
    return bucket;
}


// ============================================================
// Approximate histogram to calculate the best clip R
//
// Goal:
//   Given a histogram of bucket counts, find an optimal "clip"
//   value that separates GPU-friendly (dense) columns from
//   CPU-friendly (sparse) columns.
//
// ============================================================
//
// Definitions:
//   hist(i), i ∈ [0, B) 
//       — histogram counts for bin i
//
//   prefix(i) = Σ_{j=0..i} hist(j) * (j + 0.5)
//       — weighted prefix sum up to bin i
//
//   prefix(-1) = Σ_{j=0..B-1} hist(j) * (j + 0.5)
//       — total weighted sum across all bins
//
// Normalization:
//   p(i) = prefix(i) / prefix(-1)
//       — normalized distribution (CDF-like), ensures p(i) ∈ [0, 1]
//
// Objective function:
//   kf(i) = p(i) / (a + i) + norm * i
//
//   • First term:  p(i) / (a + i)
//       - Measures normalized distribution efficiency relative
//         to candidate clip size
//       - a = min/dist, derived from histogram bounds, adjusts
//         denominator for better stability
//
//   • Second term: norm * i
//       - Linear correction term, independent of raw data scale
//       - Helps bias the optimum clip towards desired direction
//       - norm is typically scaled with prefix(-1)/min so its
//         magnitude matches the first term
//
// Optimization:
//   Iterate i ∈ [0, B) and maximize kf(i).
//   The best bin index i* gives:
//       clip = ceil(R[i*])
//       max  = global maximum bucket count
//
// ============================================================
force_inline_ std::pair<size_t, size_t> search_clip( // -> (clip, max)
    const uint32_t* bucket,
    const size_t& bucket_size,
    const size_t& max_clip,
    float norm, // 用于偏移
    int threads
) {
    threads = std::clamp(threads, 1, MAX_THREADS());
    const int old_threads = N_THREADS();

    // 1. get min and max value in bucket
    uint32_t min_buf[DEFAULT_MAX_THREADS];
    uint32_t max_buf[DEFAULT_MAX_THREADS];
    uint32_t* mins = threads < DEFAULT_MAX_THREADS ? min_buf : new uint32_t[threads];
    uint32_t* maxs = threads < DEFAULT_MAX_THREADS ? max_buf : new uint32_t[threads];
    const size_t chunk_size = bucket_size / threads;
PARALLEL_REGION(threads)

    const int tid = THREAD_ID();
    const uint32_t* begin = bucket + tid * chunk_size;
    const size_t len = tid == threads - 1 ? bucket_size - tid * chunk_size : chunk_size;
    const auto [min, max] = simd::array_minmax(begin, len);
    mins[tid] = min;
    maxs[tid] = max;
PARALLEL_END(old_threads);
    const auto max = simd::array_max<uint32_t>(maxs, threads);
    if unlikely_(max < max_clip) return {max, max}; // dont need to calculate
    const auto min = simd::array_min<uint32_t>(mins, threads);
    
    // 2. bin
    const float dist = max - min;
    if unlikely_(dist == 0) return {max, max};
    float hist_bin[APPROX_HIST_BINS] = {0}; // 使用fp32防止溢出
    float* hist_bin_threads = hist_bin;
    if likely_(threads > 1) {
        hist_bin_threads = new float[threads * APPROX_HIST_BINS] {0};
    }

    // 3. hist
    const auto global_min_v = simd::set1<float>((float)min);
    const auto global_dist_v = simd::set1<float>((float)dist);
    const size_t real_bin_size = APPROX_HIST_BINS - 1;
    const auto bin_size_v = simd::set1<float>((float)real_bin_size);

PARALLEL_REGION(threads)

    const int tid = THREAD_ID();
    const size_t begin_idx = tid * chunk_size;
    const uint32_t* begin = bucket + begin_idx;
    const size_t len = tid == threads - 1 ? bucket_size - begin_idx : chunk_size;
    alignas_(ALIGN_SIZE) float tmp[simd::lanes<float>()]; // for simd
    float* const hist_bin_cur = hist_bin_threads + tid * APPROX_HIST_BINS;
    // simd
    const size_t step = simd::lanes<float>();
    size_t i = 0;
    for (; i + step <= len; i += step) {
        const auto v_i = simd::load<uint32_t>(begin + i);
        const auto v = simd::convert<float, uint32_t>(v_i);
        const auto dist_v = simd::sub<float>(v, global_min_v);
        const auto dist_bin_b = simd::mul<float>(dist_v, bin_size_v);
        const auto bin_v = simd::div<float>(dist_bin_b, global_dist_v);
        simd::store<float>(bin_v, tmp);
        for (size_t j = 0; j < step; j++) {
            int idx = static_cast<int>(tmp[j]);
            idx = idx > 0 ? idx : 0;
            idx = idx < APPROX_HIST_BINS ? idx : APPROX_HIST_BINS - 1;
            hist_bin_cur[idx] += 1;
        }
    }
    const size_t tail = len - i;
    if (tail > 0) {
        for (size_t j = 0; j < tail; j++) {
            float v = static_cast<float>(begin[i + j]);
            float dist_v = v - (float)min;
            float dist_bin_b = dist_v * (float)real_bin_size;
            float bin_f = dist_bin_b / (float)dist;
            int idx = static_cast<int>(bin_f);
            idx = idx > 0 ? idx : 0;
            idx = idx < APPROX_HIST_BINS ? idx : APPROX_HIST_BINS - 1;
            hist_bin_cur[idx] += 1;
        }
    }
PARALLEL_END(old_threads);
    // reduction
    if likely_(threads > 1) {
        for (size_t i = 0; i < threads; i++) {
            for (size_t j = 0; j < APPROX_HIST_BINS; j++) {
                hist_bin[j] += hist_bin_threads[i * APPROX_HIST_BINS + j];
            }
        }
        delete[] hist_bin_threads;
    }

    // 4. get best clip
    // 4.1 calculate prefix sum
    float prefix_sum[APPROX_HIST_BINS] = {0};
    // center(i) = min + (i + 0.5) * dist / BINS
    const float w = float(dist) / float(real_bin_size);
    const float center0 = float(min) + 0.5f * w;
    prefix_sum[0] = float(hist_bin[0]) * center0;
    for (size_t i = 1; i < APPROX_HIST_BINS; i++) {
        const float current = float(min) + (float(i) + 0.5f) * w;
        prefix_sum[i] = prefix_sum[i - 1] + float(hist_bin[i]) * current;
    }

    // calculate R_i
    float f[APPROX_HIST_BINS];
    float R[APPROX_HIST_BINS];
    const size_t step = simd::lanes<float>();
    const auto min_v = simd::set1<float>((float)min);
    const auto w_v = simd::set1<float>(w);
    alignas_(ALIGN_SIZE) float idx_v_buf[simd::lanes<float>()];

    // norm: norm * prefix_sum[-1] / min
    norm = norm * prefix_sum[APPROX_HIST_BINS - 1] / (float(min) > 0 ? min : 1.0f);
    const auto norm_v = simd::set1<float>(norm);

    for (uint8_t i = 0; i < simd::lanes<float>(); i++) {
        idx_v_buf[i] = float(i + 1.0f);
    }
    const size_t bin_chunk = APPROX_HIST_BINS / threads;
PARALLEL_REGION(threads)
    const auto idx_v = simd::load<float>(idx_v_buf);
    const int tid = THREAD_ID();
    size_t i = tid * bin_chunk;
    const size_t end = (tid == threads - 1) ? APPROX_HIST_BINS : (i + bin_chunk);
    for (; i + step < end; i += step) {
        const auto idx_v_cur = simd::add<float>(idx_v, simd::set1<float>((float)i));
        const auto prefix_sum_v = simd::load<float>(prefix_sum + i);
        const auto R_v = simd::add<float>(simd::mul<float>(w_v, idx_v_cur), min_v);
        const auto f_v = simd::div<float>(prefix_sum_v, R_v);
        const auto f_norm_v = simd::add<float>(f_v, simd::mul<float>(norm_v, idx_v));
        simd::store<float>(f_norm_v, f + i);
        simd::store<float>(R_v, R + i);
    }
    if likely_(i < end) {
        for (; i < end; ++i) {
            const float idx_cur = float(i + 1.0f);
            const float R_cur = w * idx_cur + float(min);
            const float f_cur = prefix_sum[i] / R_cur;
            const float f_norm_cur = f_cur + idx_cur * norm;
            f[i] = f_norm_cur;
            R[i] = R_cur;
        }
    }
PARALLEL_END(old_threads);

    // find max f
    float max_f = f[0];
    size_t max_i = 0;
    for (size_t i = 1; i < APPROX_HIST_BINS; i++) {
        if (f[i] > max_f) {
            max_f = f[i];
            max_i = i;
        }
    }
    return {(size_t)std::ceil(R[max_i]), max};
}


// 根据clip移动内存到cpu和gpu
using BadPoints = std::tuple<
    std::vector<size_t>, // group 坐标
    std::vector<size_t>, // col 坐标
    std::vector<size_t>  // indptr 每一段长度
>;
template<class T>
force_inline_ BadPoints move_memory(
    const T* data,
    const int64_t* indices,
    const int64_t* indptr,
    const size_t& R,
    const size_t& C,
    const size_t& nnz,
    const uint32_t* bucket,
    const size_t& bucket_size,
    torch::Tensor& cpu_buf, // 坏点
    torch::Tensor& gpu_buf // 提前开辟
);

// #define REF_NORM 0.5
// #define TAR_NORM 0
// MWUResult mannwhitneyu_with_cuda(
//     const std::variant<view::CscView, view::CsrView>& A,
//     const torch::Tensor& group_id,
//     const size_t& n_groups,
//     const MannWhitneyuOption& option,
//     const int threads,
//     size_t* progress_ptr
// ) {
//     const torch::Tensor* data_ptr    = nullptr;
//     const torch::Tensor* indices_ptr = nullptr;
//     const torch::Tensor* indptr_ptr  = nullptr;
//     size_t R = 0, C = 0, nnz = 0;
//     const size_t G = n_groups;
//     bool is_csc = false;

//     if (std::holds_alternative<view::CscView>(A)) {
//         const auto& A_view = std::get<view::CscView>(A);
//         data_ptr    = &A_view.data_;
//         indices_ptr = &A_view.indices_;
//         indptr_ptr  = &A_view.indptr_;
//         R = A_view.rows();
//         C = A_view.cols();
//         nnz  = A_view.nnz();
//         is_csc = true;
//     } else {
//         const auto& A_view = std::get<view::CsrView>(A);
//         data_ptr    = &A_view.data_;
//         indices_ptr = &A_view.indices_;
//         indptr_ptr  = &A_view.indptr_;
//         C = A_view.rows(); // reverse and see as CscView
//         R = A_view.cols();
//         nnz  = A_view.nnz();
//     }

//     const auto& data = *data_ptr;
//     const auto& indices = *indices_ptr;
//     const auto& indptr = *indptr_ptr;

//     const auto bucket = group_nnz(
//         group_id.data_ptr<int64_t>(),
//         indptr.data_ptr<int64_t>(),
//         R,
//         C,
//         n_groups,
//         threads
//     );

//     auto [ref_clip, ref_max] = search_clip(bucket.data(), R, option.max_clip, REF_NORM, threads);
//     auto [tar_clip, tar_max] = search_clip(bucket.data() + C, C * (G - 1), option.max_clip, TAR_NORM, threads);
//     tar_clip = tar_max < option.max_clip ? tar_max : tar_clip;

//     return {torch::Tensor(), torch::Tensor()};
// }

template<class T>
force_inline_ T array_normal_sf(T* z, const size_t& n) {
    // 上尾概率：P(Z >= z)
    // 使用可选 fast_erfc 包装，避免在热路径里多次触发 libc 调用开销
    return 0.5 * fast_erfc(z / std::sqrt(2.0));

    const size_t step = simd::lanes<T>();
    size_t i = 0;
    for (; i + step <= n; i += step) {
        const auto v = simd::load<T>(z + i);
        const auto v_res = erfc_v(v);
        simd::store<T>(v_res, z + i);
    }
    if (i < n) {
        const auto m = simd::mask_from_count<T>(n - i);
        const auto v = simd::masked_load<T>(m, z + i);
        const auto v_res = erfc_v(v);
        simd::masked_store<T>(m, v_res, z + i);
    }
}


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




using MWUCoreResult = std::tuple<std::vector<double>, std::vector<double>>;
template<class T>
force_inline_ MWUCoreResult 
mwu_core(
    const T* data,
    const int64_t* indices,
    const int64_t* indptr,
    const size_t& R,
    const size_t& C,
    const size_t& nnz
) {
    
}











// ========================
// cpu implementation of 
// ========================
MWUResult mannwhitneyu(
    const std::variant<view::CscView, view::CsrView>& A,
    const torch::Tensor& group_id,
    const size_t& n_groups,
    const MannWhitneyuOption& option,
    const int threads,
    size_t* progress_ptr
) {
    const torch::Tensor* data_ptr    = nullptr;
    const torch::Tensor* indices_ptr = nullptr;
    const torch::Tensor* indptr_ptr  = nullptr;
    size_t R = 0, C = 0, nnz = 0;
    const size_t G = n_groups;
    bool is_csc = false;

    if (std::holds_alternative<view::CscView>(A)) {
        const auto& A_view = std::get<view::CscView>(A);
        data_ptr    = &A_view.data_;
        indices_ptr = &A_view.indices_;
        indptr_ptr  = &A_view.indptr_;
        R = A_view.rows();
        C = A_view.cols();
        nnz  = A_view.nnz();
        is_csc = true;
    } else {
        const auto& A_view = std::get<view::CsrView>(A);
        data_ptr    = &A_view.data_;
        indices_ptr = &A_view.indices_;
        indptr_ptr  = &A_view.indptr_;
        C = A_view.rows(); // reverse and see as CscView
        R = A_view.cols();
        nnz  = A_view.nnz();
    }

    
}






PROJECT_END
