// mannwhitneyu.cpp - Mann-Whitney U test (parallel and cuda speeding)
#include "mannwhitneyu.hpp"
#include "common.hpp"
#include "sparse.hpp"
#include "macro.hpp"
#include "simd.hpp"
#include <cstdint>
#include <sys/types.h>

PROJECT_BEGIN

// 获得 [G, C] 的nnz计数分布
force_inline_ std::vector<uint32_t> max_len(
    const int64_t* group_id,     // [R]
    const int64_t* indices,      // [nnz]
    const int64_t* indptr,       // [C + 1]
    const size_t other_dim_size, // R
    const size_t contiguous_dim_size, // C
    const size_t n_groups,       // G
    const uint8_t threads
) {
    const size_t R = other_dim_size;
    const size_t C = contiguous_dim_size;
    const size_t G = n_groups;

    std::vector<uint32_t> bucket(G * C, 0);

PARALLEL_REGION(threads)
{
    PARALLEL_FOR(static,, for (size_t c = 0; c < C; c++) {
        const int64_t c_off  = indptr[c];
        const int64_t r_len  = indptr[c + 1] - c_off;
        const int64_t* g_id  = group_id + c_off;

        for (size_t r = 0; r < r_len; r++) {
            const int64_t g = g_id[r];
            if likely_(g >= 0 && g < (int64_t)G) {
                bucket[c * G + g] += 1;
            }
        }
    })
}
    return bucket;
}

// return {clip, max}
#ifndef QUANTILE_BINS
#define QUANTILE_BINS 1024
#endif
force_inline_ std::pair<size_t, size_t> quantile(
    const std::vector<uint32_t>& bucket,
    double q,                 // 0.0 ~ 1.0
    const size_t max_clip,
    const int threads = 1
) {
    if unlikely_(bucket.empty()) return {0, 0};

    const uint32_t* ptr = bucket.data();
    const size_t n = bucket.size();

    size_t global_max = 0;
    size_t global_min = std::numeric_limits<size_t>::max();

    const size_t tail = n % simd::lanes<uint32_t>();
    // 提前处理tail
    if likely_(tail > 0) {
        const auto m = simd::mask_from_count<uint32_t>(tail);
        const auto v = simd::masked_load<uint32_t>(m, ptr + n - tail);
        global_max = simd::reduce_max<uint32_t>(v);
        global_min = simd::reduce_min<uint32_t>(v);
    }

    PARALLEL_REGION(threads)
{
    PARALLEL_FOR(static, reduction(min:global_min) reduction(max:global_max),
    for (size_t i = tail; i < n; i += simd::lanes<uint32_t>()) {
        const size_t step = simd::lanes<uint32_t>();
        const auto v = simd::load<uint32_t>(ptr + i);
        const uint32_t v_max = simd::reduce_max<uint32_t>(v);
        const uint32_t v_min = simd::reduce_min<uint32_t>(v);
    })


    // hist
    


}

}


// return {clip, max}
force_inline_ std::pair<size_t, size_t> norm(
    const std::vector<uint32_t>& bucket,
    const double k,
    const size_t max_clip,
    const int threads = 1
) {
    if unlikely_(bucket.empty()) return {0, 0};

    const uint32_t* bucket_ptr = bucket.data();
    const size_t n = bucket.size();

    size_t global_max   = 0;
    double global_sum   = 0.0;
    double global_sqr_sum = 0.0;

    const size_t tail = n % simd::lanes<uint32_t>();
    // 提前处理tail
    if likely_(tail > 0) {
        const auto m = simd::mask_from_count<uint32_t>(tail);
        const auto v = simd::masked_load<uint32_t>(m, bucket_ptr + n - tail);
        global_sum     += static_cast<double>(simd::reduce_sum<uint32_t>(v));
        global_sqr_sum += static_cast<double>(
            simd::reduce_sum<uint32_t>(simd::mul<uint32_t>(v, v)));
        const size_t max_v = simd::reduce_max<uint32_t>(v);
        global_max = max_v;
    }

    // ===== 并行 SIMD 规约 =====
    PARALLEL_REGION(threads)
{
    PARALLEL_FOR(static, reduction(+:global_sum,global_sqr_sum) reduction(max:global_max),
    for (size_t i = tail; i < n; i += simd::lanes<uint32_t>()) {
        const size_t step = simd::lanes<uint32_t>();
        const auto v = simd::load<uint32_t>(bucket_ptr + i);

        // 局部累积
        global_sum     += static_cast<double>(simd::reduce_sum<uint32_t>(v));
        global_sqr_sum += static_cast<double>(
            simd::reduce_sum<uint32_t>(simd::mul<uint32_t>(v, v)));
        const size_t max_v = simd::reduce_max<uint32_t>(v);
        global_max = global_max > max_v ? global_max : max_v;
    })
}
    if (global_max < max_clip) return {global_max, global_max};

    // ===== 均值 / 标准差 =====
    const double mean = global_sum / n;
    double variance   = global_sqr_sum / n - mean * mean;
    variance = variance < 0.0 ? 0.0 : variance;  // 数值稳定保护
    const double stdev = std::sqrt(variance);

    // ===== μ + kσ, 至少返回 mean =====
    const double cutoff = std::max(mean + k * stdev, mean);
    return {static_cast<size_t>(std::ceil(cutoff)), global_max};
}



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
    size_t rows = 0, cols = 0, nnz = 0;
    bool is_csc = false;

    if (std::holds_alternative<view::CscView>(A)) {
        const auto& A_view = std::get<view::CscView>(A);
        data_ptr    = &A_view.data_;
        indices_ptr = &A_view.indices_;
        indptr_ptr  = &A_view.indptr_;
        rows = A_view.rows();
        cols = A_view.cols();
        nnz  = A_view.nnz();
        is_csc = true;
    } else {
        const auto& A_view = std::get<view::CsrView>(A);
        data_ptr    = &A_view.data_;
        indices_ptr = &A_view.indices_;
        indptr_ptr  = &A_view.indptr_;
        rows = A_view.rows();
        cols = A_view.cols();
        nnz  = A_view.nnz();
    }

    const auto& data = *data_ptr;
    const auto& indices = *indices_ptr;
    const auto& indptr = *indptr_ptr;

    std::vector<uint32_t> bucket = max_len(
        group_id.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(),
        indptr.data_ptr<int64_t>(),
        is_csc ? rows : cols,
        is_csc ? cols : rows,
        n_groups,
        threads
    );

    size_t clip = 0;
    size_t max = 0;
    if (option.threshold_method == MannWhitneyuOption::ThresholdMethod::quantile) {
        auto [clip, max] = quantile(bucket, option.quantile);
    } else if (option.threshold_method == MannWhitneyuOption::ThresholdMethod::norm) {
        auto [clip, max] = norm(bucket, option.norm_k);
    } else {
        clip = option.max_clip;
        
    }
    clip = std::min(clip, option.max_clip);

    // 根据 clip 移动内存
    
    

    
}




PROJECT_END
