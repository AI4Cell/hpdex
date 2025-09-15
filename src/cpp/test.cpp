#include "config.hpp"
#include "macro.hpp"
#include "simd.hpp"
#include "common.hpp"
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>


PROJECT_BEGIN


// approx hist to calculate best clip R
#define MIN_SEARCH_SIZE 32
force_inline_ std::pair<size_t, size_t> search_clip( // -> (clip, max)
    const std::vector<uint32_t>& bucket,
    const size_t max_clip,
    int threads
) {
    threads = std::clamp(threads, 1, MAX_THREADS());

    // 1. get min and max value in bucket
    uint32_t min_buf[DEFAULT_MAX_THREADS];
    uint32_t max_buf[DEFAULT_MAX_THREADS];
    uint32_t* mins = threads < DEFAULT_MAX_THREADS ? min_buf : new uint32_t[threads];
    uint32_t* maxs = threads < DEFAULT_MAX_THREADS ? max_buf : new uint32_t[threads];
    const size_t chunk_size = bucket.size() / threads;
PARALLEL_REGION(threads)
{
    const int tid = THREAD_ID();
    const uint32_t* begin = bucket.data() + tid * chunk_size;
    const size_t len = tid == threads - 1 ? bucket.size() - tid * chunk_size : chunk_size;
    auto [min, max] = simd::array_minmax(begin, len);
    mins[tid] = min;
    maxs[tid] = max;
}
    const auto max = simd::array_max<uint32_t>(maxs, threads);
    if unlikely_(max < max_clip) return {max, max}; // dont need to calculate
    const auto min = simd::array_min<uint32_t>(mins, threads);
    
    // 2. bin
    const auto dist = max - min;
    size_t hist_bin[APPROX_HIST_BINS] = {0};
    size_t* hist_bin_threads = hist_bin;
    if likely_(threads > 1) {
        hist_bin_threads = new size_t[threads * APPROX_HIST_BINS] {0};
    }

    // 3. hist
    const auto global_min_v = simd::set1<uint32_t>(min);
    const auto global_dist_v = simd::set1<uint32_t>(dist);
    const auto bin_size_v = simd::set1<uint32_t>(APPROX_HIST_BINS);
PARALLEL_REGION(threads)
{
    const int tid = THREAD_ID();
    const uint32_t* begin = bucket.data() + tid * chunk_size;
    const size_t len = tid == threads - 1 ? bucket.size() - tid * chunk_size : chunk_size;
    alignas_(ALIGN_SIZE) uint32_t tmp[simd::lanes<uint32_t>()]; // for simd
    size_t* const hist_bin_cur = hist_bin_threads + tid * APPROX_HIST_BINS;
    // simd
    const size_t step = simd::lanes<uint32_t>();
    size_t i = 0;
    for (; i + step <= len; i += step) {
        const auto v = simd::load(begin + i);
        const auto dist_v = simd::sub<uint32_t>(v, global_min_v);
        const auto dist_bin_b = simd::mul<uint32_t>(dist_v, bin_size_v);
        const auto bin_v = simd::div<uint32_t>(dist_bin_b, global_dist_v);
        simd::store(bin_v, tmp);
        for (size_t j = 0; j < step; j++) {
            hist_bin_cur[tmp[j]] += 1;
        }
    }
    const size_t tail = len - i;
    if (tail > 0) {
        const auto m = simd::mask_from_count<uint32_t>(tail);
        const auto v = simd::masked_load(m, begin + i);
        const auto dist_v = simd::sub<uint32_t>(v, global_min_v);
        const auto dist_bin_b = simd::mul<uint32_t>(dist_v, bin_size_v);
        const auto bin_v = simd::div<uint32_t>(dist_bin_b, global_dist_v);
        simd::store(bin_v, tmp);
        for (size_t j = 0; j < tail; j++) {
            hist_bin_cur[tmp[j]] += 1;
        }
    }
}
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
    // 我们的目标函数是：
    //     f(R) = (Σ_{v_i < R} v_i) / R
    // 在直方图近似下：
    //
    // 1. bin 宽度： w = (max - min) / APPROX_HIST_BINS
    // 2. bin i 的中心： center(i) = min + (i + 0.5) * w
    // 3. bin i 的上界： R_i = min + (i+1) * w
    // 4. 前缀和： prefix_sum[i] = Σ_{k=0}^i hist[k] * center(k)
    // 5. 函数值： f(R_i) = prefix_sum[i] / R_i
    //
    // 最优 clip R* = argmax_{i} f(R_i)
    //
    // 由于bins已排序，前缀和的一阶差分一定单调递增
    // 所以该函数 f(R_i) = prefix_sum[i] / R_i 是准凸函数
    // 因此我们使用并行多分搜索法来找到最优的 clip R*

    // 4.1 计算前缀和（递归计算无法并行，所以这里直接计算）
    double prefix_sum[APPROX_HIST_BINS] = {0};
    // use bin center for better approximation: center(i) = min + (i + 0.5) * dist / BINS
    const double w = double(dist) / double(APPROX_HIST_BINS);
    const double center0 = double(min) + 0.5 * w;
    prefix_sum[0] = double(hist_bin[0]) * center0;
    for (size_t i = 1; i < APPROX_HIST_BINS; i++) {
        const double current = double(min) + (double(i) + 0.5) * w;
        prefix_sum[i] = prefix_sum[i - 1] + double(hist_bin[i]) * current;
    }
    
    // 4.2 多分搜索
    size_t begin = 0;
    size_t end = APPROX_HIST_BINS;

    bool res_buf[DEFAULT_MAX_THREADS]; // default threads buf
    bool* res_ptr = res_buf;
    if unlikely_(threads > DEFAULT_MAX_THREADS) {
        res_ptr = new bool[threads];
    }

    while (end - begin > MIN_SEARCH_SIZE) {
        const size_t chunk_size = (end - begin) / threads;
        if unlikely_(chunk_size < 3) break; // too small

        // search
PARALLEL_REGION(threads)
{
        const int tid = THREAD_ID();
        const bool tid_is_first = tid == 0;
        const bool tid_is_last = tid == threads - 1;
        const size_t l = tid_is_first ? begin : begin + tid * chunk_size - 1;
        const size_t r = tid_is_last ? end : l + chunk_size + 2; // guarded below
        // Clamp indices to valid range [begin, end)
        const size_t l0 = std::min(std::max(l, begin), end - 1);
        const size_t l1 = std::min(l0 + 1, end - 1);
        const size_t r1 = std::min(std::max<size_t>(r, begin + 1), end) - 1; // r-1 within [begin, end-1]
        const size_t r0 = std::max(r1 - 1, begin);                            // r-2 or begin
        // f(i) = prefix_sum[i] / (min + (i+1) * dist / BINS)
        const double denom_l0 = double(min) + double(l0 + 1) * double(dist) / double(APPROX_HIST_BINS);
        const double denom_l1 = double(min) + double(l1 + 1) * double(dist) / double(APPROX_HIST_BINS);
        const double denom_r0 = double(min) + double(r0 + 1) * double(dist) / double(APPROX_HIST_BINS);
        const double denom_r1 = double(min) + double(r1 + 1) * double(dist) / double(APPROX_HIST_BINS);
        const double fl1 = prefix_sum[l0] / denom_l0;
        const double fl2 = prefix_sum[l1] / denom_l1;
        const double fr1 = prefix_sum[r0] / denom_r0;
        const double fr2 = prefix_sum[r1] / denom_r1;
        const bool l_up = fl2 > fl1;
        const bool r_up = fr2 > fr1;
        if (l_up && !r_up && !tid_is_last) { // xor
            res_ptr[tid] = true;
        } else {
            res_ptr[tid] = false;
        }
}
        #if DEBUG
        size_t count = 0;
        for (size_t i = 0; i < threads; i++) {
            count += res_ptr[i] ? 1 : 0;
        }
        if (count > 1) {
            throw std::runtime_error("search_clip: multiple true");
        }
        #endif

        // find first true
        size_t t_pos = 0;
        for (;t_pos < threads; t_pos++) {
            if unlikely_(res_ptr[t_pos]) {
                break;
            }
        }
        #if DEBUG
        if unlikely_(t_pos == threads) {
            throw std::runtime_error("search_clip: no true");
        }
        #endif
        begin = begin + t_pos * chunk_size;
        end = std::min(begin + chunk_size, end);
    }

    // calculate R_i
    double f_buf[MIN_SEARCH_SIZE];
    uint32_t R_buf[MIN_SEARCH_SIZE];
    double* f = f_buf;
    uint32_t* R = R_buf;
    const size_t len = end - begin;
    if unlikely_(len > MIN_SEARCH_SIZE) {
        f = new double[len];
        R = new uint32_t[len];
    }
    const size_t step = simd::lanes<double>();
    size_t i = 0;
    const auto min_v_f = simd::set1<double>(min);
    const auto inv_bins_v = simd::set1<double>(double(dist) / double(APPROX_HIST_BINS));
    alignas_(ALIGN_SIZE) double idx_buf[step];
    for (size_t k = 0; k < step; k++) {
        idx_buf[k] = double(k + 1);
    }
    for (; i + step <= len; i += step) {
        const auto begin_v = simd::set1<double>(double(begin + i));
        const auto idx_i_v = simd::load<double>(idx_buf);
        const auto idx_v = simd::add<double>(begin_v, idx_i_v);
        const auto R_v = simd::add<double>(simd::mul<double>(inv_bins_v, idx_v), min_v_f);
        const auto prefix_sum_v = simd::load<double>(prefix_sum + begin + i);
        const auto f_v = simd::div<double>(prefix_sum_v, R_v);
        simd::store<double>(f_v, f + i);
        const auto R_v_u = simd::convert<uint32_t, double>(R_v);
        simd::store<uint32_t>(R_v_u, R + i);
    }
    const size_t tail = len - i;
    if (tail > 0) {
        const auto m = simd::mask_from_count<double>(tail);
        const auto begin_v = simd::set1<double>(double(begin + i));
        const auto idx_i_v = simd::masked_load<double>(m, idx_buf);
        const auto idx_v = simd::add<double>(begin_v, idx_i_v);
        const auto R_v = simd::add<double>(simd::mul<double>(inv_bins_v, idx_v), min_v_f);
        const auto prefix_sum_v = simd::load<double>(prefix_sum + begin + i);
        const auto f_v = simd::div<double>(prefix_sum_v, R_v);
        simd::store<double>(f_v, f + i);
        const auto R_v_u = simd::convert<uint32_t, double>(R_v);
        simd::store<uint32_t>(R_v_u, R + i);
    }

    // find best clip
    size_t best = 0;
    for (size_t j = 0; j < len; j++) {
        best = f[j] > f[best] ? j : best;
    }
    if unlikely_(threads > DEFAULT_MAX_THREADS) {
        delete[] mins;
        delete[] maxs;
        delete[] res_ptr;
    }
    if unlikely_(len > MIN_SEARCH_SIZE) {
        delete[] f;
        delete[] R;
    }
    return {R[best], max};
}

PROJECT_END