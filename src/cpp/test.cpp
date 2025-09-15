#include "config.hpp"
#include "macro.hpp"
#include "simd.hpp"
#include "common.hpp"
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "macro.hpp"


PROJECT_BEGIN


// approx hist to calculate best clip R
force_inline_ std::pair<size_t, size_t> search_clip( // -> (clip, max)
    const std::vector<uint32_t>& bucket,
    const size_t max_clip,
    int threads
) {
    threads = std::clamp(threads, 1, MAX_THREADS());
    const int old_threads = N_THREADS();

    // 1. get min and max value in bucket
    uint32_t min_buf[DEFAULT_MAX_THREADS];
    uint32_t max_buf[DEFAULT_MAX_THREADS];
    uint32_t* mins = threads < DEFAULT_MAX_THREADS ? min_buf : new uint32_t[threads];
    uint32_t* maxs = threads < DEFAULT_MAX_THREADS ? max_buf : new uint32_t[threads];
    const size_t chunk_size = bucket.size() / threads;
PARALLEL_REGION(threads)

    const int tid = THREAD_ID();
    const uint32_t* begin = bucket.data() + tid * chunk_size;
    const size_t len = tid == threads - 1 ? bucket.size() - tid * chunk_size : chunk_size;
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
    const uint32_t* begin = bucket.data() + begin_idx;
    const size_t len = tid == threads - 1 ? bucket.size() - begin_idx : chunk_size;
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
            if (idx < 0 || idx >= APPROX_HIST_BINS) {
                throw std::runtime_error("search_clip: idx out of range:" + std::to_string(idx));
            }
            hist_bin_cur[idx] += 1;
        }
    }
    const size_t tail = len - i;
    if likely_(tail > 0) {
        const auto m = simd::mask_from_count<uint32_t>(tail);
        const auto v_i = simd::masked_load<uint32_t>(m, begin + i);
        const auto v = simd::convert<float, uint32_t>(v_i);
        const auto dist_v = simd::sub<float>(v, global_min_v);
        const auto dist_bin_b = simd::mul<float>(dist_v, bin_size_v);
        const auto bin_v = simd::div<float>(dist_bin_b, global_dist_v);
        simd::store<float>(bin_v, tmp);
        for (size_t j = 0; j < tail; j++) {
            int idx = static_cast<int>(tmp[j]);
            idx = idx > 0 ? idx : 0;
            idx = idx < APPROX_HIST_BINS ? idx : APPROX_HIST_BINS - 1;
            if (idx < 0 || idx >= APPROX_HIST_BINS) {
                throw std::runtime_error("search_clip: idx out of range:" + std::to_string(idx));
            }
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
        simd::store<float>(f_v, f + i);
        simd::store<float>(R_v, R + i);
    }
    if likely_(i < end) {
        const auto m = simd::mask_from_count<float>(APPROX_HIST_BINS - i);
        const auto idx_v_cur = simd::add<float>(idx_v, simd::set1<float>((float)i));
        const auto prefix_sum_v = simd::masked_load<float>(m, prefix_sum + i); // 分子需要mask处理
        const auto R_v = simd::add<float>(simd::mul<float>(w_v, idx_v_cur), min_v);
        const auto f_v = simd::div<float>(prefix_sum_v, R_v);
        simd::store<float>(f_v, f + i);
        simd::store<float>(R_v, R + i);
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

PROJECT_END

#include <random>
#include <iostream>
#include <ctime>

// ===== 精确朴素版：频数 + 前缀和，f(R)=sum_{v<R}v / R 的全局最优 =====
force_inline_ std::pair<size_t, size_t> search_clip_naive(
    const std::vector<uint32_t>& bucket,
    const size_t max_clip
) {
    if (bucket.empty()) return {0, 0};

    // min/max
    auto [min_it, max_it] = std::minmax_element(bucket.begin(), bucket.end());
    const uint32_t maxv = *max_it;

    // 与近似实现保持一致：若 max < max_clip 直接返回
    if (maxv < max_clip) return {maxv, maxv};

    // 频数直方图：O(n + maxv)
    std::vector<uint64_t> freq(static_cast<size_t>(maxv) + 1, 0);
    for (uint32_t v : bucket) ++freq[v];

    // prefix_weight_lt[R] = sum_{v < R} v * freq[v]
    // 用 long double 提升数值稳定性
    std::vector<long double> prefix_weight_lt(static_cast<size_t>(maxv) + 1, 0.0L);
    long double acc = 0.0L;
    // 注意：prefix[R] 是 "< R" 的和，所以先写入，再累加当前 r
    for (uint32_t r = 0; r <= maxv; ++r) {
        prefix_weight_lt[r] = acc;
        acc += static_cast<long double>(r) * static_cast<long double>(freq[r]);
    }

    // 扫描 R ∈ [1..maxv]，找最大 f(R) = prefix[R] / R
    size_t bestR = 1;
    long double bestF = prefix_weight_lt[1] / 1.0L;
    for (uint32_t R = 2; R <= maxv; ++R) {
        const long double f = prefix_weight_lt[R] / static_cast<long double>(R);
        if (f > bestF) { bestF = f; bestR = R; }
    }
    return {bestR, maxv};
}

#include <random>
#include <iostream>
#include <chrono>

int main() {
    using clk = std::chrono::steady_clock;

    // 构造 bucket 数据
    const size_t N = 1 << 24;
    std::vector<uint32_t> bucket(N);

    std::mt19937 rng(static_cast<unsigned>(time(nullptr)));
    std::uniform_int_distribution<uint32_t> dist_u( 1000, 4000);
    for (size_t i = 0; i < N; ++i) bucket[i] = dist_u(rng);

    const size_t max_clip = 3000;
    const int threads = 10;

    // 近似（你的 SIMD+并行多分搜索）
    auto t0 = clk::now();
    auto [clip_approx, maxv_approx] = hpdex::search_clip(bucket, max_clip, threads);
    auto t1 = clk::now();

    // 朴素精确
    auto t2 = clk::now();
    auto [clip_naive, maxv_naive] = search_clip_naive(bucket, max_clip);
    auto t3 = clk::now();

    // 计算 f(R) 以量化近似误差（单次 O(n) 即可）
    auto f_value = [&](size_t R)->long double {
        if (R == 0) return 0.0L;
        long double s = 0.0L;
        for (auto v : bucket) if (v < R) s += v;
        return s / static_cast<long double>(R) / bucket.size();
    };
    long double f_approx = f_value(clip_approx);
    long double f_naive  = f_value(clip_naive);
    long double rel_err  = (f_naive == 0.0L) ? 0.0L : (f_naive - f_approx) / f_naive;

    auto ms = [](auto d){ return std::chrono::duration_cast<std::chrono::microseconds>(d).count()/1000.0; };

    std::cout << "=== search_clip 对比 ===\n";
    std::cout << "approx: clip=" << clip_approx << ", max=" << maxv_approx
              << ", time=" << ms(t1 - t0) << " ms\n";
    std::cout << "naive : clip=" << clip_naive  << ", max=" << maxv_naive
              << ", time=" << ms(t3 - t2) << " ms\n";
    std::cout << "f(approx)=" << static_cast<double>(f_approx)
              << ", f(naive)=" << static_cast<double>(f_naive)
              << ", rel_err=" << static_cast<double>(rel_err) << "\n";
    return 0;
}
