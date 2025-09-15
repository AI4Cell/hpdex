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



#include <random>
#include <iostream>
#include <chrono>

int main() {
    using clk = std::chrono::steady_clock;

    // 构造 bucket 数据
    const size_t N = 1 << 30;
    std::vector<uint32_t> bucket(N);

    std::mt19937 rng(static_cast<unsigned>(time(nullptr)));
    std::uniform_int_distribution<uint32_t> dist_u( 1000, 4000);
    for (size_t i = 0; i < N; ++i) bucket[i] = dist_u(rng);

    const size_t max_clip = 3000;
    const int threads = 1;

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
