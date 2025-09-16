// simd.hpp - Comprehensive SIMD Wrapper (Highway)
// Dependencies: config.hpp, macro.hpp
#pragma once

#include <cstddef>
#include <utility>
#include <limits>
#include <cstring>

#include "config.hpp"
#include "dependencies.hpp"
#include "macro.hpp"

// ================================================================
//      Conditional include Highway
// ================================================================

namespace hn = hwy::HWY_NAMESPACE;

namespace hpdex {
namespace simd {
using namespace hn;

template<class T>
force_inline_ T array_reduce_max(const T* x, size_t n) {
    using D = HWY_FULL(T);
    D d;

    const size_t step = Lanes(d);
    size_t i = 0;

    // 初始最大值
    T max_val = std::numeric_limits<T>::lowest();

    // SIMD 部分
    if (n >= step) {
        auto vmax = Load(d, x);
        max_val = ReduceMax(d, vmax);
        i += step;

        for (; i + step <= n; i += step) {
            auto v = Load(d, x + i);
            auto local_max = ReduceMax(d, v);
            if (local_max > max_val) max_val = local_max;
        }
    }

    // 处理尾巴
    for (; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }

    return max_val;
}

template<class T>
force_inline_ T array_reduce_min(const T* x, size_t n) {
    using D = HWY_FULL(T);
    D d;

    const size_t step = Lanes(d);
    size_t i = 0;

    // 初始最小值
    T min_val = std::numeric_limits<T>::max();

    // SIMD 部分
    if (n >= step) {
        auto vmin = Load(d, x);
        min_val = ReduceMin(d, vmin);
        i += step;

        for (; i + step <= n; i += step) {
            auto v = Load(d, x + i);
            auto local_min = ReduceMin(d, v);
            if (local_min < min_val) min_val = local_min;
        }
    }

    // 处理尾巴
    for (; i < n; ++i) {
        if (x[i] < min_val) min_val = x[i];
    }

    return min_val;
}

template<class T>
force_inline_ std::pair<T, T> array_reduce_minmax(const T* x, size_t n) {
    using D = HWY_FULL(T);
    D d;

    const size_t step = Lanes(d);
    size_t i = 0;

    // 初始 min/max
    T min_val = std::numeric_limits<T>::max();
    T max_val = std::numeric_limits<T>::lowest();

    if (n >= step) {
        auto v = Load(d, x);
        min_val = ReduceMin(d, v);
        max_val = ReduceMax(d, v);
        i += step;

        for (; i + step <= n; i += step) {
            v = Load(d, x + i);
            T local_min = ReduceMin(d, v);
            T local_max = ReduceMax(d, v);
            if (local_min < min_val) min_val = local_min;
            if (local_max > max_val) max_val = local_max;
        }
    }

    // 处理尾巴
    for (; i < n; ++i) {
        if (x[i] < min_val) min_val = x[i];
        if (x[i] > max_val) max_val = x[i];
    }

    return {min_val, max_val};
}



}
}