#pragma once
#include "macro.hpp"
#include "config.hpp"
#include "simd.hpp"

#include <cmath>

PROJECT_BEGIN

// SIMD实现erf
template<class T>
force_inline_ T erf(T x) {
#if FAST_NORM
    // Abramowitz and Stegun 7.1.26 近似
    // 误差 < 1.5e-7
    const T a1 = 0.254829592;
    const T a2 = -0.284496736;
    const T a3 = 1.421413741;
    const T a4 = -1.453152027;
    const T a5 = 1.061405429;
    const T p  = 0.3275911;

    T sign = (x < 0) ? -1 : 1;
    x = std::abs(x);

    T t = 1.0 / (1.0 + p * x);
    T y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-x * x);

    return sign * y;
#else
    return std::erf(x);
#endif
}

// SIMD批量erf
template<class T>
force_inline_ void erf_v(T* x, size_t n) {
#if FAST_NORM && USE_HIGHWAY
    using namespace simd;
    const T a1 = 0.254829592;
    const T a2 = -0.284496736;
    const T a3 = 1.421413741;
    const T a4 = -1.453152027;
    const T a5 = 1.061405429;
    const T p  = 0.3275911;

    size_t step = lanes<T>();
    size_t i = 0;
    for (; i + step <= n; i += step) {
        auto vx = load(x + i);

        // sign = (vx < 0) ? -1 : 1
        auto vzero = zero<T>();
        auto vneg1 = set1<T>(-1);
        auto vone  = set1<T>(1);
        auto sign = select(lt(vx, vzero), vneg1, vone);

        // abs(x)
        auto vabsx = abs(vx);

        // t = 1.0 / (1.0 + p * abs(x))
        auto vp = set1<T>(p);
        auto vt = vone / (vone + vp * vabsx);

        // y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-abs(x) * abs(x))
        auto va1 = set1<T>(a1);
        auto va2 = set1<T>(a2);
        auto va3 = set1<T>(a3);
        auto va4 = set1<T>(a4);
        auto va5 = set1<T>(a5);

        auto vexp = exp(-vabsx * vabsx);

        auto vy = vone - (((((va5 * vt + va4) * vt + va3) * vt + va2) * vt + va1) * vt * vexp);

        vy = sign * vy;

        store(vy, x + i);
    }
    // 处理尾部
    if (i < n) {
        size_t tail = n - i;
        auto m = mask_from_count<T>(tail);
        auto vx = masked_load(m, x + i);

        auto vzero = zero<T>();
        auto vneg1 = set1<T>(-1);
        auto vone  = set1<T>(1);
        auto sign = select(lt(vx, vzero), vneg1, vone);

        auto vabsx = abs(vx);

        auto vp = set1<T>(p);
        auto vt = vone / (vone + vp * vabsx);

        auto va1 = set1<T>(a1);
        auto va2 = set1<T>(a2);
        auto va3 = set1<T>(a3);
        auto va4 = set1<T>(a4);
        auto va5 = set1<T>(a5);

        auto vexp = exp(-vabsx * vabsx);

        auto vy = vone - (((((va5 * vt + va4) * vt + va3) * vt + va2) * vt + va1) * vt * vexp);

        vy = sign * vy;

        masked_store(m, vy, x + i);
    }
#else
    // 非FAST_NORM直接用erf
    for (size_t i = 0; i < n; ++i) {
        x[i] = erf(x[i]);
    }
#endif
}

template<class T>
force_inline_ T erfc(T x) {
    return 1.0 - erf(x);
}

// SIMD批量erfc
template<class T>
force_inline_ void erfc_v(T* x, size_t n) {
#if FAST_NORM && USE_HIGHWAY
    using namespace simd;
    const T a1 = 0.254829592;
    const T a2 = -0.284496736;
    const T a3 = 1.421413741;
    const T a4 = -1.453152027;
    const T a5 = 1.061405429;
    const T p  = 0.3275911;

    size_t step = lanes<T>();
    size_t i = 0;
    for (; i + step <= n; i += step) {
        auto vx = load(x + i);

        // sign = (vx < 0) ? -1 : 1
        auto vzero = zero<T>();
        auto vneg1 = set1<T>(-1);
        auto vone  = set1<T>(1);
        auto sign = select(lt(vx, vzero), vneg1, vone);

        // abs(x)
        auto vabsx = abs(vx);

        // t = 1.0 / (1.0 + p * abs(x))
        auto vp = set1<T>(p);
        auto vt = vone / (vone + vp * vabsx);

        // erf = sign * (1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-abs(x) * abs(x)))
        auto va1 = set1<T>(a1);
        auto va2 = set1<T>(a2);
        auto va3 = set1<T>(a3);
        auto va4 = set1<T>(a4);
        auto va5 = set1<T>(a5);

        auto vexp = exp(-vabsx * vabsx);

        auto verf = sign * (vone - (((((va5 * vt + va4) * vt + va3) * vt + va2) * vt + va1) * vt * vexp));

        // erfc = 1.0 - erf
        auto verfc = vone - verf;

        store(verfc, x + i);
    }
    // 处理尾部
    if (i < n) {
        size_t tail = n - i;
        auto m = mask_from_count<T>(tail);
        auto vx = masked_load(m, x + i);

        auto vzero = zero<T>();
        auto vneg1 = set1<T>(-1);
        auto vone  = set1<T>(1);
        auto sign = select(lt(vx, vzero), vneg1, vone);

        auto vabsx = abs(vx);

        auto vp = set1<T>(p);
        auto vt = vone / (vone + vp * vabsx);

        auto va1 = set1<T>(a1);
        auto va2 = set1<T>(a2);
        auto va3 = set1<T>(a3);
        auto va4 = set1<T>(a4);
        auto va5 = set1<T>(a5);

        auto vexp = exp(-vabsx * vabsx);

        auto verf = sign * (vone - (((((va5 * vt + va4) * vt + va3) * vt + va2) * vt + va1) * vt * vexp));

        // erfc = 1.0 - erf
        auto verfc = vone - verf;

        masked_store(m, verfc, x + i);
    }
#else
    // 非FAST_NORM直接用erfc
    for (size_t i = 0; i < n; ++i) {
        x[i] = erfc(x[i]);
    }
#endif
}

PROJECT_END