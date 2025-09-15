#pragma once
#include "macro.hpp"
#include "config.hpp"
#include <cmath>
#include "simd.hpp"


namespace hpdex {

// ----------- 标量 fast_erfc -----------
force_inline_ double fast_erfc(double x) {
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
        + t * ( 0.17087277 ))))))))));
    double r = (x >= 0.0) ? tau : 2.0 - tau;
    if (r < 0.0) r = 0.0;
    if (r > 2.0) r = 2.0;
    return r;
}

force_inline_ double normal_sf(double z) {
    return 0.5 * fast_erfc(z / std::sqrt(2.0));
}

force_inline_ void precompute_mu_inv_sd(
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

template<class T>
force_inline_ double p_asymptotic_two_sided(
    double U1, size_t n1, size_t n2,
    double tie_sum, double cc,
    double& mu_out, double& invsd_out
) {
    precompute_mu_inv_sd(n1, n2, tie_sum, mu_out, invsd_out);
    if (invsd_out == 0.0) return 1.0;
    const double z = (std::fabs(U1 - mu_out) - cc) * invsd_out;
    return 2.0 * normal_sf(z);
}

template<class T>
force_inline_ double p_asymptotic_greater(
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
force_inline_ double p_asymptotic_less(
    double U1, size_t n1, size_t n2,
    double tie_sum, double cc,
    double& mu_out, double& invsd_out
) {
    precompute_mu_inv_sd(n1, n2, tie_sum, mu_out, invsd_out);
    if (invsd_out == 0.0) return 1.0;
    const double z = (U1 - mu_out + cc) * invsd_out;
    return 1.0 - normal_sf(z);
}



// -------- SIMD fast_erfc --------
template<class D>
force_inline_ hn::Vec<D> fast_erfc_v(D d, hn::Vec<D> x) {
    using T = hn::TFromD<D>;

    auto ax = Abs(x);
    auto half = hn::Set(d, T(0.5));
    auto one  = hn::Set(d, T(1.0));
    auto two  = hn::Set(d, T(2.0));

    auto t = one / (one + half * ax);

    // Abramowitz & Stegun 7.1.26-like approximation
    auto tau = t * Exp(-ax*ax
        - hn::Set(d, T(1.26551223))
        + t * (hn::Set(d, T(1.00002368))
        + t * (hn::Set(d, T(0.37409196))
        + t * (hn::Set(d, T(0.09678418))
        + t * (hn::Set(d, T(-0.18628806))
        + t * (hn::Set(d, T(0.27886807))
        + t * (hn::Set(d, T(-1.13520398))
        + t * (hn::Set(d, T(1.48851587))
        + t * (hn::Set(d, T(-0.82215223))
        + t * (hn::Set(d, T(0.17087277))))))))))));

    auto mask_pos = GtEq(x, hn::Zero(d));
    auto r = IfThenElse(mask_pos, tau, two - tau);

    // Clamp [0, 2]
    r = Min(Max(r, hn::Zero(d)), two);
    return r;
}

template<class D>
force_inline_ hn::Vec<D> normal_sf_v(D d, hn::Vec<D> z) {
    using T = hn::TFromD<D>;
    auto inv_sqrt2 = hn::Set(d, T(1.0 / std::sqrt(2.0)));
    auto arg = z * inv_sqrt2;
    return hn::Mul(hn::Set(d, T(0.5)), fast_erfc_v<D>(d, arg));
}

template<class D>
force_inline_ void precompute_mu_inv_sd_v(
    D d,
    hn::Vec<D> n1, hn::Vec<D> n2, hn::Vec<D> tie_sum,
    hn::Vec<D>& mu, hn::Vec<D>& inv_sd
) {
    using T = hn::TFromD<D>;
    auto dn1 = ConvertTo(d, n1);
    auto dn2 = ConvertTo(d, n2);
    auto N   = dn1 + dn2;

    mu = hn::Mul(hn::Set(d, T(0.5)), dn1 * dn2);

    auto denom = N * (N - hn::Set(d, T(1.0)));
    auto base  = dn1 * dn2 / hn::Set(d, T(12.0));

    auto var = IfThenElse(
        Gt(denom, hn::Zero(d)),
        base * (N + hn::Set(d, T(1.0)) - tie_sum / denom),
        dn1 * dn2 * (N + hn::Set(d, T(1.0))) / hn::Set(d, T(12.0))
    );

    inv_sd = IfThenElse(
        Gt(var, hn::Zero(d)),
        hn::Div(hn::Set(d, T(1.0)), Sqrt(var)),
        hn::Zero(d)
    );
}

template<class D>
force_inline_ hn::Vec<D> p_asymptotic_two_sided_v(
    D d,
    hn::Vec<D> U1,
    hn::Vec<D> n1,
    hn::Vec<D> n2,
    hn::Vec<D> tie_sum,
    hn::Vec<D> cc
) {
    using T = hn::TFromD<D>;
    hn::Vec<D> mu, invsd;
    precompute_mu_inv_sd_v(d, n1, n2, tie_sum, mu, invsd);

    auto zero = hn::Zero(d);
    auto one  = hn::Set(d, T(1.0));
    auto two  = hn::Set(d, T(2.0));

    auto z = (Abs(U1 - mu) - cc) * invsd;
    auto sf = normal_sf_v<D>(d, z);
    return IfThenElse(Eq(invsd, zero), one, two * sf);
}

template<class D>
force_inline_ hn::Vec<D> p_asymptotic_greater_v(
    D d,
    hn::Vec<D> U1,
    hn::Vec<D> n1,
    hn::Vec<D> n2,
    hn::Vec<D> tie_sum,
    hn::Vec<D> cc
) {
    using T = hn::TFromD<D>;
    hn::Vec<D> mu, invsd;
    precompute_mu_inv_sd_v(d, n1, n2, tie_sum, mu, invsd);

    auto zero = hn::Zero(d);
    auto one  = hn::Set(d, T(1.0));

    auto z = (U1 - mu - cc) * invsd;
    auto sf = normal_sf_v<D>(d, z);
    return IfThenElse(Eq(invsd, zero), one, sf);
}

template<class D>
force_inline_ hn::Vec<D> p_asymptotic_less_v(
    D d,
    hn::Vec<D> U1,
    hn::Vec<D> n1,
    hn::Vec<D> n2,
    hn::Vec<D> tie_sum,
    hn::Vec<D> cc
) {
    using T = hn::TFromD<D>;
    hn::Vec<D> mu, invsd;
    precompute_mu_inv_sd_v(d, n1, n2, tie_sum, mu, invsd);

    auto zero = hn::Zero(d);
    auto one  = hn::Set(d, T(1.0));

    auto z = (U1 - mu + cc) * invsd;
    auto sf = normal_sf_v<D>(d, z);
    return IfThenElse(Eq(invsd, zero), one, one - sf);
}

// ================================================================
// array 版本：批量计算
// ================================================================
template<class T>
force_inline_ void array_p_asymptotic_two_sided(
    const T* U1, const T* n1, const T* n2,
    const T* tie_sum, const T* cc,
    T* out, size_t N
) {
    using D = HWY_FULL(T);
    D d;
    const size_t step = Lanes(d);
    size_t i = 0;

    // SIMD 主循环
    for (; i + step <= N; i += step) {
        auto vU1  = Load(d, U1 + i);
        auto vn1  = Load(d, n1 + i);
        auto vn2  = Load(d, n2 + i);
        auto vts  = Load(d, tie_sum + i);
        auto vcc  = Load(d, cc + i);

        auto vp = p_asymptotic_two_sided_v<T>(d, vU1, vn1, vn2, vts, vcc);
        Store(vp, d, out + i);
    }

    // 尾巴 fallback
    for (; i < N; ++i) {
        double mu, invsd;
        out[i] = p_asymptotic_two_sided<T>(U1[i], n1[i], n2[i], tie_sum[i], cc[i], mu, invsd);
    }
}

template<class T>
force_inline_ void array_p_asymptotic_greater(
    const T* U1, const T* n1, const T* n2,
    const T* tie_sum, const T* cc,
    T* out, size_t N
) {
    using D = HWY_FULL(T);
    D d;
    const size_t step = Lanes(d);
    size_t i = 0;

    for (; i + step <= N; i += step) {
        auto vU1  = Load(d, U1 + i);
        auto vn1  = Load(d, n1 + i);
        auto vn2  = Load(d, n2 + i);
        auto vts  = Load(d, tie_sum + i);
        auto vcc  = Load(d, cc + i);

        auto vp = p_asymptotic_greater_v<T>(d, vU1, vn1, vn2, vts, vcc);
        Store(vp, d, out + i);
    }

    for (; i < N; ++i) {
        double mu, invsd;
        out[i] = p_asymptotic_greater<T>(U1[i], n1[i], n2[i], tie_sum[i], cc[i], mu, invsd);
    }
}

template<class T>
force_inline_ void array_p_asymptotic_less(
    const T* U1, const T* n1, const T* n2,
    const T* tie_sum, const T* cc,
    T* out, size_t N
) {
    using D = HWY_FULL(T);
    D d;
    const size_t step = Lanes(d);
    size_t i = 0;

    for (; i + step <= N; i += step) {
        auto vU1  = Load(d, U1 + i);
        auto vn1  = Load(d, n1 + i);
        auto vn2  = Load(d, n2 + i);
        auto vts  = Load(d, tie_sum + i);
        auto vcc  = Load(d, cc + i);

        auto vp = p_asymptotic_less_v<T>(d, vU1, vn1, vn2, vts, vcc);
        Store(vp, d, out + i);
    }

    for (; i < N; ++i) {
        double mu, invsd;
        out[i] = p_asymptotic_less<T>(U1[i], n1[i], n2[i], tie_sum[i], cc[i], mu, invsd);
    }
}


}