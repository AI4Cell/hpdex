#pragma once
#include "macro.hpp"
#include "config.hpp"
#include <cmath>
#include "simd.hpp"

namespace hpdex {

// ================================================================
// 标量工具
// ================================================================

// ----------- fast_erfc (double 专用) -----------
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

// ================================================================
// 标量 p 计算（double 专用）
// ================================================================
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

// ================================================================
// SIMD 版本 (double 专用)
// ================================================================
template<class D>
HWY_INLINE hn::Vec<D> fast_erfc_v(D d, hn::Vec<D> x) {
    using T = hn::TFromD<D>;
    auto ax   = hn::Abs(x);
    auto half = hn::Set(d, T(0.5));
    auto one  = hn::Set(d, T(1.0));
    auto two  = hn::Set(d, T(2.0));

    auto t = one / (one + half * ax);

    auto neg_ax2 = hn::Neg(ax * ax);

    auto tau = t * hn::Exp(d,
        neg_ax2
        + hn::Neg(hn::Set(d, T(1.26551223)))
        + t * (hn::Set(d, T(1.00002368))
        + t * (hn::Set(d, T(0.37409196))
        + t * (hn::Set(d, T(0.09678418))
        + t * (hn::Neg(hn::Set(d, T(0.18628806))))
        + t * (hn::Set(d, T(0.27886807)))
        + t * (hn::Neg(hn::Set(d, T(1.13520398))))
        + t * (hn::Set(d, T(1.48851587)))
        + t * (hn::Neg(hn::Set(d, T(0.82215223))))
        + t * (hn::Set(d, T(0.17087277)))))));

    auto mask_pos = hn::Ge(x, hn::Zero(d));
    auto r = hn::IfThenElse(mask_pos, tau, two - tau);

    r = hn::Min(hn::Max(r, hn::Zero(d)), two);
    return r;
}



force_inline_ hn::Vec<HWY_FULL(double)> normal_sf_v(HWY_FULL(double) d, hn::Vec<HWY_FULL(double)> z) {
    auto inv_sqrt2 = hn::Set(d, 1.0 / std::sqrt(2.0));
    auto arg = z * inv_sqrt2;
    return hn::Mul(hn::Set(d, 0.5), fast_erfc_v(d, arg));
}

force_inline_ void precompute_mu_inv_sd_v(
    HWY_FULL(double) d,
    hn::Vec<HWY_FULL(double)> n1, hn::Vec<HWY_FULL(double)> n2, hn::Vec<HWY_FULL(double)> tie_sum,
    hn::Vec<HWY_FULL(double)>& mu, hn::Vec<HWY_FULL(double)>& inv_sd
) {
    auto N   = n1 + n2;
    mu = hn::Mul(hn::Set(d, 0.5), n1 * n2);

    auto one  = hn::Set(d, 1.0);
    auto denom = N * (N - one);
    auto base  = n1 * n2 / hn::Set(d, 12.0);

    auto var = hn::IfThenElse(
        hn::Gt(denom, hn::Zero(d)),
        base * (N + one - tie_sum / denom),
        n1 * n2 * (N + one) / hn::Set(d, 12.0)
    );

    inv_sd = hn::IfThenElse(
        hn::Gt(var, hn::Zero(d)),
        hn::Div(hn::Set(d, 1.0), hn::Sqrt(var)),
        hn::Zero(d)
    );
}

force_inline_ hn::Vec<HWY_FULL(double)> p_asymptotic_two_sided_v(
    HWY_FULL(double) d,
    hn::Vec<HWY_FULL(double)> U1,
    hn::Vec<HWY_FULL(double)> n1,
    hn::Vec<HWY_FULL(double)> n2,
    hn::Vec<HWY_FULL(double)> tie_sum,
    hn::Vec<HWY_FULL(double)> cc
) {
    hn::Vec<HWY_FULL(double)> mu, invsd;
    precompute_mu_inv_sd_v(d, n1, n2, tie_sum, mu, invsd);
    auto z = (hn::Abs(U1 - mu) - cc) * invsd;
    auto sf = normal_sf_v(d, z);
    return hn::IfThenElse(hn::Eq(invsd, hn::Zero(d)), hn::Set(d, 1.0), hn::Set(d, 2.0) * sf);
}

force_inline_ hn::Vec<HWY_FULL(double)> p_asymptotic_greater_v(
    HWY_FULL(double) d,
    hn::Vec<HWY_FULL(double)> U1,
    hn::Vec<HWY_FULL(double)> n1,
    hn::Vec<HWY_FULL(double)> n2,
    hn::Vec<HWY_FULL(double)> tie_sum,
    hn::Vec<HWY_FULL(double)> cc
) {
    hn::Vec<HWY_FULL(double)> mu, invsd;
    precompute_mu_inv_sd_v(d, n1, n2, tie_sum, mu, invsd);
    auto z = (U1 - mu - cc) * invsd;
    auto sf = normal_sf_v(d, z);
    return hn::IfThenElse(hn::Eq(invsd, hn::Zero(d)), hn::Set(d, 1.0), sf);
}

force_inline_ hn::Vec<HWY_FULL(double)> p_asymptotic_less_v(
    HWY_FULL(double) d,
    hn::Vec<HWY_FULL(double)> U1,
    hn::Vec<HWY_FULL(double)> n1,
    hn::Vec<HWY_FULL(double)> n2,
    hn::Vec<HWY_FULL(double)> tie_sum,
    hn::Vec<HWY_FULL(double)> cc
) {
    hn::Vec<HWY_FULL(double)> mu, invsd;
    precompute_mu_inv_sd_v(d, n1, n2, tie_sum, mu, invsd);
    auto z = (U1 - mu + cc) * invsd;
    auto sf = normal_sf_v(d, z);
    return hn::IfThenElse(hn::Eq(invsd, hn::Zero(d)), hn::Set(d, 1.0), hn::Set(d, 1.0) - sf);
}

// ================================================================
// array 版本：批量计算（double 专用）
// ================================================================
force_inline_ void array_p_asymptotic_two_sided(
    const double* U1, const double* n1, const double* n2,
    const double* tie_sum, const double* cc,
    double* out, size_t N
) {
    using D = HWY_FULL(double);
    D d;
    const size_t step = Lanes(d);
    size_t i = 0;

    for (; i + step <= N; i += step) {
        auto vU1  = hn::Load(d, U1 + i);
        auto vn1  = hn::Load(d, n1 + i);
        auto vn2  = hn::Load(d, n2 + i);
        auto vts  = hn::Load(d, tie_sum + i);
        auto vcc  = hn::Load(d, cc + i);

        auto vp = p_asymptotic_two_sided_v(d, vU1, vn1, vn2, vts, vcc);
        hn::Store(vp, d, out + i);
    }
    for (; i < N; ++i) {
        double mu, invsd;
        out[i] = p_asymptotic_two_sided(U1[i], size_t(n1[i]), size_t(n2[i]), tie_sum[i], cc[i], mu, invsd);
    }
}

force_inline_ void array_p_asymptotic_greater(
    const double* U1, const double* n1, const double* n2,
    const double* tie_sum, const double* cc,
    double* out, size_t N
) {
    using D = HWY_FULL(double);
    D d;
    const size_t step = Lanes(d);
    size_t i = 0;

    for (; i + step <= N; i += step) {
        auto vU1  = hn::Load(d, U1 + i);
        auto vn1  = hn::Load(d, n1 + i);
        auto vn2  = hn::Load(d, n2 + i);
        auto vts  = hn::Load(d, tie_sum + i);
        auto vcc  = hn::Load(d, cc + i);

        auto vp = p_asymptotic_greater_v(d, vU1, vn1, vn2, vts, vcc);
        hn::Store(vp, d, out + i);
    }
    for (; i < N; ++i) {
        double mu, invsd;
        out[i] = p_asymptotic_greater(U1[i], size_t(n1[i]), size_t(n2[i]), tie_sum[i], cc[i], mu, invsd);
    }
}

force_inline_ void array_p_asymptotic_less(
    const double* U1, const double* n1, const double* n2,
    const double* tie_sum, const double* cc,
    double* out, size_t N
) {
    using D = HWY_FULL(double);
    D d;
    const size_t step = Lanes(d);
    size_t i = 0;

    for (; i + step <= N; i += step) {
        auto vU1  = hn::Load(d, U1 + i);
        auto vn1  = hn::Load(d, n1 + i);
        auto vn2  = hn::Load(d, n2 + i);
        auto vts  = hn::Load(d, tie_sum + i);
        auto vcc  = hn::Load(d, cc + i);

        auto vp = p_asymptotic_less_v(d, vU1, vn1, vn2, vts, vcc);
        hn::Store(vp, d, out + i);
    }
    for (; i < N; ++i) {
        double mu, invsd;
        out[i] = p_asymptotic_less(U1[i], size_t(n1[i]), size_t(n2[i]), tie_sum[i], cc[i], mu, invsd);
    }
}

} // namespace hpdex
