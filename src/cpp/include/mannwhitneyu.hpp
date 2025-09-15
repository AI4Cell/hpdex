// mannwhitneyu.hpp - Mann-Whitney U test
#pragma once
#include "macro.hpp"
#include "sparse.hpp"
#include <cstddef>


PROJECT_BEGIN

using MWUResult = std::tuple<torch::Tensor, torch::Tensor>; // double

struct MannWhitneyuOption {
    bool ref_sorted;
    bool tar_sorted;

    bool tie_correction;
    bool use_continuity;

    size_t max_clip = 5000;
    double quantile_q = 0.8;
    double norm_k = 0.5;

    // 假设检验的方向
    enum Alternative { less = 0, greater = 1, two_sided = 2 } alternative;

    // 计算方法
    enum Method { automatic = 0, exact = 1, asymptotic = 2 } method;

    // threshold method
    enum ThresholdMethod { none = 0, quantile = 1, norm = 2 } threshold_method;

};

MWUResult mannwhitneyu(
    const std::variant<view::CscView, view::CsrView>& A,
    const torch::Tensor& group_id,
    const size_t& n_groups,
    const MannWhitneyuOption& option,
    const int threads = -1,
    size_t* progress_ptr = nullptr
);





PROJECT_END