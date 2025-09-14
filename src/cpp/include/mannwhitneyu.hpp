// mannwhitneyu.hpp - Mann-Whitney U test
#pragma once
#include "macro.hpp"
#include "sparse.hpp"
#include "common.hpp"


PROJECT_BEGIN

using MWUResult = std::tuple<torch::Tensor, torch::Tensor>; // double

struct MannWhitneyuOption {
    bool ref_sorted;
    bool tar_sorted;
    
    bool tie_correction;
    bool use_continuity;

    // 假设检验的方向
    enum Alternative { less = 0, greater = 1, two_sided = 2 } alternative;

    // 计算方法
    enum Method { automatic = 0, exact = 1, asymptotic = 2 } method;
};

MWUResult mannwhitneyu(
    const view::CscView& A,
    const torch::Tensor& group_id,
    const size_t& n_groups,
    const MannWhitneyuOption& option,
    const int threads = -1,
    size_t* progress_ptr = nullptr
);





PROJECT_END