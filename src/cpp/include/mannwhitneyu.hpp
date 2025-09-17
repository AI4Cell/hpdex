// mannwhitneyu.hpp - Mann-Whitney U test
#pragma once
#include "macro.hpp"
#include "sparse.hpp"
#include <cstddef>
#include <vector>
#include <variant>

namespace hpdex {

struct MWUResult {
    std::vector<double> U1;
    std::vector<double> P;
};

struct MannWhitneyuOption {
    bool ref_sorted;
    bool tar_sorted;

    bool tie_correction;
    bool use_continuity;

    bool fast_norm;

    enum ZeroHandling { none = 0, min = 1, max = 2, mix = 3 } zero_handling;

    // 假设检验的方向
    enum Alternative { less = 0, greater = 1, two_sided = 2 } alternative;

    // 计算方法
    enum Method { exact = 1, asymptotic = 2 } method;
};

template<class T>
MWUResult mannwhitneyu(
    const view::CscView<T>& A,
    const int32_t* group_id,
    const size_t& n_targets,
    const MannWhitneyuOption& option,
    const int threads = -1,
    size_t* progress_ptr = nullptr
);


template<class T>
std::vector<double> group_mean(
    const view::CscView<T>&,
    const int32_t*,
    const size_t&,
    bool,
    int,
    bool = false
);


}