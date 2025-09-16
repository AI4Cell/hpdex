// src/cpp/module.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <omp.h>
#include "sparse.hpp"
#include "mannwhitneyu.hpp"
#include "progress.hpp"

namespace py = pybind11;
namespace hpdex {

// ================= 工具函数 =================
static py::array_t<double> as_numpy_array(std::vector<double>&& vec) {
    // 用 new 分配，把 vector 转移给 capsule 管理
    auto* v = new std::vector<double>(std::move(vec));

    // capsule 负责释放内存
    auto capsule = py::capsule(v, [](void* p) {
        delete static_cast<std::vector<double>*>(p);
    });

    return py::array_t<double>(
        v->size(),          // 元素数
        v->data(),          // 裸指针
        capsule             // 生命周期托管
    );
}

// ================= MWU 调用 =================
template<typename T>
py::tuple call_mwu_numpy(
    py::array_t<T> data,
    py::array_t<int64_t> indices,
    py::array_t<int64_t> indptr,
    py::array_t<int32_t> group_id,
    size_t n_targets,
    const MannWhitneyuOption& opt,
    int threads,
    bool is_csr,
    bool show_progress
) {
    const size_t R   = static_cast<size_t>(group_id.size());
    const size_t C   = static_cast<size_t>(indptr.size()) - 1;
    const size_t NNZ = static_cast<size_t>(data.size());

    // CSR → 转置成 CSC
    size_t rows = is_csr ? C : R;
    size_t cols = is_csr ? R : C;

    auto V = view::CscView<T>::from_raw(
        data.data(), indices.data(), indptr.data(),
        rows, cols, NNZ
    );

    const int32_t* gid_ptr = group_id.data();

    // 进度条设置
    std::unique_ptr<ProgressTracker> tracker;
    std::unique_ptr<ProgressBar> progress_bar;
    size_t* progress_ptr = nullptr;

    if (show_progress) {
        // 确定有效的线程数
        int actual_threads = threads < 0 ? omp_get_max_threads() : 
                           (threads > omp_get_max_threads() ? omp_get_max_threads() : threads);
        
        tracker = std::make_unique<ProgressTracker>(actual_threads);
        progress_ptr = tracker->ptr();

        // 计算各阶段的总量
        const size_t u_tie_total = C;
        const size_t p_total = C * n_targets;
        std::string u_tie_name = "Calc U&tie";
        if (!opt.tie_correction && opt.method == MannWhitneyuOption::exact) {
            u_tie_name = "Calc U";
        }
        std::vector<ProgressBar::Stage> stages = {
            {"Calc U&tie", u_tie_total},
            {"Calc P", p_total}
        };

        progress_bar = std::make_unique<ProgressBar>(stages, *tracker, 100);
        progress_bar->start();
    }

    auto result = mannwhitneyu<T>(V, gid_ptr, n_targets, opt, threads, progress_ptr);

    if (progress_bar) {
        progress_bar->stop();
    }

    auto U1_arr = py::array_t<double>(result.U1.size(), result.U1.data());
    U1_arr.attr("flags").attr("writeable") = py::bool_(false);
    auto P_arr  = py::array_t<double>(result.P.size(), result.P.data());
    P_arr.attr("flags").attr("writeable") = py::bool_(false);
    return py::make_tuple(U1_arr, P_arr);
}

// ================= GroupMean 调用 =================
template<typename T>
py::array_t<double> call_group_mean_numpy(
    py::array_t<T> data,
    py::array_t<int64_t> indices,
    py::array_t<int64_t> indptr,
    py::array_t<int32_t> group_id,
    size_t n_groups,
    bool include_zeros,
    int threads,
    bool is_csr
) {
    const size_t R   = static_cast<size_t>(group_id.size());
    const size_t C   = static_cast<size_t>(indptr.size()) - 1;
    const size_t NNZ = static_cast<size_t>(data.size());

    size_t rows = is_csr ? C : R;
    size_t cols = is_csr ? R : C;

    auto V = view::CscView<T>::from_raw(
        data.data(), indices.data(), indptr.data(),
        rows, cols, NNZ
    );

    const int32_t* gid_ptr = group_id.data();

    auto result = group_mean<T>(V, gid_ptr, n_groups, include_zeros, threads);

    auto arr = py::array_t<double>(result.size(), result.data());
    arr.attr("flags").attr("writeable") = py::bool_(false);
    return arr;
}

// ---------------- dtype 分派宏 ----------------
#define DISPATCH_DTYPE_MWU(DATA, INDICES, INDPTR, GID, N_GROUPS, EXTRA, THREADS, IS_CSR, SHOW_PROGRESS, CALLER) \
    if (DATA.dtype().is(py::dtype::of<int8_t>())) \
        return CALLER<int8_t>(DATA.cast<py::array_t<int8_t>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR, SHOW_PROGRESS); \
    else if (DATA.dtype().is(py::dtype::of<uint8_t>())) \
        return CALLER<uint8_t>(DATA.cast<py::array_t<uint8_t>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR, SHOW_PROGRESS); \
    else if (DATA.dtype().is(py::dtype::of<int16_t>())) \
        return CALLER<int16_t>(DATA.cast<py::array_t<int16_t>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR, SHOW_PROGRESS); \
    else if (DATA.dtype().is(py::dtype::of<uint16_t>())) \
        return CALLER<uint16_t>(DATA.cast<py::array_t<uint16_t>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR, SHOW_PROGRESS); \
    else if (DATA.dtype().is(py::dtype::of<int32_t>())) \
        return CALLER<int32_t>(DATA.cast<py::array_t<int32_t>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR, SHOW_PROGRESS); \
    else if (DATA.dtype().is(py::dtype::of<uint32_t>())) \
        return CALLER<uint32_t>(DATA.cast<py::array_t<uint32_t>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR, SHOW_PROGRESS); \
    else if (DATA.dtype().is(py::dtype::of<int64_t>())) \
        return CALLER<int64_t>(DATA.cast<py::array_t<int64_t>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR, SHOW_PROGRESS); \
    else if (DATA.dtype().is(py::dtype::of<uint64_t>())) \
        return CALLER<uint64_t>(DATA.cast<py::array_t<uint64_t>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR, SHOW_PROGRESS); \
    else if (DATA.dtype().is(py::dtype::of<float>())) \
        return CALLER<float>(DATA.cast<py::array_t<float>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR, SHOW_PROGRESS); \
    else if (DATA.dtype().is(py::dtype::of<double>())) \
        return CALLER<double>(DATA.cast<py::array_t<double>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR, SHOW_PROGRESS); \
    else \
        throw std::runtime_error("Unsupported dtype for data");

#define DISPATCH_DTYPE_GM(DATA, INDICES, INDPTR, GID, N_GROUPS, EXTRA, THREADS, IS_CSR, CALLER) \
    if (DATA.dtype().is(py::dtype::of<int8_t>())) \
        return CALLER<int8_t>(DATA.cast<py::array_t<int8_t>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR); \
    else if (DATA.dtype().is(py::dtype::of<uint8_t>())) \
        return CALLER<uint8_t>(DATA.cast<py::array_t<uint8_t>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR); \
    else if (DATA.dtype().is(py::dtype::of<int16_t>())) \
        return CALLER<int16_t>(DATA.cast<py::array_t<int16_t>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR); \
    else if (DATA.dtype().is(py::dtype::of<uint16_t>())) \
        return CALLER<uint16_t>(DATA.cast<py::array_t<uint16_t>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR); \
    else if (DATA.dtype().is(py::dtype::of<int32_t>())) \
        return CALLER<int32_t>(DATA.cast<py::array_t<int32_t>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR); \
    else if (DATA.dtype().is(py::dtype::of<uint32_t>())) \
        return CALLER<uint32_t>(DATA.cast<py::array_t<uint32_t>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR); \
    else if (DATA.dtype().is(py::dtype::of<int64_t>())) \
        return CALLER<int64_t>(DATA.cast<py::array_t<int64_t>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR); \
    else if (DATA.dtype().is(py::dtype::of<uint64_t>())) \
        return CALLER<uint64_t>(DATA.cast<py::array_t<uint64_t>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR); \
    else if (DATA.dtype().is(py::dtype::of<float>())) \
        return CALLER<float>(DATA.cast<py::array_t<float>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR); \
    else if (DATA.dtype().is(py::dtype::of<double>())) \
        return CALLER<double>(DATA.cast<py::array_t<double>>(), INDICES.cast<py::array_t<int64_t>>(), INDPTR.cast<py::array_t<int64_t>>(), GID.cast<py::array_t<int32_t>>(), N_GROUPS, EXTRA, THREADS, IS_CSR); \
    else \
        throw std::runtime_error("Unsupported dtype for data");

// ---------------- PyBind ----------------
PYBIND11_MODULE(kernel, m) {
    m.doc() = "HPDEx C++ extension module (MWU + GroupMean, unified interface)";

    m.def("mannwhitneyu",
        [](py::array data, py::array indices, py::array indptr, py::array group_id,
           size_t n_targets, bool ref_sorted, bool tar_sorted, bool tie_correction,
           bool use_continuity, bool fast_norm, int zero_handling,
           int alternative, int method, int threads, std::string layout, bool show_progress) {
            MannWhitneyuOption opt{};
            opt.ref_sorted     = ref_sorted;
            opt.tar_sorted     = tar_sorted;
            opt.tie_correction = tie_correction;
            opt.use_continuity = use_continuity;
            opt.fast_norm      = fast_norm;
            opt.zero_handling  = static_cast<MannWhitneyuOption::ZeroHandling>(zero_handling);
            opt.alternative    = static_cast<MannWhitneyuOption::Alternative>(alternative);
            opt.method         = static_cast<MannWhitneyuOption::Method>(method);

            bool is_csr = (layout == "csr");
            DISPATCH_DTYPE_MWU(data, indices, indptr, group_id,
                               n_targets, opt, threads, is_csr, show_progress, call_mwu_numpy);
        },
        py::arg("data"), py::arg("indices"), py::arg("indptr"), py::arg("group_id"),
        py::arg("n_targets"),
        py::arg("ref_sorted") = false,
        py::arg("tar_sorted") = false,
        py::arg("tie_correction") = true,
        py::arg("use_continuity") = false,
        py::arg("fast_norm") = true,
        py::arg("zero_handling") = 0,
        py::arg("alternative") = 2,
        py::arg("method") = 2,
        py::arg("threads") = -1,
        py::arg("layout") = "csc",
        py::arg("show_progress") = false
    );

    m.def("group_mean",
        [](py::array data, py::array indices, py::array indptr, py::array group_id,
           size_t n_groups, bool include_zeros, int threads, std::string layout) {
            bool is_csr = (layout == "csr");
            DISPATCH_DTYPE_GM(data, indices, indptr, group_id,
                           n_groups, include_zeros, threads, is_csr, call_group_mean_numpy);
        },
        py::arg("data"), py::arg("indices"), py::arg("indptr"), py::arg("group_id"),
        py::arg("n_groups"), py::arg("include_zeros") = true,
        py::arg("threads") = -1,
        py::arg("layout") = "csc"
    );
}

} // namespace hpdex
