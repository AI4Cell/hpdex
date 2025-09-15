// src/cpp/module.cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <variant>
#include "sparse.hpp"
#include "mannwhitneyu.hpp"

namespace py = pybind11;
namespace hpdex {

PYBIND11_MODULE(hpdex_cpp, m) {
    m.doc() = "HPDEx C++ extension module (Mann-Whitney U test)";

    // -------- CSC 接口 --------
    m.def("mannwhitneyu_csc",
        [](const torch::Tensor& data,
           const torch::Tensor& indices,
           const torch::Tensor& indptr,
           const torch::Tensor& group_id,
           size_t n_groups,
           bool ref_sorted,
           bool tar_sorted,
           bool tie_correction,
           bool use_continuity,
           int alternative,   // 0=less, 1=greater, 2=two_sided
           int method,        // 1=exact, 2=asymptotic
           int threads) -> MWUResult
        {
            TORCH_CHECK(indices.scalar_type() == torch::kLong, "indices must be int64");
            TORCH_CHECK(indptr.scalar_type()  == torch::kLong, "indptr must be int64");
            TORCH_CHECK(group_id.scalar_type() == torch::kInt, "group_id must be int32");

            MannWhitneyuOption opt;
            opt.ref_sorted     = ref_sorted;
            opt.tar_sorted     = tar_sorted;
            opt.tie_correction = tie_correction;
            opt.use_continuity = use_continuity;
            opt.alternative    = static_cast<MannWhitneyuOption::Alternative>(alternative);
            opt.method         = static_cast<MannWhitneyuOption::Method>(method);

            py::gil_scoped_release release;
            return mannwhitneyu(
                std::variant<hpdex::view::CscView, hpdex::view::CsrView>(
                    hpdex::view::CscView::from_torch(
                        data, indices, indptr,
                        (size_t)group_id.size(0),
                        (size_t)indptr.size(0) - 1,
                        (size_t)data.size(0)
                    )
                ),
                group_id, n_groups, opt, threads, nullptr
            );
        },
        py::arg("data"), py::arg("indices"), py::arg("indptr"),
        py::arg("group_id"), py::arg("n_groups"),
        py::arg("ref_sorted") = false,
        py::arg("tar_sorted") = false,
        py::arg("tie_correction") = true,
        py::arg("use_continuity") = false,
        py::arg("alternative") = 2,
        py::arg("method") = 2,
        py::arg("threads") = -1
    );

    // -------- CSR 接口 --------
    m.def("mannwhitneyu_csr",
        [](const torch::Tensor& data,
           const torch::Tensor& indices,
           const torch::Tensor& indptr,
           const torch::Tensor& group_id,
           size_t n_groups,
           bool ref_sorted,
           bool tar_sorted,
           bool tie_correction,
           bool use_continuity,
           int alternative,
           int method,
           int threads) -> MWUResult
        {
            TORCH_CHECK(indices.scalar_type() == torch::kLong, "indices must be int64");
            TORCH_CHECK(indptr.scalar_type()  == torch::kLong, "indptr must be int64");
            TORCH_CHECK(group_id.scalar_type() == torch::kInt, "group_id must be int32");

            MannWhitneyuOption opt;
            opt.ref_sorted     = ref_sorted;
            opt.tar_sorted     = tar_sorted;
            opt.tie_correction = tie_correction;
            opt.use_continuity = use_continuity;
            opt.alternative    = static_cast<MannWhitneyuOption::Alternative>(alternative);
            opt.method         = static_cast<MannWhitneyuOption::Method>(method);

            py::gil_scoped_release release;
            return mannwhitneyu(
                std::variant<hpdex::view::CscView, hpdex::view::CsrView>(
                    hpdex::view::CsrView::from_torch(
                        data, indices, indptr,
                        (size_t)group_id.size(0),
                        (size_t)indptr.size(0) - 1,
                        (size_t)data.size(0)
                    )
                ),
                group_id, n_groups, opt, threads, nullptr
            );
        },
        py::arg("data"), py::arg("indices"), py::arg("indptr"),
        py::arg("group_id"), py::arg("n_groups"),
        py::arg("ref_sorted") = false,
        py::arg("tar_sorted") = false,
        py::arg("tie_correction") = true,
        py::arg("use_continuity") = false,
        py::arg("alternative") = 2,
        py::arg("method") = 2,
        py::arg("threads") = -1
    );
}

}