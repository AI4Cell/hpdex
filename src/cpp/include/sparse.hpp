// sparse.hpp - Sparse Matrix View
#pragma once
#include "macro.hpp"

#include <torch/torch.h>
#include <Eigen/Sparse>

namespace hpdex {

namespace view {
struct CsrView {
    torch::Tensor data_;
    torch::Tensor indices_;
    torch::Tensor indptr_;
    size_t rows_;
    size_t cols_;
    size_t nnz_;

    force_inline_ torch::Tensor data() const { return data_; }
    force_inline_ torch::Tensor indices() const { return indices_; }
    force_inline_ torch::Tensor indptr() const { return indptr_; }
    force_inline_ size_t rows() const { return rows_; }
    force_inline_ size_t cols() const { return cols_; }
    force_inline_ size_t nnz() const { return nnz_; }

    static force_inline_ CsrView from_torch(torch::Tensor data, torch::Tensor indices, torch::Tensor indptr, size_t rows, size_t cols, size_t nnz) {
        indices = indices.to(torch::kInt64);
        indptr = indptr.to(torch::kInt64);
        return CsrView(data, indices, indptr, rows, cols, nnz);
    }

    template<class T>
    force_inline_ Eigen::SparseMatrix<T> to_eigen() const {
        if unlikely_(data_.dtype() != torch::dtype<T>) {
            throw std::runtime_error("[CsrView::to_eigen] mismatch dtype");
        }
        const T* data_ptr = data_.data_ptr<T>();
        const int64_t* indices_ptr = indices_.data_ptr<int64_t>();
        const int64_t* indptr_ptr = indptr_.data_ptr<int64_t>();

        // 拷贝 indptr, indices, data 到新的内存，由Eigen持有
        std::vector<Eigen::Index> outer(rows_ + 1);
        std::vector<Eigen::Index> inner(nnz_);
        std::vector<T> values(nnz_);

        std::memcpy(outer.data(), indptr_ptr, sizeof(Eigen::Index) * (rows_ + 1));
        std::memcpy(inner.data(), indices_ptr, sizeof(Eigen::Index) * nnz_);
        std::memcpy(values.data(), data_ptr, sizeof(T) * nnz_);

        Eigen::Map<Eigen::SparseMatrix<T, Eigen::RowMajor, Eigen::Index>> mapped(
            rows_, cols_, nnz_,
            outer.data(), inner.data(), values.data()
        );
        Eigen::SparseMatrix<T, Eigen::RowMajor, Eigen::Index> mat = mapped;
        return mat;
    }

private:
    CsrView(torch::Tensor data, torch::Tensor indices, torch::Tensor indptr, size_t rows, size_t cols, size_t nnz)
        : data_(data), indices_(indices), indptr_(indptr), rows_(rows), cols_(cols), nnz_(nnz) {}
};

struct CscView {
    torch::Tensor data_;
    torch::Tensor indices_;      // 行索引 (row indices) int64_t
    torch::Tensor indptr_;       // 列指针 (column pointers) int64_t
    size_t rows_;
    size_t cols_;
    size_t nnz_;

    force_inline_ torch::Tensor data() const { return data_; }
    force_inline_ torch::Tensor indices() const { return indices_; }
    force_inline_ torch::Tensor indptr() const { return indptr_; }
    force_inline_ size_t rows() const { return rows_; }
    force_inline_ size_t cols() const { return cols_; }
    force_inline_ size_t nnz() const { return nnz_; }

    static force_inline_ CscView from_torch(torch::Tensor data, torch::Tensor indices, torch::Tensor indptr, size_t rows, size_t cols, size_t nnz) {
        indices = indices.to(torch::kInt64).to(torch::kCPU);
        indptr = indptr.to(torch::kInt64).to(torch::kCPU);
        return CscView(data, indices, indptr, rows, cols, nnz);
    }

    template<class T>
    force_inline_ Eigen::SparseMatrix<T> to_eigen() const {
        if unlikely_(data_.dtype() != torch::dtype<T>) {
            throw std::runtime_error("[CscView::to_eigen] mismatch dtype");
        }
        const T* data_ptr = data_.data_ptr<T>();
        const int64_t* indices_ptr = indices_.data_ptr<int64_t>();
        const int64_t* indptr_ptr = indptr_.data_ptr<int64_t>();

        // 拷贝 indptr, indices, data 到新的内存，由Eigen持有
        std::vector<Eigen::Index> outer(cols_ + 1);  // 列数+1
        std::vector<Eigen::Index> inner(nnz_);
        std::vector<T> values(nnz_);

        std::memcpy(outer.data(), indptr_ptr, sizeof(Eigen::Index) * (cols_ + 1));
        std::memcpy(inner.data(), indices_ptr, sizeof(Eigen::Index) * nnz_);
        std::memcpy(values.data(), data_ptr, sizeof(T) * nnz_);

        // 注意：Eigen 默认就是 ColMajor (CSC格式)，所以不需要指定 RowMajor
        Eigen::Map<Eigen::SparseMatrix<T, Eigen::ColMajor, Eigen::Index>> mapped(
            rows_, cols_, nnz_,
            outer.data(), inner.data(), values.data()
        );
        Eigen::SparseMatrix<T, Eigen::ColMajor, Eigen::Index> mat = mapped;
        return mat;
    }

private:
    CscView(torch::Tensor data, torch::Tensor indices, torch::Tensor indptr, size_t rows, size_t cols, size_t nnz)
        : data_(data), indices_(indices), indptr_(indptr), rows_(rows), cols_(cols), nnz_(nnz) {}
};
}

}
