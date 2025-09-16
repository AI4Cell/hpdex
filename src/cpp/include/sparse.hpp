// sparse.hpp - Minimal Sparse Matrix Views (no torch, no eigen)
#pragma once
#include "macro.hpp"
#include <cstddef>
#include <cstdint>

namespace hpdex {
namespace view {

// ================= CSR View =================
template<class T>
struct CsrView {
    const T*       data_;     // 非零元素数组 (nnz)
    const int64_t* indices_;  // 列索引 (nnz)
    const int64_t* indptr_;   // 行指针 (rows+1)
    size_t rows_;
    size_t cols_;
    size_t nnz_;

    force_inline_ const T*       data()    const { return data_; }
    force_inline_ const int64_t* indices() const { return indices_; }
    force_inline_ const int64_t* indptr()  const { return indptr_; }
    force_inline_ size_t rows()  const { return rows_; }
    force_inline_ size_t cols()  const { return cols_; }
    force_inline_ size_t nnz()   const { return nnz_; }

    static force_inline_ CsrView from_raw(const T* data,
                            const int64_t* indices,
                            const int64_t* indptr,
                            size_t rows, size_t cols, size_t nnz) {
        return CsrView(data, indices, indptr, rows, cols, nnz);
    }

private:
    CsrView(const T* d, const int64_t* i, const int64_t* p,
            size_t r, size_t c, size_t n)
        : data_(d), indices_(i), indptr_(p), rows_(r), cols_(c), nnz_(n) {}
};

// ================= CSC View =================
template<class T>
struct CscView {
    const T*       data_;     // 非零元素数组 (nnz)
    const int64_t* indices_;  // 行索引 (nnz)
    const int64_t* indptr_;   // 列指针 (cols+1)
    size_t rows_;
    size_t cols_;
    size_t nnz_;

    force_inline_ const T*       data()    const { return data_; }
    force_inline_ const int64_t* indices() const { return indices_; }
    force_inline_ const int64_t* indptr()  const { return indptr_; }
    force_inline_ size_t rows()  const { return rows_; }
    force_inline_ size_t cols()  const { return cols_; }
    force_inline_ size_t nnz()   const { return nnz_; }

    static force_inline_ CscView from_raw(const T* data,
                            const int64_t* indices,
                            const int64_t* indptr,
                            size_t rows, size_t cols, size_t nnz) {
        return CscView(data, indices, indptr, rows, cols, nnz);
    }

private:
    CscView(const T* d, const int64_t* i, const int64_t* p,
            size_t r, size_t c, size_t n)
        : data_(d), indices_(i), indptr_(p), rows_(r), cols_(c), nnz_(n) {}
};

} // namespace view
} // namespace hpdex
