// cuda_tools.hpp
#pragma once
#include "macro.hpp"

template<class T>
force_inline_ void copy_to_cuda(
    const T* src,
    T* dist,
    const size_t& n
);