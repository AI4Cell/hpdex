// cpu_tools.hpp
#pragma once
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>  // CUDAStream, guards

// 将裸指针 src[0:n) 拷贝到 dst_gpu 的 [dst_offset : dst_offset+n) 区间
// 仅使用 torch 的接口（copy_ + CUDA stream），可选分块 chunk_elems
template <class T>
inline void copy_cpu_segment_to_gpu(
    const T* src,                      // CPU 原始内存（可普通内存）
    size_t n,                          // 元素个数
    torch::Tensor& dst_gpu,            // 目标 GPU Tensor（预分配好）
    size_t dst_offset,                 // 目标偏移（以元素为单位）
    size_t chunk_elems = 0,            // 分块大小；0 表示一次性
    c10::optional<at::cuda::CUDAStream> stream_opt = c10::nullopt
) {
    TORCH_CHECK(dst_gpu.is_cuda(), "dst_gpu must be a CUDA tensor");
    TORCH_CHECK(dst_gpu.scalar_type() == c10::CppTypeToScalarType<T>::value,
                "dtype mismatch between T and dst_gpu");
    TORCH_CHECK(dst_offset + n <= static_cast<size_t>(dst_gpu.numel()),
                "copy range exceeds dst_gpu size");

    if (!torch::cuda::is_available()) {
        TORCH_CHECK(false, "CUDA not available: guard before calling this function.");
    }

    // 选择流（若未指定则用当前流）
    at::cuda::CUDAStream stream = stream_opt.has_value()
        ? *stream_opt
        : at::cuda::getCurrentCUDAStream();

    at::cuda::CUDAStreamGuard guard(stream);

    // 目标视图（展平后做窄切片）
    auto dst_flat = dst_gpu.view({-1});

    // 是否分块
    const size_t step = (chunk_elems == 0 || chunk_elems >= n) ? n : chunk_elems;
    size_t copied = 0;

    while (copied < n) {
        const size_t cur = std::min(step, n - copied);

        // 为了启用真正的异步 H2D，使用 pinned host bounce buffer
        auto cpu_bounce = torch::empty(
            {static_cast<long>(cur)},
            torch::TensorOptions()
                .dtype(dst_gpu.dtype())
                .device(torch::kCPU)
                .pinned_memory(true)  // 关键：页锁定内存
        );

        // 将用户的 src 拷到 pinned buffer（CPU memcpy）
        std::memcpy(cpu_bounce.data_ptr<T>(), src + copied, cur * sizeof(T));

        // 目标 GPU 切片：[dst_offset+copied, cur)
        auto dst_slice = dst_flat.narrow(0, static_cast<long>(dst_offset + copied),
                                            static_cast<long>(cur));

        // 使用 torch 的 copy_，并启用 non_blocking（配合 pinned memory 即可异步）
        dst_slice.copy_(cpu_bounce, /*non_blocking=*/true);

        copied += cur;
    }

    // 交由调用方决定是否同步：
    // at::cuda::synchronize(stream);
}
