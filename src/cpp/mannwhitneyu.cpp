// mannwhitneyu.cpp
// mwu cpu 算子实现
#include "mannwhitneyu.hpp"
#include "common.hpp"
#include "config.hpp"
#include "macro.hpp"
#include "simd.hpp"

namespace hpdex {

template<class T>
void array_p_asymptotic_fast_parallel(
    const T* U1, const T* n1, const T* n2,
    const T* tie_sum, const T* cc,
    T* out, size_t N,
    MannWhitneyuOption::Alternative alt
) {
    using D = hn::D<T>;
    D d;
    const size_t step = Lanes(d);

    // -------- chunk 大小选择 --------
    const size_t chunk_align = 64 / sizeof(T); // 64B 对齐需要多少元素
    const size_t chunk_size  = std::max(step, chunk_align) * 128; // 手动放大，避免太小

    // -------- 并行划分 --------
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();

        size_t chunk_per_thread = (N + nth - 1) / nth;
        // 对齐到 step，保证前面的 chunk 是 step 的倍数
        size_t begin = tid * chunk_per_thread;
        size_t end   = std::min(N, begin + chunk_per_thread);

        // 对齐 begin 到 step 的倍数（便于 SIMD）
        if (tid > 0) {
            size_t misalign = begin % step;
            if (misalign != 0) {
                begin += (step - misalign);
                if (begin > end) begin = end;
            }
        }

        // -------- 执行对应版本 --------
        switch (alt) {
            case Alternative::TwoSided:
                array_p_asymptotic_two_sided_fast<T>(
                    U1 + begin, n1 + begin, n2 + begin,
                    tie_sum + begin, cc + begin,
                    out + begin, end - begin
                );
                break;
            case Alternative::Greater:
                array_p_asymptotic_greater_fast<T>(
                    U1 + begin, n1 + begin, n2 + begin,
                    tie_sum + begin, cc + begin,
                    out + begin, end - begin
                );
                break;
            case Alternative::Less:
                array_p_asymptotic_less_fast<T>(
                    U1 + begin, n1 + begin, n2 + begin,
                    tie_sum + begin, cc + begin,
                    out + begin, end - begin
                );
                break;
        }
    }
}

}