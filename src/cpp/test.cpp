#include "macro.hpp"
#include "simd.hpp"
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <random>

PROJECT_BEGIN

// return {clip, max}
force_inline_ std::pair<size_t, size_t> norm(
    const std::vector<uint32_t>& bucket,
    const double k,
    const size_t max_clip,
    const int threads = 1
) {
    if unlikely_(bucket.empty()) return {0, 0};

    const uint32_t* bucket_ptr = bucket.data();
    const size_t n = bucket.size();

    size_t global_max   = 0;
    double global_sum   = 0.0;
    double global_sqr_sum = 0.0;

    const size_t tail = n % simd::lanes<uint32_t>();
    // 提前处理tail
    if likely_(tail > 0) {
        auto m = simd::mask_from_count<uint32_t>(tail);
        auto v = simd::masked_load(m, bucket_ptr + n - tail);
        global_sum     += static_cast<double>(simd::reduce_sum<uint32_t>(v));
        global_sqr_sum += static_cast<double>(
            simd::reduce_sum<uint32_t>(simd::mul<uint32_t>(v, v)));
        const size_t max_v = simd::reduce_max<uint32_t>(v);
        global_max = max_v;
    }

    // ===== 并行 SIMD 规约 =====
    PARALLEL_REGION(threads)
{
    PARALLEL_FOR(static, reduction(+:global_sum,global_sqr_sum) reduction(max:global_max),
    for (size_t i = tail; i < n; i += simd::lanes<uint32_t>()) {
        const size_t step = simd::lanes<uint32_t>();
        auto v = simd::load(bucket_ptr + i);

        // 局部累积
        global_sum     += static_cast<double>(simd::reduce_sum<uint32_t>(v));
        global_sqr_sum += static_cast<double>(
            simd::reduce_sum<uint32_t>(simd::mul<uint32_t>(v, v)));
        const size_t max_v = simd::reduce_max<uint32_t>(v);
        global_max = global_max > max_v ? global_max : max_v;
    })
}
    if (global_max < max_clip) return {global_max, global_max};

    // ===== 均值 / 标准差 =====
    const double mean = global_sum / n;
    double variance   = global_sqr_sum / n - mean * mean;
    variance = variance < 0.0 ? 0.0 : variance;  // 数值稳定保护
    const double stdev = std::sqrt(variance);

    // ===== μ + kσ, 至少返回 mean =====
    const double cutoff = std::max(mean + k * stdev, mean);
    return {static_cast<size_t>(std::ceil(cutoff)), global_max};
}

// 朴素算法版本用于验证
force_inline_ std::pair<size_t, size_t> norm_naive(
    const std::vector<uint32_t>& bucket,
    const double k,
    const size_t max_clip
) {
    if (bucket.empty()) return {0, 0};

    const size_t n = bucket.size();
    
    // 朴素计算：逐个元素遍历
    size_t global_max = 0;
    double global_sum = 0.0;
    double global_sqr_sum = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        const uint32_t val = bucket[i];
        global_sum += static_cast<double>(val);
        global_sqr_sum += static_cast<double>(val * val);
        global_max = global_max > val ? global_max : val;
    }
    
    if (global_max < max_clip) return {global_max, global_max};
    
    // 计算统计量
    const double mean = global_sum / n;
    double variance = global_sqr_sum / n - mean * mean;
    variance = variance < 0.0 ? 0.0 : variance;
    const double stdev = std::sqrt(variance);
    
    const double cutoff = std::max(mean + k * stdev, mean);
    return {static_cast<size_t>(std::ceil(cutoff)), global_max};
}

PROJECT_END

// 测试函数
void test_algorithm_consistency() {
    std::cout << "=== 算法一致性验证测试 ===" << std::endl;
    
    // 测试用例
    std::vector<std::vector<uint32_t>> test_cases = {
        {1, 2, 3, 4, 5},
        {10, 20, 30, 40, 50},
        {1, 1, 1, 1, 1},  // 相同值
        {1},              // 单个元素
        {},               // 空数组
        {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000},  // 大数值
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},  // SIMD对齐
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},  // SIMD不对齐
    };
    
    std::vector<double> k_values = {0.5, 1.0, 2.0, 3.0};
    std::vector<size_t> max_clips = {5, 10, 100, 1000};
    
    bool all_passed = true;
    int test_count = 0;
    
    for (const auto& bucket : test_cases) {
        for (double k : k_values) {
            for (size_t max_clip : max_clips) {
                test_count++;
                
                auto [clip_simd, max_simd] = hpdex::norm(bucket, k, max_clip);
                auto [clip_naive, max_naive] = hpdex::norm_naive(bucket, k, max_clip);
                
                bool clip_match = (clip_simd == clip_naive);
                bool max_match = (max_simd == max_naive);
                
                if (!clip_match || !max_match) {
                    all_passed = false;
                    std::cout << "❌ 测试失败 #" << test_count << std::endl;
                    std::cout << "   输入: [";
                    for (size_t i = 0; i < bucket.size(); ++i) {
                        std::cout << bucket[i];
                        if (i < bucket.size() - 1) std::cout << ", ";
                    }
                    std::cout << "], k=" << k << ", max_clip=" << max_clip << std::endl;
                    std::cout << "   SIMD:  clip=" << clip_simd << ", max=" << max_simd << std::endl;
                    std::cout << "   朴素:  clip=" << clip_naive << ", max=" << max_naive << std::endl;
                    std::cout << "   差异:  clip" << (clip_match ? "✓" : "✗") 
                              << ", max" << (max_match ? "✓" : "✗") << std::endl;
                    std::cout << std::endl;
                }
            }
        }
    }
    
    if (all_passed) {
        std::cout << "✅ 所有 " << test_count << " 个测试用例通过！SIMD和朴素算法结果完全一致。" << std::endl;
    } else {
        std::cout << "❌ 发现不一致的结果，需要调试SIMD算法。" << std::endl;
    }
    
    std::cout << std::endl;
}

// 性能对比测试
void test_performance() {
    std::cout << "=== 性能对比测试 ===" << std::endl;
    
    // 生成测试数据
    const size_t data_size = 1000000;
    std::vector<uint32_t> large_bucket(data_size);
    for (size_t i = 0; i < data_size; ++i) {
        large_bucket[i] = static_cast<uint32_t>(rand() % 1000 + 1);
    }
    
    const double k = 2.0;
    const size_t max_clip = 1000;
    const int iterations = 100;
    
    // 测试朴素算法
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto [clip, max] = hpdex::norm_naive(large_bucket, k, max_clip);
        (void)clip; (void)max;  // 避免优化
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto naive_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    // 测试SIMD算法
    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto [clip, max] = hpdex::norm(large_bucket, k, max_clip, 4);  // 使用4线程
        (void)clip; (void)max;  // 避免优化
    }
    end_time = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    std::cout << "数据大小: " << data_size << " 个元素" << std::endl;
    std::cout << "迭代次数: " << iterations << std::endl;
    std::cout << "朴素算法时间: " << naive_time << " 微秒" << std::endl;
    std::cout << "SIMD算法时间: " << simd_time << " 微秒" << std::endl;
    std::cout << "加速比: " << (double)naive_time / simd_time << "x" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "HPDEX 算法验证测试" << std::endl;
    std::cout << "==================" << std::endl;
    std::cout << std::endl;
    
    // 简单测试
    std::vector<uint32_t> bucket = {1, 2, 3, 4, 5};
    auto [clip, max] = hpdex::norm(bucket, 0.5, 5);
    auto [clip_naive, max_naive] = hpdex::norm_naive(bucket, 0.5, 5);
    
    std::cout << "简单测试示例:" << std::endl;
    std::cout << "输入: [1, 2, 3, 4, 5], k=0.5, max_clip=5" << std::endl;
    std::cout << "SIMD:  clip=" << clip << ", max=" << max << std::endl;
    std::cout << "朴素:  clip=" << clip_naive << ", max=" << max_naive << std::endl;
    std::cout << "结果: " << (clip == clip_naive && max == max_naive ? "一致 ✓" : "不一致 ✗") << std::endl;
    std::cout << std::endl;
    
    // 一致性验证
    test_algorithm_consistency();
    
    // 性能对比
    test_performance();
    
    return 0;
}