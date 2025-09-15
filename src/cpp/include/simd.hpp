// simd.hpp - Comprehensive SIMD Wrapper (Highway + Scalar fallback)
// Dependencies: config.hpp, macro.hpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <cstring>

#include "config.hpp" // HWY_STATIC_DEFINE
#include "macro.hpp"

// ================================================================
//      Conditional include Highway or fallback
// ================================================================
#if USE_HIGHWAY
  #include <hwy/highway.h> 
  #include <hwy/contrib/math/math-inl.h>
  #include <hwy/contrib/sort/vqsort-inl.h>
  namespace hn = hwy::HWY_NAMESPACE;
#endif

PROJECT_BEGIN
namespace simd {

// ================================================================
//      Type Traits & Utilities
// ================================================================
template<typename T>
struct is_supported_type : std::false_type {};

template<> struct is_supported_type<float>    : std::true_type {};
template<> struct is_supported_type<double>   : std::true_type {};
template<> struct is_supported_type<int8_t>   : std::true_type {};
template<> struct is_supported_type<uint8_t>  : std::true_type {};
template<> struct is_supported_type<int16_t>  : std::true_type {};
template<> struct is_supported_type<uint16_t> : std::true_type {};
template<> struct is_supported_type<int32_t>  : std::true_type {};
template<> struct is_supported_type<uint32_t> : std::true_type {};
template<> struct is_supported_type<int64_t>  : std::true_type {};
template<> struct is_supported_type<uint64_t> : std::true_type {};

template<typename T>
constexpr bool is_supported_type_v = is_supported_type<T>::value;

template<typename T>
struct is_floating_point : std::is_floating_point<T> {};

template<typename T>
constexpr bool is_floating_point_v = is_floating_point<T>::value;

template<typename T>
struct is_signed_integer : std::integral_constant<bool,
    std::is_integral_v<T> && std::is_signed_v<T>> {};

template<typename T>
constexpr bool is_signed_integer_v = is_signed_integer<T>::value;

#if USE_HIGHWAY
// ================================================================
//      SIMD Implementation (Highway)
// ================================================================
template<class T> using D    = hn::ScalableTag<T>;
template<class T> using vec  = decltype(hn::Zero(D<T>{}));
template<class T> using mask = hn::Mask<D<T>>;

// ---- Vector Properties ----
template<class T>
force_inline_ constexpr size_t lanes() { return hn::Lanes(D<T>{}); }

template<class T>
force_inline_ constexpr size_t max_lanes() { return hn::MaxLanes(D<T>{}); }

// ---- Initialization ----
template<class T>
force_inline_ vec<T> zero() { return hn::Zero(D<T>{}); }

template<class T>
force_inline_ vec<T> set1(T value) { return hn::Set(D<T>{}, value); }

template<class T>
force_inline_ vec<T> iota(T start = 0) { return hn::Iota(D<T>{}, start); }

template<class T>
force_inline_ vec<T> undefined() { return hn::Undefined(D<T>{}); }

// ---- Load / Store ----
template<class T>
force_inline_ vec<T> load(const T* p) { return hn::Load(D<T>{}, p); }

template<class T>
force_inline_ vec<T> load_aligned(const T* p) { return hn::Load(D<T>{}, p); }

template<class T>
force_inline_ vec<T> load_unaligned(const T* p) { return hn::LoadU(D<T>{}, p); }

template<class T>
force_inline_ void store(const vec<T>& v, T* p) { hn::Store(v, D<T>{}, p); }

template<class T>
force_inline_ void store_aligned(const vec<T>& v, T* p) { hn::Store(v, D<T>{}, p); }

template<class T>
force_inline_ void store_unaligned(const vec<T>& v, T* p) { hn::StoreU(v, D<T>{}, p); }

// ---- Masked Load / Store ----
template<class T>
force_inline_ vec<T> masked_load(mask<T> m, const T* p, vec<T> v0 = zero<T>()) {
    return hn::MaskedLoad(m, D<T>{}, p);
}

template<class T>
force_inline_ void masked_store(mask<T> m, const vec<T>& v, T* p) {
    hn::Store(v, D<T>{}, p);
}

// ---- Gather / Scatter ----
template<class T, class TI>
force_inline_ vec<T> gather(const T* base, const vec<TI>& indices) {
    return hn::GatherIndex(D<T>{}, base, indices);
}

template<class T, class TI>
force_inline_ void scatter(const vec<T>& v, T* base, const vec<TI>& indices) {
    hn::ScatterIndex(v, D<T>{}, base, indices);
}

// ---- Arithmetic Operations ----
template<class T>
force_inline_ vec<T> add(const vec<T>& a, const vec<T>& b) { return hn::Add(a, b); }

template<class T>
force_inline_ vec<T> sub(const vec<T>& a, const vec<T>& b) { return hn::Sub(a, b); }

template<class T>
force_inline_ vec<T> mul(const vec<T>& a, const vec<T>& b) { return hn::Mul(a, b); }

template<class T>
force_inline_ vec<T> div(const vec<T>& a, const vec<T>& b) { return hn::Div(a, b); }

template<class T>
force_inline_ vec<T> neg(const vec<T>& v) { return hn::Neg(v); }

template<class T>
force_inline_ vec<T> abs(const vec<T>& v) { return hn::Abs(v); }

// FMA (Fused Multiply-Add): a*b + c
template<class T>
force_inline_ vec<T> fma(const vec<T>& a, const vec<T>& b, const vec<T>& c) {
    return hn::MulAdd(a, b, c);
}

// FMS (Fused Multiply-Subtract): a*b - c
template<class T>
force_inline_ vec<T> fms(const vec<T>& a, const vec<T>& b, const vec<T>& c) {
    return hn::MulSub(a, b, c);
}

// FNMA (Fused Negative Multiply-Add): -(a*b) + c
template<class T>
force_inline_ vec<T> fnma(const vec<T>& a, const vec<T>& b, const vec<T>& c) {
    return hn::NegMulAdd(a, b, c);
}

// FNMS (Fused Negative Multiply-Subtract): -(a*b) - c
template<class T>
force_inline_ vec<T> fnms(const vec<T>& a, const vec<T>& b, const vec<T>& c) {
    return hn::NegMulSub(a, b, c);
}

// ---- Min/Max Operations ----
template<class T>
force_inline_ vec<T> vmax(const vec<T>& a, const vec<T>& b) { return hn::Max(a, b); }

template<class T>
force_inline_ vec<T> vmin(const vec<T>& a, const vec<T>& b) { return hn::Min(a, b); }

template<class T>
force_inline_ vec<T> clamp(const vec<T>& v, const vec<T>& lo, const vec<T>& hi) {
    return hn::Clamp(v, lo, hi);
}

// ---- Comparison Operations ----
template<class T>
force_inline_ mask<T> eq(const vec<T>& a, const vec<T>& b) { return hn::Eq(a, b); }

template<class T>
force_inline_ mask<T> ne(const vec<T>& a, const vec<T>& b) { return hn::Ne(a, b); }

template<class T>
force_inline_ mask<T> lt(const vec<T>& a, const vec<T>& b) { return hn::Lt(a, b); }

template<class T>
force_inline_ mask<T> le(const vec<T>& a, const vec<T>& b) { return hn::Le(a, b); }

template<class T>
force_inline_ mask<T> gt(const vec<T>& a, const vec<T>& b) { return hn::Gt(a, b); }

template<class T>
force_inline_ mask<T> ge(const vec<T>& a, const vec<T>& b) { return hn::Ge(a, b); }

// ---- Logical Operations (Masks) ----
template<class T>
force_inline_ mask<T> mask_and(const mask<T>& a, const mask<T>& b) { return hn::And(a, b); }

template<class T>
force_inline_ mask<T> mask_or(const mask<T>& a, const mask<T>& b) { return hn::Or(a, b); }

template<class T>
force_inline_ mask<T> mask_xor(const mask<T>& a, const mask<T>& b) { return hn::Xor(a, b); }

template<class T>
force_inline_ mask<T> mask_not(const mask<T>& m) { return hn::Not(m); }

template<class T>
force_inline_ mask<T> mask_andnot(const mask<T>& a, const mask<T>& b) { return hn::AndNot(a, b); }

// ---- Bitwise Operations ----
template<class T>
force_inline_ vec<T> bit_and(const vec<T>& a, const vec<T>& b) { return hn::And(a, b); }

template<class T>
force_inline_ vec<T> bit_or(const vec<T>& a, const vec<T>& b) { return hn::Or(a, b); }

template<class T>
force_inline_ vec<T> bit_xor(const vec<T>& a, const vec<T>& b) { return hn::Xor(a, b); }

template<class T>
force_inline_ vec<T> bit_not(const vec<T>& v) { return hn::Not(v); }

template<class T>
force_inline_ vec<T> bit_andnot(const vec<T>& a, const vec<T>& b) { return hn::AndNot(a, b); }

// ---- Shift Operations ----
template<class T>
force_inline_ vec<T> shl(const vec<T>& v, int bits) { return hn::ShiftLeft<bits>(v); }

template<class T>
force_inline_ vec<T> shr(const vec<T>& v, int bits) { return hn::ShiftRight<bits>(v); }

template<class T>
force_inline_ vec<T> rotl(const vec<T>& v, int bits) { return hn::RotateLeft<bits>(v); }

template<class T>
force_inline_ vec<T> rotr(const vec<T>& v, int bits) { return hn::RotateRight<bits>(v); }

// ---- Select / Blend ----
template<class T>
force_inline_ vec<T> select(const mask<T>& m, const vec<T>& yes, const vec<T>& no) {
    return hn::IfThenElse(m, yes, no);
}

template<class T>
force_inline_ vec<T> select_zero_else(const mask<T>& m, const vec<T>& no) {
    return hn::IfThenZeroElse(m, no);
}

template<class T>
force_inline_ vec<T> select_else_zero(const mask<T>& m, const vec<T>& yes) {
    return hn::IfThenElseZero(m, yes);
}

// ---- Math Functions (Floating Point) ----
template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> sqrt(const vec<T>& v) { return hn::Sqrt(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> rsqrt(const vec<T>& v) { return hn::ApproximateReciprocalSqrt(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> reciprocal(const vec<T>& v) { return hn::ApproximateReciprocal(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> floor(const vec<T>& v) { return hn::Floor(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> ceil(const vec<T>& v) { return hn::Ceil(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> round(const vec<T>& v) { return hn::Round(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> trunc(const vec<T>& v) { return hn::Trunc(v); }

// Advanced math functions (if available in Highway contrib)
#ifdef HWY_CONTRIB_MATH_MATH_INL_H_
template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> exp(const vec<T>& v) { return hn::Exp(D<T>{}, v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> log(const vec<T>& v) { return hn::Log(D<T>{}, v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> sin(const vec<T>& v) { return hn::Sin(D<T>{}, v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> cos(const vec<T>& v) { return hn::Cos(D<T>{}, v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> tan(const vec<T>& v) { return hn::Tan(D<T>{}, v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> asin(const vec<T>& v) { return hn::Asin(D<T>{}, v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> acos(const vec<T>& v) { return hn::Acos(D<T>{}, v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> atan(const vec<T>& v) { return hn::Atan(D<T>{}, v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> tanh(const vec<T>& v) { return hn::Tanh(D<T>{}, v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> sinh(const vec<T>& v) { return hn::Sinh(D<T>{}, v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> cosh(const vec<T>& v) { return hn::Cosh(D<T>{}, v); }
#endif

// ---- Reduction Operations ----
template<class T>
force_inline_ T reduce_sum(const vec<T>& v) { return hn::ReduceSum(D<T>{}, v); }

template<class T>
force_inline_ T reduce_min(const vec<T>& v) { return hn::ReduceMin(D<T>{}, v); }

template<class T>
force_inline_ T reduce_max(const vec<T>& v) { return hn::ReduceMax(D<T>{}, v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ T reduce_mul(const vec<T>& v) { 
    // Highway doesn't have ReduceMul, implement using GetLane
    T result = 1;
    for (size_t i = 0; i < lanes<T>(); ++i) {
        result *= hn::ExtractLane(v, i);
    }
    return result;
}

// ---- Horizontal Operations ----
template<class T>
force_inline_ vec<T> hadd(const vec<T>& a, const vec<T>& b) {
    return hn::Add(hn::OddEven(a, b), hn::OddEven(b, a));
}

// ---- Shuffle / Permute ----
template<class T>
force_inline_ vec<T> reverse(const vec<T>& v) { return hn::Reverse(D<T>{}, v); }

template<class T>
force_inline_ vec<T> shuffle(const vec<T>& v, const vec<uint8_t>& indices) {
    return hn::TableLookupBytes(v, indices);
}

// ---- Type Conversion ----
template<typename To, typename From>
force_inline_ vec<To> convert(const vec<From>& v) {
    return hn::ConvertTo(D<To>{}, v);
}

template<typename To, typename From>
force_inline_ vec<To> reinterpret(const vec<From>& v) {
    return hn::BitCast(D<To>{}, v);
}

// ---- Extract / Insert ----
template<class T>
force_inline_ T extract_lane(const vec<T>& v, size_t i) {
    return hn::ExtractLane(v, i);
}

template<class T>
force_inline_ vec<T> insert_lane(const vec<T>& v, size_t i, T value) {
    return hn::InsertLane(v, i, value);
}

// ---- Mask Operations ----
template<class T>
force_inline_ bool all_true(const mask<T>& m) { return hn::AllTrue(D<T>{}, m); }

template<class T>
force_inline_ bool all_false(const mask<T>& m) { return hn::AllFalse(D<T>{}, m); }

template<class T>
force_inline_ size_t count_true(const mask<T>& m) { return hn::CountTrue(D<T>{}, m); }

template<class T>
force_inline_ size_t find_first_true(const mask<T>& m) { return hn::FindFirstTrue(D<T>{}, m); }

// Create mask from count (first 'count' elements are true)
template<class T>
force_inline_ mask<T> mask_from_count(size_t count) {
    return hn::FirstN(D<T>{}, count);
}

// ---- Sort ----
template<class T>
force_inline_ void sort(T* data, size_t n) {
    if (n > 1) hwy::VQSort(data, n, hwy::SortAscending{});
}

template<class T>
force_inline_ void sort_descending(T* data, size_t n) {
    if (n > 1) hwy::VQSort(data, n, hwy::SortDescending{});
}

#else
// ================================================================
//      Scalar Fallback Implementation
// ================================================================
template<class T> using vec = T;
template<class T> using mask = bool;

// ---- Vector Properties ----
template<class T>
force_inline_ constexpr size_t lanes() { return 1; }

template<class T>
force_inline_ constexpr size_t max_lanes() { return 1; }

// ---- Initialization ----
template<class T>
force_inline_ vec<T> zero() { return T(0); }

template<class T>
force_inline_ vec<T> set1(T value) { return value; }

template<class T>
force_inline_ vec<T> iota(T start = 0) { return start; }

template<class T>
force_inline_ vec<T> undefined() { return T{}; }

// ---- Load / Store ----
template<class T>
force_inline_ vec<T> load(const T* p) { return *p; }

template<class T>
force_inline_ vec<T> load_aligned(const T* p) { return *p; }

template<class T>
force_inline_ vec<T> load_unaligned(const T* p) { return *p; }

template<class T>
force_inline_ void store(const vec<T>& v, T* p) { *p = v; }

template<class T>
force_inline_ void store_aligned(const vec<T>& v, T* p) { *p = v; }

template<class T>
force_inline_ void store_unaligned(const vec<T>& v, T* p) { *p = v; }

// ---- Masked Load / Store ----
template<class T>
force_inline_ vec<T> masked_load(mask<T> m, const T* p, vec<T> v0 = T(0)) {
    return m ? *p : v0;
}

template<class T>
force_inline_ void masked_store(mask<T> m, const vec<T>& v, T* p) {
    if (m) *p = v;
}

// ---- Gather / Scatter ----
template<class T, class TI>
force_inline_ vec<T> gather(const T* base, const vec<TI>& index) {
    return base[index];
}

template<class T, class TI>
force_inline_ void scatter(const vec<T>& v, T* base, const vec<TI>& index) {
    base[index] = v;
}

// ---- Arithmetic Operations ----
template<class T>
force_inline_ vec<T> add(const vec<T>& a, const vec<T>& b) { return a + b; }

template<class T>
force_inline_ vec<T> sub(const vec<T>& a, const vec<T>& b) { return a - b; }

template<class T>
force_inline_ vec<T> mul(const vec<T>& a, const vec<T>& b) { return a * b; }

template<class T>
force_inline_ vec<T> div(const vec<T>& a, const vec<T>& b) { return a / b; }

template<class T>
force_inline_ vec<T> neg(const vec<T>& v) { return -v; }

template<class T>
force_inline_ vec<T> abs(const vec<T>& v) { 
    if constexpr (std::is_unsigned_v<T>) return v;
    else return v < 0 ? -v : v;
}

template<class T>
force_inline_ vec<T> fma(const vec<T>& a, const vec<T>& b, const vec<T>& c) {
    if constexpr (is_floating_point_v<T>) {
        return std::fma(a, b, c);
    } else {
        return a * b + c;
    }
}

template<class T>
force_inline_ vec<T> fms(const vec<T>& a, const vec<T>& b, const vec<T>& c) {
    return a * b - c;
}

template<class T>
force_inline_ vec<T> fnma(const vec<T>& a, const vec<T>& b, const vec<T>& c) {
    return c - a * b;
}

template<class T>
force_inline_ vec<T> fnms(const vec<T>& a, const vec<T>& b, const vec<T>& c) {
    return -(a * b) - c;
}

// ---- Min/Max Operations ----
template<class T>
force_inline_ vec<T> vmax(const vec<T>& a, const vec<T>& b) { 
    return (a > b) ? a : b;
}

template<class T>
force_inline_ vec<T> vmin(const vec<T>& a, const vec<T>& b) { 
    return (a < b) ? a : b;
}

template<class T>
force_inline_ vec<T> clamp(const vec<T>& v, const vec<T>& lo, const vec<T>& hi) {
    return vmin(vmax(v, lo), hi);
}

// ---- Comparison Operations ----
template<class T>
force_inline_ mask<T> eq(const vec<T>& a, const vec<T>& b) { return a == b; }

template<class T>
force_inline_ mask<T> ne(const vec<T>& a, const vec<T>& b) { return a != b; }

template<class T>
force_inline_ mask<T> lt(const vec<T>& a, const vec<T>& b) { return a < b; }

template<class T>
force_inline_ mask<T> le(const vec<T>& a, const vec<T>& b) { return a <= b; }

template<class T>
force_inline_ mask<T> gt(const vec<T>& a, const vec<T>& b) { return a > b; }

template<class T>
force_inline_ mask<T> ge(const vec<T>& a, const vec<T>& b) { return a >= b; }

// ---- Logical Operations (Masks) ----
template<class T>
force_inline_ mask<T> mask_and(const mask<T>& a, const mask<T>& b) { return a && b; }

template<class T>
force_inline_ mask<T> mask_or(const mask<T>& a, const mask<T>& b) { return a || b; }

template<class T>
force_inline_ mask<T> mask_xor(const mask<T>& a, const mask<T>& b) { return a != b; }

template<class T>
force_inline_ mask<T> mask_not(const mask<T>& m) { return !m; }

template<class T>
force_inline_ mask<T> mask_andnot(const mask<T>& a, const mask<T>& b) { return a && !b; }

// ---- Bitwise Operations ----
template<class T>
force_inline_ vec<T> bit_and(const vec<T>& a, const vec<T>& b) { 
    if constexpr (std::is_integral_v<T>) {
        return a & b;
    } else {
        union { T f; uint32_t i; } ua = {a}, ub = {b}, ur;
        ur.i = ua.i & ub.i;
        return ur.f;
    }
}

template<class T>
force_inline_ vec<T> bit_or(const vec<T>& a, const vec<T>& b) { 
    if constexpr (std::is_integral_v<T>) {
        return a | b;
    } else {
        union { T f; uint32_t i; } ua = {a}, ub = {b}, ur;
        ur.i = ua.i | ub.i;
        return ur.f;
    }
}

template<class T>
force_inline_ vec<T> bit_xor(const vec<T>& a, const vec<T>& b) { 
    if constexpr (std::is_integral_v<T>) {
        return a ^ b;
    } else {
        union { T f; uint32_t i; } ua = {a}, ub = {b}, ur;
        ur.i = ua.i ^ ub.i;
        return ur.f;
    }
}

template<class T>
force_inline_ vec<T> bit_not(const vec<T>& v) { 
    if constexpr (std::is_integral_v<T>) {
        return ~v;
    } else {
        union { T f; uint32_t i; } u = {v}, ur;
        ur.i = ~u.i;
        return ur.f;
    }
}

template<class T>
force_inline_ vec<T> bit_andnot(const vec<T>& a, const vec<T>& b) { 
    return bit_and(a, bit_not(b));
}

// ---- Shift Operations ----
template<class T>
force_inline_ vec<T> shl(const vec<T>& v, int bits) { 
    if constexpr (std::is_integral_v<T>) {
        return v << bits;
    } else {
        return v; // No-op for float
    }
}

template<class T>
force_inline_ vec<T> shr(const vec<T>& v, int bits) { 
    if constexpr (std::is_integral_v<T>) {
        return v >> bits;
    } else {
        return v; // No-op for float
    }
}

template<class T>
force_inline_ vec<T> rotl(const vec<T>& v, int bits) { 
    if constexpr (std::is_integral_v<T>) {
        constexpr int mask = sizeof(T) * 8 - 1;
        bits &= mask;
        return (v << bits) | (v >> ((-bits) & mask));
    } else {
        return v; // No-op for float
    }
}

template<class T>
force_inline_ vec<T> rotr(const vec<T>& v, int bits) { 
    if constexpr (std::is_integral_v<T>) {
        constexpr int mask = sizeof(T) * 8 - 1;
        bits &= mask;
        return (v >> bits) | (v << ((-bits) & mask));
    } else {
        return v; // No-op for float
    }
}

// ---- Select / Blend ----
template<class T>
force_inline_ vec<T> select(const mask<T>& m, const vec<T>& yes, const vec<T>& no) {
    return m ? yes : no;
}

template<class T>
force_inline_ vec<T> select_zero_else(const mask<T>& m, const vec<T>& no) {
    return m ? T(0) : no;
}

template<class T>
force_inline_ vec<T> select_else_zero(const mask<T>& m, const vec<T>& yes) {
    return m ? yes : T(0);
}

// ---- Math Functions (Floating Point) ----
template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> sqrt(const vec<T>& v) { return std::sqrt(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> rsqrt(const vec<T>& v) { return T(1) / std::sqrt(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> reciprocal(const vec<T>& v) { return T(1) / v; }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> floor(const vec<T>& v) { return std::floor(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> ceil(const vec<T>& v) { return std::ceil(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> round(const vec<T>& v) { return std::round(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> trunc(const vec<T>& v) { return std::trunc(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> exp(const vec<T>& v) { return std::exp(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> log(const vec<T>& v) { return std::log(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> sin(const vec<T>& v) { return std::sin(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> cos(const vec<T>& v) { return std::cos(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> tan(const vec<T>& v) { return std::tan(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> asin(const vec<T>& v) { return std::asin(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> acos(const vec<T>& v) { return std::acos(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> atan(const vec<T>& v) { return std::atan(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> tanh(const vec<T>& v) { return std::tanh(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> sinh(const vec<T>& v) { return std::sinh(v); }

template<class T, typename = std::enable_if_t<is_floating_point_v<T>>>
force_inline_ vec<T> cosh(const vec<T>& v) { return std::cosh(v); }

// ---- Reduction Operations ----
template<class T>
force_inline_ T reduce_sum(const vec<T>& v) { return v; }

template<class T>
force_inline_ T reduce_min(const vec<T>& v) { return v; }

template<class T>
force_inline_ T reduce_max(const vec<T>& v) { return v; }

template<class T>
force_inline_ T reduce_mul(const vec<T>& v) { return v; }

// ---- Horizontal Operations ----
template<class T>
force_inline_ vec<T> hadd(const vec<T>& a, const vec<T>& b) { return a + b; }

// ---- Shuffle / Permute ----
template<class T>
force_inline_ vec<T> reverse(const vec<T>& v) { return v; }

template<class T>
force_inline_ vec<T> shuffle(const vec<T>& v, const vec<uint8_t>& indices) { 
    return v; // No shuffle in scalar
}

// ---- Type Conversion ----
template<typename To, typename From>
force_inline_ vec<To> convert(const vec<From>& v) {
    return static_cast<To>(v);
}

template<typename To, typename From>
force_inline_ vec<To> reinterpret(const vec<From>& v) {
    union { From f; To t; } u = {v};
    return u.t;
}

// ---- Extract / Insert ----
template<class T>
force_inline_ T extract_lane(const vec<T>& v, size_t i) {
    return v; // Only one lane in scalar
}

template<class T>
force_inline_ vec<T> insert_lane(const vec<T>& v, size_t i, T value) {
    return value; // Only one lane in scalar
}

// ---- Mask Operations ----
template<class T>
force_inline_ bool all_true(const mask<T>& m) { return m; }

template<class T>
force_inline_ bool all_false(const mask<T>& m) { return !m; }

template<class T>
force_inline_ size_t count_true(const mask<T>& m) { return m ? 1 : 0; }

template<class T>
force_inline_ size_t find_first_true(const mask<T>& m) { return m ? 0 : 1; }

// Create mask from count (first 'count' elements are true)
template<class T>
force_inline_ mask<T> mask_from_count(size_t count) {
    return count > 0;
}

// ---- Sort ----
template<class T>
force_inline_ void sort(T* data, size_t n) {
    if (n > 1) std::sort(data, data + n);
}

template<class T>
force_inline_ void sort_descending(T* data, size_t n) {
    if (n > 1) std::sort(data, data + n, std::greater<T>());
}

#endif // USE_HWY

// ================================================================
//      Common Helper Functions
// ================================================================

// Broadcast single value to all lanes
template<class T>
force_inline_ vec<T> broadcast(T value) { return set1(value); }

// Check if all elements are zero
template<class T>
force_inline_ bool is_zero(const vec<T>& v) {
    return all_true(eq(v, zero<T>()));
}

// Sum all elements in array using SIMD
template<class T>
force_inline_ T array_sum(const T* const data, const size_t n) {
    vec<T> sum_vec = zero<T>();
    const size_t step = lanes<T>();
    size_t i = 0;
    
    // Main SIMD loop
    for (; i + step <= n; i += step) {
        sum_vec = add(sum_vec, load(data + i));
    }

    // 尾部掩码处理
    if (i < n) {
        const size_t tail = n - i;
        const auto m = mask_from_count<T>(tail);
        sum_vec = add(sum_vec, masked_load(m, data + i));
    }

    // 向量归约为标量
    const T sum = reduce_sum(sum_vec);
    return sum;
}

// Find min/max in array using SIMD
template<class T>
force_inline_ std::pair<T, T> array_minmax(const T* const data, const size_t n) {
    if (n == 0) return {T{}, T{}};

    vec<T> min_vec = set1(data[0]);
    vec<T> max_vec = set1(data[0]);
    const size_t step = lanes<T>();
    size_t i = 0;

    // 主SIMD循环
    for (; i + step <= n; i += step) {
        const vec<T> v = load(data + i);
        min_vec = vmin(min_vec, v);
        max_vec = vmax(max_vec, v);
    }

    // 尾部掩码处理
    if (i < n) {
        const size_t tail = n - i;
        const auto m = mask_from_count<T>(tail);
        const vec<T> v = masked_load(m, data + i);
        min_vec = vmin(min_vec, v);
        max_vec = vmax(max_vec, v);
    }

    // 向量归约为标量
    const T min_val = reduce_min(min_vec);
    const T max_val = reduce_max(max_vec);

    return {min_val, max_val};
}

// 单独的最小值版本
template<class T>
force_inline_ T array_min(const T* const data, const size_t n) {
    if (n == 0) return T{};
    vec<T> min_vec = set1(data[0]);
    const size_t step = lanes<T>();
    size_t i = 0;
    for (; i + step <= n; i += step) {
        const vec<T> v = load(data + i);
        min_vec = vmin(min_vec, v);
    }
    if (i < n) {
        const size_t tail = n - i;
        const auto m = mask_from_count<T>(tail);
        const vec<T> v = masked_load(m, data + i);
        min_vec = vmin(min_vec, v);
    }
    return reduce_min(min_vec);
}

// 单独的最大值版本
template<class T>
force_inline_ T array_max(const T* const data, const size_t n) {
    if (n == 0) return T{};
    vec<T> max_vec = set1(data[0]);
    const size_t step = lanes<T>();
    size_t i = 0;
    for (; i + step <= n; i += step) {
        const vec<T> v = load(data + i);
        max_vec = vmax(max_vec, v);
    }
    if (i < n) {
        const size_t tail = n - i;
        const auto m = mask_from_count<T>(tail);
        const vec<T> v = masked_load(m, data + i);
        max_vec = vmax(max_vec, v);
    }
    return reduce_max(max_vec);
}

// SIMD点积，尾部用掩码处理
template<class T>
force_inline_ T dot_product(const T* a, const T* b, size_t n) {
    vec<T> sum_vec = zero<T>();
    size_t step = lanes<T>();
    size_t i = 0;

    // 主SIMD循环
    for (; i + step <= n; i += step) {
        vec<T> va = load(a + i);
        vec<T> vb = load(b + i);
        sum_vec = fma(va, vb, sum_vec);
    }

    // 尾部掩码处理
    if (i < n) {
        size_t tail = n - i;
        auto m = mask_from_count<T>(tail);
        vec<T> va = masked_load(m, a + i);
        vec<T> vb = masked_load(m, b + i);
        sum_vec = fma(va, vb, sum_vec);
    }

    // 向量归约为标量
    T sum = reduce_sum(sum_vec);
    return sum;
}

// 一元函数应用，尾部用掩码处理
template<class T, typename Func>
force_inline_ void array_apply(T* data, size_t n, Func func) {
    size_t step = lanes<T>();
    size_t i = 0;

    // 主SIMD循环
    for (; i + step <= n; i += step) {
        vec<T> v = load(data + i);
        v = func(v);
        store(v, data + i);
    }

    // 尾部掩码处理
    if (i < n) {
        size_t tail = n - i;
        auto m = mask_from_count<T>(tail);
        vec<T> v = masked_load(m, data + i);
        v = func(v);
        masked_store(m, v, data + i);
    }
}

// 二元函数应用，尾部用掩码处理
template<class T, typename Func>
force_inline_ void array_apply_binary(T* dst, const T* a, const T* b, size_t n, Func func) {
    size_t step = lanes<T>();
    size_t i = 0;

    // 主SIMD循环
    for (; i + step <= n; i += step) {
        vec<T> va = load(a + i);
        vec<T> vb = load(b + i);
        vec<T> result = func(va, vb);
        store(result, dst + i);
    }

    // 尾部掩码处理
    if (i < n) {
        size_t tail = n - i;
        auto m = mask_from_count<T>(tail);
        vec<T> va = masked_load(m, a + i);
        vec<T> vb = masked_load(m, b + i);
        vec<T> result = func(va, vb);
        masked_store(m, result, dst + i);
    }
}

} // namespace simd
PROJECT_END