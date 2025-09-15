#pragma once

#include "config.hpp"

#define PROJECT_BEGIN namespace PROJECT_NAME {
#define PROJECT_END }

#if USE_OPENMP
  #define PARALLEL_REGION(threads) \
    _Pragma("omp parallel num_threads(threads)")

  #define THREAD_ID() omp_get_thread_num()
  #define MAX_THREADS() omp_get_max_threads()

  // omp for
  #define PARALLEL_FOR(SCHEDULE, ARGS, LOOP) \
    _Pragma(HPDEXC_STRINGIFY(omp for schedule(SCHEDULE) ARGS)) \
    { LOOP }

  // atomic operation
  #define ATOMIC(DO) \
    _Pragma("omp atomic") { DO }

  // critical section
  #define CRITICAL(DO) \
    _Pragma("omp critical") { DO }

  #define HPDEXC_STRINGIFY(x) HPDEXC_STRINGIFY_IMPL(x)
  #define HPDEXC_STRINGIFY_IMPL(x) #x
#else
  #define PARALLEL_REGION(threads)
  #define PARALLEL_FOR(SCHEDULE, LOOP) { LOOP }
  #define THREAD_ID() 0
  #define MAX_THREADS() 1
  #define ATOMIC(DO) { DO }
  #define CRITICAL(DO) { DO }
#endif


// ============================================
// tools
// ============================================
#if defined(_MSC_VER)
  #define force_inline_ __forceinline
  #define restrict_     __restrict
#else
  #define force_inline_ inline __attribute__((always_inline))
  #define restrict_     __restrict__
#endif

#if defined(_MSC_VER)
  #define malloc_aligned_(ptr, size) \
    do { ptr = static_cast<void*>(_aligned_malloc((size), HPDEXC_ALIGN_SIZE)); } while(0)
  #define free_aligned_(ptr) \
    do { _aligned_free((ptr)); } while(0)
#else
  #define malloc_aligned_(ptr, size) \
    do { if (posix_memalign(&(ptr), HPDEXC_ALIGN_SIZE, (size)) != 0) ptr = nullptr; } while(0)
  #define free_aligned_(ptr) \
    do { free((ptr)); } while(0)
#endif

// assume_aligned
#if defined(__clang__) || defined(__GNUC__)
  #define assume_aligned_(p, N) reinterpret_cast<decltype(p)>(__builtin_assume_aligned((p), (N)))
#else
  #define assume_aligned_(p, N) (p)
#endif

// branch prediction
#if defined(__clang__) || defined(__GNUC__)
  #define likely_(x)   (__builtin_expect(!!(x), 1))
  #define unlikely_(x) (__builtin_expect(!!(x), 0))
#else
  #define likely_(x)   (x)
  #define unlikely_(x) (x)
#endif

// prefetch
#if defined(__clang__) || defined(__GNUC__)
  #define prefetch_r_(p,loc) __builtin_prefetch((p), 0, (loc))
  #define prefetch_w_(p,loc) __builtin_prefetch((p), 1, (loc))
#else
  #define prefetch_r_(p,loc) ((void)0)
  #define prefetch_w_(p,loc) ((void)0)
#endif

// export
#if defined(_MSC_VER)
  #define export_ __declspec(dllexport)
#else
  #define export_ __attribute__((visibility("default")))
#endif

// alignas
#if defined(_MSC_VER)
  #define alignas_(N) __declspec(align((N)))
#else
  #define alignas_(N) alignas((N))
#endif