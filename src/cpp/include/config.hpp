// config.hpp 配置文件

#pragma once

#ifndef FAST_NORM
#define FAST_NORM 1
#endif

#ifndef HWY_STATIC_DEFINE
#define HWY_STATIC_DEFINE 1
#endif

#ifndef DEBUG
#define DEBUG 0
#endif

#define ALIGN_SIZE 64

#define DEFAULT_MAX_THREADS 32

#ifndef APPROX_HIST_BINS
#define APPROX_HIST_BINS 1024
#endif

#define PROJECT_NAME hpdex
