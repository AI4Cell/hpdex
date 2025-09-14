cmake_minimum_required(VERSION 3.18)
project(hpdex LANGUAGES CXX)

# ============================================
# 基础设置
# ============================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 导出编译命令，生成 compile_commands.json 文件
# 这对于代码补全、静态分析和 IDE 支持非常有用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 获取项目根目录
get_filename_component(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)
set(LIB_DIR "${ROOT_DIR}/Lib")

# ============================================
# 1. Python 和 pybind11 (uv 虚拟环境)
# ============================================
if(DEFINED ENV{VIRTUAL_ENV})
    set(Python_ROOT_DIR $ENV{VIRTUAL_ENV})
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(pybind11_DIR)
    list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR})
endif()
find_package(pybind11 REQUIRED)

# ============================================
# 2. Eigen
# ============================================
set(EIGEN3_INCLUDE_DIR "${LIB_DIR}/eigen-3.4.0")
if(NOT EXISTS ${EIGEN3_INCLUDE_DIR})
    message(FATAL_ERROR "Eigen not found at ${EIGEN3_INCLUDE_DIR}. Please install it first.")
endif()
include_directories(${EIGEN3_INCLUDE_DIR})

# ============================================
# 3. LibTorch
# ============================================
list(APPEND CMAKE_PREFIX_PATH "${LIB_DIR}/libtorch")

find_package(Torch QUIET)

if(Torch_FOUND)
    set(USE_TORCH ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    add_definitions(-DUSE_TORCH)
else()
    message(WARNING "LibTorch not found at ${LIB_DIR}/libtorch. Torch features disabled.")
    set(USE_TORCH OFF)
endif()

# ============================================
# 4. Highway
# ============================================
set(HIGHWAY_DIR "${LIB_DIR}/highway")
if(EXISTS ${HIGHWAY_DIR})
    set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(${HIGHWAY_DIR} highway_build)
    set(USE_HIGHWAY ON)
else()
    message(WARNING "Highway not found at ${HIGHWAY_DIR}. Disabled.")
    set(USE_HIGHWAY OFF)
endif()

# ============================================
# 5. OpenMP
# ============================================
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    set(USE_OPENMP ON)
else()
    message(WARNING "OpenMP not found. Disabled.")
    set(USE_OPENMP OFF)
endif()

# ============================================
# 平台特定设置
# ============================================
if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    # macOS 上的 OpenMP (Homebrew)
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
    elseif(EXISTS "/usr/local/opt/libomp")
        set(OpenMP_ROOT "/usr/local/opt/libomp")
    endif()
    if(OpenMP_ROOT)
        include_directories("${OpenMP_ROOT}/include")
        link_directories("${OpenMP_ROOT}/lib")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${OpenMP_ROOT}/lib/libomp.dylib")
    endif()
elseif(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# ============================================
# 打印配置信息
# ============================================
message(STATUS "========================================")
message(STATUS "Configuration Summary:")
message(STATUS "  Python:     ${Python_EXECUTABLE}")
message(STATUS "  pybind11:   ${pybind11_VERSION}")
message(STATUS "  Eigen:      ${EIGEN3_INCLUDE_DIR}")
message(STATUS "  LibTorch:   ${USE_TORCH}")
message(STATUS "  Highway:    ${USE_HIGHWAY}")
message(STATUS "  OpenMP:     ${USE_OPENMP}")
message(STATUS "========================================")

# ============================================
# Python 扩展模块
# ============================================
pybind11_add_module(hpdex_module
    src/cpp/module.cpp
    # 这里可以继续添加你的源码
)

target_compile_features(hpdex_module PRIVATE cxx_std_17)

target_include_directories(hpdex_module PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/include
    ${EIGEN3_INCLUDE_DIR}
)

if(USE_TORCH)
    target_link_libraries(hpdex_module PRIVATE ${TORCH_LIBRARIES})
endif()

if(USE_HIGHWAY)
    target_link_libraries(hpdex_module PRIVATE hwy)
endif()

if(USE_OPENMP)
    target_link_libraries(hpdex_module PRIVATE OpenMP::OpenMP_CXX)
endif()
