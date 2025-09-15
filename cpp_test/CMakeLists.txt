cmake_minimum_required(VERSION 3.18)
project(hpdex_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 默认 Debug 构建（若未指定）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# 确保 Debug 下包含符号并禁用优化
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O3")

# 获取主项目根目录
get_filename_component(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)
set(LIB_DIR "${ROOT_DIR}/Lib")

# ===== 依赖路径（仅头文件方式） =====
set(EIGEN3_INCLUDE_DIR "${LIB_DIR}/eigen-3.4.0")
set(HIGHWAY_DIR "${LIB_DIR}/highway")

# ===== OpenMP（macOS Clang 回退到 Homebrew libomp） =====
find_package(OpenMP QUIET)
if(NOT OpenMP_CXX_FOUND)
    message(STATUS "OpenMP not detected, forcing Homebrew libomp...")
    set(OpenMP_CXX_FLAGS_LIST -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include)
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
    set(OpenMP_CXX_FOUND TRUE)
endif()

# ===== 源文件 =====
set(TEST_SOURCES
    ${ROOT_DIR}/src/cpp/test.cpp
)

add_executable(test ${TEST_SOURCES})

# 编译/链接选项（OpenMP）
if(OpenMP_CXX_FLAGS_LIST)
    target_compile_options(test PRIVATE ${OpenMP_CXX_FLAGS_LIST})
endif()
if(OpenMP_CXX_FOUND)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(test PRIVATE OpenMP::OpenMP_CXX)
    elseif(OpenMP_omp_LIBRARY)
        target_link_libraries(test PRIVATE ${OpenMP_omp_LIBRARY})
        # 运行时查找 libomp
        target_link_options(test PRIVATE -Wl,-rpath,/opt/homebrew/opt/libomp/lib)
    endif()
endif()

# 头文件路径（只包含，不构建 Highway 子项目）
target_include_directories(test PRIVATE
    ${ROOT_DIR}/src/cpp/include
    ${EIGEN3_INCLUDE_DIR}
)
if(EXISTS ${HIGHWAY_DIR})
    target_include_directories(test PRIVATE ${HIGHWAY_DIR})
endif()

# 可选：Torch（若存在则链接）
list(APPEND CMAKE_PREFIX_PATH "${LIB_DIR}/libtorch")
find_package(Torch QUIET)
if(Torch_FOUND)
    target_link_libraries(test PRIVATE ${TORCH_LIBRARIES})
endif()
