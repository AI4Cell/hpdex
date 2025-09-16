cmake_minimum_required(VERSION 3.18)
project(hpdex LANGUAGES CXX)

# -----------------------------
# 基础设置
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -----------------------------
# Python & pybind11
# -----------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# 通过 pybind11 Python 包获取 CMake config 路径（兼容 pip 安装/多平台）
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import pybind11, sys; sys.stdout.write(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(pybind11_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
endif()
find_package(pybind11 CONFIG REQUIRED)

# -----------------------------
# Highway（作为子目录）
# -----------------------------
set(HIGHWAY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Lib/highway")
if(NOT EXISTS "${HIGHWAY_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "Highway not found at ${HIGHWAY_DIR}")
endif()
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${HIGHWAY_DIR}" highway_build EXCLUDE_FROM_ALL)

# -----------------------------
# OpenMP（可选；跨平台处理）
# -----------------------------
find_package(OpenMP QUIET)

# macOS: 如果没有找到 OpenMP::OpenMP_CXX，尝试 Homebrew libomp 兜底
if(APPLE AND NOT OpenMP_CXX_FOUND)
  foreach(_omp_root "/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")
    if(EXISTS "${_omp_root}/lib/libomp.dylib")
      message(STATUS "Using fallback libomp at ${_omp_root}")
      add_library(OpenMP::OpenMP_CXX SHARED IMPORTED)
      set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
        IMPORTED_LOCATION "${_omp_root}/lib/libomp.dylib"
        INTERFACE_INCLUDE_DIRECTORIES "${_omp_root}/include"
      )
      set(OpenMP_CXX_FOUND TRUE)
      set(_OMP_RPATH "${_omp_root}/lib")
      break()
    endif()
  endforeach()
endif()

# -----------------------------
# pybind11 扩展模块
#   目标名：kernel
#   Python 导入路径：hpdex/backend/bin/kernel[.so|.pyd]
# -----------------------------
pybind11_add_module(kernel
  src/cpp/module.cpp
  src/cpp/mannwhitneyu.cpp
)

target_compile_features(kernel PRIVATE cxx_std_17)

if(MSVC)
  target_compile_options(kernel PRIVATE /O2 /permissive- /EHsc /Zc:preprocessor /bigobj)
  target_compile_definitions(kernel PRIVATE NOMINMAX)
else()
  target_compile_options(kernel PRIVATE -O3)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(kernel PRIVATE -Wno-deprecated-declarations)
  endif()
endif()

target_include_directories(kernel PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/include
)

target_link_libraries(kernel PRIVATE hwy hwy_contrib)
if(OpenMP_CXX_FOUND)
  target_link_libraries(kernel PRIVATE OpenMP::OpenMP_CXX)
endif()

# macOS: 避免链接期强解析 Python 符号；并给 libomp 设 rpath
if(APPLE)
  set_target_properties(kernel PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  if(DEFINED _OMP_RPATH)
    set_target_properties(kernel PROPERTIES
      BUILD_RPATH "${_OMP_RPATH}"
      INSTALL_RPATH "${_OMP_RPATH}"
    )
  endif()
endif()

# -----------------------------
# 安装到包内的目标位置（site-packages/hpdex/backend/bin）
# scikit-build-core 会收集此 install 产物并打进 wheel
# -----------------------------
install(TARGETS kernel
  LIBRARY DESTINATION hpdex/backend/bin      # Unix-like .so
  RUNTIME DESTINATION hpdex/backend/bin      # Windows .pyd 归类为 RUNTIME
  ARCHIVE DESTINATION hpdex/backend/bin      # （通常不会用到，但保留以防有 import lib）
)
