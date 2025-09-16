cmake_minimum_required(VERSION 3.18)
project(hpdex LANGUAGES CXX)

# ===============================
# 基础
# ===============================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ===============================
# Python & pybind11
# ===============================
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(pybind11_DIR)
  list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR})
endif()
find_package(pybind11 REQUIRED CONFIG)

# ===============================
# Highway（作为子模块/子目录）
# ===============================
set(HIGHWAY_DIR "${CMAKE_SOURCE_DIR}/Lib/highway")
if(NOT EXISTS ${HIGHWAY_DIR})
  message(FATAL_ERROR "Highway not found at ${HIGHWAY_DIR}")
endif()
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${HIGHWAY_DIR} highway_build EXCLUDE_FROM_ALL)

# ===============================
# OpenMP（可选）
# ===============================
find_package(OpenMP QUIET)

# ===============================
# macOS 专项
# ===============================
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  # Python 扩展：避免在链接期解析 Python 符号
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup")

  # 尝试 Homebrew libomp（若 find_package 失败）
  if(NOT OpenMP_CXX_FOUND)
    foreach(_omp_root "/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")
      if(EXISTS "${_omp_root}/lib/libomp.dylib")
        message(STATUS "Using libomp at ${_omp_root}")
        add_library(omp SHARED IMPORTED)
        set_target_properties(omp PROPERTIES IMPORTED_LOCATION "${_omp_root}/lib/libomp.dylib")
        set(OpenMP_CXX_FOUND TRUE)
        set(_OMP_RPATH "${_omp_root}/lib")
        include_directories("${_omp_root}/include")
      endif()
    endforeach()
  endif()
endif()

# ===============================
# Python 扩展模块
# ===============================
pybind11_add_module(hpdex_cpp
  src/cpp/module.cpp
  src/cpp/mannwhitneyu.cpp
)

target_compile_features(hpdex_cpp PRIVATE cxx_std_17)

# Release 模式启用 -O3 优化
target_compile_options(hpdex_cpp PRIVATE $<$<CONFIG:Release>:-O3>)

# 你的头文件路径（如有）
target_include_directories(hpdex_cpp PRIVATE
  ${CMAKE_SOURCE_DIR}/src/cpp
  ${CMAKE_SOURCE_DIR}/src/cpp/include
)

# Highway
target_link_libraries(hpdex_cpp PRIVATE hwy hwy_contrib)

# OpenMP（找到就连）
if(OpenMP_CXX_FOUND)
  target_link_libraries(hpdex_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# 给 libomp 配 rpath（macOS）
if(APPLE AND DEFINED _OMP_RPATH)
  set_target_properties(hpdex_cpp PROPERTIES
    BUILD_RPATH "${_OMP_RPATH}"
    INSTALL_RPATH "${_OMP_RPATH}")
endif()

# ===============================
# 安装（可选）
# ===============================
include(GNUInstallDirs)
set(PY_INSTALL_DIR "${Python_SITEARCH}")
if(NOT PY_INSTALL_DIR)
  set(PY_INSTALL_DIR "${Python_SITELIB}")
endif()
if(NOT PY_INSTALL_DIR)
  set(PY_INSTALL_DIR "${CMAKE_BINARY_DIR}")
endif()
message(STATUS "Python install dir: ${PY_INSTALL_DIR}")

install(TARGETS hpdex_cpp
  LIBRARY DESTINATION ${PY_INSTALL_DIR}
  ARCHIVE DESTINATION ${PY_INSTALL_DIR}
  RUNTIME DESTINATION ${PY_INSTALL_DIR}
)