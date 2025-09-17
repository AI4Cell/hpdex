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

# 创建 compile_commands.json 符号链接的自定义目标将在 kernel 目标定义后添加

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

# macOS: 优先使用静态链接以避免运行时符号解析问题
if(APPLE AND NOT OpenMP_CXX_FOUND)
  foreach(_omp_root "/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")
    # 优先检查静态库是否存在
    if(EXISTS "${_omp_root}/lib/libomp.a")
      message(STATUS "Using static libomp at ${_omp_root}")
      add_library(OpenMP::OpenMP_CXX STATIC IMPORTED)
      
      # 检查编译器是否支持-fopenmp标志
      include(CheckCXXCompilerFlag)
      check_cxx_compiler_flag("-fopenmp" COMPILER_SUPPORTS_FOPENMP)
      check_cxx_compiler_flag("-Xpreprocessor -fopenmp" COMPILER_SUPPORTS_XPREPROCESSOR_FOPENMP)
      
      if(COMPILER_SUPPORTS_FOPENMP)
        set(_omp_compile_flags "-fopenmp")
        set(_omp_link_flags "-fopenmp")
      elseif(COMPILER_SUPPORTS_XPREPROCESSOR_FOPENMP)
        set(_omp_compile_flags "-Xpreprocessor;-fopenmp")
        set(_omp_link_flags "-lomp")
      else()
        # 如果编译器不支持OpenMP，则只设置头文件路径和库
        message(WARNING "Compiler does not support OpenMP flags, using library-only approach")
        set(_omp_compile_flags "")
        set(_omp_link_flags "")
      endif()
      
      set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
        IMPORTED_LOCATION "${_omp_root}/lib/libomp.a"
        INTERFACE_INCLUDE_DIRECTORIES "${_omp_root}/include"
        INTERFACE_COMPILE_OPTIONS "${_omp_compile_flags}"
        INTERFACE_LINK_OPTIONS "${_omp_link_flags}"
      )
      set(OpenMP_CXX_FOUND TRUE)
      set(_OMP_STATIC_LINKING TRUE)
      break()
    # 如果没有静态库，回退到动态库
    elseif(EXISTS "${_omp_root}/lib/libomp.dylib")
      message(STATUS "Using dynamic libomp at ${_omp_root} (fallback)")
      add_library(OpenMP::OpenMP_CXX SHARED IMPORTED)
      
      # 检查编译器是否支持-fopenmp标志
      include(CheckCXXCompilerFlag)
      check_cxx_compiler_flag("-fopenmp" COMPILER_SUPPORTS_FOPENMP)
      check_cxx_compiler_flag("-Xpreprocessor -fopenmp" COMPILER_SUPPORTS_XPREPROCESSOR_FOPENMP)
      
      if(COMPILER_SUPPORTS_FOPENMP)
        set(_omp_compile_flags "-fopenmp")
        set(_omp_link_flags "-fopenmp")
      elseif(COMPILER_SUPPORTS_XPREPROCESSOR_FOPENMP)
        set(_omp_compile_flags "-Xpreprocessor;-fopenmp")
        set(_omp_link_flags "-lomp")
      else()
        # 如果编译器不支持OpenMP，则只设置头文件路径和库
        message(WARNING "Compiler does not support OpenMP flags, using library-only approach")
        set(_omp_compile_flags "")
        set(_omp_link_flags "")
      endif()
      
      set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
        IMPORTED_LOCATION "${_omp_root}/lib/libomp.dylib"
        INTERFACE_INCLUDE_DIRECTORIES "${_omp_root}/include"
        INTERFACE_COMPILE_OPTIONS "${_omp_compile_flags}"
        INTERFACE_LINK_OPTIONS "${_omp_link_flags}"
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

# 添加 include 目录
target_include_directories(kernel PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/include
  src/cpp/include
)

target_link_libraries(kernel PRIVATE hwy hwy_contrib)

# 平台特定的OpenMP处理
if(OpenMP_CXX_FOUND)
  target_link_libraries(kernel PRIVATE OpenMP::OpenMP_CXX)
  
  # 根据平台选择正确的编译标志
  if(APPLE)
    # macOS: 区分静态和动态链接的处理方式
    target_compile_definitions(kernel PRIVATE _OPENMP)
    if(DEFINED _OMP_STATIC_LINKING)
      message(STATUS "OpenMP enabled for macOS (static linking)")
      # 静态链接时可能需要额外的系统库
      target_link_libraries(kernel PRIVATE "-framework Foundation")
    else()
      message(STATUS "OpenMP enabled for macOS (dynamic linking)")
    endif()
  elseif(WIN32)
    # Windows: MSVC的OpenMP支持
    target_compile_definitions(kernel PRIVATE _OPENMP)
    message(STATUS "OpenMP enabled for Windows")
  else()
    # Linux: GCC的OpenMP支持
    target_compile_definitions(kernel PRIVATE _OPENMP)
    message(STATUS "OpenMP enabled for Linux")
  endif()
else()
  message(WARNING "OpenMP not found, parallel execution will be disabled")
  target_compile_definitions(kernel PRIVATE NO_OPENMP)
endif()

# macOS: 避免链接期强解析 Python 符号；处理 rpath（仅动态链接时需要）
if(APPLE)
  set_target_properties(kernel PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  # 只有在使用动态链接时才设置 rpath
  if(DEFINED _OMP_RPATH AND NOT DEFINED _OMP_STATIC_LINKING)
    set_target_properties(kernel PROPERTIES
      BUILD_RPATH "${_OMP_RPATH}"
      INSTALL_RPATH "${_OMP_RPATH}"
    )
    message(STATUS "Setting rpath for dynamic libomp: ${_OMP_RPATH}")
  endif()
endif()

# -----------------------------
# 创建 compile_commands.json 符号链接
# -----------------------------
if(CMAKE_GENERATOR STREQUAL "Unix Makefiles" OR CMAKE_GENERATOR STREQUAL "Ninja")
  if(CMAKE_BINARY_DIR STREQUAL "${CMAKE_SOURCE_DIR}/build")
    set(_compile_commands_src "${CMAKE_BINARY_DIR}/compile_commands.json")
    set(_compile_commands_dst "${CMAKE_SOURCE_DIR}/compile_commands.json")
    add_custom_command(
      TARGET kernel POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E create_symlink
              "${_compile_commands_src}" "${_compile_commands_dst}"
      COMMENT "在项目根目录创建 compile_commands.json 的符号链接"
      VERBATIM
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
