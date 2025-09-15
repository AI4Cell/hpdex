cmake_minimum_required(VERSION 3.10)

project(googletest-download NONE)

include(ExternalProject)
ExternalProject_Add(googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           43efa0a4efd40c78b9210d15373112081899a97c
  SOURCE_DIR        "/Users/wzq/Documents/Code/Project/westlake/hpdex/build_test_cpp/highway_build/googletest-src"
  BINARY_DIR        "/Users/wzq/Documents/Code/Project/westlake/hpdex/build_test_cpp/highway_build/googletest-build"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)
