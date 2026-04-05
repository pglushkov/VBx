cmake_minimum_required(VERSION 3.16)
project(vbx VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Build benchmarks using Google Benchmark ---
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
# --- Python bindings (nanobind) ---
option(VBX_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

# --- Eigen (header-only) ---
include(FetchContent)
FetchContent_Declare(eigen
    URL https://github.com/eigen-mirror/eigen/archive/refs/tags/5.0.1.tar.gz
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

# --- libvbx ---
add_library(vbx
    src/linalg.cpp
    src/clustering.cpp
    src/scoring.cpp
    src/vbhmm.cpp
    src/vbx.cpp
    third_party/fastcluster/fastcluster.cpp
)
target_include_directories(vbx PUBLIC
    include
    third_party/fastcluster
)
target_compile_options(vbx PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(vbx PUBLIC Eigen3::Eigen)

# --- vbx_cli ---
add_executable(vbx_cli tools/vbx_cli.cpp)
target_link_libraries(vbx_cli PRIVATE vbx)

if(VBX_BUILD_PYTHON_BINDINGS)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(nanobind CONFIG QUIET)
  if(NOT nanobind_FOUND)
    include(FetchContent)
    FetchContent_Declare(nanobind
            GIT_REPOSITORY https://github.com/wjakob/nanobind
            GIT_TAG        v2.4.0
        )
    FetchContent_MakeAvailable(nanobind)
  endif()
  nanobind_add_module(vbx_native STABLE_ABI NB_STATIC
        bindings/python/bind_vbx.cpp
    )
  target_link_libraries(vbx_native PRIVATE vbx)
  install(TARGETS vbx_native LIBRARY DESTINATION .)
endif()

if(BUILD_BENCHMARKS)
  find_package(benchmark QUIET)
  if(NOT benchmark_FOUND)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.8.3
    )

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(benchmark)
  endif()
  add_subdirectory(benchmarks)
endif()
