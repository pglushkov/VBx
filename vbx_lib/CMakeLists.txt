cmake_minimum_required(VERSION 3.16)
project(vbx VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# --- vbx_cli ---
add_executable(vbx_cli tools/vbx_cli.cpp)
target_link_libraries(vbx_cli PRIVATE vbx)

# --- Python bindings (nanobind) ---
option(VBX_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

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
