cmake_minimum_required(VERSION 3.24)
project(backtest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Python + pybind11 (pip-installed; cmake dir comes from the module) ----
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

find_package(OpenMP)

# ---- CPU engine module ----
pybind11_add_module(btcpu engine/cpu/module.cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(btcpu PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(btcpu PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/engine"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/engine"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/engine")

# ---- CUDA (optional so the CPU engine still builds on machines without it) ----
option(BT_CUDA_FMAD "allow FMA contraction in CUDA kernels (bench builds)" OFF)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89)   # RTX 4070 (Ada)
    endif()
    add_executable(cuda_smoke engine/cuda/smoke.cu)
    set_property(TARGET cuda_smoke PROPERTY CUDA_RUNTIME_LIBRARY Shared)

    pybind11_add_module(btcuda engine/cuda/cuda_module.cu)
    # Match Python/MSVC's dynamic CRT and avoid a static-cudart LIBCMT conflict.
    set_property(TARGET btcuda PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(NOT BT_CUDA_FMAD)
        # keep float op-for-op identical to the CPU engines (SPEC.md section 5)
        target_compile_options(btcuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    endif()
    set_target_properties(btcuda PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/engine"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/engine"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/engine")
else()
    message(STATUS "CUDA compiler not found - GPU targets skipped")
endif()
