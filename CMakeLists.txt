cmake_minimum_required(VERSION 3.22)
project(vkwr_ops LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(utils)
include(cpu_extension)

if(NOT DEFINED VKWR_TARGET_DEVICE)
    set(VKWR_TARGET_DEVICE "cuda" CACHE STRING "Target device: cuda or cpu")
endif()

if(NOT DEFINED VKWR_CUDA_ARCH)
    set(VKWR_CUDA_ARCH "89" CACHE STRING "CUDA architecture (e.g., 80, 86, 89, 90, 120)")
endif()

# Find Python first (Torch's cmake also queries Python internally).
# CMAKE_PYTHON_EXECUTABLE should be passed from setup.py so this picks up
# the correct venv interpreter and matching headers/libs.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Import torch cmake configuration first.
# Torch also imports CUDA language with proper configuration,
# so there is no need to call enable_language(CUDA) explicitly.
find_package(Torch REQUIRED)

# Phase 1: RWKV7
add_vkwr_extension(
    _rwkv7_C
    SOURCES
        csrc/rwkv7/torch_bindings.cpp
        csrc/rwkv7/state_fwd_fp16.cu
    DEFINITIONS _N_=64 ENABLE_L2_PREFETCH=0
)

# Phase 3: Sampling
add_vkwr_extension(
    _sampling_C
    SOURCES
        csrc/sampling/torch_bindings.cpp
        csrc/sampling/sampling.cu
)