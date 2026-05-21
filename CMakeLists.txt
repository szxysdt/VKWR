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

# v1-1: WKV core (fp16 + fp32) + advance_i32
add_vkwr_extension(
    _v1_wkv_C
    SOURCES
        csrc/v1/wkv/torch_bindings.cpp
        csrc/v1/wkv/wkv_fp16_v2.cu
        csrc/v1/wkv/wkv_fp32_v2.cu
        csrc/v1/wkv/advance_i32_ops.cu
    DEFINITIONS _IO_FP16_
)

# v1-2: Linear (needs cuBLAS + cuBLASLt)
add_vkwr_extension(
    _v1_linear_C
    SOURCES
        csrc/v1/linear/torch_bindings.cpp
        csrc/v1/linear/linear_ops.cu
    LINK_LIBRARIES cublas cublasLt
)

# v1-3: LayerNorm + Add+LN fusion
add_vkwr_extension(
    _v1_norm_C
    SOURCES
        csrc/v1/norm/torch_bindings.cpp
        csrc/v1/norm/norm_ops.cu
)

# v1-4: TMIX + CMIX + Activation
add_vkwr_extension(
    _v1_mix_C
    SOURCES
        csrc/v1/mix/torch_bindings.cpp
        csrc/v1/mix/mix_ops.cu
)

# v1-5: Low-Rank fused (pure CUDA kernels, no cuBLAS needed)
add_vkwr_extension(
    _v1_rank_C
    SOURCES
        csrc/v1/rank/torch_bindings.cpp
        csrc/v1/rank/rank_ops.cu
)