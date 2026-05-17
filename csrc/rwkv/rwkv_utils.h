#pragma once

#include <cuda_fp16.h>
#include <ATen/ATen.h>

typedef at::Half fp16;

inline __half* cast(fp16* ptr) {
    return reinterpret_cast<__half*>(ptr);
}
