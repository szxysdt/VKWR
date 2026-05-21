#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <climits>

#include "v1/common/warp_primitives.cuh"

using dtype = at::Half;

namespace {

constexpr int LN_THREADS = 256;
constexpr int LN_SMALL_THREADS = 1024;
constexpr int LN_SMALL512_THREADS = 512;
constexpr int LN_SMALL_C = 4096;

__device__ __forceinline__ float bf16_bits_to_float_dev(uint16_t bits) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = static_cast<uint32_t>(bits) << 16;
  return v.f;
}
__global__ void emb_ln0_bf16_to_f16_kernel(
    int V,
    int C,
    const uint16_t* __restrict__ emb,
    const uint16_t* __restrict__ weight,
    const uint16_t* __restrict__ bias,
    dtype* __restrict__ out,
    float eps) {
  // Precision path: bf16 inputs -> fp32 two-pass stats/affine -> fp16 output.
  const int tok = blockIdx.x;
  const int tid = threadIdx.x;
  if (tok >= V) {
    return;
  }
  const uint16_t* er = emb + static_cast<int64_t>(tok) * C;
  float sum = 0.0f;
  for (int c = tid; c < C; c += blockDim.x) {
    sum += bf16_bits_to_float_dev(er[c]);
  }
  const float mean = block_sum_t<256>(sum) / static_cast<float>(C);
  float var = 0.0f;
  for (int c = tid; c < C; c += blockDim.x) {
    const float d = bf16_bits_to_float_dev(er[c]) - mean;
    var += d * d;
  }
  const float rstd = rsqrtf(block_sum_t<256>(var) / static_cast<float>(C) + eps);
  dtype* yr = out + static_cast<int64_t>(tok) * C;
  for (int c = tid; c < C; c += blockDim.x) {
    const float x = bf16_bits_to_float_dev(er[c]);
    const float w = bf16_bits_to_float_dev(weight[c]);
    const float b = bf16_bits_to_float_dev(bias[c]);
    yr[c] = static_cast<dtype>((x - mean) * rstd * w + b);
  }
}

__global__ void add_f16_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ y,
    dtype* __restrict__ out,
    int64_t n_pairs) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n_pairs) {
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[i]);
    const float2 yv = __half22float2(reinterpret_cast<const __half2*>(y)[i]);
    reinterpret_cast<__half2*>(out)[i] = __floats2half2_rn(xv.x + yv.x, xv.y + yv.y);
  }
}

__global__ void layer_norm_f16_kernel(
    int C,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ y,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * C;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
    sum += v;
  }
  sum = block_sum_t<LN_THREADS>(sum);
  const float inv_c = 1.0f / static_cast<float>(C);
  const float mean = sum * inv_c;
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<LN_THREADS>(sum_var);
  const float var = sum_var * inv_c;
  const float rstd = rsqrtf(var + eps);
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
    const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
    const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
    *reinterpret_cast<__half*>(y + base + c) = __float2half_rn((v - mean) * rstd * w + b);
  }
}

__global__ void add_layer_norm_f16_kernel(
    int C,
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ x_out,
    dtype* __restrict__ y,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * C;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    sum += v;
  }
  sum = block_sum_t<LN_THREADS>(sum);
  const float inv_c = 1.0f / static_cast<float>(C);
  const float mean = sum * inv_c;
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<LN_THREADS>(sum_var);
  const float rstd = rsqrtf(sum_var * inv_c + eps);
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
    const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
    *reinterpret_cast<__half*>(x_out + base + c) = __float2half_rn(v);
    *reinterpret_cast<__half*>(y + base + c) = __float2half_rn((v - mean) * rstd * w + b);
  }
}

template <int Threads, bool VecStats, bool VecOut>
__global__ __launch_bounds__(Threads, 1) void layer_norm_f16_small_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ y,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  float sum = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 v = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      sum += v.x + v.y;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
      sum += v;
    }
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 v = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float dx = v.x - mean;
      const float dy = v.y - mean;
      sum_var += dx * dx + dy * dy;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
      const float d = v - mean;
      sum_var += d * d;
    }
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
  if constexpr (VecOut) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 v = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[idx]);
      const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[idx]);
      reinterpret_cast<__half2*>(y + base)[idx] = __floats2half2_rn(
          (v.x - mean) * rstd * w.x + b.x,
          (v.y - mean) * rstd * w.y + b.y);
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
      const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
      const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
      *reinterpret_cast<__half*>(y + base + c) = __float2half_rn((v - mean) * rstd * w + b);
    }
  }
}

template <int Threads, bool VecStats, bool VecOut>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_f16_small_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ x_out,
    dtype* __restrict__ y,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  float sum = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + base)[idx]);
      sum += xv.x + rv.x + xv.y + rv.y;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + base + c));
      sum += v;
    }
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + base)[idx]);
      const float dx = xv.x + rv.x - mean;
      const float dy = xv.y + rv.y - mean;
      sum_var += dx * dx + dy * dy;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + base + c));
      const float d = v - mean;
      sum_var += d * d;
    }
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
  if constexpr (VecOut) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + base)[idx]);
      const float sx = xv.x + rv.x;
      const float sy = xv.y + rv.y;
      const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[idx]);
      const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[idx]);
      reinterpret_cast<__half2*>(x_out + base)[idx] = __floats2half2_rn(sx, sy);
      reinterpret_cast<__half2*>(y + base)[idx] = __floats2half2_rn(
          (sx - mean) * rstd * w.x + b.x,
          (sy - mean) * rstd * w.y + b.y);
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + base + c));
      const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
      const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
      *reinterpret_cast<__half*>(x_out + base + c) = __float2half_rn(v);
      *reinterpret_cast<__half*>(y + base + c) = __float2half_rn((v - mean) * rstd * w + b);
    }
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_cmix_mix_f16_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ x_out,
    dtype* __restrict__ mixed,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  float sum = 0.0f;
  const int64_t base2 = base >> 1;
  constexpr int pairs = LN_SMALL_C >> 1;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    sum += xv.x + rv.x + xv.y + rv.y;
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const float d0 = x0 - mean;
    const float d1 = x1 - mean;
    sum_var += d0 * d0 + d1 * d1;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float2 mix = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x, (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(mixed)[base2 + p] =
        __floats2half2_rn(yv.x + (prev.x - yv.x) * mix.x, yv.y + (prev.y - yv.y) * mix.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_cmix_mix_f16_scalar_stats_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ x_out,
    dtype* __restrict__ mixed,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  const int64_t base2 = base >> 1;
  constexpr int pairs = LN_SMALL_C >> 1;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
    const int c = threadIdx.x + k * Threads;
    sum += __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
           __half2float(*reinterpret_cast<const __half*>(residual + base + c));
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
    const int c = threadIdx.x + k * Threads;
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float2 mix = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x, (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(mixed)[base2 + p] =
        __floats2half2_rn(yv.x + (prev.x - yv.x) * mix.x, yv.y + (prev.y - yv.y) * mix.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_tmix_mix6_f16_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ x_out,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base2 = row * (LN_SMALL_C >> 1);
  constexpr int pairs = LN_SMALL_C >> 1;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    sum += xv.x + rv.x + xv.y + rv.y;
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const float d0 = x0 - mean;
    const float d1 = x1 - mean;
    sum_var += d0 * d0 + d1 * d1;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x, (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    const float dx0 = prev.x - yv.x;
    const float dx1 = prev.y - yv.y;
    const float2 mr = __half22float2(reinterpret_cast<const __half2*>(x_r)[p]);
    const float2 mw = __half22float2(reinterpret_cast<const __half2*>(x_w)[p]);
    const float2 mk = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float2 mv = __half22float2(reinterpret_cast<const __half2*>(x_v)[p]);
    const float2 ma = __half22float2(reinterpret_cast<const __half2*>(x_a)[p]);
    const float2 mg = __half22float2(reinterpret_cast<const __half2*>(x_g)[p]);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(out_r)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mr.x, yv.y + dx1 * mr.y);
    reinterpret_cast<__half2*>(out_w)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mw.x, yv.y + dx1 * mw.y);
    reinterpret_cast<__half2*>(out_k)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mk.x, yv.y + dx1 * mk.y);
    reinterpret_cast<__half2*>(out_v)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mv.x, yv.y + dx1 * mv.y);
    reinterpret_cast<__half2*>(out_a)[base2 + p] = __floats2half2_rn(yv.x + dx0 * ma.x, yv.y + dx1 * ma.y);
    reinterpret_cast<__half2*>(out_g)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mg.x, yv.y + dx1 * mg.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_tmix_mix6_f16_scalar_stats_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ x_out,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  const int64_t base2 = row * (LN_SMALL_C >> 1);
  constexpr int pairs = LN_SMALL_C >> 1;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
    const int c = threadIdx.x + k * Threads;
    sum += __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
           __half2float(*reinterpret_cast<const __half*>(residual + base + c));
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
    const int c = threadIdx.x + k * Threads;
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x, (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    const float dx0 = prev.x - yv.x;
    const float dx1 = prev.y - yv.y;
    const float2 mr = __half22float2(reinterpret_cast<const __half2*>(x_r)[p]);
    const float2 mw = __half22float2(reinterpret_cast<const __half2*>(x_w)[p]);
    const float2 mk = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float2 mv = __half22float2(reinterpret_cast<const __half2*>(x_v)[p]);
    const float2 ma = __half22float2(reinterpret_cast<const __half2*>(x_a)[p]);
    const float2 mg = __half22float2(reinterpret_cast<const __half2*>(x_g)[p]);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(out_r)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mr.x, yv.y + dx1 * mr.y);
    reinterpret_cast<__half2*>(out_w)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mw.x, yv.y + dx1 * mw.y);
    reinterpret_cast<__half2*>(out_k)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mk.x, yv.y + dx1 * mk.y);
    reinterpret_cast<__half2*>(out_v)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mv.x, yv.y + dx1 * mv.y);
    reinterpret_cast<__half2*>(out_a)[base2 + p] = __floats2half2_rn(yv.x + dx0 * ma.x, yv.y + dx1 * ma.y);
    reinterpret_cast<__half2*>(out_g)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mg.x, yv.y + dx1 * mg.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

template <int Threads, bool VecStats, bool VecOut>
__global__ __launch_bounds__(Threads, 1) void add_last_layer_norm_f16_small_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ y,
    int64_t B,
    int64_t T,
    float eps) {
  const int64_t bidx = blockIdx.x;
  if (bidx >= B) {
    return;
  }
  const int64_t src = (bidx * T + (T - 1)) * LN_SMALL_C;
  const int64_t dst = bidx * LN_SMALL_C;
  float sum = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + src)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + src)[idx]);
      sum += xv.x + rv.x + xv.y + rv.y;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + src + c));
      sum += v;
    }
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + src)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + src)[idx]);
      const float dx = xv.x + rv.x - mean;
      const float dy = xv.y + rv.y - mean;
      sum_var += dx * dx + dy * dy;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + src + c));
      const float d = v - mean;
      sum_var += d * d;
    }
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
  if constexpr (VecOut) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + src)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + src)[idx]);
      const float sx = xv.x + rv.x;
      const float sy = xv.y + rv.y;
      const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[idx]);
      const float2 bb = __half22float2(reinterpret_cast<const __half2*>(bias)[idx]);
      reinterpret_cast<__half2*>(y + dst)[idx] = __floats2half2_rn(
          (sx - mean) * rstd * w.x + bb.x,
          (sy - mean) * rstd * w.y + bb.y);
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + src + c));
      const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
      const float bb = __half2float(*reinterpret_cast<const __half*>(bias + c));
      *reinterpret_cast<__half*>(y + dst + c) = __float2half_rn((v - mean) * rstd * w + bb);
    }
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_last_layer_norm_f16_generic_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ y,
    int64_t B,
    int64_t T,
    int C,
    float eps) {
  const int64_t bidx = blockIdx.x;
  if (bidx >= B) {
    return;
  }
  const int64_t src = (bidx * T + (T - 1)) * static_cast<int64_t>(C);
  const int64_t dst = bidx * static_cast<int64_t>(C);
  float sum = 0.0f;
  for (int c = threadIdx.x; c < C; c += Threads) {
    sum += __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
           __half2float(*reinterpret_cast<const __half*>(residual + src + c));
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum / static_cast<float>(C);
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += Threads) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + src + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var / static_cast<float>(C) + eps);
  const int pairs = C >> 1;
  for (int p = threadIdx.x; p < pairs; p += Threads) {
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + src)[p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + src)[p]);
    const float sx = xv.x + rv.x;
    const float sy = xv.y + rv.y;
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 bb = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    reinterpret_cast<__half2*>(y + dst)[p] = __floats2half2_rn(
        (sx - mean) * rstd * w.x + bb.x,
        (sy - mean) * rstd * w.y + bb.y);
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_cmix_mix_f16_generic_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ x_out,
    dtype* __restrict__ mixed,
    int64_t rows,
    int C,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * static_cast<int64_t>(C);
  float sum = 0.0f;
  for (int c = threadIdx.x; c < C; c += Threads) {
    sum += __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
           __half2float(*reinterpret_cast<const __half*>(residual + base + c));
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum / static_cast<float>(C);
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += Threads) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var / static_cast<float>(C) + eps);
  const int pairs = C >> 1;
  const int64_t base2 = base >> 1;
  for (int p = threadIdx.x; p < pairs; p += Threads) {
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float2 mix = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x, (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(mixed)[base2 + p] =
        __floats2half2_rn(yv.x + (prev.x - yv.x) * mix.x, yv.y + (prev.y - yv.y) * mix.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_tmix_mix6_f16_generic_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ x_out,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g,
    int64_t rows,
    int C,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * static_cast<int64_t>(C);
  float sum = 0.0f;
  for (int c = threadIdx.x; c < C; c += Threads) {
    sum += __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
           __half2float(*reinterpret_cast<const __half*>(residual + base + c));
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum / static_cast<float>(C);
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += Threads) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var / static_cast<float>(C) + eps);
  const int pairs = C >> 1;
  const int64_t base2 = base >> 1;
  for (int p = threadIdx.x; p < pairs; p += Threads) {
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x, (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    const float dx0 = prev.x - yv.x;
    const float dx1 = prev.y - yv.y;
    const float2 mr = __half22float2(reinterpret_cast<const __half2*>(x_r)[p]);
    const float2 mw = __half22float2(reinterpret_cast<const __half2*>(x_w)[p]);
    const float2 mk = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float2 mv = __half22float2(reinterpret_cast<const __half2*>(x_v)[p]);
    const float2 ma = __half22float2(reinterpret_cast<const __half2*>(x_a)[p]);
    const float2 mg = __half22float2(reinterpret_cast<const __half2*>(x_g)[p]);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(out_r)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mr.x, yv.y + dx1 * mr.y);
    reinterpret_cast<__half2*>(out_w)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mw.x, yv.y + dx1 * mw.y);
    reinterpret_cast<__half2*>(out_k)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mk.x, yv.y + dx1 * mk.y);
    reinterpret_cast<__half2*>(out_v)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mv.x, yv.y + dx1 * mv.y);
    reinterpret_cast<__half2*>(out_a)[base2 + p] = __floats2half2_rn(yv.x + dx0 * ma.x, yv.y + dx1 * ma.y);
    reinterpret_cast<__half2*>(out_g)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mg.x, yv.y + dx1 * mg.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

} // namespace

at::Tensor add_f16_cuda(at::Tensor x, at::Tensor y) {
  TORCH_CHECK((x.numel() % 2) == 0, "add_f16 requires even numel");
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t n_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  add_f16_kernel<<<static_cast<int>(ceil_div(n_pairs, threads)), threads, 0, stream>>>(
      x.data_ptr<dtype>(), y.data_ptr<dtype>(), out.data_ptr<dtype>(), n_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor layer_norm_f16_cuda(at::Tensor x, at::Tensor weight, at::Tensor bias, double eps) {
  auto y = at::empty_like(x);
  const int64_t c64 = x.size(-1);
  TORCH_CHECK(c64 <= INT_MAX, "C too large");
  const int C = static_cast<int>(c64);
  const int64_t rows = x.numel() / C;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (C == LN_SMALL_C) {
    if (rows >= 1024) {
      layer_norm_f16_small_kernel<LN_SMALL512_THREADS, true, true><<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(),
          weight.data_ptr<dtype>(),
          bias.data_ptr<dtype>(),
          y.data_ptr<dtype>(),
          rows,
          static_cast<float>(eps));
    } else if (rows >= 512) {
      layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(),
          weight.data_ptr<dtype>(),
          bias.data_ptr<dtype>(),
          y.data_ptr<dtype>(),
          rows,
          static_cast<float>(eps));
    } else {
      layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(),
          weight.data_ptr<dtype>(),
          bias.data_ptr<dtype>(),
          y.data_ptr<dtype>(),
          rows,
          static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
  }
  layer_norm_f16_kernel<<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
      C,
      x.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      y.data_ptr<dtype>(),
      rows,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor layer_norm_f16_small_cuda(at::Tensor x, at::Tensor weight, at::Tensor bias, double eps) {
  auto y = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
      x.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      y.data_ptr<dtype>(),
      rows,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor layer_norm_f16_small512_cuda(at::Tensor x, at::Tensor weight, at::Tensor bias, double eps) {
  auto y = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
      x.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      y.data_ptr<dtype>(),
      rows,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor emb_ln0_bf16_to_f16_cuda(at::Tensor emb, at::Tensor weight, at::Tensor bias, double eps) {
  auto out = at::empty(emb.sizes(), emb.options().dtype(at::kHalf));
  const int64_t v64 = emb.size(0);
  const int64_t c64 = emb.size(1);
  TORCH_CHECK(v64 <= INT_MAX && c64 <= INT_MAX, "emb shape too large");
  const int V = static_cast<int>(v64);
  const int C = static_cast<int>(c64);
  auto stream = at::cuda::getCurrentCUDAStream();
  emb_ln0_bf16_to_f16_kernel<<<V, 256, 0, stream>>>(
      V, C,
      reinterpret_cast<const uint16_t*>(emb.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint16_t*>(weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint16_t*>(bias.data_ptr<at::BFloat16>()),
      out.data_ptr<dtype>(),
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<at::Tensor> add_layer_norm_f16_cuda(at::Tensor x, at::Tensor residual, at::Tensor weight, at::Tensor bias, double eps) {
  auto x_out = at::empty_like(x);
  auto y = at::empty_like(x);
  const int64_t c64 = x.size(-1);
  TORCH_CHECK(c64 <= INT_MAX, "C too large");
  const int C = static_cast<int>(c64);
  const int64_t rows = x.numel() / C;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (C == LN_SMALL_C) {
    if (rows >= 1024) {
      add_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, true, true><<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
          x_out.data_ptr<dtype>(), y.data_ptr<dtype>(), rows, static_cast<float>(eps));
    } else if (rows >= 512) {
      add_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
          x_out.data_ptr<dtype>(), y.data_ptr<dtype>(), rows, static_cast<float>(eps));
    } else {
      add_layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
          x_out.data_ptr<dtype>(), y.data_ptr<dtype>(), rows, static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {x_out, y};
  }
  add_layer_norm_f16_kernel<<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
      C,
      x.data_ptr<dtype>(),
      residual.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      x_out.data_ptr<dtype>(),
      y.data_ptr<dtype>(),
      rows,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, y};
}

at::Tensor add_last_layer_norm_f16_cuda(at::Tensor x, at::Tensor residual, at::Tensor weight, at::Tensor bias, double eps) {
  const int64_t B = x.size(0);
  const int64_t T = x.size(1);
  const int64_t C = x.size(2);
  TORCH_CHECK((C % 2) == 0, "add_last_layer_norm_f16 requires even C");
  auto y = at::empty({B, C}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (C != LN_SMALL_C) {
    add_last_layer_norm_f16_generic_kernel<LN_THREADS><<<static_cast<int>(B), LN_THREADS, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
        y.data_ptr<dtype>(), B, T, static_cast<int>(C), static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
  }
  if (B >= 1024) {
    add_last_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, true, true><<<static_cast<int>(B), LN_SMALL512_THREADS, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
        y.data_ptr<dtype>(), B, T, static_cast<float>(eps));
  } else if (B >= 512) {
    add_last_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false><<<static_cast<int>(B), LN_SMALL512_THREADS, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
        y.data_ptr<dtype>(), B, T, static_cast<float>(eps));
  } else {
    add_last_layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false><<<static_cast<int>(B), LN_SMALL_THREADS, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
        y.data_ptr<dtype>(), B, T, static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

std::vector<at::Tensor> add_layer_norm_cmix_mix_f16_cuda(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor shift_state,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor x_k,
    double eps) {
  auto x_out = at::empty_like(x);
  auto mixed = at::empty_like(x);
  const int64_t C = x.size(-1);
  TORCH_CHECK((C % 2) == 0, "add_layer_norm_cmix_mix_f16 requires even C");
  const int64_t rows = x.numel() / C;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (C == LN_SMALL_C) {
    add_layer_norm_cmix_mix_f16_scalar_stats_kernel<LN_SMALL_THREADS><<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
        x.data_ptr<dtype>(),
        residual.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        weight.data_ptr<dtype>(),
        bias.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(),
        mixed.data_ptr<dtype>(),
        rows,
        static_cast<float>(eps));
  } else {
    add_layer_norm_cmix_mix_f16_generic_kernel<LN_THREADS><<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
        x.data_ptr<dtype>(),
        residual.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        weight.data_ptr<dtype>(),
        bias.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(),
        mixed.data_ptr<dtype>(),
        rows,
        static_cast<int>(C),
        static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, mixed};
}

std::vector<at::Tensor> add_layer_norm_cmix_mix_f16_scalar_stats_cuda(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor shift_state,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor x_k,
    double eps) {
  auto x_out = at::empty_like(x);
  auto mixed = at::empty_like(x);
  const int64_t C = x.size(-1);
  TORCH_CHECK((C % 2) == 0, "add_layer_norm_cmix_mix_f16 requires even C");
  const int64_t rows = x.numel() / C;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (C == LN_SMALL_C) {
    add_layer_norm_cmix_mix_f16_scalar_stats_kernel<LN_SMALL_THREADS><<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
        x.data_ptr<dtype>(),
        residual.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        weight.data_ptr<dtype>(),
        bias.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(),
        mixed.data_ptr<dtype>(),
        rows,
        static_cast<float>(eps));
  } else {
    add_layer_norm_cmix_mix_f16_generic_kernel<LN_THREADS><<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
        x.data_ptr<dtype>(),
        residual.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        weight.data_ptr<dtype>(),
        bias.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(),
        mixed.data_ptr<dtype>(),
        rows,
        static_cast<int>(C),
        static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, mixed};
}

std::vector<at::Tensor> add_layer_norm_tmix_mix6_f16_cuda(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor shift_state,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g,
    double eps) {
  auto x_out = at::empty_like(x);
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  const int64_t C = x.size(-1);
  TORCH_CHECK((C % 2) == 0, "add_layer_norm_tmix_mix6_f16 requires even C");
  const int64_t rows = x.numel() / C;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (C == LN_SMALL_C) {
    add_layer_norm_tmix_mix6_f16_scalar_stats_kernel<LN_SMALL_THREADS><<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
        x.data_ptr<dtype>(),
        residual.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        weight.data_ptr<dtype>(),
        bias.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(),
        x_w.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(),
        x_a.data_ptr<dtype>(),
        x_g.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(),
        out_w.data_ptr<dtype>(),
        out_k.data_ptr<dtype>(),
        out_v.data_ptr<dtype>(),
        out_a.data_ptr<dtype>(),
        out_g.data_ptr<dtype>(),
        rows,
        static_cast<float>(eps));
  } else {
    add_layer_norm_tmix_mix6_f16_generic_kernel<LN_THREADS><<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
        x.data_ptr<dtype>(),
        residual.data_ptr<dtype>(),
        shift_state.data_ptr<dtype>(),
        weight.data_ptr<dtype>(),
        bias.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(),
        x_w.data_ptr<dtype>(),
        x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(),
        x_a.data_ptr<dtype>(),
        x_g.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(),
        out_w.data_ptr<dtype>(),
        out_k.data_ptr<dtype>(),
        out_v.data_ptr<dtype>(),
        out_a.data_ptr<dtype>(),
        out_g.data_ptr<dtype>(),
        rows,
        static_cast<int>(C),
        static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, out_r, out_w, out_k, out_v, out_a, out_g};
}

std::vector<at::Tensor> add_layer_norm_tmix_mix6_f16_cfg_cuda(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor shift_state,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g,
    double eps,
    int threads) {
  auto x_out = at::empty_like(x);
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (threads == 256) {
    add_layer_norm_tmix_mix6_f16_kernel<256><<<static_cast<int>(rows), 256, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(), out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(),
        out_k.data_ptr<dtype>(), out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(),
        out_g.data_ptr<dtype>(), rows, static_cast<float>(eps));
  } else if (threads == 512) {
    add_layer_norm_tmix_mix6_f16_kernel<512><<<static_cast<int>(rows), 512, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(), out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(),
        out_k.data_ptr<dtype>(), out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(),
        out_g.data_ptr<dtype>(), rows, static_cast<float>(eps));
  } else {
    add_layer_norm_tmix_mix6_f16_kernel<1024><<<static_cast<int>(rows), 1024, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(), out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(),
        out_k.data_ptr<dtype>(), out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(),
        out_g.data_ptr<dtype>(), rows, static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, out_r, out_w, out_k, out_v, out_a, out_g};
}

std::vector<at::Tensor> add_layer_norm_tmix_mix6_f16_scalar_stats_cuda(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor shift_state,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g,
    double eps) {
  auto x_out = at::empty_like(x);
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  add_layer_norm_tmix_mix6_f16_scalar_stats_kernel<LN_SMALL_THREADS><<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
      x.data_ptr<dtype>(),
      residual.data_ptr<dtype>(),
      shift_state.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      x_r.data_ptr<dtype>(),
      x_w.data_ptr<dtype>(),
      x_k.data_ptr<dtype>(),
      x_v.data_ptr<dtype>(),
      x_a.data_ptr<dtype>(),
      x_g.data_ptr<dtype>(),
      x_out.data_ptr<dtype>(),
      out_r.data_ptr<dtype>(),
      out_w.data_ptr<dtype>(),
      out_k.data_ptr<dtype>(),
      out_v.data_ptr<dtype>(),
      out_a.data_ptr<dtype>(),
      out_g.data_ptr<dtype>(),
      rows,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, out_r, out_w, out_k, out_v, out_a, out_g};
}

std::vector<at::Tensor> add_layer_norm_cmix_mix_f16_cfg_cuda(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor shift_state,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor x_k,
    double eps,
    int threads) {
  auto x_out = at::empty_like(x);
  auto mixed = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (threads == 256) {
    add_layer_norm_cmix_mix_f16_kernel<256><<<static_cast<int>(rows), 256, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        weight.data_ptr<dtype>(), bias.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(), mixed.data_ptr<dtype>(), rows, static_cast<float>(eps));
  } else if (threads == 512) {
    add_layer_norm_cmix_mix_f16_kernel<512><<<static_cast<int>(rows), 512, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        weight.data_ptr<dtype>(), bias.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(), mixed.data_ptr<dtype>(), rows, static_cast<float>(eps));
  } else {
    add_layer_norm_cmix_mix_f16_kernel<1024><<<static_cast<int>(rows), 1024, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        weight.data_ptr<dtype>(), bias.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(), mixed.data_ptr<dtype>(), rows, static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, mixed};
}
