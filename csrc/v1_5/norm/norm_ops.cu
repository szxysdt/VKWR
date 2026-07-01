#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

#include <climits>

#include "common/warp_primitives.cuh"

using dtype = at::Half;

namespace {

constexpr int LN_THREADS = 256;
constexpr int LN_SMALL_THREADS = 1024;
constexpr int LN_SMALL512_THREADS = 512;
constexpr int LN_SMALL_C = 4096;

// ===== Fused add_ln + tmix_mix6 varlen kernels (two-pass) =====

// Pass 1: compute add+LN, write y_tmp[row] and x_out[row]
template <int Threads>
__global__
__launch_bounds__(Threads, 1) void add_layer_norm_tmix_mix6_f16_ln_pass_varlen(
    const dtype* __restrict__ x, const dtype* __restrict__ residual,
    const dtype* __restrict__ weight, const dtype* __restrict__ bias,
    dtype* __restrict__ x_out, dtype* __restrict__ y_tmp, int64_t rows, int C,
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
    const float v =
        __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
        __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var / static_cast<float>(C) + eps);
  const int pairs = C >> 1;
  const int64_t base2 = base >> 1;
  for (int p = threadIdx.x; p < pairs; p += Threads) {
    const float2 xv =
        __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv =
        __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w =
        __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x,
                                         (x1 - mean) * rstd * w.y + b.y);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(y_tmp)[base2 + p] = y2;
  }
}

// Pass 2: mix pass — reads y_tmp[row] for cur, y_tmp[row-1] or shift_state for
// prev
template <int Threads>
__global__
__launch_bounds__(Threads, 1) void add_layer_norm_tmix_mix6_f16_mix_pass_varlen(
    const dtype* __restrict__ y_tmp, dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_r, const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k, const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a, const dtype* __restrict__ x_g,
    dtype* __restrict__ out_r, dtype* __restrict__ out_w,
    dtype* __restrict__ out_k, dtype* __restrict__ out_v,
    dtype* __restrict__ out_a, dtype* __restrict__ out_g,
    const int* __restrict__ query_start_loc, const int* __restrict__ req_id,
    int64_t rows, int C) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int bid = req_id[row];
  const int t_local = row - query_start_loc[bid];
  const int my_t = query_start_loc[bid + 1] - query_start_loc[bid];
  const int64_t base2 = static_cast<int64_t>(row) * (C >> 1);
  const int64_t bid_base2 = static_cast<int64_t>(bid) * (C >> 1);
  const int pairs = C >> 1;
  for (int p = threadIdx.x; p < pairs; p += Threads) {
    const float2 yv =
        __half22float2(reinterpret_cast<const __half2*>(y_tmp)[base2 + p]);
    float2 prev;
    if (t_local == 0) {
      prev = __half22float2(
          reinterpret_cast<const __half2*>(shift_state)[bid_base2 + p]);
    } else {
      prev = __half22float2(reinterpret_cast<const __half2*>(
          y_tmp)[static_cast<int64_t>(row - 1) * (C >> 1) + p]);
    }
    const float dx0 = prev.x - yv.x;
    const float dx1 = prev.y - yv.y;
    const float2 mr = __half22float2(reinterpret_cast<const __half2*>(x_r)[p]);
    const float2 mw = __half22float2(reinterpret_cast<const __half2*>(x_w)[p]);
    const float2 mk = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float2 mv = __half22float2(reinterpret_cast<const __half2*>(x_v)[p]);
    const float2 ma = __half22float2(reinterpret_cast<const __half2*>(x_a)[p]);
    const float2 mg = __half22float2(reinterpret_cast<const __half2*>(x_g)[p]);
    reinterpret_cast<__half2*>(out_r)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mr.x, yv.y + dx1 * mr.y);
    reinterpret_cast<__half2*>(out_w)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mw.x, yv.y + dx1 * mw.y);
    reinterpret_cast<__half2*>(out_k)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mk.x, yv.y + dx1 * mk.y);
    reinterpret_cast<__half2*>(out_v)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mv.x, yv.y + dx1 * mv.y);
    reinterpret_cast<__half2*>(out_a)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * ma.x, yv.y + dx1 * ma.y);
    reinterpret_cast<__half2*>(out_g)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mg.x, yv.y + dx1 * mg.y);
    if (t_local == my_t - 1) {
      reinterpret_cast<__half2*>(shift_state)[bid_base2 + p] =
          reinterpret_cast<const __half2*>(y_tmp)[base2 + p];
    }
  }
}

// ===== Fused add_ln + cmix_mix varlen kernels (two-pass) =====

// Pass 1: compute add+LN, write y_tmp[row] and x_out[row]
template <int Threads>
__global__
__launch_bounds__(Threads, 1) void add_layer_norm_cmix_mix_f16_ln_pass_varlen(
    const dtype* __restrict__ x, const dtype* __restrict__ residual,
    const dtype* __restrict__ weight, const dtype* __restrict__ bias,
    dtype* __restrict__ x_out, dtype* __restrict__ y_tmp, int64_t rows, int C,
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
    const float v =
        __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
        __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var / static_cast<float>(C) + eps);
  const int pairs = C >> 1;
  const int64_t base2 = base >> 1;
  for (int p = threadIdx.x; p < pairs; p += Threads) {
    const float2 xv =
        __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv =
        __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w =
        __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x,
                                         (x1 - mean) * rstd * w.y + b.y);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(y_tmp)[base2 + p] = y2;
  }
}

// Pass 2: mix pass
template <int Threads>
__global__
__launch_bounds__(Threads, 1) void add_layer_norm_cmix_mix_f16_mix_pass_varlen(
    const dtype* __restrict__ y_tmp, dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_k, dtype* __restrict__ mixed,
    const int* __restrict__ query_start_loc, const int* __restrict__ req_id,
    int64_t rows, int C) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int bid = req_id[row];
  const int t_local = row - query_start_loc[bid];
  const int my_t = query_start_loc[bid + 1] - query_start_loc[bid];
  const int64_t base2 = static_cast<int64_t>(row) * (C >> 1);
  const int64_t bid_base2 = static_cast<int64_t>(bid) * (C >> 1);
  const int pairs = C >> 1;
  for (int p = threadIdx.x; p < pairs; p += Threads) {
    const float2 yv =
        __half22float2(reinterpret_cast<const __half2*>(y_tmp)[base2 + p]);
    float2 prev;
    if (t_local == 0) {
      prev = __half22float2(
          reinterpret_cast<const __half2*>(shift_state)[bid_base2 + p]);
    } else {
      prev = __half22float2(reinterpret_cast<const __half2*>(
          y_tmp)[static_cast<int64_t>(row - 1) * (C >> 1) + p]);
    }
    const float2 mix = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    reinterpret_cast<__half2*>(mixed)[base2 + p] = __floats2half2_rn(
        yv.x + (prev.x - yv.x) * mix.x, yv.y + (prev.y - yv.y) * mix.y);
    if (t_local == my_t - 1) {
      reinterpret_cast<__half2*>(shift_state)[bid_base2 + p] =
          reinterpret_cast<const __half2*>(y_tmp)[base2 + p];
    }
  }
}

// ===== Scalar stats variants (C == LN_SMALL_C == 4096) =====

// tmix Pass 1: LN for C=4096
template <int Threads>
__global__
__launch_bounds__(Threads, 1) void add_layer_norm_tmix_mix6_f16_ln_pass_scalar_varlen(
    const dtype* __restrict__ x, const dtype* __restrict__ residual,
    const dtype* __restrict__ weight, const dtype* __restrict__ bias,
    dtype* __restrict__ x_out, dtype* __restrict__ y_tmp, int64_t rows,
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
    const float v =
        __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
        __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd =
      rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv =
        __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv =
        __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w =
        __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x,
                                         (x1 - mean) * rstd * w.y + b.y);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(y_tmp)[base2 + p] = y2;
  }
}

// tmix Pass 2: mix for C=4096
template <int Threads>
__global__
__launch_bounds__(Threads, 1) void add_layer_norm_tmix_mix6_f16_mix_pass_scalar_varlen(
    const dtype* __restrict__ y_tmp, dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_r, const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k, const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a, const dtype* __restrict__ x_g,
    dtype* __restrict__ out_r, dtype* __restrict__ out_w,
    dtype* __restrict__ out_k, dtype* __restrict__ out_v,
    dtype* __restrict__ out_a, dtype* __restrict__ out_g,
    const int* __restrict__ query_start_loc, const int* __restrict__ req_id,
    int64_t rows) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int bid = req_id[row];
  const int t_local = row - query_start_loc[bid];
  const int my_t = query_start_loc[bid + 1] - query_start_loc[bid];
  const int64_t base2 = row * (LN_SMALL_C >> 1);
  const int64_t bid_base2 = static_cast<int64_t>(bid) * (LN_SMALL_C >> 1);
  constexpr int pairs = LN_SMALL_C >> 1;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 yv =
        __half22float2(reinterpret_cast<const __half2*>(y_tmp)[base2 + p]);
    float2 prev;
    if (t_local == 0) {
      prev = __half22float2(
          reinterpret_cast<const __half2*>(shift_state)[bid_base2 + p]);
    } else {
      prev = __half22float2(reinterpret_cast<const __half2*>(
          y_tmp)[((row - 1) * (LN_SMALL_C >> 1)) + p]);
    }
    const float dx0 = prev.x - yv.x;
    const float dx1 = prev.y - yv.y;
    const float2 mr = __half22float2(reinterpret_cast<const __half2*>(x_r)[p]);
    const float2 mw = __half22float2(reinterpret_cast<const __half2*>(x_w)[p]);
    const float2 mk = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float2 mv = __half22float2(reinterpret_cast<const __half2*>(x_v)[p]);
    const float2 ma = __half22float2(reinterpret_cast<const __half2*>(x_a)[p]);
    const float2 mg = __half22float2(reinterpret_cast<const __half2*>(x_g)[p]);
    reinterpret_cast<__half2*>(out_r)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mr.x, yv.y + dx1 * mr.y);
    reinterpret_cast<__half2*>(out_w)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mw.x, yv.y + dx1 * mw.y);
    reinterpret_cast<__half2*>(out_k)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mk.x, yv.y + dx1 * mk.y);
    reinterpret_cast<__half2*>(out_v)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mv.x, yv.y + dx1 * mv.y);
    reinterpret_cast<__half2*>(out_a)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * ma.x, yv.y + dx1 * ma.y);
    reinterpret_cast<__half2*>(out_g)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mg.x, yv.y + dx1 * mg.y);
    if (t_local == my_t - 1) {
      reinterpret_cast<__half2*>(shift_state)[bid_base2 + p] =
          reinterpret_cast<const __half2*>(y_tmp)[base2 + p];
    }
  }
}

// cmix Pass 1: LN for C=4096
template <int Threads>
__global__
__launch_bounds__(Threads, 1) void add_layer_norm_cmix_mix_f16_ln_pass_scalar_varlen(
    const dtype* __restrict__ x, const dtype* __restrict__ residual,
    const dtype* __restrict__ weight, const dtype* __restrict__ bias,
    dtype* __restrict__ x_out, dtype* __restrict__ y_tmp, int64_t rows,
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
    const float v =
        __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
        __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd =
      rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv =
        __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv =
        __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w =
        __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x,
                                         (x1 - mean) * rstd * w.y + b.y);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(y_tmp)[base2 + p] = y2;
  }
}

// cmix Pass 2: mix for C=4096
template <int Threads>
__global__
__launch_bounds__(Threads, 1) void add_layer_norm_cmix_mix_f16_mix_pass_scalar_varlen(
    const dtype* __restrict__ y_tmp, dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_k, dtype* __restrict__ mixed,
    const int* __restrict__ query_start_loc, const int* __restrict__ req_id,
    int64_t rows) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int bid = req_id[row];
  const int t_local = row - query_start_loc[bid];
  const int my_t = query_start_loc[bid + 1] - query_start_loc[bid];
  const int64_t base2 = row * (LN_SMALL_C >> 1);
  const int64_t bid_base2 = static_cast<int64_t>(bid) * (LN_SMALL_C >> 1);
  constexpr int pairs = LN_SMALL_C >> 1;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 yv =
        __half22float2(reinterpret_cast<const __half2*>(y_tmp)[base2 + p]);
    float2 prev;
    if (t_local == 0) {
      prev = __half22float2(
          reinterpret_cast<const __half2*>(shift_state)[bid_base2 + p]);
    } else {
      prev = __half22float2(reinterpret_cast<const __half2*>(
          y_tmp)[((row - 1) * (LN_SMALL_C >> 1)) + p]);
    }
    const float2 mix = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    reinterpret_cast<__half2*>(mixed)[base2 + p] = __floats2half2_rn(
        yv.x + (prev.x - yv.x) * mix.x, yv.y + (prev.y - yv.y) * mix.y);
    if (t_local == my_t - 1) {
      reinterpret_cast<__half2*>(shift_state)[bid_base2 + p] =
          reinterpret_cast<const __half2*>(y_tmp)[base2 + p];
    }
  }
}

// ===== add_last_layer_norm varlen kernels =====

template <int Threads, bool VecStats, bool VecOut>
__global__
__launch_bounds__(Threads, 1) void add_last_layer_norm_f16_small_kernel_varlen(
    const dtype* __restrict__ x, const dtype* __restrict__ residual,
    const dtype* __restrict__ weight, const dtype* __restrict__ bias,
    dtype* __restrict__ y, int64_t B, const int* __restrict__ query_start_loc,
    float eps) {
  const int64_t bidx = blockIdx.x;
  if (bidx >= B) {
    return;
  }
  const int last_token = query_start_loc[bidx + 1] - 1;
  const int64_t src = last_token * LN_SMALL_C;
  const int64_t dst = bidx * LN_SMALL_C;
  float sum = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv =
          __half22float2(reinterpret_cast<const __half2*>(x + src)[idx]);
      const float2 rv =
          __half22float2(reinterpret_cast<const __half2*>(residual + src)[idx]);
      sum += xv.x + rv.x + xv.y + rv.y;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v =
          __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
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
      const float2 xv =
          __half22float2(reinterpret_cast<const __half2*>(x + src)[idx]);
      const float2 rv =
          __half22float2(reinterpret_cast<const __half2*>(residual + src)[idx]);
      const float dx = xv.x + rv.x - mean;
      const float dy = xv.y + rv.y - mean;
      sum_var += dx * dx + dy * dy;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v =
          __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
          __half2float(*reinterpret_cast<const __half*>(residual + src + c));
      const float d = v - mean;
      sum_var += d * d;
    }
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd =
      rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
  if constexpr (VecOut) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv =
          __half22float2(reinterpret_cast<const __half2*>(x + src)[idx]);
      const float2 rv =
          __half22float2(reinterpret_cast<const __half2*>(residual + src)[idx]);
      const float sx = xv.x + rv.x;
      const float sy = xv.y + rv.y;
      const float2 w =
          __half22float2(reinterpret_cast<const __half2*>(weight)[idx]);
      const float2 bb =
          __half22float2(reinterpret_cast<const __half2*>(bias)[idx]);
      reinterpret_cast<__half2*>(y + dst)[idx] = __floats2half2_rn(
          (sx - mean) * rstd * w.x + bb.x, (sy - mean) * rstd * w.y + bb.y);
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v =
          __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
          __half2float(*reinterpret_cast<const __half*>(residual + src + c));
      const float w =
          __half2float(*reinterpret_cast<const __half*>(weight + c));
      const float bb = __half2float(*reinterpret_cast<const __half*>(bias + c));
      *reinterpret_cast<__half*>(y + dst + c) =
          __float2half_rn((v - mean) * rstd * w + bb);
    }
  }
}

template <int Threads>
__global__
__launch_bounds__(Threads, 1) void add_last_layer_norm_f16_generic_kernel_varlen(
    const dtype* __restrict__ x, const dtype* __restrict__ residual,
    const dtype* __restrict__ weight, const dtype* __restrict__ bias,
    dtype* __restrict__ y, int64_t B, const int* __restrict__ query_start_loc,
    int C, float eps) {
  const int64_t bidx = blockIdx.x;
  if (bidx >= B) {
    return;
  }
  const int last_token = query_start_loc[bidx + 1] - 1;
  const int64_t src = last_token * static_cast<int64_t>(C);
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
    const float v =
        __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
        __half2float(*reinterpret_cast<const __half*>(residual + src + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var / static_cast<float>(C) + eps);
  const int pairs = C >> 1;
  for (int p = threadIdx.x; p < pairs; p += Threads) {
    const float2 xv =
        __half22float2(reinterpret_cast<const __half2*>(x + src)[p]);
    const float2 rv =
        __half22float2(reinterpret_cast<const __half2*>(residual + src)[p]);
    const float sx = xv.x + rv.x;
    const float sy = xv.y + rv.y;
    const float2 w =
        __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 bb = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    reinterpret_cast<__half2*>(y + dst)[p] = __floats2half2_rn(
        (sx - mean) * rstd * w.x + bb.x, (sy - mean) * rstd * w.y + bb.y);
  }
}

}  // namespace

// ===== Host functions =====

std::vector<at::Tensor> add_layer_norm_tmix_mix6_f16_cuda_varlen(
    at::Tensor x, at::Tensor residual, at::Tensor shift_state,
    at::Tensor weight, at::Tensor bias, at::Tensor x_r, at::Tensor x_w,
    at::Tensor x_k, at::Tensor x_v, at::Tensor x_a, at::Tensor x_g,
    at::Tensor query_start_loc, at::Tensor req_id, double eps) {
  auto x_out = at::empty_like(x);
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  const int64_t C = x.size(-1);
  TORCH_CHECK((C % 2) == 0,
              "add_layer_norm_tmix_mix6_f16_varlen requires even C");
  const int64_t rows = x.numel() / C;
  const int* qsl_ptr = query_start_loc.data_ptr<int>();
  const int* rid_ptr = req_id.data_ptr<int>();
  auto stream = at::cuda::getCurrentCUDAStream();
  auto y_tmp = at::empty_like(x);
  if (C == LN_SMALL_C) {
    add_layer_norm_tmix_mix6_f16_ln_pass_scalar_varlen<LN_SMALL_THREADS>
        <<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
            x_out.data_ptr<dtype>(), y_tmp.data_ptr<dtype>(), rows,
            static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    add_layer_norm_tmix_mix6_f16_mix_pass_scalar_varlen<LN_SMALL_THREADS>
        <<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
            y_tmp.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
            x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
            x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
            out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(),
            out_k.data_ptr<dtype>(), out_v.data_ptr<dtype>(),
            out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(), qsl_ptr, rid_ptr,
            rows);
  } else {
    add_layer_norm_tmix_mix6_f16_ln_pass_varlen<LN_THREADS>
        <<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
            x_out.data_ptr<dtype>(), y_tmp.data_ptr<dtype>(), rows,
            static_cast<int>(C), static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    add_layer_norm_tmix_mix6_f16_mix_pass_varlen<LN_THREADS>
        <<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
            y_tmp.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
            x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
            x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
            out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(),
            out_k.data_ptr<dtype>(), out_v.data_ptr<dtype>(),
            out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(), qsl_ptr, rid_ptr,
            rows, static_cast<int>(C));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, out_r, out_w, out_k, out_v, out_a, out_g};
}

std::vector<at::Tensor> add_layer_norm_cmix_mix_f16_cuda_varlen(
    at::Tensor x, at::Tensor residual, at::Tensor shift_state,
    at::Tensor weight, at::Tensor bias, at::Tensor x_k,
    at::Tensor query_start_loc, at::Tensor req_id, double eps) {
  auto x_out = at::empty_like(x);
  auto mixed = at::empty_like(x);
  const int64_t C = x.size(-1);
  TORCH_CHECK((C % 2) == 0,
              "add_layer_norm_cmix_mix_f16_varlen requires even C");
  const int64_t rows = x.numel() / C;
  const int* qsl_ptr = query_start_loc.data_ptr<int>();
  const int* rid_ptr = req_id.data_ptr<int>();
  auto stream = at::cuda::getCurrentCUDAStream();
  auto y_tmp = at::empty_like(x);
  if (C == LN_SMALL_C) {
    add_layer_norm_cmix_mix_f16_ln_pass_scalar_varlen<LN_SMALL_THREADS>
        <<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
            x_out.data_ptr<dtype>(), y_tmp.data_ptr<dtype>(), rows,
            static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    add_layer_norm_cmix_mix_f16_mix_pass_scalar_varlen<LN_SMALL_THREADS>
        <<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
            y_tmp.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
            x_k.data_ptr<dtype>(), mixed.data_ptr<dtype>(), qsl_ptr, rid_ptr,
            rows);
  } else {
    add_layer_norm_cmix_mix_f16_ln_pass_varlen<LN_THREADS>
        <<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
            x_out.data_ptr<dtype>(), y_tmp.data_ptr<dtype>(), rows,
            static_cast<int>(C), static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    add_layer_norm_cmix_mix_f16_mix_pass_varlen<LN_THREADS>
        <<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
            y_tmp.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
            x_k.data_ptr<dtype>(), mixed.data_ptr<dtype>(), qsl_ptr, rid_ptr,
            rows, static_cast<int>(C));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, mixed};
}

at::Tensor add_last_layer_norm_f16_cuda_varlen(
    at::Tensor x, at::Tensor residual, at::Tensor weight, at::Tensor bias,
    at::Tensor query_start_loc, double eps) {
  const int64_t B = query_start_loc.size(0) - 1;
  const int64_t C = x.size(-1);
  TORCH_CHECK((C % 2) == 0, "add_last_layer_norm_f16_varlen requires even C");
  auto y = at::empty({B, C}, x.options());
  const int* qsl_ptr = query_start_loc.data_ptr<int>();
  auto stream = at::cuda::getCurrentCUDAStream();
  if (C != LN_SMALL_C) {
    add_last_layer_norm_f16_generic_kernel_varlen<LN_THREADS>
        <<<static_cast<int>(B), LN_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
            y.data_ptr<dtype>(), B, qsl_ptr, static_cast<int>(C),
            static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
  }
  if (B >= 1024) {
    add_last_layer_norm_f16_small_kernel_varlen<LN_SMALL512_THREADS, true, true>
        <<<static_cast<int>(B), LN_SMALL512_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
            y.data_ptr<dtype>(), B, qsl_ptr, static_cast<float>(eps));
  } else if (B >= 512) {
    add_last_layer_norm_f16_small_kernel_varlen<LN_SMALL512_THREADS, false,
                                                false>
        <<<static_cast<int>(B), LN_SMALL512_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
            y.data_ptr<dtype>(), B, qsl_ptr, static_cast<float>(eps));
  } else {
    add_last_layer_norm_f16_small_kernel_varlen<LN_SMALL_THREADS, false, false>
        <<<static_cast<int>(B), LN_SMALL_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
            y.data_ptr<dtype>(), B, qsl_ptr, static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
