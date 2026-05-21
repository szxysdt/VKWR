#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "v1/common/warp_primitives.cuh"

using dtype = at::Half;

namespace {

template <int Threads>
__global__ __launch_bounds__(Threads, 2) void linear_wag_rank_in_f16_kernel(
    int M,
    int K,
    int Rw,
    int Ra,
    int Rg,
    int Rmax,
    const dtype* __restrict__ xw,
    const dtype* __restrict__ xa,
    const dtype* __restrict__ xg,
    const dtype* __restrict__ w1_t,
    const dtype* __restrict__ a1_t,
    const dtype* __restrict__ g1_t,
    dtype* __restrict__ w1,
    dtype* __restrict__ a1,
    dtype* __restrict__ g1) {
  const int r = blockIdx.x;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int R = Rw;
  const dtype* x = xw;
  const dtype* wt = w1_t;
  dtype* y = w1;
  if (group == 1) {
    R = Ra;
    x = xa;
    wt = a1_t;
    y = a1;
  } else if (group == 2) {
    R = Rg;
    x = xg;
    wt = g1_t;
    y = g1;
  }
  if (m >= M || r >= R || r >= Rmax) {
    return;
  }
  float acc = 0.0f;
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const dtype* w_row = wt + static_cast<int64_t>(r) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(w_row + k));
    acc = fmaf(xv.x, wv.x, acc);
    acc = fmaf(xv.y, wv.y, acc);
  }
  if ((K & 1) && threadIdx.x == 0) {
    acc = fmaf(__half2float(*reinterpret_cast<const __half*>(x_row + K - 1)),
               __half2float(*reinterpret_cast<const __half*>(w_row + K - 1)),
               acc);
  }
  acc = block_sum_t<Threads>(acc);
  if (threadIdx.x == 0) {
    *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * R + r) = __float2half_rn(acc);
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 2) void linear_wagv_rank_in_f16_kernel(
    int M,
    int K,
    int Rw,
    int Ra,
    int Rg,
    int Rv,
    int Rmax,
    const dtype* __restrict__ xw,
    const dtype* __restrict__ xa,
    const dtype* __restrict__ xg,
    const dtype* __restrict__ xv,
    const dtype* __restrict__ w1_t,
    const dtype* __restrict__ a1_t,
    const dtype* __restrict__ g1_t,
    const dtype* __restrict__ v1_t,
    dtype* __restrict__ w1,
    dtype* __restrict__ a1,
    dtype* __restrict__ g1,
    dtype* __restrict__ v1) {
  const int r = blockIdx.x;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int R = Rw;
  const dtype* x = xw;
  const dtype* wt = w1_t;
  dtype* y = w1;
  if (group == 1) {
    R = Ra;
    x = xa;
    wt = a1_t;
    y = a1;
  } else if (group == 2) {
    R = Rg;
    x = xg;
    wt = g1_t;
    y = g1;
  } else if (group == 3) {
    R = Rv;
    x = xv;
    wt = v1_t;
    y = v1;
  }
  if (m >= M || r >= R || r >= Rmax) {
    return;
  }
  float acc = 0.0f;
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const dtype* w_row = wt + static_cast<int64_t>(r) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv2 = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(w_row + k));
    acc = fmaf(xv2.x, wv.x, acc);
    acc = fmaf(xv2.y, wv.y, acc);
  }
  if ((K & 1) && threadIdx.x == 0) {
    acc = fmaf(__half2float(*reinterpret_cast<const __half*>(x_row + K - 1)),
               __half2float(*reinterpret_cast<const __half*>(w_row + K - 1)),
               acc);
  }
  acc = block_sum_t<Threads>(acc);
  if (threadIdx.x == 0) {
    *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * R + r) = __float2half_rn(acc);
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_wag_rank_out_f16_kernel(
    int M,
    int C,
    int Kw,
    int Ka,
    int Kg,
    const dtype* __restrict__ w1,
    const dtype* __restrict__ a1,
    const dtype* __restrict__ g1,
    const dtype* __restrict__ w2_t,
    const dtype* __restrict__ a2_t,
    const dtype* __restrict__ g2_t,
    dtype* __restrict__ w,
    dtype* __restrict__ a,
    dtype* __restrict__ g) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int K = Kw;
  const dtype* x = w1;
  const dtype* wt = w2_t;
  dtype* y = w;
  if (group == 1) {
    K = Ka;
    x = a1;
    wt = a2_t;
    y = a;
  } else if (group == 2) {
    K = Kg;
    x = g1;
    wt = g2_t;
    y = g;
  }
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
    if (group == 0) {
      xv = tanhf(xv);
    } else if (group == 2) {
      xv = 1.0f / (1.0f + expf(-xv));
    }
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < C) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(wt + static_cast<int64_t>(n) * K + k)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int u = 0; u < Threads / 32; ++u) {
        sum += partial[u][j];
      }
      const int n = n0 + j;
      if (n < C) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * C + n) = __float2half_rn(sum);
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_wagv_rank_out_f16_kernel(
    int M,
    int C,
    int Kw,
    int Ka,
    int Kg,
    int Kv,
    const dtype* __restrict__ w1,
    const dtype* __restrict__ a1,
    const dtype* __restrict__ g1,
    const dtype* __restrict__ v1,
    const dtype* __restrict__ w2_t,
    const dtype* __restrict__ a2_t,
    const dtype* __restrict__ g2_t,
    const dtype* __restrict__ v2_t,
    const dtype* __restrict__ v,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
    dtype* __restrict__ w,
    dtype* __restrict__ a,
    dtype* __restrict__ g,
    dtype* __restrict__ v_out) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int K = Kw;
  const dtype* x = w1;
  const dtype* wt = w2_t;
  dtype* y = w;
  if (group == 1) {
    K = Ka;
    x = a1;
    wt = a2_t;
    y = a;
  } else if (group == 2) {
    K = Kg;
    x = g1;
    wt = g2_t;
    y = g;
  } else if (group == 3) {
    K = Kv;
    x = v1;
    wt = v2_t;
    y = v_out;
  }
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
    if (group == 0) {
      xv = tanhf(xv);
    } else if (group == 2) {
      xv = 1.0f / (1.0f + expf(-xv));
    }
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < C) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(wt + static_cast<int64_t>(n) * K + k)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int u = 0; u < Threads / 32; ++u) {
        sum += partial[u][j];
      }
      const int n = n0 + j;
      if (n < C) {
        if (group == 3) {
          const int64_t idx = static_cast<int64_t>(m) * C + n;
          const float vv = __half2float(*reinterpret_cast<const __half*>(v + idx));
          const float vf = __half2float(*reinterpret_cast<const __half*>(v_first + idx));
          const float gate = 1.0f / (1.0f + expf(-(__half2float(*reinterpret_cast<const __half*>(v0 + n)) + sum)));
          *reinterpret_cast<__half*>(y + idx) = __float2half_rn(vv + (vf - vv) * gate);
        } else {
          *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * C + n) = __float2half_rn(sum);
        }
      }
    }
  }
}

} // namespace

void rwkv7_v3a_linear_wag_rank_in_f16_launch(
    cudaStream_t stream, int M, int K, int Rw, int Ra, int Rg,
    const dtype* xw, const dtype* xa, const dtype* xg,
    const dtype* w1_t, const dtype* a1_t, const dtype* g1_t,
    dtype* w1, dtype* a1, dtype* g1) {
  const int Rmax = std::max(Rw, std::max(Ra, Rg));
  linear_wag_rank_in_f16_kernel<256><<<dim3(Rmax, M, 3), 256, 0, stream>>>(
      M, K, Rw, Ra, Rg, Rmax, xw, xa, xg, w1_t, a1_t, g1_t, w1, a1, g1);
}

void rwkv7_v3a_linear_wagv_rank_in_f16_launch(
    cudaStream_t stream, int M, int K, int Rw, int Ra, int Rg, int Rv,
    const dtype* xw, const dtype* xa, const dtype* xg, const dtype* xv,
    const dtype* w1_t, const dtype* a1_t, const dtype* g1_t, const dtype* v1_t,
    dtype* w1, dtype* a1, dtype* g1, dtype* v1) {
  const int Rmax = std::max(std::max(Rw, Ra), std::max(Rg, Rv));
  linear_wagv_rank_in_f16_kernel<256><<<dim3(Rmax, M, 4), 256, 0, stream>>>(
      M, K, Rw, Ra, Rg, Rv, Rmax, xw, xa, xg, xv, w1_t, a1_t, g1_t, v1_t, w1, a1, g1, v1);
}

void rwkv7_v3a_linear_wag_rank_out_f16_launch(
    cudaStream_t stream, int M, int C, int Kw, int Ka, int Kg,
    const dtype* w1, const dtype* a1, const dtype* g1,
    const dtype* w2_t, const dtype* a2_t, const dtype* g2_t,
    dtype* w, dtype* a, dtype* g) {
  linear_wag_rank_out_f16_kernel<128, 4><<<dim3(ceil_div(C, 4), M, 3), 128, 0, stream>>>(
      M, C, Kw, Ka, Kg, w1, a1, g1, w2_t, a2_t, g2_t, w, a, g);
}

void rwkv7_v3a_linear_wagv_rank_out_f16_launch(
    cudaStream_t stream, int M, int C, int Kw, int Ka, int Kg, int Kv,
    const dtype* w1, const dtype* a1, const dtype* g1, const dtype* v1,
    const dtype* w2_t, const dtype* a2_t, const dtype* g2_t, const dtype* v2_t,
    const dtype* v, const dtype* v_first, const dtype* v0,
    dtype* w, dtype* a, dtype* g, dtype* v_out) {
  linear_wagv_rank_out_f16_kernel<128, 4><<<dim3(ceil_div(C, 4), M, 4), 128, 0, stream>>>(
      M, C, Kw, Ka, Kg, Kv, w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0, w, a, g, v_out);
}

std::vector<at::Tensor> linear_wag_rank_in_f16_cuda(
    int64_t M, int64_t K, int64_t Rw, int64_t Ra, int64_t Rg,
    at::Tensor xw, at::Tensor xa, at::Tensor xg,
    at::Tensor w1_t, at::Tensor a1_t, at::Tensor g1_t) {
  auto w1 = at::empty({M, Rw}, xw.options());
  auto a1 = at::empty({M, Ra}, xa.options());
  auto g1 = at::empty({M, Rg}, xg.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  rwkv7_v3a_linear_wag_rank_in_f16_launch(
      stream,
      static_cast<int>(M), static_cast<int>(K),
      static_cast<int>(Rw), static_cast<int>(Ra), static_cast<int>(Rg),
      static_cast<const dtype*>(xw.data_ptr()),
      static_cast<const dtype*>(xa.data_ptr()),
      static_cast<const dtype*>(xg.data_ptr()),
      static_cast<const dtype*>(w1_t.data_ptr()),
      static_cast<const dtype*>(a1_t.data_ptr()),
      static_cast<const dtype*>(g1_t.data_ptr()),
      static_cast<dtype*>(w1.data_ptr()),
      static_cast<dtype*>(a1.data_ptr()),
      static_cast<dtype*>(g1.data_ptr()));
  return {w1, a1, g1};
}

std::vector<at::Tensor> linear_wagv_rank_in_f16_cuda(
    int64_t M, int64_t K, int64_t Rw, int64_t Ra, int64_t Rg, int64_t Rv,
    at::Tensor xw, at::Tensor xa, at::Tensor xg, at::Tensor xv,
    at::Tensor w1_t, at::Tensor a1_t, at::Tensor g1_t, at::Tensor v1_t) {
  auto w1 = at::empty({M, Rw}, xw.options());
  auto a1 = at::empty({M, Ra}, xa.options());
  auto g1 = at::empty({M, Rg}, xg.options());
  auto v1 = at::empty({M, Rv}, xv.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  rwkv7_v3a_linear_wagv_rank_in_f16_launch(
      stream,
      static_cast<int>(M), static_cast<int>(K),
      static_cast<int>(Rw), static_cast<int>(Ra), static_cast<int>(Rg), static_cast<int>(Rv),
      static_cast<const dtype*>(xw.data_ptr()),
      static_cast<const dtype*>(xa.data_ptr()),
      static_cast<const dtype*>(xg.data_ptr()),
      static_cast<const dtype*>(xv.data_ptr()),
      static_cast<const dtype*>(w1_t.data_ptr()),
      static_cast<const dtype*>(a1_t.data_ptr()),
      static_cast<const dtype*>(g1_t.data_ptr()),
      static_cast<const dtype*>(v1_t.data_ptr()),
      static_cast<dtype*>(w1.data_ptr()),
      static_cast<dtype*>(a1.data_ptr()),
      static_cast<dtype*>(g1.data_ptr()),
      static_cast<dtype*>(v1.data_ptr()));
  return {w1, a1, g1, v1};
}

std::vector<at::Tensor> linear_wag_rank_out_f16_cuda(
    int64_t M, int64_t C, int64_t Kw, int64_t Ka, int64_t Kg,
    at::Tensor w1, at::Tensor a1, at::Tensor g1,
    at::Tensor w2_t, at::Tensor a2_t, at::Tensor g2_t) {
  auto w = at::empty({M, C}, w1.options());
  auto a = at::empty({M, C}, a1.options());
  auto g = at::empty({M, C}, g1.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  rwkv7_v3a_linear_wag_rank_out_f16_launch(
      stream,
      static_cast<int>(M), static_cast<int>(C),
      static_cast<int>(Kw), static_cast<int>(Ka), static_cast<int>(Kg),
      static_cast<const dtype*>(w1.data_ptr()),
      static_cast<const dtype*>(a1.data_ptr()),
      static_cast<const dtype*>(g1.data_ptr()),
      static_cast<const dtype*>(w2_t.data_ptr()),
      static_cast<const dtype*>(a2_t.data_ptr()),
      static_cast<const dtype*>(g2_t.data_ptr()),
      static_cast<dtype*>(w.data_ptr()),
      static_cast<dtype*>(a.data_ptr()),
      static_cast<dtype*>(g.data_ptr()));
  return {w, a, g};
}

std::vector<at::Tensor> linear_wagv_rank_out_f16_cuda(
    int64_t M, int64_t C, int64_t Kw, int64_t Ka, int64_t Kg, int64_t Kv,
    at::Tensor w1, at::Tensor a1, at::Tensor g1, at::Tensor v1,
    at::Tensor w2_t, at::Tensor a2_t, at::Tensor g2_t, at::Tensor v2_t,
    at::Tensor v, at::Tensor v_first, at::Tensor v0) {
  auto w = at::empty({M, C}, w1.options());
  auto a = at::empty({M, C}, a1.options());
  auto g = at::empty({M, C}, g1.options());
  auto v_out = at::empty({M, C}, v1.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  rwkv7_v3a_linear_wagv_rank_out_f16_launch(
      stream,
      static_cast<int>(M), static_cast<int>(C),
      static_cast<int>(Kw), static_cast<int>(Ka), static_cast<int>(Kg), static_cast<int>(Kv),
      static_cast<const dtype*>(w1.data_ptr()),
      static_cast<const dtype*>(a1.data_ptr()),
      static_cast<const dtype*>(g1.data_ptr()),
      static_cast<const dtype*>(v1.data_ptr()),
      static_cast<const dtype*>(w2_t.data_ptr()),
      static_cast<const dtype*>(a2_t.data_ptr()),
      static_cast<const dtype*>(g2_t.data_ptr()),
      static_cast<const dtype*>(v2_t.data_ptr()),
      static_cast<const dtype*>(v.data_ptr()),
      static_cast<const dtype*>(v_first.data_ptr()),
      static_cast<const dtype*>(v0.data_ptr()),
      static_cast<dtype*>(w.data_ptr()),
      static_cast<dtype*>(a.data_ptr()),
      static_cast<dtype*>(g.data_ptr()),
      static_cast<dtype*>(v_out.data_ptr()));
  return {w, a, g, v_out};
}
