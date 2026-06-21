#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <assert.h>
#include <cuda_fp16.h>

#include <vector>

#include "v1/common/warp_primitives.cuh"

using dtype = at::Half;

namespace {

constexpr int HEAD_SIZE = 64;
constexpr float TMIX_LN_X_EPS = 64.0e-5f;
constexpr int FFN_SPMV_THREADS = 128;
constexpr int FFN_TILE = 128;

__device__ inline __half2 load_h2(const dtype* ptr) {
  return *reinterpret_cast<const __half2*>(ptr);
}

__device__ inline void store_h2(dtype* ptr, float x0, float x1) {
  *reinterpret_cast<__half2*>(ptr) = __floats2half2_rn(x0, x1);
}

__global__ void tmix_mix6_kernel_varlen(
    int C, const dtype* __restrict__ x, dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_r, const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k, const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a, const dtype* __restrict__ x_g,
    dtype* __restrict__ out_r, dtype* __restrict__ out_w,
    dtype* __restrict__ out_k, dtype* __restrict__ out_v,
    dtype* __restrict__ out_a, dtype* __restrict__ out_g,
    const int* __restrict__ query_start_loc, const int* __restrict__ req_id,
    int64_t total_pairs) {
  const int64_t pair_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  const int c_pairs = C >> 1;
  const int64_t bt = pair_idx / c_pairs;
  const int c = static_cast<int>(pair_idx - bt * c_pairs) << 1;

  const int bid = req_id[bt];
  const int t_local = bt - query_start_loc[bid];
  const int64_t idx = bt * C + c;

  const __half2 cur2 = load_h2(x + idx);

  __half2 prev2;
  if (t_local == 0) {
    prev2 = load_h2(shift_state + static_cast<int64_t>(bid) * C + c);
  } else {
    prev2 = load_h2(x + idx - C);
  }

  const float2 cur = __half22float2(cur2);
  const float2 prev = __half22float2(prev2);
  const float dx0 = prev.x - cur.x;
  const float dx1 = prev.y - cur.y;

  const float2 xr = __half22float2(load_h2(x_r + c));
  const float2 xw = __half22float2(load_h2(x_w + c));
  const float2 xk = __half22float2(load_h2(x_k + c));
  const float2 xv = __half22float2(load_h2(x_v + c));
  const float2 xa = __half22float2(load_h2(x_a + c));
  const float2 xg = __half22float2(load_h2(x_g + c));

  store_h2(out_r + idx, cur.x + dx0 * xr.x, cur.y + dx1 * xr.y);
  store_h2(out_w + idx, cur.x + dx0 * xw.x, cur.y + dx1 * xw.y);
  store_h2(out_k + idx, cur.x + dx0 * xk.x, cur.y + dx1 * xk.y);
  store_h2(out_v + idx, cur.x + dx0 * xv.x, cur.y + dx1 * xv.y);
  store_h2(out_a + idx, cur.x + dx0 * xa.x, cur.y + dx1 * xa.y);
  store_h2(out_g + idx, cur.x + dx0 * xg.x, cur.y + dx1 * xg.y);
}

__global__ void tmix_mix6_update_state_kernel_varlen(
    int B, int C, const dtype* __restrict__ x, dtype* __restrict__ shift_state,
    const int* __restrict__ query_start_loc) {
  const int b = blockIdx.x;
  if (b >= B) return;
  const int last_row = query_start_loc[b + 1] - 1;
  const int64_t src = static_cast<int64_t>(last_row) * C;
  const int64_t dst = static_cast<int64_t>(b) * C;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    shift_state[dst + c] = x[src + c];
  }
}

template <bool HalfMath, int Vec>
__global__ void tmix_mix6_t1_c4096_kernel_varlen(
    const dtype* __restrict__ x, dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_r, const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k, const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a, const dtype* __restrict__ x_g,
    dtype* __restrict__ out_r, dtype* __restrict__ out_w,
    dtype* __restrict__ out_k, dtype* __restrict__ out_v,
    dtype* __restrict__ out_a, dtype* __restrict__ out_g,
    const int* __restrict__ req_id, int64_t total_pairs) {
  const int64_t base_pair =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * Vec;
#pragma unroll
  for (int u = 0; u < Vec; ++u) {
    const int64_t pair_idx = base_pair + u;
    if (pair_idx >= total_pairs) {
      return;
    }
    const int c = static_cast<int>(pair_idx & 2047) << 1;
    const int64_t idx = pair_idx << 1;
    const int64_t bt = pair_idx >> 11;
    const int bid = req_id[bt];
    const int64_t state_base = static_cast<int64_t>(bid) * 4096;

    const __half2 cur2 = load_h2(x + idx);
    const __half2 prev2 = load_h2(shift_state + state_base + c);
    if constexpr (HalfMath) {
      const __half2 dx = __hsub2(prev2, cur2);
      *reinterpret_cast<__half2*>(out_r + idx) =
          __hfma2(dx, load_h2(x_r + c), cur2);
      *reinterpret_cast<__half2*>(out_w + idx) =
          __hfma2(dx, load_h2(x_w + c), cur2);
      *reinterpret_cast<__half2*>(out_k + idx) =
          __hfma2(dx, load_h2(x_k + c), cur2);
      *reinterpret_cast<__half2*>(out_v + idx) =
          __hfma2(dx, load_h2(x_v + c), cur2);
      *reinterpret_cast<__half2*>(out_a + idx) =
          __hfma2(dx, load_h2(x_a + c), cur2);
      *reinterpret_cast<__half2*>(out_g + idx) =
          __hfma2(dx, load_h2(x_g + c), cur2);
    } else {
      const float2 cur = __half22float2(cur2);
      const float2 prev = __half22float2(prev2);
      const float dx0 = prev.x - cur.x;
      const float dx1 = prev.y - cur.y;
      const float2 xr = __half22float2(load_h2(x_r + c));
      const float2 xw = __half22float2(load_h2(x_w + c));
      const float2 xk = __half22float2(load_h2(x_k + c));
      const float2 xv = __half22float2(load_h2(x_v + c));
      const float2 xa = __half22float2(load_h2(x_a + c));
      const float2 xg = __half22float2(load_h2(x_g + c));
      store_h2(out_r + idx, cur.x + dx0 * xr.x, cur.y + dx1 * xr.y);
      store_h2(out_w + idx, cur.x + dx0 * xw.x, cur.y + dx1 * xw.y);
      store_h2(out_k + idx, cur.x + dx0 * xk.x, cur.y + dx1 * xk.y);
      store_h2(out_v + idx, cur.x + dx0 * xv.x, cur.y + dx1 * xv.y);
      store_h2(out_a + idx, cur.x + dx0 * xa.x, cur.y + dx1 * xa.y);
      store_h2(out_g + idx, cur.x + dx0 * xg.x, cur.y + dx1 * xg.y);
    }
  }
}

__global__ void cmix_mix_kernel_varlen(int C, const dtype* __restrict__ x,
                                       dtype* __restrict__ shift_state,
                                       const dtype* __restrict__ x_k,
                                       dtype* __restrict__ out,
                                       const int* __restrict__ query_start_loc,
                                       const int* __restrict__ req_id,
                                       int64_t total_pairs) {
  const int64_t pair_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  const int c_pairs = C >> 1;
  const int64_t bt = pair_idx / c_pairs;
  const int c = static_cast<int>(pair_idx - bt * c_pairs) << 1;
  const int bid = req_id[bt];
  const int t_local = bt - query_start_loc[bid];
  const int my_t = query_start_loc[bid + 1] - query_start_loc[bid];
  const int64_t idx = bt * C + c;

  const __half2 cur2 = load_h2(x + idx);
  const __half2 prev2 =
      (t_local == 0) ? load_h2(shift_state + static_cast<int64_t>(bid) * C + c)
                     : load_h2(x + idx - C);
  const float2 cur = __half22float2(cur2);
  const float2 prev = __half22float2(prev2);
  const float2 mix = __half22float2(load_h2(x_k + c));
  store_h2(out + idx, cur.x + (prev.x - cur.x) * mix.x,
           cur.y + (prev.y - cur.y) * mix.y);
}

__global__ void cmix_mix_update_state_kernel_varlen(
    int B, int C, const dtype* __restrict__ x, dtype* __restrict__ shift_state,
    const int* __restrict__ query_start_loc) {
  const int b = blockIdx.x;
  if (b >= B) return;
  const int last_row = query_start_loc[b + 1] - 1;
  const int64_t src = static_cast<int64_t>(last_row) * C;
  const int64_t dst = static_cast<int64_t>(b) * C;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    shift_state[dst + c] = x[src + c];
  }
}

__global__ void cmix_sparse_up_rows_kernel_varlen(
    int C, int F, const dtype* __restrict__ x, dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_k, const dtype* __restrict__ key_fc,
    dtype* __restrict__ act, const int* __restrict__ query_start_loc,
    const int* __restrict__ req_id, int64_t total_rows) {
  const int f = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float acc = 0.0f;

  const int bid = req_id[row];
  const int t_local = row - query_start_loc[bid];

  const auto x2 =
      reinterpret_cast<const __half2*>(x + static_cast<int64_t>(row) * C);
  const auto p2 = (t_local == 0)
                      ? reinterpret_cast<const __half2*>(
                            shift_state + static_cast<int64_t>(bid) * C)
                      : reinterpret_cast<const __half2*>(
                            x + static_cast<int64_t>(row - 1) * C);
  const auto k2 = reinterpret_cast<const __half2*>(x_k);
  const auto w2 =
      reinterpret_cast<const __half2*>(key_fc + static_cast<int64_t>(f) * C);
  const int n = C / 2;
  for (int j = tid; j < n; j += 64) {
    const float2 xv = __half22float2(x2[j]);
    const float2 pv = __half22float2(p2[j]);
    const float2 kv = __half22float2(k2[j]);
    const float2 wv = __half22float2(w2[j]);
    acc = fmaf(xv.x + (pv.x - xv.x) * kv.x, wv.x, acc);
    acc = fmaf(xv.y + (pv.y - xv.y) * kv.y, wv.y, acc);
  }

  acc = warp_sum(acc);
  __shared__ float warp_sums[2];
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    float total = lane < 2 ? warp_sums[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) {
      act[static_cast<int64_t>(row) * F + f] = __float2half_rn(total);
    }
  }
}

__global__ void cmix_sparse_copy_zero_rows_kernel_varlen(
    int B, int C, const dtype* __restrict__ x, dtype* __restrict__ shift_state,
    dtype* __restrict__ out, const int* __restrict__ query_start_loc,
    int64_t out_vec4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < out_vec4) {
    reinterpret_cast<int4*>(out)[i] = make_int4(0, 0, 0, 0);
  }
  const int64_t state_vec4 = static_cast<int64_t>(B) * (C / 8);
  if (i < state_vec4) {
    const int b = static_cast<int>(i / (C / 8));
    const int c4 = static_cast<int>(i - static_cast<int64_t>(b) * (C / 8));
    const int my_t = query_start_loc[b + 1] - query_start_loc[b];
    assert(my_t > 0 && "query_start_loc[b+1] must be > query_start_loc[b]");
    const int last_token = query_start_loc[b + 1] - 1;
    reinterpret_cast<int4*>(shift_state)[i] = reinterpret_cast<const int4*>(
        x + static_cast<int64_t>(last_token) * C)[c4];
  }
}

__global__
__launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_relu_rows_kernel_varlen(
    int C, int F, const dtype* __restrict__ preact,
    const dtype* __restrict__ value_fc, dtype* __restrict__ out, int64_t rows) {
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;
  const dtype* pre_row = preact + static_cast<int64_t>(row) * F;

  if (tid < FFN_TILE) {
    const float v = fmaxf(__half2float(pre_row[start_f + tid]), 0.0f);
    vec_slice[tid] = __float2half_rn(v * v);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  for (int i = 0; i < nnz_count; ++i) {
    const int actual_f = start_f + nnz_ids[i];
    const __half2 mat = *reinterpret_cast<const __half2*>(
        value_fc + static_cast<int64_t>(actual_f) * C +
        c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
    acc = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, acc);
  }
  atomicAdd(
      reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C +
                                 c_block * (2 * FFN_SPMV_THREADS) + tid * 2),
      acc);
}

__global__
__launch_bounds__(256, 2) void cmix_sparse_spmv_relu_rows_t512_kernel_varlen(
    int C, int F, const dtype* __restrict__ preact,
    const dtype* __restrict__ value_fc, dtype* __restrict__ out, int64_t rows) {
  constexpr int TILE = 512;
  constexpr int THREADS = 256;
  __shared__ __align__(256) __half vec_slice[TILE];
  __shared__ __align__(256) int nnz_ids[TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[TILE / 32];
  __shared__ int warp_prefix[TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * TILE;
  const dtype* pre_row = preact + static_cast<int64_t>(row) * F;

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const float v = fmaxf(__half2float(preact[start_f + local_f]), 0.0f);
    vec_slice[local_f] = __float2half_rn(v * v);
  }
  __syncthreads();

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const bool nonzero = bool(__half_as_ushort(vec_slice[local_f]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    if (lane == 0) {
      warp_counts[warp_id + u * (THREADS / 32)] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const bool nonzero = bool(__half_as_ushort(vec_slice[local_f]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    const int local_pos = __popc(mask & ((1u << lane) - 1u));
    const int group = warp_id + u * (THREADS / 32);
    if (nonzero) {
      nnz_ids[warp_prefix[group] + local_pos] = local_f;
    }
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  for (int i = 0; i < nnz_count; ++i) {
    const int local_f = nnz_ids[i];
    const int actual_f = start_f + local_f;
    const __half2 mat = *reinterpret_cast<const __half2*>(
        value_fc + static_cast<int64_t>(actual_f) * C +
        c_block * (2 * THREADS) + tid * 2);
    acc = __hfma2(__half2half2(vec_slice[local_f]), mat, acc);
  }
  atomicAdd(reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C +
                                       c_block * (2 * THREADS) + tid * 2),
            acc);
}

__global__ void zero_vec4_kernel(dtype* __restrict__ out, int64_t n_vec4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n_vec4) {
    reinterpret_cast<int4*>(out)[i] = make_int4(0, 0, 0, 0);
  }
}

}  // namespace

std::vector<at::Tensor> tmix_mix6_cuda_varlen(
    int B, int C, at::Tensor x, at::Tensor shift_state, at::Tensor x_r,
    at::Tensor x_w, at::Tensor x_k, at::Tensor x_v, at::Tensor x_a,
    at::Tensor x_g, at::Tensor query_start_loc, at::Tensor req_id) {
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.size(0) * (C / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  tmix_mix6_kernel_varlen<<<static_cast<int>(ceil_div(total_pairs, threads)),
                            threads, 0, stream>>>(
      C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
      x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
      x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
      out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(),
      out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(),
      query_start_loc.data_ptr<int>(), req_id.data_ptr<int>(), total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  tmix_mix6_update_state_kernel_varlen<<<B, threads, 0, stream>>>(
      B, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
      query_start_loc.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_r, out_w, out_k, out_v, out_a, out_g};
}

template <int Vec>
std::vector<at::Tensor> tmix_mix6_t1_c4096_cuda_varlen_impl(
    int B, at::Tensor x, at::Tensor shift_state, at::Tensor x_r, at::Tensor x_w,
    at::Tensor x_k, at::Tensor x_v, at::Tensor x_a, at::Tensor x_g,
    at::Tensor query_start_loc, at::Tensor req_id, int threads,
    bool half_math) {
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  const int64_t total_pairs = x.size(0) * (4096 / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = static_cast<int>(
      ceil_div(total_pairs, static_cast<int64_t>(threads) * Vec));
  if (half_math) {
    tmix_mix6_t1_c4096_kernel_varlen<true, Vec><<<blocks, threads, 0, stream>>>(
        x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
        out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(),
        out_k.data_ptr<dtype>(), out_v.data_ptr<dtype>(),
        out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(),
        req_id.data_ptr<int>(), total_pairs);
  } else {
    tmix_mix6_t1_c4096_kernel_varlen<false, Vec>
        <<<blocks, threads, 0, stream>>>(
            x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
            x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
            x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
            out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(),
            out_k.data_ptr<dtype>(), out_v.data_ptr<dtype>(),
            out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(),
            req_id.data_ptr<int>(), total_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  tmix_mix6_update_state_kernel_varlen<<<B, threads, 0, stream>>>(
      B, 4096, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
      query_start_loc.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_r, out_w, out_k, out_v, out_a, out_g};
}

std::vector<at::Tensor> tmix_mix6_t1_c4096_cuda_varlen(
    int B, at::Tensor x, at::Tensor shift_state, at::Tensor x_r, at::Tensor x_w,
    at::Tensor x_k, at::Tensor x_v, at::Tensor x_a, at::Tensor x_g,
    at::Tensor query_start_loc, at::Tensor req_id, int threads, int vec,
    bool half_math) {
  if (vec == 2) {
    return tmix_mix6_t1_c4096_cuda_varlen_impl<2>(
        B, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, query_start_loc,
        req_id, threads, half_math);
  }
  if (vec == 4) {
    return tmix_mix6_t1_c4096_cuda_varlen_impl<4>(
        B, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, query_start_loc,
        req_id, threads, half_math);
  }
  if (vec == 8) {
    return tmix_mix6_t1_c4096_cuda_varlen_impl<8>(
        B, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, query_start_loc,
        req_id, threads, half_math);
  }
  return tmix_mix6_t1_c4096_cuda_varlen_impl<1>(
      B, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, query_start_loc, req_id,
      threads, half_math);
}

at::Tensor cmix_mix_cuda_varlen(int B, int C, at::Tensor x,
                                at::Tensor shift_state, at::Tensor x_k,
                                at::Tensor query_start_loc, at::Tensor req_id) {
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.size(0) * (C / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  cmix_mix_kernel_varlen<<<static_cast<int>(ceil_div(total_pairs, threads)),
                           threads, 0, stream>>>(
      C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
      x_k.data_ptr<dtype>(), out.data_ptr<dtype>(),
      query_start_loc.data_ptr<int>(), req_id.data_ptr<int>(), total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  cmix_mix_update_state_kernel_varlen<<<B, threads, 0, stream>>>(
      B, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
      query_start_loc.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_rows_cuda_varlen(int B, int C, int F, at::Tensor x,
                                        at::Tensor shift_state, at::Tensor x_k,
                                        at::Tensor key_fc, at::Tensor value_fc,
                                        at::Tensor query_start_loc,
                                        at::Tensor req_id) {
  const int64_t rows = x.size(0);
  auto act = at::empty({static_cast<int64_t>(rows), static_cast<int64_t>(F)},
                       x.options());
  auto out = at::empty({static_cast<int64_t>(rows), static_cast<int64_t>(C)},
                       x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  cmix_sparse_up_rows_kernel_varlen<<<dim3(F, rows, 1), 64, 0, stream>>>(
      C, F, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
      x_k.data_ptr<dtype>(), key_fc.data_ptr<dtype>(), act.data_ptr<dtype>(),
      query_start_loc.data_ptr<int>(), req_id.data_ptr<int>(), rows);
  const int64_t out_vec4 = rows * (C / 8);
  cmix_sparse_copy_zero_rows_kernel_varlen<<<
      static_cast<int>(ceil_div(out_vec4, 128)), 128, 0, stream>>>(
      B, C, x.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
      out.data_ptr<dtype>(), query_start_loc.data_ptr<int>(), out_vec4);
  cmix_sparse_spmv_relu_rows_kernel_varlen<<<
      dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), rows), FFN_SPMV_THREADS, 0,
      stream>>>(C, F, act.data_ptr<dtype>(), value_fc.data_ptr<dtype>(),
                out.data_ptr<dtype>(), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_down_relu_rows_cuda_varlen(int rows, int C, int F,
                                                  at::Tensor preact,
                                                  at::Tensor value_fc) {
  auto out = at::empty({static_cast<int64_t>(rows), static_cast<int64_t>(C)},
                       preact.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  zero_vec4_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0,
                     stream>>>(out.data_ptr<dtype>(), out_vec4);
  cmix_sparse_spmv_relu_rows_kernel_varlen<<<
      dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), rows), FFN_SPMV_THREADS, 0,
      stream>>>(C, F, preact.data_ptr<dtype>(), value_fc.data_ptr<dtype>(),
                out.data_ptr<dtype>(), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cmix_sparse_down_relu_rows_t512_cuda_varlen(int rows, int C, int F,
                                                       at::Tensor preact,
                                                       at::Tensor value_fc) {
  auto out = at::empty({static_cast<int64_t>(rows), static_cast<int64_t>(C)},
                       preact.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  zero_vec4_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0,
                     stream>>>(out.data_ptr<dtype>(), out_vec4);
  cmix_sparse_spmv_relu_rows_t512_kernel_varlen<<<dim3(F / 512, C / 512, rows),
                                                  256, 0, stream>>>(
      C, F, preact.data_ptr<dtype>(), value_fc.data_ptr<dtype>(),
      out.data_ptr<dtype>(), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
