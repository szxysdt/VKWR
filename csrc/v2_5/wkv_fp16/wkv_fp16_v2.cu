#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <assert.h>
#include <cuda_fp16.h>

namespace {

constexpr int N = 64;
constexpr int HALF2_N = N / 2;
constexpr int LDG_ELEMS = sizeof(int4) / sizeof(half);
constexpr float TWO_NEG_41 = 4.547473508864641e-13f;
constexpr float NEXP_HALF_LOG2_E = -0.8750387749145276f;
constexpr float NLOG2_E = -1.4426950408889634f;
constexpr int ROT1 = static_cast<int>(2654435769);
using F = half;
#define CLONE_N 64

__device__ __forceinline__ float rotator1(int x) {
  return TWO_NEG_41 * float(ROT1 * x);
}

__device__ __forceinline__ half w_delta(float w, int phase) {
  float d = exp2f(NEXP_HALF_LOG2_E / (1.0f + exp2f(NLOG2_E * w))) - 1.0f +
            rotator1(phase);
  return __float2half_rn(d);
}

template <bool AddW0>
__device__ __forceinline__ half w_delta_maybe_w0(
    half w_raw, const half* __restrict__ w0_ptr, int c, int phase) {
  float w = __half2float(w_raw);
  if constexpr (AddW0) {
    w += __half2float(w0_ptr[c]);
  }
  return w_delta(w, phase);
}

template <int Bytes>
__device__ __forceinline__ void clone_cp_async(void const* smem_addr,
                                               void const* global_ptr,
                                               bool cond) {
  static_assert(Bytes == 16 || Bytes == 8 || Bytes == 4);
  int bytes = cond ? Bytes : 0;
  unsigned int addr = __cvta_generic_to_shared(smem_addr);
  if constexpr (Bytes == 16) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;" ::"r"(addr),
                 "l"(global_ptr), "n"(Bytes), "r"(bytes));
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::"r"(addr),
                 "l"(global_ptr), "n"(Bytes), "r"(bytes));
  }
}

template <int NWait>
__device__ __forceinline__ void clone_cp_wait() {
  if constexpr (NWait == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(NWait));
  }
}

__device__ __forceinline__ void clone_cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <bool AddW0 = false>
__global__ void __launch_bounds__(CLONE_N, 2) wkv_fp16_v1_clone_kernel(
    const int* __restrict__ query_start_loc, const int* __restrict__ slot_indices,
    const int C, const int H, F* __restrict__ state_ptr,
    const F* __restrict__ r_ptr, const F* __restrict__ w_ptr,
    const F* __restrict__ w0_ptr, const F* __restrict__ k_ptr,
    const F* __restrict__ v_ptr, const F* __restrict__ a_ptr,
    const F* __restrict__ b_ptr, F* __restrict__ y_ptr,
    const int* __restrict__ elapsed_t) {
  const int b = blockIdx.x / H;
  const int h = blockIdx.x % H;
  const int i = threadIdx.x;
  const int lane = i % 32;
  const int slot_id = slot_indices[b];

  __shared__ __align__(256) half2 state_smem[CLONE_N][CLONE_N / 2];

  state_ptr += slot_id * C * CLONE_N + h * CLONE_N * CLONE_N;
  constexpr int ldg_size = sizeof(int4) / sizeof(F);
#pragma unroll
  for (int j0 = 0; j0 < CLONE_N / ldg_size; j0++) {
    int4 state_vec = ((int4*)state_ptr)[j0 * CLONE_N + i];
#pragma unroll
    for (int j1 = 0; j1 < ldg_size / 2; j1++) {
      int row = j0 * ldg_size + i * ldg_size / CLONE_N;
      int col = i * ldg_size % CLONE_N / 2 + j1;
      state_smem[row][(row % 32) ^ col] = ((half2*)&state_vec)[j1];
    }
  }
  __syncthreads();

  half2 state[CLONE_N / 2];
#pragma unroll
  for (int j = 0; j < CLONE_N / 2; j++) {
    state[j] = state_smem[i][lane ^ j];
  }

  __shared__ __align__(128) half2 r[CLONE_N / 2], k[CLONE_N / 2],
      w[CLONE_N / 2], a[CLONE_N / 2], bvec[CLONE_N / 2];

  const int my_t = query_start_loc[b + 1] - query_start_loc[b];
#pragma unroll
  for (int tt = 0; tt < my_t; tt++) {
    int t = (query_start_loc[b] + tt) * C + h * CLONE_N;
    __syncthreads();
    clone_cp_async<4>((half2*)(i < 32 ? w : a) + lane,
                      (half2*)((i < 32 ? w_ptr : a_ptr) + t) + lane, true);
    clone_cp_commit();
    clone_cp_async<4>((half2*)(i < 32 ? r : k) + lane,
                      (half2*)((i < 32 ? r_ptr : k_ptr) + t) + lane, true);
    clone_cp_async<4>((half2*)bvec + lane, (half2*)(b_ptr + t) + lane, i < 32);
    clone_cp_commit();

    half vv = v_ptr[t + i];
    half2 vv2 = {vv, vv};
    half2 y2 = {0.0, 0.0};
    half2 sa2 = {0.0, 0.0};
    clone_cp_wait<1>();
    __syncthreads();
#pragma unroll
    for (int j = 0; j < CLONE_N / 2; j++) {
      sa2 = __hfma2(a[j], state[j], sa2);
    }
    half sa = sa2.x + sa2.y;
    sa2 = {sa, sa};
    ((F*)w)[i] = w_delta_maybe_w0<AddW0>(((F*)w)[i], w0_ptr, h * CLONE_N + i,
                                          elapsed_t[slot_id] + h * CLONE_N + i + tt);

    clone_cp_wait<0>();
    __syncthreads();
#pragma unroll
    for (int j = 0; j < CLONE_N / 2; j++) {
      half2& s = state[j];
      s = __hfma2(s, w[j], __hfma2(k[j], vv2, __hfma2(sa2, bvec[j], s)));
      y2 = __hfma2(s, r[j], y2);
    }
    y_ptr[t + i] = y2.x + y2.y;
  }

#pragma unroll
  for (int j = 0; j < CLONE_N / 2; j++) {
    state_smem[i][lane ^ j] = state[j];
  }
  __syncthreads();
#pragma unroll
  for (int j0 = 0; j0 < CLONE_N / ldg_size; j0++) {
    int4 state_vec;
#pragma unroll
    for (int j1 = 0; j1 < ldg_size / 2; j1++) {
      int row = j0 * ldg_size + i * ldg_size / CLONE_N;
      int col = i * ldg_size % CLONE_N / 2 + j1;
      ((half2*)&state_vec)[j1] = state_smem[row][(row % 32) ^ col];
    }
    ((int4*)state_ptr)[j0 * CLONE_N + i] = state_vec;
  }
}

template <int Bytes>
__device__ __forceinline__ void cp_async(void* smem, const void* global,
                                         bool pred) {
  static_assert(Bytes == 16 || Bytes == 8 || Bytes == 4);
  int bytes = pred ? Bytes : 0;
  unsigned addr = __cvta_generic_to_shared(smem);
  if constexpr (Bytes == 16) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;" ::"r"(addr),
                 "l"(global), "n"(Bytes), "r"(bytes));
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::"r"(addr),
                 "l"(global), "n"(Bytes), "r"(bytes));
  }
}

__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int NWait>
__device__ __forceinline__ void cp_wait() {
  if constexpr (NWait == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(NWait));
  }
}

__device__ __forceinline__ void prefetch_token(
    int tid, int lane, int token, half2* r, half2* w, half2* k, half2* a,
    half2* b, const half* r_ptr, const half* w_ptr, const half* k_ptr,
    const half* a_ptr, const half* b_ptr) {
  cp_async<4>((tid < 32 ? w : a) + lane,
              (const half2*)(tid < 32 ? w_ptr + token : a_ptr + token) + lane,
              true);
  cp_commit();
  cp_async<4>((tid < 32 ? r : k) + lane,
              (const half2*)(tid < 32 ? r_ptr + token : k_ptr + token) + lane,
              true);
  cp_async<4>(b + lane, (const half2*)(b_ptr + token) + lane, tid < 32);
  cp_commit();
}

template <bool AddW0 = false>
__global__ __launch_bounds__(N, 2) void wkv_fp16_v1_exact_kernel(
    const int* __restrict__ query_start_loc, const int* __restrict__ slot_indices,
    const int C, const int H, half* __restrict__ state_ptr,
    const half* __restrict__ r_ptr, const half* __restrict__ w_ptr,
    const half* __restrict__ w0_ptr, const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr, const half* __restrict__ a_ptr,
    const half* __restrict__ b_ptr, half* __restrict__ y_ptr,
    const int* __restrict__ elapsed_t) {
  const int b_id = blockIdx.x / H;
  const int h = blockIdx.x % H;
  const int i = threadIdx.x;
  const int lane = i % 32;
  const int slot_id = slot_indices[b_id];

  __shared__ __align__(256) half2 state_smem[N][HALF2_N];
  state_ptr += slot_id * C * N + h * N * N;

#pragma unroll
  for (int j0 = 0; j0 < N / LDG_ELEMS; j0++) {
    int4 state_vec = ((int4*)state_ptr)[j0 * N + i];
#pragma unroll
    for (int j1 = 0; j1 < LDG_ELEMS / 2; j1++) {
      int row = j0 * LDG_ELEMS + i * LDG_ELEMS / N;
      int col = i * LDG_ELEMS % N / 2 + j1;
      state_smem[row][(row % 32) ^ col] = ((half2*)&state_vec)[j1];
    }
  }
  __syncthreads();

  half2 state[HALF2_N];
#pragma unroll
  for (int j = 0; j < HALF2_N; j++) {
    state[j] = state_smem[i][lane ^ j];
  }

  __shared__ __align__(128) half2 r[HALF2_N], k[HALF2_N], w[HALF2_N],
      a[HALF2_N], bvec[HALF2_N];

  const int my_t = query_start_loc[b_id + 1] - query_start_loc[b_id];
#pragma unroll
  for (int tt = 0; tt < my_t; tt++) {
    int t = (query_start_loc[b_id] + tt) * C + h * N;
    __syncthreads();
    cp_async<4>((half2*)(i < 32 ? w : a) + lane,
                (half2*)((i < 32 ? w_ptr : a_ptr) + t) + lane, true);
    cp_commit();
    cp_async<4>((half2*)(i < 32 ? r : k) + lane,
                (half2*)((i < 32 ? r_ptr : k_ptr) + t) + lane, true);
    cp_async<4>((half2*)bvec + lane, (half2*)(b_ptr + t) + lane, i < 32);
    cp_commit();

    half vv = v_ptr[t + i];
    half2 vv2 = {vv, vv};
    half2 y2 = {0.0, 0.0};
    half2 sa2 = {0.0, 0.0};
    cp_wait<1>();
    __syncthreads();
#pragma unroll
    for (int j = 0; j < HALF2_N; j++) {
      sa2 = __hfma2(a[j], state[j], sa2);
    }
    half sa = sa2.x + sa2.y;
    sa2 = {sa, sa};
    ((half*)w)[i] = w_delta_maybe_w0<AddW0>(((half*)w)[i], w0_ptr, h * N + i,
                                             elapsed_t[slot_id] + h * N + i + tt);

    cp_wait<0>();
    __syncthreads();
#pragma unroll
    for (int j = 0; j < HALF2_N; j++) {
      half2& s = state[j];
      s = __hfma2(s, w[j], __hfma2(k[j], vv2, __hfma2(sa2, bvec[j], s)));
      y2 = __hfma2(s, r[j], y2);
    }
    y_ptr[t + i] = y2.x + y2.y;
  }

#pragma unroll
  for (int j = 0; j < HALF2_N; j++) {
    state_smem[i][lane ^ j] = state[j];
  }
  __syncthreads();
#pragma unroll
  for (int j0 = 0; j0 < N / LDG_ELEMS; j0++) {
    int4 state_vec;
#pragma unroll
    for (int j1 = 0; j1 < LDG_ELEMS / 2; j1++) {
      int row = j0 * LDG_ELEMS + i * LDG_ELEMS / N;
      int col = i * LDG_ELEMS % N / 2 + j1;
      ((half2*)&state_vec)[j1] = state_smem[row][(row % 32) ^ col];
    }
    ((int4*)state_ptr)[j0 * N + i] = state_vec;
  }
}

template <bool AddW0 = false>
__global__ __launch_bounds__(N, 2) void wkv_fp16_seq_v2_kernel(
    const int* __restrict__ query_start_loc, const int* __restrict__ slot_indices,
    int C, int H, half* __restrict__ state_ptr, const half* __restrict__ r_ptr,
    const half* __restrict__ w_ptr, const half* __restrict__ w0_ptr,
    const half* __restrict__ k_ptr, const half* __restrict__ v_ptr,
    const half* __restrict__ a_ptr, const half* __restrict__ b_ptr,
    half* __restrict__ y_ptr, const int* __restrict__ elapsed_t) {
  const int bh = blockIdx.x;
  const int b_id = bh / H;
  const int h = bh - b_id * H;
  const int i = threadIdx.x;
  const int lane = i & 31;
  const int slot_id = slot_indices[b_id];

  __shared__ __align__(256) half2 state_smem[N][HALF2_N];
  state_ptr += static_cast<int64_t>(slot_id) * C * N + h * N * N;

#pragma unroll
  for (int j0 = 0; j0 < N / LDG_ELEMS; ++j0) {
    int4 state_vec = ((int4*)state_ptr)[j0 * N + i];
#pragma unroll
    for (int j1 = 0; j1 < LDG_ELEMS / 2; ++j1) {
      int row = j0 * LDG_ELEMS + i * LDG_ELEMS / N;
      int col = i * LDG_ELEMS % N / 2 + j1;
      state_smem[row][(row & 31) ^ col] = ((half2*)&state_vec)[j1];
    }
  }
  __syncthreads();

  half2 state[HALF2_N];
#pragma unroll
  for (int j = 0; j < HALF2_N; ++j) {
    state[j] = state_smem[i][lane ^ j];
  }

  __shared__ __align__(128) half2 r[2][HALF2_N], w[2][HALF2_N], k[2][HALF2_N],
      a[2][HALF2_N], bvec[2][HALF2_N];

  const int my_t = query_start_loc[b_id + 1] - query_start_loc[b_id];
  const int token_base = query_start_loc[b_id];
  int token = token_base * C + h * N;
  prefetch_token(i, lane, token, r[0], w[0], k[0], a[0], bvec[0], r_ptr, w_ptr,
                 k_ptr, a_ptr, b_ptr);

  for (int tt = 0; tt < my_t; ++tt) {
    const int cur = tt & 1;
    cp_wait<0>();
    __syncthreads();

    half2 sa2 = {0.0f, 0.0f};
#pragma unroll
    for (int j = 0; j < HALF2_N; ++j) {
      sa2 = __hfma2(a[cur][j], state[j], sa2);
    }
    half sa = sa2.x + sa2.y;
    sa2 = {sa, sa};
    ((half*)w[cur])[i] =
        w_delta_maybe_w0<AddW0>(((half*)w[cur])[i], w0_ptr, h * N + i,
                                 elapsed_t[slot_id] + h * N + i + tt);
    __syncthreads();

    if (tt + 1 < my_t) {
      int next_token = token + C;
      prefetch_token(i, lane, next_token, r[cur ^ 1], w[cur ^ 1], k[cur ^ 1],
                     a[cur ^ 1], bvec[cur ^ 1], r_ptr, w_ptr, k_ptr, a_ptr,
                     b_ptr);
    }

    half vv = v_ptr[token + i];
    half2 vv2 = {vv, vv};
    half2 y2 = {0.0f, 0.0f};
#pragma unroll
    for (int j = 0; j < HALF2_N; ++j) {
      half2 s = state[j];
      s = __hfma2(s, w[cur][j],
                  __hfma2(k[cur][j], vv2, __hfma2(sa2, bvec[cur][j], s)));
      state[j] = s;
      y2 = __hfma2(s, r[cur][j], y2);
    }
    y_ptr[token + i] = y2.x + y2.y;
    token += C;
  }

#pragma unroll
  for (int j = 0; j < HALF2_N; ++j) {
    state_smem[i][lane ^ j] = state[j];
  }
  __syncthreads();
#pragma unroll
  for (int j0 = 0; j0 < N / LDG_ELEMS; ++j0) {
    int4 state_vec;
#pragma unroll
    for (int j1 = 0; j1 < LDG_ELEMS / 2; ++j1) {
      int row = j0 * LDG_ELEMS + i * LDG_ELEMS / N;
      int col = i * LDG_ELEMS % N / 2 + j1;
      ((half2*)&state_vec)[j1] = state_smem[row][(row & 31) ^ col];
    }
    ((int4*)state_ptr)[j0 * N + i] = state_vec;
  }
}

static bool use_v2_seq(int B, int max_t) {
  // B==64/128 && T==1 scenario is dispatched to v1 wkv_one at Python layer,
  // will not reach here. This only covers T>1 seq scenarios.
  // TODO: this router need optim
  return (B == 1 && max_t >= 8) || (B == 4 && max_t >= 4) ||
         (B == 8 && max_t >= 8) || (max_t >= 16);
}

}  // namespace

void wkv_seq_v2_cuda_impl_varlen(int B, int max_t, const int* query_start_loc,
                                  const int* slot_indices, int C, int H,
                                  at::Tensor state, at::Tensor r, at::Tensor w,
                                  const half* w0_ptr, bool add_w0, at::Tensor k,
                                  at::Tensor v, at::Tensor a, at::Tensor b,
                                  at::Tensor y, at::Tensor elapsed_t) {
  assert(C == H * N);
  // Note: all-one (all requests T==1) scenario is dispatched to v1's
  // wkv_one_fp16 at Python wrapper layer, will not enter this function.

  auto stream = at::cuda::getCurrentCUDAStream();
  if (use_v2_seq(B, max_t)) {
    if (add_w0) {
      wkv_fp16_seq_v2_kernel<true><<<dim3(B * H), dim3(N), 0, stream>>>(
          query_start_loc, slot_indices, C, H,
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(w.data_ptr()), w0_ptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(y.data_ptr()), elapsed_t.data_ptr<int>());
    } else {
      wkv_fp16_seq_v2_kernel<false><<<dim3(B * H), dim3(N), 0, stream>>>(
          query_start_loc, slot_indices, C, H,
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(w.data_ptr()), nullptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(y.data_ptr()), elapsed_t.data_ptr<int>());
    }
  } else {
    if (add_w0) {
      wkv_fp16_v1_exact_kernel<true><<<dim3(B * H), dim3(N), 0, stream>>>(
          query_start_loc, slot_indices, C, H,
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(w.data_ptr()), w0_ptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(y.data_ptr()), elapsed_t.data_ptr<int>());
    } else {
      wkv_fp16_v1_exact_kernel<false><<<dim3(B * H), dim3(N), 0, stream>>>(
          query_start_loc, slot_indices, C, H,
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(w.data_ptr()), nullptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(y.data_ptr()), elapsed_t.data_ptr<int>());
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wkv_seq_v2_cuda_varlen(int B, int max_t, const int* query_start_loc,
                            const int* slot_indices, int C, int H,
                            at::Tensor state, at::Tensor r, at::Tensor w,
                            at::Tensor k, at::Tensor v, at::Tensor a,
                            at::Tensor b, at::Tensor y, at::Tensor elapsed_t) {
  wkv_seq_v2_cuda_impl_varlen(B, max_t, query_start_loc, slot_indices, C, H,
                              state, r, w, nullptr, false, k, v, a, b, y,
                              elapsed_t);
}

void wkv_seq_w0_v2_cuda_varlen(int B, int max_t, const int* query_start_loc,
                               const int* slot_indices, int C, int H,
                               at::Tensor state, at::Tensor r, at::Tensor w,
                               at::Tensor w0, at::Tensor k, at::Tensor v,
                               at::Tensor a, at::Tensor b, at::Tensor y,
                               at::Tensor elapsed_t) {
  wkv_seq_v2_cuda_impl_varlen(B, max_t, query_start_loc, slot_indices, C, H,
                              state, r, w,
                              reinterpret_cast<const half*>(w0.data_ptr()),
                              true, k, v, a, b, y, elapsed_t);
}
