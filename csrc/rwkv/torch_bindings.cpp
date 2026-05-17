#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include "rwkv_ops.h"
#include "core/register.h"

typedef at::Half fp16;

template <typename F>
void cuda_wkv_forward(int B, int T, int C,
                      float *w, float *u, F *k, F *v, F *y,
                      float *aa, float *bb, float *pp);

template <typename F>
void cuda_mm8_seq(int B, int N, int M,
                  F *x, int x_stride,
                  uint8_t *w, int w_stride,
                  F *mx, F *rx,
                  F *my, F *ry,
                  F *y, int y_stride);

template <typename F>
void cuda_mm8_one(int N, int M,
                  F *x,
                  uint8_t *w, int w_stride,
                  F *mx, F *rx,
                  F *my, F *ry,
                  float *y);

void wkv_forward(int64_t B, int64_t T, int64_t C,
                 torch::Tensor &w, torch::Tensor &u,
                 torch::Tensor &k, torch::Tensor &v, torch::Tensor &y,
                 torch::Tensor &aa, torch::Tensor &bb, torch::Tensor &pp) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
    switch (k.scalar_type()) {
        case c10::ScalarType::Half:
            cuda_wkv_forward(B, T, C,
                w.data_ptr<float>(), u.data_ptr<float>(),
                k.data_ptr<fp16>(), v.data_ptr<fp16>(), y.data_ptr<fp16>(),
                aa.data_ptr<float>(), bb.data_ptr<float>(), pp.data_ptr<float>());
            break;
        case c10::ScalarType::Float:
            cuda_wkv_forward(B, T, C,
                w.data_ptr<float>(), u.data_ptr<float>(),
                k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(),
                aa.data_ptr<float>(), bb.data_ptr<float>(), pp.data_ptr<float>());
            break;
        default:
            throw std::runtime_error("wkv_forward: only FP16 and FP32 supported");
    }
}

void mm8_seq(int64_t B, int64_t N, int64_t M,
             torch::Tensor &x, torch::Tensor &w,
             torch::Tensor &mx, torch::Tensor &rx,
             torch::Tensor &my, torch::Tensor &ry,
             torch::Tensor &y) {
    TORCH_CHECK(x.stride(1) == 1, "mm8_seq: x must be contiguous along dim 1, got stride ", x.stride(1));
    TORCH_CHECK(w.stride(1) == 1, "mm8_seq: w must be contiguous along dim 1, got stride ", w.stride(1));
    TORCH_CHECK(mx.stride(0) == 1, "mm8_seq: mx must be contiguous along dim 0, got stride ", mx.stride(0));
    TORCH_CHECK(rx.stride(0) == 1, "mm8_seq: rx must be contiguous along dim 0, got stride ", rx.stride(0));
    TORCH_CHECK(my.stride(0) == 1, "mm8_seq: my must be contiguous along dim 0, got stride ", my.stride(0));
    TORCH_CHECK(ry.stride(0) == 1, "mm8_seq: ry must be contiguous along dim 0, got stride ", ry.stride(0));
    TORCH_CHECK(y.stride(1) == 1, "mm8_seq: y must be contiguous along dim 1, got stride ", y.stride(1));
    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
    switch (x.scalar_type()) {
        case c10::ScalarType::Half:
            cuda_mm8_seq(B, N, M,
                x.data_ptr<fp16>(), x.stride(0),
                w.data_ptr<uint8_t>(), w.stride(0),
                mx.data_ptr<fp16>(), rx.data_ptr<fp16>(),
                my.data_ptr<fp16>(), ry.data_ptr<fp16>(),
                y.data_ptr<fp16>(), y.stride(0));
            break;
        case c10::ScalarType::Float:
            cuda_mm8_seq(B, N, M,
                x.data_ptr<float>(), x.stride(0),
                w.data_ptr<uint8_t>(), w.stride(0),
                mx.data_ptr<float>(), rx.data_ptr<float>(),
                my.data_ptr<float>(), ry.data_ptr<float>(),
                y.data_ptr<float>(), y.stride(0));
            break;
        default:
            throw std::runtime_error("mm8_seq: only FP16 and FP32 supported");
    }
}

void mm8_one(int64_t N, int64_t M,
             torch::Tensor &x, torch::Tensor &w,
             torch::Tensor &mx, torch::Tensor &rx,
             torch::Tensor &my, torch::Tensor &ry,
             torch::Tensor &y) {
    TORCH_CHECK(x.stride(0) == 1, "mm8_one: x must be contiguous along dim 0, got stride ", x.stride(0));
    TORCH_CHECK(w.stride(1) == 1, "mm8_one: w must be contiguous along dim 1, got stride ", w.stride(1));
    TORCH_CHECK(mx.stride(0) == 1, "mm8_one: mx must be contiguous along dim 0, got stride ", mx.stride(0));
    TORCH_CHECK(rx.stride(0) == 1, "mm8_one: rx must be contiguous along dim 0, got stride ", rx.stride(0));
    TORCH_CHECK(my.stride(0) == 1, "mm8_one: my must be contiguous along dim 0, got stride ", my.stride(0));
    TORCH_CHECK(ry.stride(0) == 1, "mm8_one: ry must be contiguous along dim 0, got stride ", ry.stride(0));
    TORCH_CHECK(y.stride(0) == 1, "mm8_one: y must be contiguous along dim 0, got stride ", y.stride(0));
    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
    switch (x.scalar_type()) {
        case c10::ScalarType::Half:
            cuda_mm8_one(N, M,
                x.data_ptr<fp16>(),
                w.data_ptr<uint8_t>(), w.stride(0),
                mx.data_ptr<fp16>(), rx.data_ptr<fp16>(),
                my.data_ptr<fp16>(), ry.data_ptr<fp16>(),
                y.data_ptr<float>());
            break;
        case c10::ScalarType::Float:
            cuda_mm8_one(N, M,
                x.data_ptr<float>(),
                w.data_ptr<uint8_t>(), w.stride(0),
                mx.data_ptr<float>(), rx.data_ptr<float>(),
                my.data_ptr<float>(), ry.data_ptr<float>(),
                y.data_ptr<float>());
            break;
        default:
            throw std::runtime_error("mm8_one: only FP16 and FP32 supported");
    }
}

TORCH_LIBRARY(vkwr_rwkv, m) {
    m.def("wkv_forward(int B, int T, int C, "
          "Tensor w, Tensor u, Tensor k, Tensor v, Tensor(a!) y, "
          "Tensor aa, Tensor bb, Tensor pp) -> ()");
    m.impl("wkv_forward", c10::kCUDA, &wkv_forward);

    m.def("mm8_seq(int B, int N, int M, "
          "Tensor x, Tensor w, Tensor mx, Tensor rx, Tensor my, Tensor ry, Tensor(a!) y) -> ()");
    m.impl("mm8_seq", c10::kCUDA, &mm8_seq);

    m.def("mm8_one(int N, int M, "
          "Tensor x, Tensor w, Tensor mx, Tensor rx, Tensor my, Tensor ry, Tensor(a!) y) -> ()");
    m.impl("mm8_one", c10::kCUDA, &mm8_one);

#ifdef ENABLE_CUBLAS_GEMM
    m.def("gemm_fp16_cublas(Tensor a, Tensor b, Tensor(a!) c) -> ()");
    m.impl("gemm_fp16_cublas", c10::kCUDA, &gemm_fp16_cublas_impl);
#endif
}

REGISTER_EXTENSION(_rwkv_C)
