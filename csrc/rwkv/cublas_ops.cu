#ifdef ENABLE_CUBLAS_GEMM
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#define CUBLAS_CHECK(condition)                                                \
  for (cublasStatus_t _status = (condition);                                   \
       _status != CUBLAS_STATUS_SUCCESS;)                                      \
    throw std::runtime_error("cuBLAS error " +                                 \
                             std::to_string(_status) + " at " +                \
                             std::to_string(__LINE__));

void gemm_fp16_cublas_impl(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const auto cuda_data_type = CUDA_R_16F;
  const auto cuda_c_data_type =
      c.dtype() == torch::kFloat32 ? CUDA_R_32F : CUDA_R_16F;
  const auto compute_type = CUDA_R_32F;
  const float sp_alpha = 1.f;
  std::swap(a, b);
  const cublasOperation_t cublas_trans_a = CUBLAS_OP_N;
  const cublasOperation_t cublas_trans_b = CUBLAS_OP_N;
  const int m = a.size(-1);
  const int k = a.size(-2);
  const int n = b.size(-2);
  const int cublas_lda = m;
  const int cublas_ldb = k;
  const int cublas_ldc = m;
  cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();

#if CUDA_VERSION >= 11000
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
#else
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
#endif
  const float sp_beta = 0.f;
  if (a.sizes().size() == 2 && b.sizes().size() == 2) {
    CUBLAS_CHECK(cublasGemmEx(
        cublas_handle, cublas_trans_a, cublas_trans_b, m, n, k, &sp_alpha,
        a.data_ptr(), cuda_data_type, cublas_lda, b.data_ptr(), cuda_data_type,
        cublas_ldb, &sp_beta, c.data_ptr(), cuda_c_data_type, cublas_ldc,
        compute_type, algo));
  } else {
    TORCH_CHECK(a.sizes().size() == 3 && b.sizes().size() == 3,
        "gemm_fp16_cublas: expected 3D tensors, got a.dim()=", a.sizes().size(),
        " b.dim()=", b.sizes().size());
    const long long int cublas_stride_a = m * k;
    const long long int cublas_stride_b = k * n;
    const long long int cublas_stride_c = m * n;
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cublas_handle, cublas_trans_a, cublas_trans_b, m,
        n, k, &sp_alpha, a.data_ptr(), cuda_data_type, cublas_lda,
        cublas_stride_a, b.data_ptr(), cuda_data_type, cublas_ldb, cublas_stride_b,
        &sp_beta, c.data_ptr(), cuda_c_data_type, cublas_ldc, cublas_stride_c,
        a.size(0), compute_type, algo));
  }
}
#endif
