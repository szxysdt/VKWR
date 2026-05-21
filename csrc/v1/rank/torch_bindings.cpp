#include <torch/extension.h>
#include <torch/library.h>

#include "core/register.h"
#include "v1/common/v1_check.h"
#include "rank_ops.h"

std::vector<torch::Tensor> linear_wag_rank_in_f16_fn(
    int64_t M, int64_t K, int64_t Rw, int64_t Ra, int64_t Rg,
    torch::Tensor xw, torch::Tensor xa, torch::Tensor xg,
    torch::Tensor w1_t, torch::Tensor a1_t, torch::Tensor g1_t) {
  check_half_cuda_contig(xw, "xw");
  check_half_cuda_contig(xa, "xa");
  check_half_cuda_contig(xg, "xg");
  TORCH_CHECK(xw.dim() == 2 && xw.size(0) == M && xw.size(1) == K, "xw shape mismatch");
  TORCH_CHECK(xa.dim() == 2 && xa.size(0) == M && xa.size(1) == K, "xa shape mismatch");
  TORCH_CHECK(xg.dim() == 2 && xg.size(0) == M && xg.size(1) == K, "xg shape mismatch");
  check_half_cuda_contig(w1_t, "w1_t");
  check_half_cuda_contig(a1_t, "a1_t");
  check_half_cuda_contig(g1_t, "g1_t");
  TORCH_CHECK(w1_t.dim() == 2 && w1_t.size(0) == Rw && w1_t.size(1) == K, "w1_t shape mismatch");
  TORCH_CHECK(a1_t.dim() == 2 && a1_t.size(0) == Ra && a1_t.size(1) == K, "a1_t shape mismatch");
  TORCH_CHECK(g1_t.dim() == 2 && g1_t.size(0) == Rg && g1_t.size(1) == K, "g1_t shape mismatch");
  return linear_wag_rank_in_f16_cuda(
      M, K, Rw, Ra, Rg, xw, xa, xg, w1_t, a1_t, g1_t);
}

std::vector<torch::Tensor> linear_wagv_rank_in_f16_fn(
    int64_t M, int64_t K, int64_t Rw, int64_t Ra, int64_t Rg, int64_t Rv,
    torch::Tensor xw, torch::Tensor xa, torch::Tensor xg, torch::Tensor xv,
    torch::Tensor w1_t, torch::Tensor a1_t, torch::Tensor g1_t, torch::Tensor v1_t) {
  check_half_cuda_contig(xw, "xw");
  check_half_cuda_contig(xa, "xa");
  check_half_cuda_contig(xg, "xg");
  check_half_cuda_contig(xv, "xv");
  TORCH_CHECK(xw.dim() == 2 && xw.size(0) == M && xw.size(1) == K, "xw shape mismatch");
  TORCH_CHECK(xa.dim() == 2 && xa.size(0) == M && xa.size(1) == K, "xa shape mismatch");
  TORCH_CHECK(xg.dim() == 2 && xg.size(0) == M && xg.size(1) == K, "xg shape mismatch");
  TORCH_CHECK(xv.dim() == 2 && xv.size(0) == M && xv.size(1) == K, "xv shape mismatch");
  check_half_cuda_contig(w1_t, "w1_t");
  check_half_cuda_contig(a1_t, "a1_t");
  check_half_cuda_contig(g1_t, "g1_t");
  check_half_cuda_contig(v1_t, "v1_t");
  TORCH_CHECK(w1_t.dim() == 2 && w1_t.size(0) == Rw && w1_t.size(1) == K, "w1_t shape mismatch");
  TORCH_CHECK(a1_t.dim() == 2 && a1_t.size(0) == Ra && a1_t.size(1) == K, "a1_t shape mismatch");
  TORCH_CHECK(g1_t.dim() == 2 && g1_t.size(0) == Rg && g1_t.size(1) == K, "g1_t shape mismatch");
  TORCH_CHECK(v1_t.dim() == 2 && v1_t.size(0) == Rv && v1_t.size(1) == K, "v1_t shape mismatch");
  return linear_wagv_rank_in_f16_cuda(
      M, K, Rw, Ra, Rg, Rv, xw, xa, xg, xv, w1_t, a1_t, g1_t, v1_t);
}

std::vector<torch::Tensor> linear_wag_rank_out_f16_fn(
    int64_t M, int64_t C, int64_t Kw, int64_t Ka, int64_t Kg,
    torch::Tensor w1, torch::Tensor a1, torch::Tensor g1,
    torch::Tensor w2_t, torch::Tensor a2_t, torch::Tensor g2_t) {
  check_half_cuda_contig(w1, "w1");
  check_half_cuda_contig(a1, "a1");
  check_half_cuda_contig(g1, "g1");
  TORCH_CHECK(w1.dim() == 2 && w1.size(0) == M && w1.size(1) == Kw, "w1 shape mismatch");
  TORCH_CHECK(a1.dim() == 2 && a1.size(0) == M && a1.size(1) == Ka, "a1 shape mismatch");
  TORCH_CHECK(g1.dim() == 2 && g1.size(0) == M && g1.size(1) == Kg, "g1 shape mismatch");
  check_half_cuda_contig(w2_t, "w2_t");
  check_half_cuda_contig(a2_t, "a2_t");
  check_half_cuda_contig(g2_t, "g2_t");
  TORCH_CHECK(w2_t.dim() == 2 && w2_t.size(0) == C && w2_t.size(1) == Kw, "w2_t shape mismatch");
  TORCH_CHECK(a2_t.dim() == 2 && a2_t.size(0) == C && a2_t.size(1) == Ka, "a2_t shape mismatch");
  TORCH_CHECK(g2_t.dim() == 2 && g2_t.size(0) == C && g2_t.size(1) == Kg, "g2_t shape mismatch");
  return linear_wag_rank_out_f16_cuda(
      M, C, Kw, Ka, Kg, w1, a1, g1, w2_t, a2_t, g2_t);
}

std::vector<torch::Tensor> linear_wagv_rank_out_f16_fn(
    int64_t M, int64_t C, int64_t Kw, int64_t Ka, int64_t Kg, int64_t Kv,
    torch::Tensor w1, torch::Tensor a1, torch::Tensor g1, torch::Tensor v1,
    torch::Tensor w2_t, torch::Tensor a2_t, torch::Tensor g2_t, torch::Tensor v2_t,
    torch::Tensor v, torch::Tensor v_first, torch::Tensor v0) {
  check_half_cuda_contig(w1, "w1");
  check_half_cuda_contig(a1, "a1");
  check_half_cuda_contig(g1, "g1");
  check_half_cuda_contig(v1, "v1");
  TORCH_CHECK(w1.dim() == 2 && w1.size(0) == M && w1.size(1) == Kw, "w1 shape mismatch");
  TORCH_CHECK(a1.dim() == 2 && a1.size(0) == M && a1.size(1) == Ka, "a1 shape mismatch");
  TORCH_CHECK(g1.dim() == 2 && g1.size(0) == M && g1.size(1) == Kg, "g1 shape mismatch");
  TORCH_CHECK(v1.dim() == 2 && v1.size(0) == M && v1.size(1) == Kv, "v1 shape mismatch");
  check_half_cuda_contig(w2_t, "w2_t");
  check_half_cuda_contig(a2_t, "a2_t");
  check_half_cuda_contig(g2_t, "g2_t");
  check_half_cuda_contig(v2_t, "v2_t");
  TORCH_CHECK(w2_t.dim() == 2 && w2_t.size(0) == C && w2_t.size(1) == Kw, "w2_t shape mismatch");
  TORCH_CHECK(a2_t.dim() == 2 && a2_t.size(0) == C && a2_t.size(1) == Ka, "a2_t shape mismatch");
  TORCH_CHECK(g2_t.dim() == 2 && g2_t.size(0) == C && g2_t.size(1) == Kg, "g2_t shape mismatch");
  TORCH_CHECK(v2_t.dim() == 2 && v2_t.size(0) == C && v2_t.size(1) == Kv, "v2_t shape mismatch");
  check_half_cuda_contig(v, "v");
  check_half_cuda_contig(v_first, "v_first");
  TORCH_CHECK(v.dim() == 2 && v.size(0) == M && v.size(1) == C, "v shape mismatch");
  TORCH_CHECK(v_first.dim() == 2 && v_first.size(0) == M && v_first.size(1) == C, "v_first shape mismatch");
  check_half_cuda_contig(v0, "v0");
  TORCH_CHECK(v0.dim() == 1 && v0.size(0) == C, "v0 shape mismatch");
  return linear_wagv_rank_out_f16_cuda(
      M, C, Kw, Ka, Kg, Kv, w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0);
}

TORCH_LIBRARY(vkwr_v1_rank, m) {
  m.def("linear_wag_rank_in_f16(int M, int K, int Rw, int Ra, int Rg, "
      "Tensor xw, Tensor xa, Tensor xg, Tensor w1_t, Tensor a1_t, Tensor g1_t) -> Tensor[]");
  m.impl("linear_wag_rank_in_f16", c10::kCUDA, &linear_wag_rank_in_f16_fn);

  m.def("linear_wagv_rank_in_f16(int M, int K, int Rw, int Ra, int Rg, int Rv, "
      "Tensor xw, Tensor xa, Tensor xg, Tensor xv, "
      "Tensor w1_t, Tensor a1_t, Tensor g1_t, Tensor v1_t) -> Tensor[]");
  m.impl("linear_wagv_rank_in_f16", c10::kCUDA, &linear_wagv_rank_in_f16_fn);

  m.def("linear_wag_rank_out_f16(int M, int C, int Kw, int Ka, int Kg, "
      "Tensor w1, Tensor a1, Tensor g1, Tensor w2_t, Tensor a2_t, Tensor g2_t) -> Tensor[]");
  m.impl("linear_wag_rank_out_f16", c10::kCUDA, &linear_wag_rank_out_f16_fn);

  m.def("linear_wagv_rank_out_f16(int M, int C, int Kw, int Ka, int Kg, int Kv, "
      "Tensor w1, Tensor a1, Tensor g1, Tensor v1, "
      "Tensor w2_t, Tensor a2_t, Tensor g2_t, Tensor v2_t, "
      "Tensor v, Tensor v_first, Tensor v0) -> Tensor[]");
  m.impl("linear_wagv_rank_out_f16", c10::kCUDA, &linear_wagv_rank_out_f16_fn);
}

REGISTER_EXTENSION(_v1_rank_C)
