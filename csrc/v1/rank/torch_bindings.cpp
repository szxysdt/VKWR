#include <torch/extension.h>
#include <torch/library.h>

#include "core/register.h"
#include "v1_check.h"
#include "rank_ops.h"

std::vector<torch::Tensor> linear_wag_rank_in_f16_fn(
    torch::Tensor xw, torch::Tensor xa, torch::Tensor xg,
    torch::Tensor w1_t, torch::Tensor a1_t, torch::Tensor g1_t) {
  check_half_cuda_contig(xw, "xw");
  check_half_cuda_contig(xa, "xa");
  check_half_cuda_contig(xg, "xg");
  TORCH_CHECK(xw.sizes() == xa.sizes() && xw.sizes() == xg.sizes(),
              "xw/xa/xg shape mismatch");
  TORCH_CHECK(xw.dim() >= 2, "xw must have at least 2 dimensions");
  check_half_cuda_contig(w1_t, "w1_t");
  check_half_cuda_contig(a1_t, "a1_t");
  check_half_cuda_contig(g1_t, "g1_t");
  TORCH_CHECK(w1_t.dim() == 2 && a1_t.dim() == 2 && g1_t.dim() == 2,
              "weight_t must be 2D");
  TORCH_CHECK(xw.size(-1) == w1_t.size(1) && xw.size(-1) == a1_t.size(1) &&
              xw.size(-1) == g1_t.size(1),
              "rank-in K mismatch");
  return linear_wag_rank_in_f16_cuda(xw, xa, xg, w1_t, a1_t, g1_t);
}

std::vector<torch::Tensor> linear_wagv_rank_in_f16_fn(
    torch::Tensor xw, torch::Tensor xa, torch::Tensor xg, torch::Tensor xv,
    torch::Tensor w1_t, torch::Tensor a1_t, torch::Tensor g1_t, torch::Tensor v1_t) {
  check_half_cuda_contig(xw, "xw");
  check_half_cuda_contig(xa, "xa");
  check_half_cuda_contig(xg, "xg");
  check_half_cuda_contig(xv, "xv");
  TORCH_CHECK(xw.sizes() == xa.sizes() && xw.sizes() == xg.sizes() && xw.sizes() == xv.sizes(),
              "xw/xa/xg/xv shape mismatch");
  TORCH_CHECK(xw.dim() >= 2, "xw must have at least 2 dimensions");
  check_half_cuda_contig(w1_t, "w1_t");
  check_half_cuda_contig(a1_t, "a1_t");
  check_half_cuda_contig(g1_t, "g1_t");
  check_half_cuda_contig(v1_t, "v1_t");
  TORCH_CHECK(w1_t.dim() == 2 && a1_t.dim() == 2 && g1_t.dim() == 2 && v1_t.dim() == 2,
              "weight_t must be 2D");
  TORCH_CHECK(xw.size(-1) == w1_t.size(1) && xw.size(-1) == a1_t.size(1) &&
              xw.size(-1) == g1_t.size(1) && xw.size(-1) == v1_t.size(1),
              "rank-in K mismatch");
  return linear_wagv_rank_in_f16_cuda(xw, xa, xg, xv, w1_t, a1_t, g1_t, v1_t);
}

std::vector<torch::Tensor> linear_wag_rank_out_f16_fn(
    torch::Tensor w1, torch::Tensor a1, torch::Tensor g1,
    torch::Tensor w2_t, torch::Tensor a2_t, torch::Tensor g2_t) {
  check_half_cuda_contig(w1, "w1");
  check_half_cuda_contig(a1, "a1");
  check_half_cuda_contig(g1, "g1");
  TORCH_CHECK(w1.dim() >= 2 && a1.dim() == w1.dim() && g1.dim() == w1.dim(),
              "w1/a1/g1 dim mismatch");
  TORCH_CHECK(w1.sizes().slice(0, w1.dim() - 1) == a1.sizes().slice(0, a1.dim() - 1),
              "w1/a1 batch mismatch");
  TORCH_CHECK(w1.sizes().slice(0, w1.dim() - 1) == g1.sizes().slice(0, g1.dim() - 1),
              "w1/g1 batch mismatch");
  check_half_cuda_contig(w2_t, "w2_t");
  check_half_cuda_contig(a2_t, "a2_t");
  check_half_cuda_contig(g2_t, "g2_t");
  TORCH_CHECK(w2_t.dim() == 2 && a2_t.dim() == 2 && g2_t.dim() == 2,
              "weight_t must be 2D");
  TORCH_CHECK(w2_t.size(0) == a2_t.size(0) && w2_t.size(0) == g2_t.size(0),
              "output C mismatch");
  TORCH_CHECK(w1.size(-1) == w2_t.size(1), "w rank mismatch");
  TORCH_CHECK(a1.size(-1) == a2_t.size(1), "a rank mismatch");
  TORCH_CHECK(g1.size(-1) == g2_t.size(1), "g rank mismatch");
  return linear_wag_rank_out_f16_cuda(w1, a1, g1, w2_t, a2_t, g2_t);
}

std::vector<torch::Tensor> linear_wagv_rank_out_f16_fn(
    torch::Tensor w1, torch::Tensor a1, torch::Tensor g1, torch::Tensor v1,
    torch::Tensor w2_t, torch::Tensor a2_t, torch::Tensor g2_t, torch::Tensor v2_t,
    torch::Tensor v, torch::Tensor v_first, torch::Tensor v0) {
  check_half_cuda_contig(w1, "w1");
  check_half_cuda_contig(a1, "a1");
  check_half_cuda_contig(g1, "g1");
  check_half_cuda_contig(v1, "v1");
  TORCH_CHECK(w1.dim() >= 2 && a1.dim() == w1.dim() && g1.dim() == w1.dim() && v1.dim() == w1.dim(),
              "rank dim mismatch");
  check_half_cuda_contig(w2_t, "w2_t");
  check_half_cuda_contig(a2_t, "a2_t");
  check_half_cuda_contig(g2_t, "g2_t");
  check_half_cuda_contig(v2_t, "v2_t");
  TORCH_CHECK(w2_t.dim() == 2 && a2_t.dim() == 2 && g2_t.dim() == 2 && v2_t.dim() == 2,
              "weight_t must be 2D");
  TORCH_CHECK(w2_t.size(0) == a2_t.size(0) && w2_t.size(0) == g2_t.size(0) && w2_t.size(0) == v2_t.size(0),
              "output C mismatch");
  TORCH_CHECK(w1.size(-1) == w2_t.size(1) && a1.size(-1) == a2_t.size(1) &&
              g1.size(-1) == g2_t.size(1) && v1.size(-1) == v2_t.size(1),
              "rank mismatch");
  check_half_cuda_contig(v, "v");
  check_half_cuda_contig(v_first, "v_first");
  check_half_cuda_contig(v0, "v0");
  TORCH_CHECK(v.sizes() == v_first.sizes(), "v/v_first shape mismatch");
  TORCH_CHECK(v.dim() >= 2 && v.size(-1) == w2_t.size(0), "v shape mismatch");
  TORCH_CHECK(v0.dim() == 1 && v0.size(0) == w2_t.size(0), "v0 shape mismatch");
  return linear_wagv_rank_out_f16_cuda(w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0);
}

TORCH_LIBRARY(vkwr_v1_rank, m) {
  m.def("linear_wag_rank_in_f16(Tensor xw, Tensor xa, Tensor xg, "
      "Tensor w1_t, Tensor a1_t, Tensor g1_t) -> Tensor[]");
  m.impl("linear_wag_rank_in_f16", c10::kCUDA, &linear_wag_rank_in_f16_fn);

  m.def("linear_wagv_rank_in_f16(Tensor xw, Tensor xa, Tensor xg, Tensor xv, "
      "Tensor w1_t, Tensor a1_t, Tensor g1_t, Tensor v1_t) -> Tensor[]");
  m.impl("linear_wagv_rank_in_f16", c10::kCUDA, &linear_wagv_rank_in_f16_fn);

  m.def("linear_wag_rank_out_f16(Tensor w1, Tensor a1, Tensor g1, "
      "Tensor w2_t, Tensor a2_t, Tensor g2_t) -> Tensor[]");
  m.impl("linear_wag_rank_out_f16", c10::kCUDA, &linear_wag_rank_out_f16_fn);

  m.def("linear_wagv_rank_out_f16(Tensor w1, Tensor a1, Tensor g1, Tensor v1, "
      "Tensor w2_t, Tensor a2_t, Tensor g2_t, Tensor v2_t, "
      "Tensor v, Tensor v_first, Tensor v0) -> Tensor[]");
  m.impl("linear_wagv_rank_out_f16", c10::kCUDA, &linear_wagv_rank_out_f16_fn);
}

REGISTER_EXTENSION(_v1_rank_C)
