#include <torch/extension.h>
#include <torch/library.h>

#include "core/register.h"
#include "v1/common/v1_check.h"
#include "norm_ops.h"

torch::Tensor layer_norm_f16_fn(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dim");
  const int64_t c = x.size(-1);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  TORCH_CHECK(c > 0 && c <= 8192, "unsupported C");
  return layer_norm_f16_cuda(x, weight, bias, eps);
}

torch::Tensor layer_norm_f16_small_fn(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dim");
  const int64_t c = x.size(-1);
  TORCH_CHECK(c == 4096, "small LN currently requires C=4096");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  return layer_norm_f16_small_cuda(x, weight, bias, eps);
}

torch::Tensor layer_norm_f16_small512_fn(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dim");
  const int64_t c = x.size(-1);
  TORCH_CHECK(c == 4096, "small512 LN currently requires C=4096");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  return layer_norm_f16_small512_cuda(x, weight, bias, eps);
}

torch::Tensor emb_ln0_bf16_to_f16_fn(torch::Tensor emb, torch::Tensor weight, torch::Tensor bias, double eps) {
  check_bf16_cuda_contig(emb, "emb");
  check_bf16_cuda_contig(weight, "weight");
  check_bf16_cuda_contig(bias, "bias");
  TORCH_CHECK(emb.dim() == 2, "emb must have shape [V, C]");
  const int64_t c = emb.size(1);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  return emb_ln0_bf16_to_f16_cuda(emb, weight, bias, eps);
}

torch::Tensor add_f16_fn(torch::Tensor x, torch::Tensor y) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(y, "y");
  TORCH_CHECK(x.sizes() == y.sizes(), "add_f16 shape mismatch");
  return add_f16_cuda(x, y);
}

std::vector<torch::Tensor> add_layer_norm_f16_fn(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, torch::Tensor bias, double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  TORCH_CHECK(x.sizes() == residual.sizes(), "add_layer_norm_f16 x/residual shape mismatch");
  TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dim");
  const int64_t c = x.size(-1);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  TORCH_CHECK(c > 0 && c <= 8192, "unsupported C");
  return add_layer_norm_f16_cuda(x, residual, weight, bias, eps);
}

torch::Tensor add_last_layer_norm_f16_fn(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, torch::Tensor bias, double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  TORCH_CHECK(x.sizes() == residual.sizes(), "add_last_layer_norm_f16 x/residual shape mismatch");
  TORCH_CHECK(x.dim() == 3, "x must have shape [B,T,C]");
  const int64_t c = x.size(2);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  TORCH_CHECK(c > 0 && c <= 8192, "unsupported C");
  return add_last_layer_norm_f16_cuda(x, residual, weight, bias, eps);
}

std::vector<torch::Tensor> add_layer_norm_cmix_mix_f16_fn(torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state, torch::Tensor weight, torch::Tensor bias, torch::Tensor x_k, double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(shift_state, "shift_state");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  check_half_cuda_contig(x_k, "x_k");
  TORCH_CHECK(x.sizes() == residual.sizes(), "add_layer_norm_cmix_mix_f16 x/residual shape mismatch");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1, "add_layer_norm_cmix_mix_f16 requires shape [B,1,C]");
  const int64_t c = x.size(2);
  TORCH_CHECK((c % 2) == 0 && c > 0 && c <= 8192, "unsupported C");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == x.size(0) && shift_state.size(1) == c,
              "shift_state shape mismatch");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  TORCH_CHECK(x_k.dim() == 1 && x_k.size(0) == c, "x_k shape mismatch");
  return add_layer_norm_cmix_mix_f16_cuda(x, residual, shift_state, weight, bias, x_k, eps);
}

std::vector<torch::Tensor> add_layer_norm_cmix_mix_f16_cfg_fn(torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state, torch::Tensor weight, torch::Tensor bias, torch::Tensor x_k, double eps, int64_t threads) {
  TORCH_CHECK(threads == 256 || threads == 512 || threads == 1024, "threads must be 256, 512, or 1024");
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(shift_state, "shift_state");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  check_half_cuda_contig(x_k, "x_k");
  TORCH_CHECK(x.sizes() == residual.sizes(), "add_layer_norm_cmix_mix_f16_cfg x/residual shape mismatch");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1 && x.size(2) == 4096, "add_layer_norm_cmix_mix_f16_cfg requires shape [B,1,4096]");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == x.size(0) && shift_state.size(1) == 4096,
              "shift_state shape mismatch");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == 4096, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == 4096, "bias shape mismatch");
  TORCH_CHECK(x_k.dim() == 1 && x_k.size(0) == 4096, "x_k shape mismatch");
  return add_layer_norm_cmix_mix_f16_cfg_cuda(x, residual, shift_state, weight, bias, x_k, eps, static_cast<int>(threads));
}

std::vector<torch::Tensor> add_layer_norm_cmix_mix_f16_scalar_stats_fn(torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state, torch::Tensor weight, torch::Tensor bias, torch::Tensor x_k, double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(shift_state, "shift_state");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  check_half_cuda_contig(x_k, "x_k");
  TORCH_CHECK(x.sizes() == residual.sizes(), "add_layer_norm_cmix_mix_f16_scalar_stats x/residual shape mismatch");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1 && x.size(2) == 4096, "add_layer_norm_cmix_mix_f16_scalar_stats requires shape [B,1,4096]");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == x.size(0) && shift_state.size(1) == 4096,
              "shift_state shape mismatch");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == 4096, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == 4096, "bias shape mismatch");
  TORCH_CHECK(x_k.dim() == 1 && x_k.size(0) == 4096, "x_k shape mismatch");
  return add_layer_norm_cmix_mix_f16_scalar_stats_cuda(x, residual, shift_state, weight, bias, x_k, eps);
}

std::vector<torch::Tensor> add_layer_norm_tmix_mix6_f16_fn(
    torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor x_r, torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a, torch::Tensor x_g,
    double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(shift_state, "shift_state");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  check_half_cuda_contig(x_r, "x_r");
  check_half_cuda_contig(x_w, "x_w");
  check_half_cuda_contig(x_k, "x_k");
  check_half_cuda_contig(x_v, "x_v");
  check_half_cuda_contig(x_a, "x_a");
  check_half_cuda_contig(x_g, "x_g");
  TORCH_CHECK(x.sizes() == residual.sizes(), "add_layer_norm_tmix_mix6_f16 x/residual shape mismatch");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1, "add_layer_norm_tmix_mix6_f16 requires shape [B,1,C]");
  const int64_t c = x.size(2);
  TORCH_CHECK((c % 2) == 0 && c > 0 && c <= 8192, "unsupported C");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == x.size(0) && shift_state.size(1) == c,
              "shift_state shape mismatch");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  TORCH_CHECK(x_r.numel() == c && x_w.numel() == c && x_k.numel() == c &&
              x_v.numel() == c && x_a.numel() == c && x_g.numel() == c,
              "mix vector shape mismatch");
  return add_layer_norm_tmix_mix6_f16_cuda(
      x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g, eps);
}

std::vector<torch::Tensor> add_layer_norm_tmix_mix6_f16_cfg_fn(
    torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor x_r, torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a, torch::Tensor x_g,
    double eps, int64_t threads) {
  TORCH_CHECK(threads == 256 || threads == 512 || threads == 1024, "threads must be 256, 512, or 1024");
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(shift_state, "shift_state");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  check_half_cuda_contig(x_r, "x_r");
  check_half_cuda_contig(x_w, "x_w");
  check_half_cuda_contig(x_k, "x_k");
  check_half_cuda_contig(x_v, "x_v");
  check_half_cuda_contig(x_a, "x_a");
  check_half_cuda_contig(x_g, "x_g");
  TORCH_CHECK(x.sizes() == residual.sizes(), "add_layer_norm_tmix_mix6_f16_cfg x/residual shape mismatch");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1 && x.size(2) == 4096, "add_layer_norm_tmix_mix6_f16_cfg requires shape [B,1,4096]");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == x.size(0) && shift_state.size(1) == 4096,
              "shift_state shape mismatch");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == 4096, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == 4096, "bias shape mismatch");
  TORCH_CHECK(x_r.numel() == 4096 && x_w.numel() == 4096 && x_k.numel() == 4096 &&
              x_v.numel() == 4096 && x_a.numel() == 4096 && x_g.numel() == 4096,
              "mix vector shape mismatch");
  return add_layer_norm_tmix_mix6_f16_cfg_cuda(
      x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g, eps, static_cast<int>(threads));
}

std::vector<torch::Tensor> add_layer_norm_tmix_mix6_f16_scalar_stats_fn(
    torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor x_r, torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a, torch::Tensor x_g,
    double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(shift_state, "shift_state");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  check_half_cuda_contig(x_r, "x_r");
  check_half_cuda_contig(x_w, "x_w");
  check_half_cuda_contig(x_k, "x_k");
  check_half_cuda_contig(x_v, "x_v");
  check_half_cuda_contig(x_a, "x_a");
  check_half_cuda_contig(x_g, "x_g");
  TORCH_CHECK(x.sizes() == residual.sizes(), "add_layer_norm_tmix_mix6_f16_scalar_stats x/residual shape mismatch");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1 && x.size(2) == 4096, "add_layer_norm_tmix_mix6_f16_scalar_stats requires shape [B,1,4096]");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == x.size(0) && shift_state.size(1) == 4096,
              "shift_state shape mismatch");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == 4096, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == 4096, "bias shape mismatch");
  TORCH_CHECK(x_r.numel() == 4096 && x_w.numel() == 4096 && x_k.numel() == 4096 &&
              x_v.numel() == 4096 && x_a.numel() == 4096 && x_g.numel() == 4096,
              "mix vector shape mismatch");
  return add_layer_norm_tmix_mix6_f16_scalar_stats_cuda(
      x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g, eps);
}

TORCH_LIBRARY(vkwr_v1_norm, m) {
  m.def("layer_norm_f16(Tensor x, Tensor weight, Tensor bias, float eps=1e-5) -> Tensor");
  m.impl("layer_norm_f16", c10::kCUDA, &layer_norm_f16_fn);

  m.def("layer_norm_f16_small(Tensor x, Tensor weight, Tensor bias, float eps=1e-5) -> Tensor");
  m.impl("layer_norm_f16_small", c10::kCUDA, &layer_norm_f16_small_fn);

  m.def("layer_norm_f16_small512(Tensor x, Tensor weight, Tensor bias, float eps=1e-5) -> Tensor");
  m.impl("layer_norm_f16_small512", c10::kCUDA, &layer_norm_f16_small512_fn);

  m.def("emb_ln0_bf16_to_f16(Tensor emb, Tensor weight, Tensor bias, float eps=1e-5) -> Tensor");
  m.impl("emb_ln0_bf16_to_f16", c10::kCUDA, &emb_ln0_bf16_to_f16_fn);

  m.def("add_f16(Tensor x, Tensor y) -> Tensor");
  m.impl("add_f16", c10::kCUDA, &add_f16_fn);

  m.def("add_layer_norm_f16(Tensor x, Tensor residual, Tensor weight, Tensor bias, float eps=1e-5) -> Tensor[]");
  m.impl("add_layer_norm_f16", c10::kCUDA, &add_layer_norm_f16_fn);

  m.def("add_last_layer_norm_f16(Tensor x, Tensor residual, Tensor weight, Tensor bias, float eps=1e-5) -> Tensor");
  m.impl("add_last_layer_norm_f16", c10::kCUDA, &add_last_layer_norm_f16_fn);

  m.def("add_layer_norm_cmix_mix_f16(Tensor x, Tensor residual, Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_k, float eps=1e-5) -> Tensor[]");
  m.impl("add_layer_norm_cmix_mix_f16", c10::kCUDA, &add_layer_norm_cmix_mix_f16_fn);

  m.def("add_layer_norm_cmix_mix_f16_cfg(Tensor x, Tensor residual, Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_k, float eps, int threads) -> Tensor[]");
  m.impl("add_layer_norm_cmix_mix_f16_cfg", c10::kCUDA, &add_layer_norm_cmix_mix_f16_cfg_fn);

  m.def("add_layer_norm_cmix_mix_f16_scalar_stats(Tensor x, Tensor residual, Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_k, float eps=1e-5) -> Tensor[]");
  m.impl("add_layer_norm_cmix_mix_f16_scalar_stats", c10::kCUDA, &add_layer_norm_cmix_mix_f16_scalar_stats_fn);

  m.def("add_layer_norm_tmix_mix6_f16(Tensor x, Tensor residual, Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, float eps=1e-5) -> Tensor[]");
  m.impl("add_layer_norm_tmix_mix6_f16", c10::kCUDA, &add_layer_norm_tmix_mix6_f16_fn);

  m.def("add_layer_norm_tmix_mix6_f16_cfg(Tensor x, Tensor residual, Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, float eps, int threads) -> Tensor[]");
  m.impl("add_layer_norm_tmix_mix6_f16_cfg", c10::kCUDA, &add_layer_norm_tmix_mix6_f16_cfg_fn);

  m.def("add_layer_norm_tmix_mix6_f16_scalar_stats(Tensor x, Tensor residual, Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, float eps=1e-5) -> Tensor[]");
  m.impl("add_layer_norm_tmix_mix6_f16_scalar_stats", c10::kCUDA, &add_layer_norm_tmix_mix6_f16_scalar_stats_fn);
}

REGISTER_EXTENSION(_v1_norm_C)
