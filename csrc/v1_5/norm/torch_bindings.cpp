#include <torch/extension.h>
#include <torch/library.h>

#include "core/register.h"
#include "norm_ops.h"
#include "v1/common/v1_check.h"

std::vector<torch::Tensor> add_layer_norm_tmix_mix6_f16_varlen_fn(
    int64_t total_tokens, torch::Tensor x, torch::Tensor residual,
    torch::Tensor shift_state, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor x_r, torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v,
    torch::Tensor x_a, torch::Tensor x_g, torch::Tensor query_start_loc,
    torch::Tensor req_id, double eps) {
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
  TORCH_CHECK(x.sizes() == residual.sizes(),
              "add_layer_norm_tmix_mix6_f16_varlen x/residual shape mismatch");
  TORCH_CHECK(query_start_loc.is_cuda(),
              "query_start_loc must be CUDA tensor (device-side read)");
  TORCH_CHECK(req_id.is_cuda(),
              "req_id must be CUDA tensor (device-side read)");
  int64_t B = query_start_loc.size(0) - 1;
  const int64_t c = x.size(-1);
  TORCH_CHECK(x.dim() == 2 && x.size(0) == total_tokens && x.size(1) == c,
              "x must have shape [total_tokens, C]");
  TORCH_CHECK((c % 2) == 0 && c > 0 && c <= 8192, "unsupported C");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B &&
                  shift_state.size(1) == c,
              "shift_state shape mismatch");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c,
              "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  TORCH_CHECK(x_r.numel() == c && x_w.numel() == c && x_k.numel() == c &&
                  x_v.numel() == c && x_a.numel() == c && x_g.numel() == c,
              "mix vector shape mismatch");
  TORCH_CHECK(query_start_loc.dim() == 1 && query_start_loc.size(0) == B + 1,
              "query_start_loc must have shape [B+1]");
  TORCH_CHECK(query_start_loc.scalar_type() == torch::kInt32,
              "query_start_loc must be int32");
  TORCH_CHECK(req_id.dim() == 1 && req_id.size(0) == total_tokens,
              "req_id must have shape [total_tokens]");
  TORCH_CHECK(req_id.scalar_type() == torch::kInt32, "req_id must be int32");
  return add_layer_norm_tmix_mix6_f16_cuda_varlen(
      x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g,
      query_start_loc, req_id, eps);
}

std::vector<torch::Tensor> add_layer_norm_cmix_mix_f16_varlen_fn(
    int64_t total_tokens, torch::Tensor x, torch::Tensor residual,
    torch::Tensor shift_state, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor x_k, torch::Tensor query_start_loc, torch::Tensor req_id,
    double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(shift_state, "shift_state");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  check_half_cuda_contig(x_k, "x_k");
  TORCH_CHECK(x.sizes() == residual.sizes(),
              "add_layer_norm_cmix_mix_f16_varlen x/residual shape mismatch");
  TORCH_CHECK(query_start_loc.is_cuda(),
              "query_start_loc must be CUDA tensor (device-side read)");
  TORCH_CHECK(req_id.is_cuda(),
              "req_id must be CUDA tensor (device-side read)");
  int64_t B = query_start_loc.size(0) - 1;
  const int64_t c = x.size(-1);
  TORCH_CHECK(x.dim() == 2 && x.size(0) == total_tokens && x.size(1) == c,
              "x must have shape [total_tokens, C]");
  TORCH_CHECK((c % 2) == 0 && c > 0 && c <= 8192, "unsupported C");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B &&
                  shift_state.size(1) == c,
              "shift_state shape mismatch");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c,
              "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  TORCH_CHECK(x_k.dim() == 1 && x_k.size(0) == c, "x_k shape mismatch");
  TORCH_CHECK(query_start_loc.dim() == 1 && query_start_loc.size(0) == B + 1,
              "query_start_loc must have shape [B+1]");
  TORCH_CHECK(query_start_loc.scalar_type() == torch::kInt32,
              "query_start_loc must be int32");
  TORCH_CHECK(req_id.dim() == 1 && req_id.size(0) == total_tokens,
              "req_id must have shape [total_tokens]");
  TORCH_CHECK(req_id.scalar_type() == torch::kInt32, "req_id must be int32");
  return add_layer_norm_cmix_mix_f16_cuda_varlen(x, residual, shift_state,
                                                 weight, bias, x_k,
                                                 query_start_loc, req_id, eps);
}

torch::Tensor add_last_layer_norm_f16_varlen_fn(
    int64_t total_tokens, torch::Tensor x, torch::Tensor residual,
    torch::Tensor weight, torch::Tensor bias, torch::Tensor query_start_loc,
    double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  TORCH_CHECK(x.sizes() == residual.sizes(),
              "add_last_layer_norm_f16_varlen x/residual shape mismatch");
  TORCH_CHECK(query_start_loc.is_cuda(),
              "query_start_loc must be CUDA tensor (device-side read)");
  int64_t B = query_start_loc.size(0) - 1;
  const int64_t c = x.size(-1);
  TORCH_CHECK(x.dim() == 2 && x.size(0) == total_tokens && x.size(1) == c,
              "x must have shape [total_tokens, C]");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c,
              "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  TORCH_CHECK(c > 0 && c <= 8192, "unsupported C");
  TORCH_CHECK(query_start_loc.dim() == 1 && query_start_loc.size(0) == B + 1,
              "query_start_loc must have shape [B+1]");
  TORCH_CHECK(query_start_loc.scalar_type() == torch::kInt32,
              "query_start_loc must be int32");
  return add_last_layer_norm_f16_cuda_varlen(x, residual, weight, bias,
                                             query_start_loc, eps);
}

TORCH_LIBRARY_FRAGMENT(vkwr_v1_5_norm, m) {
  m.def(
      "add_layer_norm_tmix_mix6_f16_varlen(int total_tokens, Tensor x, Tensor "
      "residual, "
      "Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_r, Tensor "
      "x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, Tensor "
      "query_start_loc, Tensor req_id, float eps=1e-5) -> Tensor[]");
  m.impl("add_layer_norm_tmix_mix6_f16_varlen", c10::kCUDA,
         &add_layer_norm_tmix_mix6_f16_varlen_fn);

  m.def(
      "add_layer_norm_cmix_mix_f16_varlen(int total_tokens, Tensor x, Tensor "
      "residual, "
      "Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_k, Tensor "
      "query_start_loc, Tensor req_id, float eps=1e-5) -> Tensor[]");
  m.impl("add_layer_norm_cmix_mix_f16_varlen", c10::kCUDA,
         &add_layer_norm_cmix_mix_f16_varlen_fn);

  m.def(
      "add_last_layer_norm_f16_varlen(int total_tokens, Tensor x, Tensor "
      "residual, Tensor "
      "weight, Tensor bias, Tensor query_start_loc, float eps=1e-5) -> Tensor");
  m.impl("add_last_layer_norm_f16_varlen", c10::kCUDA,
         &add_last_layer_norm_f16_varlen_fn);
}

REGISTER_EXTENSION(_v1_5_norm_C)
