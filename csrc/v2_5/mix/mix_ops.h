#pragma once

#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> tmix_mix6_cuda_varlen(
    int B, int C, at::Tensor x, at::Tensor shift_state, at::Tensor x_r,
    at::Tensor x_w, at::Tensor x_k, at::Tensor x_v, at::Tensor x_a,
    at::Tensor x_g, at::Tensor query_start_loc, at::Tensor req_id,
    at::Tensor slot_indices);

std::vector<at::Tensor> tmix_mix6_t1_c4096_cuda_varlen(
    int B, at::Tensor x, at::Tensor shift_state, at::Tensor x_r, at::Tensor x_w,
    at::Tensor x_k, at::Tensor x_v, at::Tensor x_a, at::Tensor x_g,
    at::Tensor query_start_loc, at::Tensor req_id, at::Tensor slot_indices,
    int threads, int vec, bool half_math);

at::Tensor cmix_mix_cuda_varlen(int B, int C, at::Tensor x,
                                at::Tensor shift_state, at::Tensor x_k,
                                at::Tensor query_start_loc, at::Tensor req_id,
                                at::Tensor slot_indices);

at::Tensor cmix_sparse_rows_cuda_varlen(int B, int C, int F, at::Tensor x,
                                        at::Tensor shift_state, at::Tensor x_k,
                                        at::Tensor key_fc, at::Tensor value_fc,
                                        at::Tensor query_start_loc,
                                        at::Tensor req_id,
                                        at::Tensor slot_indices);

at::Tensor cmix_sparse_down_relu_rows_cuda_varlen(int rows, int C, int F,
                                                   at::Tensor preact,
                                                   at::Tensor value_fc);

at::Tensor cmix_sparse_down_relu_rows_t512_cuda_varlen(int rows, int C, int F,
                                                        at::Tensor preact,
                                                        at::Tensor value_fc);
