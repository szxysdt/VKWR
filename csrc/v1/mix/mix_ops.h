#pragma once

#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> tmix_mix6_cuda(
    int B, int T, int C,
    at::Tensor x, at::Tensor shift_state,
    at::Tensor x_r, at::Tensor x_w, at::Tensor x_k, at::Tensor x_v, at::Tensor x_a, at::Tensor x_g);
std::vector<at::Tensor> tmix_mix6_cfg_cuda(
    int B, int T, int C,
    at::Tensor x, at::Tensor shift_state,
    at::Tensor x_r, at::Tensor x_w, at::Tensor x_k, at::Tensor x_v, at::Tensor x_a, at::Tensor x_g,
    int threads);
std::vector<at::Tensor> tmix_mix6_t1_c4096_cuda(
    int B,
    at::Tensor x, at::Tensor shift_state,
    at::Tensor x_r, at::Tensor x_w, at::Tensor x_k, at::Tensor x_v, at::Tensor x_a, at::Tensor x_g,
    int threads, int vec, bool half_math);

std::vector<at::Tensor> tmix_kk_a_gate_cuda(
    int B, int T, int C, int H,
    at::Tensor k, at::Tensor k_k, at::Tensor a0, at::Tensor a12, at::Tensor k_a,
    at::Tensor x, at::Tensor shift_state, bool update_shift);

at::Tensor tmix_lnx_rkvres_xg_cuda(
    int B, int T, int C, int H,
    at::Tensor x, at::Tensor r, at::Tensor k, at::Tensor v,
    at::Tensor r_k, at::Tensor weight, at::Tensor bias, at::Tensor g);

at::Tensor tmix_vres_gate_cuda(
    int B, int T, int C,
    at::Tensor v, at::Tensor v_first, at::Tensor v0, at::Tensor v12);

at::Tensor cmix_sparse_one_cuda(
    int C, int F,
    at::Tensor x, at::Tensor shift_state, at::Tensor x_k, at::Tensor key_fc, at::Tensor value_fc);

at::Tensor cmix_sparse_rows_cuda(
    int B, int T, int C, int F,
    at::Tensor x, at::Tensor shift_state, at::Tensor x_k, at::Tensor key_fc, at::Tensor value_fc);

at::Tensor cmix_sparse_down_one_cuda(
    int C, int F,
    at::Tensor act, at::Tensor value_fc);

at::Tensor cmix_sparse_down_rows_cuda(
    int B, int T, int C, int F,
    at::Tensor act, at::Tensor value_fc);

at::Tensor cmix_sparse_down_relu_one_cuda(
    int C, int F,
    at::Tensor preact, at::Tensor value_fc);

at::Tensor cmix_sparse_down_relu_rows_cuda(
    int B, int T, int C, int F,
    at::Tensor preact, at::Tensor value_fc);

at::Tensor cmix_sparse_down_relu_rows_t512_cuda(
    int B, int T, int C, int F,
    at::Tensor preact, at::Tensor value_fc);

at::Tensor cmix_mix_cuda(
    int B, int T, int C,
    at::Tensor x, at::Tensor shift_state, at::Tensor x_k);

at::Tensor cmix_mix_cfg_cuda(
    int B, int T, int C,
    at::Tensor x, at::Tensor shift_state, at::Tensor x_k, int threads);

at::Tensor relu_square_cuda(at::Tensor x);

at::Tensor act_tanh_cuda(at::Tensor x);

at::Tensor act_sigmoid_cuda(at::Tensor x);

at::Tensor add_vec_cuda(int C, at::Tensor x, at::Tensor vec);