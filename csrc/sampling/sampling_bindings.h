#pragma once

#include <torch/extension.h>

at::Tensor _setup_rand(int64_t seed, int64_t B);
at::Tensor _temperature_topk_topp(at::Tensor &logits, at::Tensor &states,
                                   at::Scalar temperature, int64_t top_k, at::Scalar top_p);
at::Tensor _repetition_topk_topp(at::Tensor &logits, at::Tensor &penalties, at::Tensor &states,
                                  at::Scalar presence_penalty, at::Scalar repetition_penalty,
                                  at::Scalar penalty_decay, at::Scalar temperature,
                                  int64_t top_k, at::Scalar top_p);
