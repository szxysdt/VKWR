#include <torch/extension.h>
#include <torch/library.h>
#include "sampling_bindings.h"
#include "core/register.h"

at::Tensor setup_rand_impl(int64_t seed, int64_t B);
at::Tensor sampling_temperature_topk_topp_impl(at::Tensor& logits, at::Tensor& states,
                                                double temperature, int64_t top_k, double top_p);
at::Tensor sampling_repetition_topk_topp_impl(at::Tensor& logits, at::Tensor& penalties,
                                               at::Tensor& states, double presence_penalty,
                                               double repetition_penalty, double penalty_decay,
                                               double temperature, int64_t top_k, double top_p);

at::Tensor _setup_rand(int64_t seed, int64_t B) {
    return setup_rand_impl(seed, B);
}

at::Tensor _temperature_topk_topp(at::Tensor& logits, at::Tensor& states,
                                   at::Scalar temperature, int64_t top_k, at::Scalar top_p) {
    return sampling_temperature_topk_topp_impl(logits, states,
                                               temperature.toDouble(), top_k, top_p.toDouble());
}

at::Tensor _repetition_topk_topp(at::Tensor& logits, at::Tensor& penalties, at::Tensor& states,
                                  at::Scalar presence_penalty, at::Scalar repetition_penalty,
                                  at::Scalar penalty_decay, at::Scalar temperature,
                                  int64_t top_k, at::Scalar top_p) {
    return sampling_repetition_topk_topp_impl(logits, penalties, states,
                                               presence_penalty.toDouble(),
                                               repetition_penalty.toDouble(),
                                               penalty_decay.toDouble(),
                                               temperature.toDouble(), top_k,
                                               top_p.toDouble());
}

TORCH_LIBRARY(vkwr_sampling, m) {
    m.def("setup_rand(int seed, int B) -> Tensor");
    m.impl("setup_rand", &_setup_rand);

    m.def("temperature_topk_topp(Tensor logits, Tensor states, "
          "Scalar temperature, int top_k, Scalar top_p) -> Tensor");
    m.impl("temperature_topk_topp", c10::kCUDA, &_temperature_topk_topp);

    m.def("repetition_topk_topp(Tensor logits, Tensor penalties, Tensor states, "
          "Scalar presence_penalty, Scalar repetition_penalty, Scalar penalty_decay, "
          "Scalar temperature, int top_k, Scalar top_p) -> Tensor");
    m.impl("repetition_topk_topp", c10::kCUDA, &_repetition_topk_topp);
}

REGISTER_EXTENSION(_sampling_C)
