#include <algorithm>
#include <stdio.h>
#include <torch/extension.h>
#include "rwkv_utils.h"

template <typename F>
__global__ void kernel_wkv_forward(const int B, const int T, const int C,
                           const float *__restrict__ const _w, const float *__restrict__ const _u, const F *__restrict__ const _k, const F *__restrict__ const _v,
                           F *__restrict__ const _y, float *__restrict__ const _aa, float *__restrict__ const _bb, float *__restrict__ const _pp) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;
    const int _state_offset = _b * C + _c;

    float u = _u[_c];
    float w = _w[_c];
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    F *__restrict__ const y = _y + _offset;

    float aa = _aa[_state_offset];
    float bb = _bb[_state_offset];
    float pp = _pp[_state_offset];
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        const float kk = float(k[ii]);
        const float vv = float(v[ii]);
        float ww = u + kk;
        float p = max(pp, ww);
        float e1 = exp(pp - p);
        float e2 = exp(ww - p);
        y[ii] = F((e1 * aa + e2 * vv) / (e1 * bb + e2));
        ww = w + pp;
        p = max(ww, kk);
        e1 = exp(ww - p);
        e2 = exp(kk - p);
        aa = e1 * aa + e2 * vv;
        bb = e1 * bb + e2;
        pp = p;
    }
    _aa[_state_offset] = aa;
    _bb[_state_offset] = bb;
    _pp[_state_offset] = pp;
}

template <typename F>
void cuda_wkv_forward(int B, int T, int C, float *w, float *u, F *k, F *v, F *y, float *aa, float *bb, float *pp) {
    dim3 threadsPerBlock(std::min(C, 32));
    TORCH_CHECK(B * C % threadsPerBlock.x == 0,
        "wkv_forward: B*C must be divisible by threadsPerBlock, got B*C=", B * C, " threadsPerBlock=", threadsPerBlock.x);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_wkv_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y, aa, bb, pp);
}

template void cuda_wkv_forward<fp16>(
    int B, int T, int C,
    float *w, float *u, fp16 *k, fp16 *v, fp16 *y,
    float *aa, float *bb, float *pp);
template void cuda_wkv_forward<float>(
    int B, int T, int C,
    float *w, float *u, float *k, float *v, float *y,
    float *aa, float *bb, float *pp);
