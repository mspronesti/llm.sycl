/*
(Approximate) GeLU non-linearity layer
*/
#ifndef LLM_SYCL_GELU_HPP
#define LLM_SYCL_GELU_HPP


#include <cassert>
#include "sycl_common.hpp"
#include "sycl_utils.hpp"

// ----------------------------------------------------------------------------
// SYCL kernels

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward_kernel2(sycl::nd_item<1> id, floatX* out, const floatX* inp) {
    int idx = (id.get_global_id(0)) * x128::size;

    x128 packed_out;
    x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
    for(int k = 0; k < packed_inp.size; ++k) {
        float xi = (float)packed_inp[k];
        float cube = 0.044715f * xi * xi * xi;
        packed_out[k] = (floatX)(0.5f * xi * (1.0f + sycl::tanh(GELU_SCALING_FACTOR * (xi + cube))));
    }
    // store instead of storecs (without cache streaming) in case it is useful for the
    // data to be in the cache for the next operation after this GeLU
    store128(out + idx, packed_out);
}

void gelu_backward_inplace_kernel(sycl::nd_item<1> id, floatX* d_in_out, const floatX* inp) {
    int idx = (id.get_global_id(0)) * x128::size;

    x128 packed_dinp;
    x128 packed_inp = load128cs(inp + idx);
    x128 packed_dout = load128(d_in_out + idx);
    for (int k = 0; k < packed_inp.size; ++k) {
        float x = (float)packed_inp[k];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = sycl::tanh(tanh_arg);
        float coshf_out = sycl::cosh(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        packed_dinp[k] = (floatX)(local_grad * (float)packed_dout[k]);
    }
    store128(d_in_out + idx, packed_dinp);
}

// ----------------------------------------------------------------------------
// kernel launchers

void gelu_forward(sycl::queue *stream, floatX* out, const floatX* inp, int N) {
    const int block_size = 512;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        gelu_forward_kernel2(id, out, inp);
    }).wait();
}

void gelu_backward_inplace(sycl::queue *stream, floatX* d_in_out, const floatX* inp, const int N) {
    const int block_size = 128;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        gelu_backward_inplace_kernel(id, d_in_out, inp);
    }).wait();
}

#endif //LLM_SYCL_GELU_HPP
