/*
Kernels for layernorm forward pass.

Compile example:
icpx -O3 -fsycl layernorm_forward.cpp -o layernorm_forward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./layernorm_forward 1

version 2 parallelizes over all of B,T,C
./layernorm_forward 2

version 3 uses cooperative groups to parallelize over all of B,T,C
./layernorm_forward 3

version 4 uses a more clever way to estimate variance, var(x) = mean(x**2) - mean(x)**2
          (allowing us to do a single pass over x on load)
./layernorm_forward 4
*/

#include <sycl/sycl.hpp>
#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <cassert>
#include <chrono>

#include "common.hpp"


// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 layernorm forward pass
void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m / C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v / C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// naive drag and drop implementation into kernel, parallelize over B,T, loop over C
void layernorm_forward_kernel1(sycl::nd_item<1> id, float* out, float* mean, float* rstd,
                               const float* inp, const float* weight, const float* bias,
                               int N, int C) {
    int idx = id.get_global_id(0);
    float eps = 1e-5f;

    if (idx < N) {
        // seek to the input position inp[idx,:]
        const float* x = inp + idx * C;
        // calculate the mean
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += x[i];
        }
        m = m / C;
        // calculate the variance (without any bias correction)
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;
        // calculate the rstd
        float s = 1.0f / sqrtf(v + eps);
        // seek to the output position in out[idx,:]
        float* out_idx = out + idx * C;
        for (int i = 0; i < C; i++) {
            float n = (s * (x[i] - m)); // normalized output
            float o = n * weight[i] + bias[i]; // scale and shift it
            out_idx[i] = o; // write
        }
        // cache the mean and rstd for the backward pass later
        mean[idx] = m;
        rstd[idx] = s;
    }
}

void mean_kernel(sycl::nd_item<1> id, float* mean, const float* inp, int N, int C, int block_size) {
    int idx = id.get_group(0); // range [0, B*T)
    int tid = id.get_local_id(0); // range [0, block_size)
    const float* x = inp + idx * C;
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sum += x[i];
    }
    sum = sycl::reduce_over_group(id.get_group(), sum, sycl::plus<float>());

    // write the final result (at thread 0) to global memory
    if (id.get_group().leader()) {
        mean[idx] = sum / C;
    }
}

void rstd_kernel(sycl::nd_item<1> id, float* rstd, const float* inp, const float* mean, int N, int C, int block_size) {
    int idx = id.get_group(0); // range [0, B*T)
    int tid = id.get_local_linear_id(); // range [0, block_size)
    const float* x = inp + idx * C;
    float m = mean[idx];
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = sycl::reduce_over_group(id.get_group(), sum, sycl::plus<float>());
    // write the final result (at thread 0) to global memory
    if (id.get_group().leader()) {
        rstd[idx] = 1.0f / sqrtf(sum / C + 1e-5f);
    }
}

void normalization_kernel(sycl::nd_item<1> id, float* out, const float* inp, float* mean, float* rstd,
                          const float* weight, const float* bias, int B, int T, int C) {
    int idx = id.get_global_id(0);

    int bt = idx / C;
    int c = idx % C;

    float m = mean[bt];
    float s = rstd[bt];
    float xi = inp[idx];
    float n = s * (xi - m);
    float o = n * weight[c] + bias[c];

    out[idx] = o;
}

// parallelize over all of B, T, C
void layernorm_forward_kernel3(sycl::nd_item<1> id, float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                               const float*  __restrict__ inp, const float*  __restrict__ weight,
                               const float* __restrict__ bias, int N, int C) {
    sycl::sub_group warp = id.get_sub_group();
    int idx = id.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();
    if(idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;

    // mean
    float sum = 0.0f;
    for (int i = warp.get_local_linear_id(); i < C; i += warp.get_max_local_range()[0]) {
        sum += x[i];
    }
    sum = sycl::reduce_over_group(warp, sum, sycl::plus<float>{});
    float m = sum / C;
    if(warp.leader() && mean != nullptr) {
        mean[idx] = m;
    }

    // rstd
    sum = 0.0f;
    for (int i = warp.get_local_linear_id(); i < C; i += warp.get_max_local_range()[0]) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = sycl::reduce_over_group(warp, sum, sycl::plus<float>{});
    float s = sycl::rsqrt(sum / C + 1e-5f);
    if(warp.leader() && rstd != nullptr) {
        rstd[idx] = s;
    }

    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = warp.get_local_linear_id(); c < C; c += warp.get_max_local_range()[0]) {
        float n = s * (x[c] - m);
        o[c] = n * weight[c] + bias[c];
    }
}

// same as kernel 3 but uses var(x) == mean(x**2) - mean(x)**2
void layernorm_forward_kernel4(sycl::nd_item<1> id, float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                               const float*  __restrict__ inp, const float*  __restrict__ weight,
                               const float* __restrict__ bias, int N, int C) {
    sycl::sub_group warp = id.get_sub_group();
    int idx = id.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();
    if(idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;

    // thread coarsening through the row, reduce the sum in series
    float sum = 0.0f;  // stores sum(x)
    float sum2 = 0.0f; // stores sum(x**2)
    for (int i = warp.get_local_linear_id(); i < C; i += warp.get_max_local_range()[0]) {
        float xi = x[i];
        sum += xi;
        sum2 += xi * xi;
    }
    // warp-level reduction at the end
    sum = sycl::reduce_over_group(warp, sum, sycl::plus<float>{});   // sum(x)
    sum2 = sycl::reduce_over_group(warp, sum2, sycl::plus<float>{}); // sum(x**2)
    sum /= C;   // mean(x)
    sum2 /= C;  // mean(x**2)

    // mean, var, rstd
    float m = sum;
    float var = sum2 - sum * sum;
    float s = sycl::rsqrt(var + 1e-5f);

    // store the mean, no need to cache it
    if(warp.leader() && mean != nullptr) {
        mean[idx] = m;
    }
    // store the rstd, no need to cache it
    if(warp.leader() && rstd != nullptr) {
        rstd[idx] = s;
    }

    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = warp.get_local_linear_id(); c < C; c += warp.get_max_local_range()[0]) {
        float n = s * (x[c] - m);
        o[c] = n * weight[c] + bias[c];
    }
}

// like 4, but in kernel 5 we have each block doing one row, not just a single warp
void layernorm_forward_kernel5(sycl::nd_item<1> id, float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                               const float*  __restrict__ inp, const float*  __restrict__ weight,
                               const float* __restrict__ bias, int N, int C) {
    int thread_idx_x = id.get_local_id(0);

    sycl::group block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();

    int idx = id.get_group(0); // simply one block per row
    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;
    // thread coarsening through the row, reduce the sum in series
    float thread_sum = 0.0; // stores sum(x)
    float thread_sum2 = 0.0; // stores sum(x**2)
    for (int i = thread_idx_x; i < C; i += id.get_local_range(0)) {
        float xi = x[i];
        thread_sum += xi;
        thread_sum2 += xi * xi;
    }
    // block-level reduction
    float block_sum = sycl::reduce_over_group(block, thread_sum, sycl::plus<float>{}); // sum(x)
    float block_sum2 = sycl::reduce_over_group(block, thread_sum2, sycl::plus<float>{}); // sum(x**2)
    // mean, var, rstd
    block_sum /= C; // mean(x)
    block_sum2 /= C; // mean(x**2)
    float m = block_sum;
    float var = block_sum2 - m * m;
    float s = sycl::rsqrt(var + 1e-5f);
    // store the mean, no need to cache it
    if(thread_idx_x == 0 && mean != nullptr) {
        mean[idx] = m;
    }
    // store the rstd, no need to cache it
    if(thread_idx_x == 0 && rstd != nullptr) {
        rstd[idx] = s;
    }
    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int i = thread_idx_x; i < C; i += id.get_local_range(0)) {
        float n = s * (x[i] - m);
        o[i] = n * weight[i] + bias[i];
    }
}

// Inspired by `fused_residual_forward_kernel5` in fused_residual_forward.cpp
void layernorm_forward_kernel6(sycl::nd_item<2> item, float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                          const float*  __restrict__ inp, const float*  __restrict__ weight,
                                          const float* __restrict__ bias, int N, int C,
                                          sycl::local_accessor<char> local_acc) {
    constexpr const int WarpSize = 32;
    assert(item.get_local_range(1) == WarpSize);

    auto sg = item.get_sub_group();
    int threadIdx_x = item.get_local_id(1);
    int threadIdx_y = item.get_local_id(0);

    // load weights and biases into shared memory
    // do this before we allow any threads to exit!
    auto params = (char *)local_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();

    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params) + (C / x128::size);
    x128* s_in = reinterpret_cast<x128*>(params) + ((2 + threadIdx_y) * C / x128::size);

    int sidx = (threadIdx_x + WarpSize * threadIdx_y) * x128::size;
    for(int i = sidx; i < C; i += item.get_local_range(0) * WarpSize * x128::size) {
        s_weight[i/x128::size] = load128(weight + i);
        s_bias[i/x128::size] = load128(bias + i);
    }
    sycl::group_barrier(item.get_group());

    int idx = item.get_group(0) * item.get_local_range(0) + threadIdx_y;
    if(idx >= N) { return; } // guard

    // adjust pointers to current token
    inp += idx * C;
    out += idx * C;

    const float eps = 1e-5f;
    float sum = 0.0f;
    for(int c = threadIdx_x * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 in_data = load128cs(inp + c);
        for(int k = 0; k < x128::size; ++k) {
            sum += (float)in_data[k];
        }
        s_in[c / x128::size] = in_data;
    }

    sum =  sycl::reduce_over_group(sg, sum, sycl::plus<float>());
    float m = sum / C;
    float v = 0.f;

    for(int c = threadIdx_x * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        for(int k = 0; k < x128::size; ++k) {
            v += ((float)in_data[k] - m) * ((float)in_data[k] - m);
        }
    }

    v = sycl::reduce_over_group(sg, v, sycl::plus<float>()) / C;
    float s = sycl::rsqrt(v + eps);

    for(int c = threadIdx_x * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        const x128 w = s_weight[c / x128::size];
        const x128 b = s_bias[c / x128::size];
        x128 out_data;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)in_data[k] - m); // normalized output
            float o = n * (float)w[k] + (float)b[k]; // scale and shift it
            out_data[k] = o;
        }

        store128cs(out + c, out_data);
    }
    // cache the mean and rstd for the backward pass later
    if(threadIdx_x == 0 && mean != nullptr) {
        mean[idx] = m;
    }
    // store the rstd, no need to cache it
    if(threadIdx_x == 0 && rstd != nullptr) {
        rstd[idx] = s;
    }
}



// ----------------------------------------------------------------------------
// kernel launcher

void layernorm_forward1(sycl::queue &q, float* out, float* mean, float* rstd,
                        const float* inp, const float* weight, const float* bias,
                        int B, int T, int C,
                        const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        layernorm_forward_kernel1(id, out, mean, rstd, inp, weight, bias, N, C);
    }).wait();
}

void layernorm_forward2(sycl::queue &q, float* out, float* mean, float* rstd,
                        const float* inp, const float* weight, const float* bias,
                        int B, int T, int C,
                        const int block_size) {
    int N = B * T;
    // in mean and rstd, threads cooperate within blocks via reductions
    // Do these still need the block_size param in the kernels?
    q.parallel_for(sycl::nd_range<1>(N * block_size, block_size), [=](sycl::nd_item<1> id) {
        mean_kernel(id, mean, inp, N, C, block_size);
    }).wait();
    q.parallel_for(sycl::nd_range<1>(N * block_size, block_size), [=](sycl::nd_item<1> id) {
        rstd_kernel(id, rstd, inp, mean, N, C, block_size);
    }).wait();
    // in the normalization, everything just gets flattened out
    const int block_size2 = 256;
    const int grid_size = ceil_div(B * T * C, block_size2);
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size2, block_size2), [=](sycl::nd_item<1> id) {
        normalization_kernel(id, out, inp, mean, rstd, weight, bias, B, T, C);
    }).wait();
}

void layernorm_forward3(sycl::queue &q, float* out, float* mean, float* rstd,
                        const float* inp, const float* weight, const float* bias,
                        int B, int T, int C,
                        const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        layernorm_forward_kernel3(id, out, mean, rstd, inp, weight, bias, N, C);
    }).wait();
}

void layernorm_forward4(sycl::queue &q, float* out, float* mean, float* rstd,
                        const float* inp, const float* weight, const float* bias,
                        int B, int T, int C,
                        const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        layernorm_forward_kernel4(id, out, mean, rstd, inp, weight, bias, N, C);
    }).wait();
}

void layernorm_forward5(sycl::queue &q, float* out, float* mean, float* rstd,
                        const float* inp, const float* weight, const float* bias,
                        int B, int T, int C,
                        const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = N;
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        layernorm_forward_kernel5(id, out, mean, rstd, inp, weight, bias, N, C);
    }).wait();
}

void layernorm_forward6(sycl::queue &q, float* out, float* mean, float* rstd,
                        const float* inp, const float* weight, const float* bias,
                        int B, int T, int C,
                        const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    int block_y = block_size / 32;
    const int grid_size = ceil_div(N, block_y);
    size_t smem = (2 + block_y) * C * sizeof(float);

    auto local_mem = q.get_device().get_info<sycl::info::device::local_mem_size>();
    if (local_mem > smem) {
        sycl::nd_range<2> grid = sycl::nd_range<2>(
                sycl::range<2>(grid_size * block_y, 32),
                sycl::range<2>(block_y, 32)
        );

        q.submit([&](sycl::handler &h) {
            sycl::local_accessor<char> local_acc(smem, h);
            h.parallel_for(grid, [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(32)]] {
                layernorm_forward_kernel6(item, out, mean, rstd, inp, weight, bias, N, C, local_acc);
            });
        });
    } else {
        const int grid_size = N;
        std::cout << "Not enough unified shared memory, falling back to kernel 5\n";
        q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
            layernorm_forward_kernel5(item, out, mean, rstd, inp, weight, bias, N, C);
        });
    }
    q.wait();
}



// kernel version dispatch
void layernorm_forward(int kernel_num,
                       sycl::queue &q,
                       float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    switch (kernel_num) {
        case 1:
            layernorm_forward1(q, out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 2:
            layernorm_forward2(q, out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 3:
            layernorm_forward3(q, out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 4:
            layernorm_forward4(q, out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 5:
            layernorm_forward5(q, out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 6:
            layernorm_forward6(q, out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        default:
            std::cout << "Invalid kernel number\n";
            std::exit(1);
    }
}

// ----------------------------------------------------------------------------
// Main

int main(int argc, char** argv) {
    srand(0);

    int B = 32; // batch size
    int T = 128; // sequence length
    int C = 768; // embedding size
    int N = B * T;

    sycl::queue q(sycl::default_selector_v);

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);

    // Device memory allocation
    float* d_out = sycl::malloc_device<float>(B * T * C, q);
    float* d_mean = sycl::malloc_device<float>(B * T, q);
    float* d_rstd = sycl::malloc_device<float>(B * T, q);
    float* d_inp = sycl::malloc_device<float>(B * T * C, q);
    float* d_weight = sycl::malloc_device<float>(C, q);
    float* d_bias = sycl::malloc_device<float>(C, q);

    // Copy data to device
    q.memcpy(d_inp, inp, B * T * C * sizeof(float)).wait();
    q.memcpy(d_weight, weight, C * sizeof(float)).wait();
    q.memcpy(d_bias, bias, C * sizeof(float)).wait();

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1)
        kernel_num = atoi(argv[1]);
    std::cout << "Using kernel version: " << kernel_num << std::endl;

    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

    int block_sizes[] = {32, 64, 128, 256, 512};
    for (int block_size : block_sizes) {
        std::cout << "Checking block size " <<  block_size << '\n';

        layernorm_forward(kernel_num, q, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);

        validate_result(d_out, out, "out", B * T * C, 1e-5f);
        validate_result(d_mean, mean, "mean", B * T, 1e-5f);
        validate_result(d_rstd, rstd, "rstd", B * T, 1e-5f);
    }

    std::cout << "All results match. Starting benchmarks.\n\n";

    // time the kernel at different block sizes
    for (int block_size : block_sizes) {
        int repeat_times = 2000;
        float elapsed_time = benchmark_kernel(
                repeat_times,
                layernorm_forward, // kernel
                // kernel params
                kernel_num, q, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias,
                B, T, C, block_size
        );

        // napkin math: estimate the memory bandwidth achieved
        long memory_ops = (2 * B * T * C) * 4; // *4 for float
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        std::cout << "block_size " << block_size << " | time " << elapsed_time << " ms | bandwidth " << memory_bandwidth << " GB/s" << std::endl;
    }

    // free memory
    free(out);
    free(mean);
    free(rstd);
    free(inp);
    free(weight);
    free(bias);

    sycl::free(d_out, q);
    sycl::free(d_mean, q);
    sycl::free(d_rstd, q);
    sycl::free(d_inp, q);
    sycl::free(d_weight, q);
    sycl::free(d_bias, q);

    return 0;
}
