#include <sycl/sycl.hpp>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstdlib>

#define ENABLE_BF16
#include "common.hpp"

// ----------------------------------------------------------------------------
// CPU code reference

void residual_forward_cpu(float* out, const float* inp1, const float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

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
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
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

void residual_forward_kernel1(sycl::nd_item<1> item, floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int idx = item.get_global_id(0);
    if (idx < N) {
        out[idx] = (floatX)((float)inp1[idx] + (float)inp2[idx]);
    }
}

void layernorm_forward_kernel1(sycl::nd_item<1> item, floatX* out, floatX* mean, floatX* rstd,
                               const floatX* inp, const floatX* weight, const floatX* bias,
                               int N, int C) {
    int idx = item.get_global_id(0);
    float eps = 1e-5f;

    if (idx < N) {
        // seek to the input position inp[idx,:]
        const floatX* x = inp + idx * C;
        // calculate the mean
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += (float)x[i];
        }
        m = m / C;
        // calculate the variance (without any bias correction)
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = (float)x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;
        // calculate the rstd
        float s = 1.0f / sqrtf(v + eps);
        // seek to the output position in out[idx,:]
        floatX* out_idx = out + idx * C;
        for (int i = 0; i < C; i++) {
            float n = (s * ((float)x[i] - m)); // normalized output
            float o = n * (float)weight[i] + (float)bias[i]; // scale and shift it
            out_idx[i] = o; // write
        }
        // cache the mean and rstd for the backward pass later
        mean[idx] = m;
        rstd[idx] = s;
    }
}

void fused_residual_forward_kernel2(sycl::nd_item<1> item, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                    const floatX* inp1, const floatX* inp2,
                                    const floatX* weight, const floatX* bias,
                                    int N, int C) {

    int idx = item.get_global_id(0);
    if(idx > N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    float eps = 1e-5f;

    float m = 0.0f;
    for(int c = 0; c < C; ++c) {
        float out = (float)inp1[c] + (float)inp2[c];
        m += out;
        residual[c] = out;
    }

    m = m / C;
    float v = 0.0f;
    for (int c = 0; c < C; c++) {
        float xshift = (float)residual[c] - m;
        v += xshift * xshift;
    }
    v = v / C;

    // calculate the rstd
    float s = 1.0f / sqrtf(v + eps);
    for (int c = 0; c < C; c++) {
        float n = (s * ((float)residual[c] - m)); // normalized output
        float o = n * (float)weight[c] + (float)bias[c]; // scale and shift it
        normed[c] = o; // write
    }
    // cache the mean and rstd for the backward pass later
    mean[idx] = m;
    rstd[idx] = s;
}

// This is not working
void fused_residual_forward_kernel3(sycl::nd_item<2> item, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C) {
    constexpr const int WarpSize = 32;
    int idx = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (idx >= N) return;

    auto sg = item.get_sub_group();

    int thread_id = item.get_local_id(1);

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    float eps = 1e-5f;
    float m = 0.0f;
    for (int c = thread_id; c < C; c += WarpSize) {
        float out = static_cast<float>(inp1[c]) + static_cast<float>(inp2[c]);
        m += out;
        residual[c] = out;
    }

    m = sycl::reduce_over_group(sg, m, sycl::plus<float>());

    m = m / C;
    float v = 0.0f;
    for (int c = thread_id; c < C; c += WarpSize) {
        float xshift = static_cast<float>(residual[c]) - m;
        v += xshift * xshift;
    }

    v = sycl::reduce_over_group(sg, v, sycl::plus<float>());
    v = v / C;

    // calculate the rstd
    float s = 1.0f / sycl::sqrt(v + eps);
    for (int c = thread_id; c < C; c += WarpSize) {
        float n = s * (static_cast<float>(residual[c]) - m); // normalized output
        float o = n * static_cast<float>(weight[c]) + static_cast<float>(bias[c]); // scale and shift it
        normed[c] = o; // write
    }
    // cache the mean and rstd for the backward pass later
    if (thread_id == 0) {
        mean[idx] = m;
        rstd[idx] = s;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher
void fused_residual_forward1(sycl::queue &q, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    const int grid_size_resid = ceil_div(N * C, block_size);
    q.parallel_for(sycl::nd_range<1>(grid_size_resid * block_size, block_size), [=](sycl::nd_item<1> id){
        residual_forward_kernel1(id, residual, inp1, inp2, N * C);
    }).wait();

    const int grid_size_ln = ceil_div(N, block_size);
    q.parallel_for(sycl::nd_range<1>(grid_size_ln * block_size, block_size), [=](sycl::nd_item<1> id) {
        layernorm_forward_kernel1(id, normed, mean, rstd, residual, weight, bias, N, C);
    }).wait();
}

void fused_residual_forward2(sycl::queue &q, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        fused_residual_forward_kernel2(id, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
    }).wait();
}

void fused_residual_forward3(sycl::queue &q, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                      const floatX* inp1, const floatX* inp2,
                                      const floatX* weight, const floatX* bias,
                                      int N, int C, const int block_size) {
    int block_y = block_size / 32;
    int grid_size = ceil_div(N, block_y);
    sycl::nd_range<2> grid = sycl::nd_range<2>(
            sycl::range<2>(grid_size * block_y, 32),
            sycl::range<2>(block_y, 32)
    );

    q.parallel_for(grid, [=](sycl::nd_item<2> item) {
            fused_residual_forward_kernel3(item, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
    }).wait();
}

// kernel version dispatch
void fused_residual_forward(int kernel_num, sycl::queue &q, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                            const floatX* inp1, const floatX* inp2,
                            const floatX* weight, const floatX* bias,
                            int N, int C, const int block_size) {
    switch (kernel_num) {
        case 1:
            fused_residual_forward1(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 2:
            fused_residual_forward2(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 3:
            fused_residual_forward3(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        default:
            std::cout << "Invalid kernel number\n";
            std::exit(1);
    }
}

int main(int argc, const char **argv) {
    int B = 8;
    int T = 1024;
    int C = 768;

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = std::atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << std::endl;

    // create host memory of random numbers
    float* residual = (float*)malloc(B * T * C * sizeof(float));
    float* normed = (float*)malloc(B * T * C * sizeof(float));
    float* inp1 = make_random_float(B * T * C);
    float* inp2 = make_random_float(B * T * C);
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);

    // select device and create queue
    sycl::queue q(sycl::default_selector_v);

    // allocate device memory
    floatX* d_residual = sycl::malloc_device<floatX>(B * T * C, q);
    floatX* d_normed = sycl::malloc_device<floatX>(B * T * C, q);
    floatX* d_inp1 = sycl::malloc_device<floatX>(B * T * C, q);
    floatX* d_inp2 = sycl::malloc_device<floatX>(B * T * C, q);
    floatX* d_mean = sycl::malloc_device<floatX>(B * T, q);
    floatX* d_rstd = sycl::malloc_device<floatX>(B * T, q);
    floatX* d_weight = sycl::malloc_device<floatX>(C, q);
    floatX* d_bias = sycl::malloc_device<floatX>(C, q);

    // copy data to device
    memcpy_convert(d_inp1, inp1, B * T * C, q);
    memcpy_convert(d_inp2, inp2, B * T * C, q);
    memcpy_convert(d_weight, weight, C, q);
    memcpy_convert(d_bias, bias, C, q);

    // first check the correctness of the kernel
    residual_forward_cpu(residual, inp1, inp2, B * T * C);
    layernorm_forward_cpu(normed, mean, rstd, residual, weight, bias, B, T, C);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};

    for (int block_size: block_sizes) {
        std::cout << "Checking block size " << block_size << "." << std::endl;

        q.memset(d_residual, 0, B * T * C * sizeof(floatX)).wait();
        fused_residual_forward(kernel_num, q, d_residual, d_normed, d_mean, d_rstd, d_inp1, d_inp2, d_weight, d_bias,
                               B * T, C, block_size);

        float tol = std::is_same_v<floatX, float> ? 1e-5 : 5e-2;
        validate_result(d_residual, residual, "residual", B * T * C, tol);
        validate_result(d_mean, mean, "mean", B * T, tol);
        validate_result(d_rstd, rstd, "rstd", B * T, tol);
        validate_result(d_normed, normed, "normed", B * T * C, tol);
    }

    std::cout << "All results match. Starting benchmarks.\n\n";

    for (int block_size: block_sizes) {
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(
                repeat_times,
                fused_residual_forward,
                kernel_num, q, d_residual, d_normed, d_mean, d_rstd, d_inp1, d_inp2,
                d_weight, d_bias, B * T, C, block_size
        );

        // napkin math: estimate the memory bandwidth achieved
        long memory_ops = B * T * (C * 4 + 2) * sizeof(floatX);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;
        float toks_per_msec = B * T / elapsed_time / 1e3;

        std::cout << "block_size " << block_size
                  << " | time " << elapsed_time << " ms"
                  << " | bandwidth " << memory_bandwidth << " GB/s"
                  << " | elements: " << toks_per_msec << " ktok/ms\n";
    }

    // free memory
    free(residual);
    free(normed);
    free(mean);
    free(rstd);
    free(weight);
    free(bias);
    free(inp1);
    free(inp2);
    sycl::free(d_residual, q);
    sycl::free(d_normed, q);
    sycl::free(d_mean, q);
    sycl::free(d_rstd, q);
    sycl::free(d_weight, q);
    sycl::free(d_bias, q);
    sycl::free(d_inp1, q);
    sycl::free(d_inp2, q);

    return 0;
}
