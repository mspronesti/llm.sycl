#include <sycl/sycl.hpp>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstdlib>
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

void residual_forward_kernel1(sycl::queue &q, float* out, const float* inp1, const float* inp2, int N, int grid_size, int block_size) {
    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size * block_size), sycl::range<1>(block_size)), [=](sycl::nd_item<1> item) {
            int idx = item.get_global_id(0);
            if (idx < N) {
                out[idx] = static_cast<float>(static_cast<float>(inp1[idx]) + static_cast<float>(inp2[idx]));
            }
        });
    }).wait();
}

void layernorm_forward_kernel1(sycl::queue &q, float* out, float* mean, float* rstd,
                               const float* inp, const float* weight, const float* bias,
                               int N, int C, int grid_size, int block_size) {
    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size * block_size), sycl::range<1>(block_size)), [=](sycl::nd_item<1> item) {
            int idx = item.get_global_id(0);
            if (idx < N) {
                const float* x = inp + idx * C;
                float m = 0.0f;
                for (int i = 0; i < C; i++) {
                    m += static_cast<float>(x[i]);
                }
                m = m / C;
                float v = 0.0f;
                for (int i = 0; i < C; i++) {
                    float xshift = static_cast<float>(x[i]) - m;
                    v += xshift * xshift;
                }
                v = v / C;
                float s = 1.0f / sycl::sqrt(v + 1e-5f);
                float* out_idx = out + idx * C;
                for (int i = 0; i < C; i++) {
                    float n = (s * (static_cast<float>(x[i]) - m));
                    float o = n * static_cast<float>(weight[i]) + static_cast<float>(bias[i]);
                    out_idx[i] = o;
                }
                mean[idx] = m;
                rstd[idx] = s;
            }
        });
    }).wait();
}

void fused_residual_forward_kernel2(sycl::queue &q, float* residual, float* normed, float* mean, float* rstd,
                                    const float* inp1, const float* inp2,
                                    const float* weight, const float* bias,
                                    int N, int C, int grid_size, int block_size) {
    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size * block_size), sycl::range<1>(block_size)), [=](sycl::nd_item<1> item) {
            int idx = item.get_global_id(0);
            if (idx >= N) return;

            float* residual_ptr = residual + idx * C;
            float* normed_ptr = normed + idx * C;
            const float* inp1_ptr = inp1 + idx * C;
            const float* inp2_ptr = inp2 + idx * C;

            float m = 0.0f;
            for (int c = 0; c < C; ++c) {
                float out = static_cast<float>(inp1_ptr[c]) + static_cast<float>(inp2_ptr[c]);
                m += out;
                residual_ptr[c] = out;
            }

            m = m / C;
            float v = 0.0f;
            for (int c = 0; c < C; c++) {
                float xshift = static_cast<float>(residual_ptr[c]) - m;
                v += xshift * xshift;
            }
            v = v / C;
            float s = 1.0f / sycl::sqrt(v + 1e-5f);
            for (int c = 0; c < C; c++) {
                float n = (s * (static_cast<float>(residual_ptr[c]) - m));
                float o = n * static_cast<float>(weight[c]) + static_cast<float>(bias[c]);
                normed_ptr[c] = o;
            }
            mean[idx] = m;
            rstd[idx] = s;
        });
    }).wait();
}

// ----------------------------------------------------------------------------
// kernel launcher
void fused_residual_forward1(sycl::queue &q, float* residual, float* normed, float* mean, float* rstd,
                             const float* inp1, const float* inp2,
                             const float* weight, const float* bias,
                             int N, int C, const int block_size) {
    const int grid_size_resid = ceil_div(N * C, block_size);
    residual_forward_kernel1(q, residual, inp1, inp2, N * C, grid_size_resid, block_size);
    const int grid_size_ln = ceil_div(N, block_size);
    layernorm_forward_kernel1(q, normed, mean, rstd, residual, weight, bias, N, C, grid_size_ln, block_size);
}

void fused_residual_forward2(sycl::queue &q, float* residual, float* normed, float* mean, float* rstd,
                             const float* inp1, const float* inp2,
                             const float* weight, const float* bias,
                             int N, int C, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    fused_residual_forward_kernel2(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, grid_size, block_size);
}

// kernel version dispatch
void fused_residual_forward(int kernel_num, sycl::queue &q, float* residual, float* normed, float* mean, float* rstd,
                            const float* inp1, const float* inp2,
                            const float* weight, const float* bias,
                            int N, int C, const int block_size) {
    switch (kernel_num) {
        case 1:
            fused_residual_forward1(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 2:
            fused_residual_forward2(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        default:
            std::cout << "Invalid kernel number\n";
            exit(1);
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
    float* d_residual = sycl::malloc_device<float>(B * T * C, q);
    float* d_normed = sycl::malloc_device<float>(B * T * C, q);
    float* d_inp1 = sycl::malloc_device<float>(B * T * C, q);
    float* d_inp2 = sycl::malloc_device<float>(B * T * C, q);
    float* d_mean = sycl::malloc_device<float>(B * T, q);
    float* d_rstd = sycl::malloc_device<float>(B * T, q);
    float* d_weight = sycl::malloc_device<float>(C, q);
    float* d_bias = sycl::malloc_device<float>(C, q);

    // copy data to device
    q.memcpy(d_inp1, inp1, B * T * C * sizeof(float)).wait();
    q.memcpy(d_inp2, inp2, B * T * C * sizeof(float)).wait();
    q.memcpy(d_weight, weight, C * sizeof(float)).wait();
    q.memcpy(d_bias, bias, C * sizeof(float)).wait();

    // first check the correctness of the kernel
    residual_forward_cpu(residual, inp1, inp2, B * T * C);
    layernorm_forward_cpu(normed, mean, rstd, residual, weight, bias, B, T, C);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        std::cout << "Checking block size " << block_size << "." << std::endl;

        q.memset(d_residual, 0, B * T * C * sizeof(float)).wait();
        fused_residual_forward(kernel_num, q, d_residual, d_normed, d_mean, d_rstd, d_inp1, d_inp2, d_weight, d_bias,
                               B * T, C, block_size);

        float tol = 1e-5;
        validate_result(d_residual, residual, "residual", B * T * C, tol);
        validate_result(d_mean, mean, "mean", B * T, tol);
        validate_result(d_rstd, rstd, "rstd", B * T, tol);
        validate_result(d_normed, normed, "normed", B * T * C, tol);
    }

    std::cout << "All results match. Starting benchmarks.\n\n";

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, fused_residual_forward, kernel_num,
                                              q, d_residual, d_normed, d_mean, d_rstd, d_inp1, d_inp2, d_weight, d_bias,
                                              B * T, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        long memory_ops = B * T * (C * 4 + 2) * sizeof(float);
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
