#include <sycl/sycl.hpp>
#include <iostream>
#include <cstdlib>
#include "oneapi/mkl/blas.hpp"
#include "mkl.h"
#include <omp.h>
#include "common.hpp"

// ----------------------------------------------------------------------------
// CPU code reference

void matmul_backward_cpu(float* dinp, float* dweight, float* dbias,
                         float* dout, float* inp, float* weight,
                         int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * OC + t * OC;
            float* dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float* wrow = weight + o*C;
                float d = dout_bt[o];
                for (int i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
#pragma omp parallel for
    for (int o = 0; o < OC; o++) {
        double sum = 0.0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* dout_bt = dout + b * T * OC + t * OC;
                float* inp_bt = inp + b * T * C + t * C;
                float* dwrow = dweight + o*C;
                float d = dout_bt[o];
                if (dbias != nullptr) { sum += d; }
                for (int i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
        if (dbias != nullptr){dbias[o] = sum;}
    }
}

// ----------------------------------------------------------------------------
// GPU kernels
// naive kernel to backpropagate only the bias, it's just a sum :'(
void matmul_backward_bias_kernel_naive(sycl::queue &q, float* dbias, const float* dout, int B, int T, int OC) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(OC), [=](sycl::id<1> o) {
            if (o < OC) {
                double sum = 0.0;
                for (int b = 0; b < B; b++) {
                    for (int t = 0; t < T; t++) {
                        sum += dout[b * T * OC + t * OC + o];
                    }
                }
                dbias[o] = sum;
            }
        });
    });
}

// use shared memory and coarsening + reductions
void matmul_backward_bias_kernel_faster(sycl::queue &q, float* dbias, const float* dout, int B, int T, int OC, int block_size) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> shared(sycl::range<1>(block_size), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(OC * block_size), sycl::range<1>(block_size)), [=](sycl::nd_item<1> item) {
            int o = item.get_group(0);
            int tid = item.get_local_id(0);
            float sum = 0.0f; // Change from double to float
            for (int i = tid; i < B * T; i += block_size) {
                sum += dout[i * OC + o];
            }
            shared[tid] = sum;
            item.barrier(sycl::access::fence_space::local_space);
            for (int stride = block_size / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    shared[tid] += shared[tid + stride];
                }
                item.barrier(sycl::access::fence_space::local_space);
            }
            if (tid == 0) {
                dbias[o] = shared[0];
            }
        });
    }).wait();
}


// ----------------------------------------------------------------------------
// kernel launcher

void matmul_backward1(sycl::queue &q, float* dinp, float* dweight, float* dbias,
                      float* dout, float* inp, float* weight, float* ones,
                      int B, int T, int C, int OC) {
    float alpha = 1.0f;
    float beta = 1.0f; // note we must use beta = 1.0 so that we do a +=, as we should, because gradients add

    // backward to input
    oneapi::mkl::blas::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                            C, B * T, OC, alpha, weight, C, dout, OC, beta, dinp, C).wait();
    // backward to weight
    oneapi::mkl::blas::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
                            C, OC, B * T, alpha, inp, C, dout, OC, beta, dweight, C).wait();

    // backward to bias, if given
    if (dbias != nullptr) {
        const int block_size = 512;
        matmul_backward_bias_kernel_faster(q, dbias, dout, B, T, OC, block_size);
    }
}

void matmul_backward(int kernel_num, sycl::queue &q,
                     float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight, float* ones,
                     int B, int T, int C, int OC) {
    switch (kernel_num) {
        case 1:
            matmul_backward1(q, dinp, dweight, dbias, dout, inp, weight, ones, B, T, C, OC);
            break;
        default:
            std::cout << "Invalid kernel number\n";
            exit(1);
    }
}

int main(int argc, char** argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    int OC = 768 * 4; // expansion of 4, e.g., in the MLP

    // set up the device
    sycl::queue q(sycl::default_selector_v);
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // create host memory of random numbers
    float* dinp = make_zeros_float(B * T * C);
    float* dweight = make_zeros_float(OC * C);
    float* dbias = make_zeros_float(OC);
    float* dout = make_random_float(B * T * OC);
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(OC * C);
    float* ones = make_ones_float(OC);

    // move to GPU
    float* d_dinp = sycl::malloc_device<float>(B * T * C, q);
    float* d_dweight = sycl::malloc_device<float>(OC * C, q);
    float* d_dbias = sycl::malloc_device<float>(OC, q);
    float* d_dout = sycl::malloc_device<float>(B * T * OC, q);
    float* d_inp = sycl::malloc_device<float>(B * T * C, q);
    float* d_weight = sycl::malloc_device<float>(OC * C, q);
    float* d_ones = sycl::malloc_device<float>(OC, q);

    q.memcpy(d_dinp, dinp, B * T * C * sizeof(float)).wait();
    q.memcpy(d_dweight, dweight, OC * C * sizeof(float)).wait();
    q.memcpy(d_dbias, dbias, OC * sizeof(float)).wait();
    q.memcpy(d_dout, dout, B * T * OC * sizeof(float)).wait();
    q.memcpy(d_inp, inp, B * T * C * sizeof(float)).wait();
    q.memcpy(d_weight, weight, OC * C * sizeof(float)).wait();
    q.memcpy(d_ones, ones, OC * sizeof(float)).wait();

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = std::atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << std::endl;

    // calculate the CPU reference
    matmul_backward_cpu(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);

    // calculate the GPU version
    matmul_backward(kernel_num, q, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_ones, B, T, C, OC);

    // compare
    std::cout << "Checking correctness..." << std::endl;
    std::cout << "dinp:" << std::endl;
    validate_result(d_dinp, dinp, "dinp", B * T * C, 1e-3f);
    std::cout << "dweight:" << std::endl;
    validate_result(d_dweight, dweight, "dweight", OC * C, 1e-3f);
    std::cout << "dbias:" << std::endl;
    validate_result(d_dbias, dbias, "dbias", OC, 1e-3f);
    std::cout << "All results match." << std::endl << std::endl;

    // now benchmark the kernel
    int repeat_times = 100;
    float elapsed_time = benchmark_kernel(repeat_times, matmul_backward, kernel_num,
                                          q, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_ones,
                                          B, T, C, OC);
    std::cout << "time " << elapsed_time << " ms" << std::endl;

    // cleanups
    free(dinp);
    free(dweight);
    free(dbias);
    free(dout);
    free(inp);
    free(weight);
    sycl::free(d_dinp, q);
    sycl::free(d_dweight, q);
    sycl::free(d_dbias, q);
    sycl::free(d_dout, q);
    sycl::free(d_inp, q);
    sycl::free(d_weight, q);
    sycl::free(d_ones, q);

    return 0;
}