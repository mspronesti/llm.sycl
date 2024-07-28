/*
Kernels for the positional encoder backward pass in GPT-2.

Compile example:
icpx -O3 encoder_backward_sycl.cpp -o encoder_backward_sycl

version 1 is naive port from CPU code to kernel: parallelizes over B,T,C, uses atomics to add to dwte, dwpe
./encoder_backward_sycl 1

version 2 is another naive port: parallelizes over C, loops over B,T
./encoder_backward_sycl 2
*/
#include <sycl/sycl.hpp>
#include <iostream>
#include <cmath>

#include "common.hpp"

// ----------------------------------------------------------------------------
// CPU code reference

void encoder_backward_cpu(float* dwte, float* dwpe,
                          float* dout, int* inp,
                          int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* dwte_ix = dwte + ix * C;
            float* dwpe_t = dwpe + t * C;
            for (int i = 0; i < C; i++) {
                float d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// naive implementation with atomics
void encoder_backward_kernel1(sycl::nd_item<1> item,
                              float* dwte, float* dwpe,
                              const float* dout, const int* inp,
                              int B, int T, int C) {
    int idx = item.get_global_id(0);
    const int N = B * T * C;
    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        const float* dout_btc = dout + b * T * C + t * C + c;
        float* dwte_ix = dwte + ix * C + c;
        float* dwpe_tc = dwpe + t * C + c;

        atomicAdd(dwte_ix, *dout_btc);
        atomicAdd(dwpe_tc, *dout_btc);
    }
}

// naive implementation that parallelizes over C and loops over B,T,
// but it gets rid of atomics
void encoder_backward_kernel2(sycl::nd_item<1> item,
                              float* dwte, float* dwpe,
                              const float* dout, const int* inp,
                              int B, int T, int C) {
    int c = item.get_global_id(0);
    if (c >= C) return;
    int BT = B * T;
    for (int i = 0; i < BT; i++) {
        int t = i % T;
        int ix = inp[i];
        float dout_btc = dout[i * C + c];
        dwte[ix * C + c] += dout_btc;
        dwpe[t * C + c] += dout_btc;
    }
}


void encoder_backward1(sycl::queue& q,
                              float* dwte, float* dwpe,
                              const float* dout, const int* inp,
                              int B, int T, int C,
                              int block_size) {
    const int N = B * T * C;
    const int grid_size =  ceil_div(N, block_size);;

    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> item) {
        encoder_backward_kernel1(item, dwte, dwpe, dout, inp, B, T, C);
    }).wait();
}

void encoder_backward2(sycl::queue& q,
                              float* dwte, float* dwpe,
                              const float* dout, const int* inp,
                              int B, int T, int C,
                              int block_size) {
    const int grid_size = ceil_div(C, block_size);
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> item) {
             encoder_backward_kernel2(item, dwte, dwpe, dout, inp, B, T, C);
    }).wait();
}


void encoder_backward(int kernel_num,
                      sycl::queue& q,
                      float* dwte, float* dwpe,
                      const float* dout, const int* inp,
                      int B, int T, int C,
                      int block_size) {
    switch (kernel_num) {
        case 1:
            encoder_backward1(q, dwte, dwpe, dout, inp, B, T, C, block_size);
            break;
        case 2:
            encoder_backward2(q, dwte, dwpe, dout, inp, B, T, C, block_size);
            break;
        default:
            std::cerr << "Invalid kernel number\n";
            std::exit(1);
    }
}

int main(int argc, char** argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    int V = 50257;

    sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order());

    // Allocate host memory and initialize with random values
    float* dout = make_random_float(B * T * C);
    int* inp = make_random_int(B * T, V);
    float* dwte = make_zeros_float(V * C);
    float* dwpe = make_zeros_float(T * C);

    // Allocate device memory
    float* d_dout = sycl::malloc_device<float>(B * T * C, q);
    int* d_inp = sycl::malloc_device<int>(B * T, q);
    float* d_dwte = sycl::malloc_device<float>(V * C, q);
    float* d_dwpe = sycl::malloc_device<float>(T * C, q);

    // Copy data from host to device
    q.memcpy(d_dout, dout, B * T * C * sizeof(float)).wait();
    q.memcpy(d_inp, inp, B * T * sizeof(int)).wait();

    // Read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = std::atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << std::endl;

    // first check the correctness of the kernel
    encoder_backward_cpu(dwte, dwpe, dout, inp, B, T, C);

    // Set up block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};

    // Check the correctness of the kernel
    for (int block_size : block_sizes) {
        std::cout << "Checking block size " << block_size << "." << std::endl;
        q.memset(d_dwte, 0, V * C * sizeof(float));
        q.memset(d_dwpe, 0, T * C * sizeof(float));
        encoder_backward(kernel_num, q, d_dwte, d_dwpe, d_dout, d_inp, B, T, C, block_size);
        validate_result(d_dwte, dwte, "dwte", V * C, 1e-5f);
        validate_result(d_dwpe, dwpe, "dwpe", T * C, 1e-5f);
    }

    std::cout << "All results match. Starting benchmarks.\n\n";

    for (int block_size : block_sizes) {
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, encoder_backward,
                                              kernel_num, q, d_dwte, d_dwpe, d_dout, d_inp, B, T, C, block_size);

        std::cout << "block_size " << block_size << " | time " << elapsed_time << " ms\n";
    }

    // Free host memory
    free(dout);
    free(inp);
    free(dwte);
    free(dwpe);

    // Free device memory
    sycl::free(d_dout, q);
    sycl::free(d_inp, q);
    sycl::free(d_dwte, q);
    sycl::free(d_dwpe, q);

    return 0;
}

