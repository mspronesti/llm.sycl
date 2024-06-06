/*
Kernels for the positional encoder forward pass in GPT-2.

Compile example:
icpx -O3 encoder_forward_sycl.cpp -o encoder_forward_sycl

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./encoder_forward_sycl 1

version 2 is more optimized, parallelizes over all of B,T,C
./encoder_forward_sycl 2

version 3 is like version 2 but uses float reads/writes
./encoder_forward_sycl 3
*/

#include <sycl/sycl.hpp>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <chrono>

#define ENABLE_BF16
#include "common.hpp"


// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 positional encoder forward pass
void encoder_forward_cpu(float* out,
                   const int* inp, const float* wte, const float* wpe,
                   int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * C + t * C;
            int ix = inp[b * T + t];
            const float* wte_ix = wte + ix * C;
            const float* wpe_t = wpe + t * C;
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// naive implementation into kernel, parallelize over B,T, loop over C
void encoder_forward_kernel1(sycl::nd_item<1> item, float* out,
                             const int* inp, const float* wte, const float* wpe,
                             int B, int T, int C) {
        int idx = item.get_global_id(0);
        int N = B * T;

        if (idx < N) {
            int b = idx / T;
            int t = idx % T;
            float *out_bt = out + b * T * C + t * C;
            int ix = inp[b * T + t];
            const float *wte_ix = wte + ix * C;
            const float *wpe_t = wpe + t * C;
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
}

// optimized implementation: parallelize over all of B,T,C
void encoder_forward_kernel2(sycl::nd_item<1> item, float* out,
                             const int* inp, const float* wte, const float* wpe,
                             int B, int T, int C) {
    int idx = item.get_global_id(0);
    int N = B * T * C;

    if (idx < N){
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        float* out_btc = out + b * T * C + t * C + c;
        const float* wte_ix = wte + ix * C + c;
        const float* wpe_tc = wpe + t * C + c;
        *out_btc = wte_ix[0] + wpe_tc[0];
    }
}



// ----------------------------------------------------------------------------
// kernel launcher

void encoder_forward1(sycl::queue &q, float* out,
                     const int* inp, const float* wte, const float* wpe,
                     int B, int T, int C, int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        encoder_forward_kernel1(id, out, inp, wte, wpe, B, T, C);
    }).wait();
}

void encoder_forward2(sycl::queue &q, float* out,
                     const int* inp, const float* wte, const float* wpe,
                     int B, int T, int C, int block_size) {
    const int N = B * T * C;
    const int grid_size = ceil_div(N, block_size);
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        encoder_forward_kernel2(id, out, inp, wte, wpe, B, T, C);
    }).wait();
}



// kernel version dispatch
void encoder_forward(sycl::queue &q, int kernel_num,
                     float* out,
                     const int* inp, const float* wte, const float* wpe,
                     int B, int T, int C,  const int block_size) {
    switch (kernel_num) {
        case 1:
            encoder_forward1(q, out, inp, wte, wpe, B, T, C, block_size);
            break;
        case 2:
            encoder_forward2(q, out, inp, wte, wpe, B, T, C, block_size);
            break;
        default:
            std::cerr << "Invalid kernel number" << std::endl;
            exit(1);
    }
}

int main(int argc, char **argv) {
    int B = 8;
    int T = 1024;
    int C = 768;
    int V = 50257;

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    int* inp = make_random_int(B * T, V);
    float* wte = make_random_float(V * C);
    float* wpe = make_random_float(T * C);

    // select device and create queue
    sycl::queue q(sycl::default_selector_v);
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // move to GPU
    float* d_out = sycl::malloc_device<float>(B * T * C, q);
    int* d_inp = sycl::malloc_device<int>(B * T, q);
    float* d_wte = sycl::malloc_device<float>(V * C, q);
    float* d_wpe = sycl::malloc_device<float>(T * C, q);

    q.memcpy(d_inp, inp, B * T * sizeof(int)).wait();
    q.memcpy(d_wte, wte, V * C * sizeof(float)).wait();
    q.memcpy(d_wpe, wpe, T * C * sizeof(float)).wait();

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1) {
        kernel_num = std::atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << std::endl;

    // first check the correctness of the kernel
    encoder_forward_cpu(out, inp, wte, wpe, B, T, C);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};

    for (int block_size : block_sizes) {
        std::cout << "Checking block size " << block_size << "." << std::endl;
        encoder_forward(q, kernel_num, d_out, d_inp, d_wte, d_wpe, B, T, C, block_size);

        float tol = 1e-5;
	    validate_result(out, out, "out", B * T * C, tol);
    }

    std::cout << "All results match. Starting benchmarks." << std::endl;

    for (int block_size : block_sizes) {
        int repeat_times = 1000;
        double elapsed_time = benchmark_kernel(
            repeat_times,
       	    encoder_forward, // kernel
            q, kernel_num, d_out, d_inp, d_wte, d_wpe, B, T, C, block_size // params
    	);
        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 3 reads and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 4 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        std::cout << "block_size " << block_size << " | time " << elapsed_time << " ms | bandwidth " << memory_bandwidth << " GB/s" << std::endl;
    }

    // free memory
    free(out);
    free(inp);
    free(wte);
    free(wpe);
    sycl::free(d_out, q);
    sycl::free(d_inp, q);
    sycl::free(d_wte, q);
    sycl::free(d_wpe, q);

    return 0;
}
