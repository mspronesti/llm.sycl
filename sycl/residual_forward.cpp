/*
Kernels for residual forward pass.

Compile example:
icpx -O3 -fsycl residual_forward.cpp -o residual_forward

version 1 is naive port from CPU code to kernel
./residual_forward 1
version 2 packs input into 128 bit memory reads
./residual_forward 2
*/

#include <sycl/sycl.hpp>
#include <iostream>
#include <cmath>
#include "common.hpp"

// ----------------------------------------------------------------------------
// CPU code reference

void residual_forward_cpu(float* out, const float* inp1, const float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

// ----------------------------------------------------------------------------
// SYCL kernels

void residual_forward_kernel1(sycl::queue& q, float* out, const float* inp1, const float* inp2, int N) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            int i = idx[0];
            out[i] = inp1[i] + inp2[i];
        });
    }).wait();
}

void residual_forward_kernel2(sycl::queue& q, float* out, const float* inp1, const float* inp2, int N) {
    constexpr int vector_size = 4;
    int packed_size = N / vector_size;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(packed_size), [=](sycl::id<1> idx) {
            int i = idx[0] * vector_size;
            sycl::vec<float, vector_size> v_inp1 = *reinterpret_cast<const sycl::vec<float, vector_size>*>(inp1 + i);
            sycl::vec<float, vector_size> v_inp2 = *reinterpret_cast<const sycl::vec<float, vector_size>*>(inp2 + i);
            sycl::vec<float, vector_size> v_out = v_inp1 + v_inp2;
            *reinterpret_cast<sycl::vec<float, vector_size>*>(out + i) = v_out;
        });
    }).wait();
}

// ----------------------------------------------------------------------------
// kernel launcher

void residual_forward(int kernel_num, sycl::queue& q, float* d_out, const float* d_inp1, const float* d_inp2, int N, const int block_size) {
    switch (kernel_num) {
        case 1:
            residual_forward_kernel1(q, d_out, d_inp1, d_inp2, N);
            break;
        case 2:
            residual_forward_kernel2(q, d_out, d_inp1, d_inp2, N);
            break;
        default:
            std::cerr << "Invalid kernel number\n";
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    int B = 8;
    int T = 1024;
    int C = 768;
    int N = B * T * C;

    // create host memory of random numbers
    float* out = new float[N];
    float* inp1 = make_random_float(N);
    float* inp2 = make_random_float(N);

    // move to device
    sycl::queue q;
    float* d_out = sycl::malloc_device<float>(N, q);
    float* d_inp1 = sycl::malloc_device<float>(N, q);
    float* d_inp2 = sycl::malloc_device<float>(N, q);
    q.memcpy(d_inp1, inp1, N * sizeof(float)).wait();
    q.memcpy(d_inp2, inp2, N * sizeof(float)).wait();

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << "\n";

    // first check the correctness of the kernel
    residual_forward_cpu(out, inp1, inp2, N);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    for (int block_size : block_sizes) {
        std::cout << "Checking block size " << block_size << ".\n";
        residual_forward(kernel_num, q, d_out, d_inp1, d_inp2, N, block_size);
        float tol = 1e-5;
        validate_result(d_out, out, "out", N, tol);
    }

    std::cout << "All results match. Starting benchmarks.\n\n";

    for (int block_size : block_sizes) {
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, residual_forward, kernel_num, q, d_out, d_inp1, d_inp2, N, block_size);

        long memory_ops = N * 3 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        std::cout << "block_size " << block_size << " | time " << elapsed_time << " ms | bandwidth " << memory_bandwidth << " GB/s\n";
    }

    sycl::free(d_out, q);
    sycl::free(d_inp1, q);
    sycl::free(d_inp2, q);
    delete[] out;
    delete[] inp1;
    delete[] inp2;

    return 0;
}

