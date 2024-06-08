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

#define ENABLE_BF16
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

void residual_forward_kernel1(sycl::nd_item<1> id, floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int idx = id.get_global_id(0);
    if (idx < N) {
        out[idx] = inp1[idx] + inp2[idx];
    }
}

void residual_forward_kernel2(sycl::nd_item<1> id, floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int idx = id.get_global_id(0) * x128::size;
    if (idx < N) {
        x128 packed_out;
        x128 packed_inp1 = load128cs(inp1 + idx);
        x128 packed_inp2 = load128cs(inp2 + idx);
        for (int k = 0; k < packed_inp1.size; ++k)
        {
            packed_out[k] = (floatX)((float)packed_inp1[k] + (float)packed_inp2[k]);
        }
        store128(out + idx, packed_out);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void residual_forward1(sycl::queue &q, floatX* out, const floatX* inp1, const floatX* inp2, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        residual_forward_kernel1(id, out, inp1, inp2, N);
    }).wait();
}

void residual_forward2(sycl::queue &q, floatX* out, const floatX* inp1, const floatX* inp2, int N, const int block_size) {
    const int grid_size = ceil_div(N, (int)(block_size * x128::size));
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        residual_forward_kernel2(id, out, inp1, inp2, N);
    }).wait();
}


// kernel version dispatch
void residual_forward(int kernel_num, sycl::queue& q, floatX* d_out, const floatX* d_inp1, const floatX* d_inp2, int N, const int block_size) {
    switch (kernel_num) {
        case 1:
            residual_forward1(q, d_out, d_inp1, d_inp2, N, block_size);
            break;
        case 2:
            residual_forward2(q, d_out, d_inp1, d_inp2, N, block_size);
            break;
        default:
            std::cerr << "Invalid kernel number\n";
            std::exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    int N = B * T * C;
    
    // create host memory of random numbers
    float* out = new float[N];
    float* inp1 = make_random_float(N);
    float* inp2 = make_random_float(N);

    // move to device
    sycl::queue q(sycl::default_selector_v);
    floatX* d_out = sycl::malloc_device<floatX>(N, q);
    floatX* d_inp1 = sycl::malloc_device<floatX>(N, q);
    floatX* d_inp2 = sycl::malloc_device<floatX>(N, q);
    memcpy_convert(d_inp1, inp1, N, q);
    memcpy_convert(d_inp2, inp2, N, q);;

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << "\n";

    // first check the correctness of the kernel
    residual_forward_cpu(out, inp1, inp2, N);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};

    for (int block_size : block_sizes) {
        std::cout << "Checking block size " << block_size << ".\n";
        residual_forward(kernel_num, q, d_out, d_inp1, d_inp2, N, block_size);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_out, out, "out", N, tol);
    }

    std::cout << "All results match. Starting benchmarks.\n\n";

    for (int block_size : block_sizes) {
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(
                repeat_times,
                residual_forward,
                kernel_num, q, d_out, d_inp1, d_inp2, N, block_size
        );

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 2 read and 1 write, 4 bytes each
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

