#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#define ENABLE_BF16

#include "common.hpp"

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

// ----------------------------------------------------------------------------
// CPU code reference

void gelu_forward_cpu(float* out, const float* inp, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

// ----------------------------------------------------------------------------
// SYCL kernels

void gelu_forward_kernel1(sycl::nd_item<1> id, floatX* out, const floatX* inp, int N) {
    int i = id.get_global_id(0);
    if (i < N) {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + sycl::tanh(GELU_SCALING_FACTOR * (xi + cube)));
    }
}

void gelu_forward_kernel2(sycl::nd_item<1> id, floatX* out, const floatX* inp, int N) {
    int i = (id.get_global_id(0)) * x128::size;
    if (i < N) {
        x128 packed_out;
        x128 packed_inp = load128cs(inp + i); // load and do not keep in cache
        for(int k = 0; k < packed_inp.size; ++k) {
            float xi = (float)packed_inp[k];
            float cube = 0.044715f * xi * xi * xi;
            packed_out[k] = (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
        }
        // store instead of storecs (without cache streaming) in case it is useful for the
        // data to be in the cache for the next operation after this GeLU
        store128(out + i, packed_out);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void gelu_forward1(sycl::queue &q, floatX* out, const floatX* inp, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        gelu_forward_kernel1(id, out, inp, N);
    }).wait();
}

void gelu_forward2(sycl::queue &q, floatX* out, const floatX* inp, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size * x128::size);
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        gelu_forward_kernel2(id, out, inp, N);
    }).wait();
}

// kernel version dispatch
void gelu_forward(int kernel_num,
                  sycl::queue &q,
                  floatX* out,
                  const floatX* inp,
                  int B, int T, int C,
                  int block_size) {
    switch (kernel_num) {
        case 1:
            gelu_forward1(q, out, inp, B * T * C, block_size);
            break;
        case 2:
            gelu_forward2(q, out, inp, B * T * C, block_size);
            break;
        default:
            std::cout << "Invalid kernel number\n";
            std::exit(1);
    }
}

// ----------------------------------------------------------------------------
// Main function
int main(int argc, char** argv) {
    int B = 8;
    int T = 1024;
    int C = 768;
    int N = B * T * C;

    // Create host memory of random numbers
    float* out = new float[N];
    float* inp = make_random_float(N);

    // SYCL queue
    sycl::queue q(sycl::default_selector_v);

    auto d_out = sycl::malloc_device<floatX>(B * T * C, q);
    auto d_inp = sycl::malloc_device<floatX>(B * T * C, q);
    memcpy_convert(d_inp, inp, B * T * C, q);

    // Read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = std::atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << std::endl;

    // First check the correctness of the kernel
    gelu_forward_cpu(out, inp, N);


    // Time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};
    for (int block_size : block_sizes) {
        std::cout << "Checking block size " << block_size << "." << std::endl;
        gelu_forward(kernel_num, q, d_out, d_inp, B, T, C, block_size);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_out, out, "out", B * T * C, tol);
    }

    std::cout << "All results match. Starting benchmarks.\n\n";

    for (int block_size : block_sizes) {
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(
            repeat_times,
            gelu_forward, // kernel
            kernel_num, q, d_out, d_inp, B, T, C, block_size // kernel params
        );

        // Napkin math: estimate the memory bandwidth achieved
        long memory_ops = N * 2 * sizeof(float);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        std::cout << "block_size " << block_size << " | time " << elapsed_time << " ms | bandwidth " << memory_bandwidth << " GB/s" << std::endl;
    }

    // Free memory
    delete[] out;
    delete[] inp;

    sycl::free(d_out, q);
    sycl::free(d_inp, q);

    return 0;
}

