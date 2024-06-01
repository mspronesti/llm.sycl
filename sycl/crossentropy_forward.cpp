#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>

#define ENABLE_BF16
#include "common.hpp"

// ----------------------------------------------------------------------------
// CPU code reference

void crossentropy_forward_cpu(float* losses,
                            const float* probs, const int* targets,
                            int B, int T, int V) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,V) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // loss = -log(probs[target])
            const float* probs_bt = probs + b * T * V + t * V;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

// ----------------------------------------------------------------------------
// SYCL kernel

void crossentropy_forward_kernel1(sycl::queue& q,
                                  float* d_losses,
                                  const float* d_probs,
                                  const int* d_targets,
                                  int B, int T, int V) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(B * T), [=](sycl::id<1> idx) {
            int i = idx[0];
            int b = i / T;
            int t = i % T;
            const float* probs_bt = d_probs + b * T * V + t * V;
            int ix = d_targets[b * T + t];
            d_losses[b * T + t] = -logf(probs_bt[ix]);
        });
    }).wait();
}

// ----------------------------------------------------------------------------
// kernel launcher

void crossentropy_forward1(sycl::queue& q,
                           float* d_losses,
                           const float* d_probs,
                           const int* d_targets,
                           int B, int T, int V,
                           const int block_size) {
    crossentropy_forward_kernel1(q, d_losses, d_probs, d_targets, B, T, V);
}

// kernel version dispatch
void crossentropy_forward(sycl::queue& q, int kernel_num,
                          float* d_losses,
                          const float* d_probs,
                          const int* d_targets,
                          int B, int T, int V,
                          const int block_size) {
    switch (kernel_num) {
        case 1:
            crossentropy_forward1(q, d_losses, d_probs, d_targets, B, T, V, block_size);
            break;
        default:
            std::cout << "Invalid kernel number" << std::endl;
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int V = 50257;

    sycl::queue q(sycl::gpu_selector_v);

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * sizeof(float));
    float* probs = make_random_float_01(B * T * V);
    int* targets = make_random_int(B * T, V);

    // Allocate device memory
    float* d_out = sycl::malloc_device<float>(B * T, q);
    float* d_probs = sycl::malloc_device<float>(B * T * V, q);
    int* d_targets = sycl::malloc_device<int>(B * T, q);

    // Copy data to device
    q.memcpy(d_probs, probs, B * T * V * sizeof(float)).wait();
    q.memcpy(d_targets, targets, B * T * sizeof(int)).wait();

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = std::atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << std::endl;

    // first check the correctness of the kernel
    crossentropy_forward_cpu(out, probs, targets, B, T, V);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        std::cout << "Checking block size " << block_size << "." << std::endl;
        crossentropy_forward(q, kernel_num, d_out, d_probs, d_targets, B, T, V, block_size);
        validate_result(d_out, out, "out", B * T, 1e-5f);
    }

    std::cout << "All results match. Starting benchmarks." << std::endl << std::endl;
    crossentropy_forward(q, kernel_num, d_out, d_probs, d_targets, B, T, V, 32);

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
       int block_size = block_sizes[j];

       int repeat_times = 1000;
       float elapsed_time = benchmark_kernel(
           repeat_times, 
           crossentropy_forward, 
           q, kernel_num, d_out, d_probs, d_targets, B, T, V, block_size
       );

      std::cout << "block_size " << block_size << " | time " << elapsed_time << " ms | per token " << (elapsed_time * 1'000'000 / (B * T)) << " ns" << std::endl;
    }

    // Free device memory
    sycl::free(d_out, q);
    sycl::free(d_probs, q);
    sycl::free(d_targets, q);

    // Free host memory
    free(out);
    free(probs);
    free(targets);

    return 0;
}

