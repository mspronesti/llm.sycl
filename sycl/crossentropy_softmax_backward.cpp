#include <CL/sycl.hpp>
#include <iostream>
#include <cmath>
#include "common.hpp"

void crossentropy_softmax_backward_cpu(float* dlogits,
                                       const float* dlosses, const float* probs, const int* targets,
                                       int B, int T, int V) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * V + t * V;
            const float* probs_bt = probs + b * T * V + t * V;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

void crossentropy_softmax_backward_kernel1(sycl::queue& q,
                                           float* dlogits,
                                           const float* dlosses,
                                           const float* probs,
                                           const int* targets,
                                           int B, int T, int V,
                                           int block_size) {
    const int N = B * T * V;
    const int grid_size = (N + block_size - 1) / block_size;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size * block_size), sycl::range<1>(block_size)), [=](sycl::nd_item<1> item) {
            int i = item.get_global_id(0);
            if (i < B * T * V) {
                int b = i / (T * V);
                int t = (i / V) % T;
                int v = i % V;
                float* dlogits_bt = dlogits + b * T * V + t * V;
                const float* probs_bt = probs + b * T * V + t * V;
                float dloss = dlosses[b * T + t];
                int ix = targets[b * T + t];
                float p = probs_bt[v];
                float indicator = v == ix ? 1.0f : 0.0f;
                dlogits_bt[v] += (p - indicator) * dloss;
            }
        });
    }).wait();
}

void crossentropy_softmax_backward(int kernel_num,
                                   sycl::queue& q,
                                   float* dlogits,
                                   const float* dlosses,
                                   const float* probs,
                                   const int* targets,
                                   int B, int T, int V,
                                   int block_size) {
    switch (kernel_num) {
        case 1:
            crossentropy_softmax_backward_kernel1(q, dlogits, dlosses, probs, targets, B, T, V, block_size);
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
    int V = 50257;

    sycl::queue q;

    // Allocate host memory and initialize with random values
    float* probs = make_random_float(B * T * V);
    int* targets = make_random_int(B * T, V);
    float* dlosses = make_random_float(B * T);
    float* dlogits = make_zeros_float(B * T * V);

    // Allocate device memory
    float* d_probs = sycl::malloc_device<float>(B * T * V, q);
    int* d_targets = sycl::malloc_device<int>(B * T, q);
    float* d_dlosses = sycl::malloc_device<float>(B * T, q);
    float* d_dlogits = sycl::malloc_device<float>(B * T * V, q);

    // Copy data from host to device
    q.memcpy(d_probs, probs, B * T * V * sizeof(float)).wait();
    q.memcpy(d_targets, targets, B * T * sizeof(int)).wait();
    q.memcpy(d_dlosses, dlosses, B * T * sizeof(float)).wait();

    // Read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = std::atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << std::endl;

    // Check the correctness of the kernel
    crossentropy_softmax_backward_cpu(dlogits, dlosses, probs, targets, B, T, V);

    // Time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};

    for (int block_size : block_sizes) {
        q.memset(d_dlogits, 0, B * T * V * sizeof(float)).wait();
        std::cout << "Checking block size " << block_size << "." << std::endl;
        crossentropy_softmax_backward(kernel_num, q, d_dlogits, d_dlosses, d_probs, d_targets, B, T, V, block_size);
        float* h_dlogits = (float*)malloc(B * T * V * sizeof(float));
        q.memcpy(h_dlogits, d_dlogits, B * T * V * sizeof(float)).wait();
        validate_result(h_dlogits, dlogits, "dlogits", B * T * V, 1e-5f);
        free(h_dlogits);
    }

    std::cout << "All results match. Starting benchmarks.\n\n";

    for (int block_size : block_sizes) {
        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, crossentropy_softmax_backward,
                                              kernel_num, q, d_dlogits, d_dlosses, d_probs, d_targets,
                                              B, T, V, block_size);

        std::cout << "block_size " << block_size << " | time " << elapsed_time << " ms | per token " << elapsed_time * 1'000 / (B * T) << " µs\n";
    }

    // Free host memory
    free(probs);
    free(targets);
    free(dlosses);
    free(dlogits);

    // Free device memory
    sycl::free(d_probs, q);
    sycl::free(d_targets, q);
    sycl::free(d_dlosses, q);
    sycl::free(d_dlogits, q);

    return 0;
}

