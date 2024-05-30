#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

// Utility functions to generate random data
std::vector<float> make_random_float(int size) {
    std::vector<float> data(size);
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    return data;
}

std::vector<int> make_random_int(int size, int max_val) {
    std::vector<int> data(size);
    for (int i = 0; i < size; ++i) {
        data[i] = rand() % max_val;
    }
    return data;
}

std::vector<float> make_zeros_float(int size) {
    return std::vector<float>(size, 0.0f);
}

// CPU reference implementation
void crossentropy_softmax_backward_cpu(float* dlogits, const float* dlosses, const float* probs, const int* targets, int B, int T, int V) {
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

// SYCL kernel implementation
void crossentropy_softmax_backward_sycl(sycl::queue& q, float* d_dlogits, const float* d_dlosses, const float* d_probs, const int* d_targets, int B, int T, int V, int block_size) {
    int N = B * T * V;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((N + block_size - 1) / block_size * block_size), sycl::range<1>(block_size)),
                         [=](sycl::nd_item<1> item) {
                             int i = item.get_global_id(0);
                             if (i < N) {
                                 int b = i / (T * V);
                                 int t = (i / V) % T;
                                 int v = i % V;
                                 float* dlogits_bt = d_dlogits + b * T * V + t * V;
                                 const float* probs_bt = d_probs + b * T * V + t * V;
                                 float dloss = d_dlosses[b * T + t];
                                 int ix = d_targets[b * T + t];
                                 float p = probs_bt[v];
                                 float indicator = v == ix ? 1.0f : 0.0f;
                                 dlogits_bt[v] += (p - indicator) * dloss;
                             }
                         });
    }).wait();
}


int main(int argc, char** argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int V = 50257;

    sycl::queue q;

    // Create host memory of random numbers
    std::vector<float> probs = make_random_float(B * T * V);
    std::vector<int> targets = make_random_int(B * T, V);
    std::vector<float> dlosses = make_random_float(B * T);
    std::vector<float> dlogits = make_zeros_float(B * T * V);

    // Allocate device memory
    float* d_probs = sycl::malloc_device<float>(B * T * V, q);
    int* d_targets = sycl::malloc_device<int>(B * T, q);
    float* d_dlosses = sycl::malloc_device<float>(B * T, q);
    float* d_dlogits = sycl::malloc_device<float>(B * T * V, q);

    // Move data to device
    q.memcpy(d_probs, probs.data(), B * T * V * sizeof(float)).wait();
    q.memcpy(d_targets, targets.data(), B * T * sizeof(int)).wait();
    q.memcpy(d_dlosses, dlosses.data(), B * T * sizeof(float)).wait();

    // First check the correctness of the kernel
    crossentropy_softmax_backward_cpu(dlogits.data(), dlosses.data(), probs.data(), targets.data(), B, T, V);

    // Time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};
    std::vector<float> dlogits_host(B * T * V);

    for (int block_size : block_sizes) {
        std::cout << "Checking block size " << block_size << ".\n";
        q.memset(d_dlogits, 0, B * T * V * sizeof(float)).wait();
        crossentropy_softmax_backward_sycl(q, d_dlogits, d_dlosses, d_probs, d_targets, B, T, V, block_size);
        q.memcpy(dlogits_host.data(), d_dlogits, B * T * V * sizeof(float)).wait();
        validate_result(dlogits_host, dlogits, "dlogits", B * T * V, 1e-5f);
    }

    std::cout << "All results match. Starting benchmarks.\n\n";

    for (int block_size : block_sizes) {
        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(
                repeat_times,
                crossentropy_softmax_backward_sycl, // kernel,
                q, d_dlogits, d_dlosses, d_probs, d_targets, B, T, V, block_size // params
        );

        std::cout << "block_size " << block_size << " | time " << elapsed_time << " ms | per token " << (elapsed_time * 1'000 / (B*T)) << " µs\n";
    }

    // Free memory
    sycl::free(d_probs, q);
    sycl::free(d_targets, q);
    sycl::free(d_dlosses, q);
    sycl::free(d_dlogits, q);

    return 0;
}

