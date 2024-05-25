#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

// Function to generate random float numbers between 0 and 1
std::vector<float> make_random_float_01(int size) {
    std::vector<float> data(size);
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    return data;
}

// Function to generate random integer numbers
std::vector<int> make_random_int(int size, int max_val) {
    std::vector<int> data(size);
    for (int i = 0; i < size; ++i) {
        data[i] = rand() % max_val;
    }
    return data;
}

// CPU reference implementation
void crossentropy_forward_cpu(float* losses, const float* probs, const int* targets, int B, int T, int V) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* probs_bt = probs + b * T * V + t * V;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

// GPU kernel in SYCL
void crossentropy_forward_sycl(sycl::queue& q, float* d_losses, const float* d_probs, const int* d_targets, int B, int T, int V, int block_size) {
    int N = B * T;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((N + block_size - 1) / block_size * block_size), sycl::range<1>(block_size)),
                         [=](sycl::nd_item<1> item) {
                             int i = item.get_global_id(0);
                             if (i < N) {
                                 int b = i / T;
                                 int t = i % T;
                                 const float* probs_bt = d_probs + b * T * V + t * V;
                                 int ix = d_targets[b * T + t];
                                 d_losses[b * T + t] = -logf(probs_bt[ix]);
                             }
                         });
    }).wait();
}

// Validation function
void validate_result(const std::vector<float>& result, const std::vector<float>& reference, const std::string& name, int size, float tol) {
    for (int i = 0; i < size; ++i) {
        if (std::abs(result[i] - reference[i]) > tol) {
            std::cerr << "Validation failed for " << name << " at index " << i << ": " << result[i] << " != " << reference[i] << "\n";
            exit(1);
        }
    }
    std::cout << name << " validation passed.\n";
}

// Benchmarking function
float benchmark_kernel(int repeat_times, sycl::queue& q, float* d_losses, const float* d_probs, const int* d_targets, int B, int T, int V, int block_size) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat_times; ++i) {
        crossentropy_forward_sycl(q, d_losses, d_probs, d_targets, B, T, V, block_size);
    }
    q.wait();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    return duration.count() / repeat_times;
}

int main(int argc, char** argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int V = 50257;

    sycl::queue q;

    // Create host memory of random numbers
    std::vector<float> out(B * T);
    std::vector<float> probs = make_random_float_01(B * T * V);
    std::vector<int> targets = make_random_int(B * T, V);

    // Allocate device memory
    float* d_out = sycl::malloc_device<float>(B * T, q);
    float* d_probs = sycl::malloc_device<float>(B * T * V, q);
    int* d_targets = sycl::malloc_device<int>(B * T, q);

    // Move data to device
    q.memcpy(d_probs, probs.data(), B * T * V * sizeof(float)).wait();
    q.memcpy(d_targets, targets.data(), B * T * sizeof(int)).wait();

    // First check the correctness of the kernel
    crossentropy_forward_cpu(out.data(), probs.data(), targets.data(), B, T, V);

    // Time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};
    std::vector<float> d_out_host(B * T);

    for (int block_size : block_sizes) {
        std::cout << "Checking block size " << block_size << ".\n";
        crossentropy_forward_sycl(q, d_out, d_probs, d_targets, B, T, V, block_size);
        q.memcpy(d_out_host.data(), d_out, B * T * sizeof(float)).wait();
        validate_result(d_out_host, out, "out", B * T, 1e-5f);
    }

    std::cout << "All results match. Starting benchmarks.\n\n";

    for (int block_size : block_sizes) {
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, q, d_out, d_probs, d_targets, B, T, V, block_size);

        std::cout << "block_size " << block_size << " | time " << elapsed_time << " ms | per token " << (elapsed_time * 1'000'000 / (B*T)) << " ns\n";
    }

    // Free memory
    sycl::free(d_out, q);
    sycl::free(d_probs, q);
    sycl::free(d_targets, q);

    return 0;
}

