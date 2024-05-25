#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

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

void gelu_forward_kernel1(sycl::queue& q, sycl::buffer<float, 1>& out_buf, sycl::buffer<const float, 1>& inp_buf, int N) {
    q.submit([&](sycl::handler& h) {
        auto out = out_buf.get_access<sycl::access::mode::write>(h);
        auto inp = inp_buf.get_access<sycl::access::mode::read>(h);

        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            float xi = inp[i];
            float cube = 0.044715f * xi * xi * xi;
            out[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
        });
    });
}

void gelu_forward_kernel2(sycl::queue& q, sycl::buffer<float, 1>& out_buf, sycl::buffer<const float, 1>& inp_buf, int N) {
    const int vec_size = 4;  // Assuming x128::size is 4 for this conversion
    q.submit([&](sycl::handler& h) {
        auto out = out_buf.get_access<sycl::access::mode::write>(h);
        auto inp = inp_buf.get_access<sycl::access::mode::read>(h);

        h.parallel_for(sycl::nd_range<1>(N / vec_size, vec_size), [=](sycl::nd_item<1> item) {
            int i = item.get_global_id(0) * vec_size;
            if (i < N) {
                sycl::vec<float, vec_size> packed_inp;
                for (int k = 0; k < vec_size; ++k) {
                    packed_inp[k] = inp[i + k];
                }

                sycl::vec<float, vec_size> packed_out;
                for (int k = 0; k < vec_size; ++k) {
                    float xi = packed_inp[k];
                    float cube = 0.044715f * xi * xi * xi;
                    packed_out[k] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
                }

                for (int k = 0; k < vec_size; ++k) {
                    out[i + k] = packed_out[k];
                }
            }
        });
    });
}

// ----------------------------------------------------------------------------
// Kernel launcher

void gelu_forward(int kernel_num, sycl::queue& q, float* out, const float* inp, int N, const int block_size) {
    sycl::buffer<float, 1> out_buf(out, sycl::range<1>(N));
    sycl::buffer<const float, 1> inp_buf(inp, sycl::range<1>(N));

    switch (kernel_num) {
        case 1:
            gelu_forward_kernel1(q, out_buf, inp_buf, N);
            break;
        case 2:
            gelu_forward_kernel2(q, out_buf, inp_buf, N);
            break;
        default:
            std::cerr << "Invalid kernel number" << std::endl;
            exit(1);
    }

    q.wait();
}

// ----------------------------------------------------------------------------
// Utility functions

float* make_random_float(long num_elements) {
    float* data = new float[num_elements];
    for (long i = 0; i < num_elements; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    return data;
}

void validate_result(float* result, float* reference, const char* name, long num_elements, float tol) {
    for (long i = 0; i < num_elements; i++) {
        if (std::fabs(result[i] - reference[i]) > tol) {
            std::cerr << "Validation failed for " << name << " at index " << i << std::endl;
            exit(1);
        }
    }
    std::cout << name << " validation passed." << std::endl;
}

float benchmark_kernel(int repeat_times, void (*kernel)(int, sycl::queue&, float*, const float*, int, const int), int kernel_num, sycl::queue& q, float* out, const float* inp, int N, const int block_size) {
    float elapsed_time = 0.0f;

    for (int i = 0; i < repeat_times; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        kernel(kernel_num, q, out, inp, N, block_size);
        q.wait();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end - start;
        elapsed_time += duration.count();
    }

    return elapsed_time / repeat_times;
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

    // Read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = std::atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << std::endl;

    // First check the correctness of the kernel
    gelu_forward_cpu(out, inp, N);

    // SYCL queue
    sycl::queue q;

    // Time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int block_size : block_sizes) {
        std::cout << "Checking block size " << block_size << "." << std::endl;
        gelu_forward(kernel_num, q, out, inp, N, block_size);
        validate_result(out, out, "out", N, 1e-5f);
    }

    std::cout << "All results match. Starting benchmarks.\n\n";

    for (int block_size : block_sizes) {
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, gelu_forward, kernel_num, q, out, inp, N, block_size);

        // Napkin math: estimate the memory bandwidth achieved
        long memory_ops = N * 2 * sizeof(float);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        std::cout << "block_size " << block_size << " | time " << elapsed_time << " ms | bandwidth " << memory_bandwidth << " GB/s" << std::endl;
    }

    // Free memory
    delete[] out;
    delete[] inp;

    return 0;
}

