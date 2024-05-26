#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <CL/sycl.hpp>

#define ENABLE_BF16


// ----------------------------------------------------------------------------
// CPU code reference

void residual_forward_cpu(float* out, const float* inp1, const float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

void validate_result(const float* result, const float* reference, const std::string& name, size_t num_elements, float tolerance) {
    for (long i = 0; i < num_elements; ++i) {
        if (fabs(result[i] - reference[i]) > tolerance) {
            std::cerr << "Validation failed for " << name << " at index " << i << ": " << result[i] << " != " << reference[i] << std::endl;
            exit(1);
        }
    }
}

// GPU kernels
void residual_forward_kernel(sycl::queue& q, sycl::buffer<float, 1>& out, sycl::buffer<float, 1>& inp1, sycl::buffer<float, 1>& inp2, int N, int block_size) {
    q.submit([&](sycl::handler& cgh) {
        auto out_acc = out.get_access<sycl::access::mode::write>(cgh);
        auto inp1_acc = inp1.get_access<sycl::access::mode::read>(cgh);
        auto inp2_acc = inp2.get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for<class ResidualForwardKernel>(
            sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(block_size)),
            [=](sycl::nd_item<1> item) {
                size_t idx = item.get_global_id(0);
                if (idx < N) {
                    out_acc[idx] = inp1_acc[idx] + inp2_acc[idx];
                }
            });
    });
}

void residual_forward(int kernel_num, sycl::queue& q, sycl::buffer<float, 1>& out, sycl::buffer<float, 1>& inp1, sycl::buffer<float, 1>& inp2, int N, int block_size) {
    switch (kernel_num) {
        case 1:
            residual_forward_kernel(q, out, inp1, inp2, N, block_size);
            break;
        case 2:
            residual_forward_kernel(q, out, inp1, inp2, N, block_size / 4);
            break;
        default:
            std::cerr << "Invalid kernel number" << std::endl;
            exit(1);
    }
}


float benchmark_kernel(int repeat_times, sycl::queue& q, int kernel_num, float* out, const float* inp1, const float* inp2, int N, int block_size) {
    sycl::buffer<float, 1> d_out(out, sycl::range<1>(N));
    sycl::buffer<float, 1> d_inp1(inp1, sycl::range<1>(N));
    sycl::buffer<float, 1> d_inp2(inp2, sycl::range<1>(N));

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < repeat_times; ++i) {
        residual_forward(kernel_num, q, d_out, d_inp1, d_inp2, N, block_size);
        q.wait();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_ms = end - start;

    return duration_ms.count() / repeat_times;
}


int main(int argc, char **argv) {
    sycl::queue q{sycl::gpu_selector_v};

    int B = 8;
    int T = 1024;
    int C = 768;

    // create host memory of random numbers
    float* out = new float[B * T * C];
    float* inp1 = new float[B * T * C];
    float* inp2 = new float[B * T * C];
    for (int i = 0; i < B * T * C; ++i) {
        inp1[i] = static_cast<float>(rand()) / RAND_MAX;
        inp2[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << std::endl;

    // first check the correctness of the kernel
    residual_forward_cpu(out, inp1, inp2, B * T * C);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        std::cout << "Checking block size " << block_size << "." << std::endl;
        float elapsed_time = benchmark_kernel(1000, q, kernel_num, out, inp1, inp2, B *
        T * C, block_size);

        float tol = 1e-5;
        validate_result(out, out, "out", B * T * C, tol);
    }

    std::cout << "All results match. Starting benchmarks." << std::endl;

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        float elapsed_time = benchmark_kernel(1000, q, kernel_num, out, inp1, inp2, B * T * C, block_size);
        size_t memory_ops = B * T * C * 3 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        std::cout << "block_size " << block_size << " | time " << elapsed_time << " ms | bandwidth " << memory_bandwidth << " GB/s" << std::endl;
    }

    delete[] out;
    delete[] inp1;
    delete[] inp2;

    return 0;
}

