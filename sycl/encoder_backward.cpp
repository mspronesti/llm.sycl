/*
Kernels for the positional encoder backward pass in GPT-2.

Compile example:
dpcpp -O3 encoder_backward_sycl.cpp -o encoder_backward_sycl

version 1 is naive port from CPU code to kernel: parallelizes over B,T,C, uses atomics to add to dwte, dwpe
./encoder_backward_sycl 1

version 2 is another naive port: parallelizes over C, loops over B,T
./encoder_backward_sycl 2
*/

#include <CL/sycl.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define ENABLE_BF16

int* make_random_int(size_t N, int V) {
    int* arr = (int*)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    return arr;
}

float* make_random_float(long num_elements) {
    float* data = new float[num_elements];
    for (long i = 0; i < num_elements; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    return data;
}

float* make_zeros_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float)); // all zero
    return arr;
}
// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 positional encoder backward pass
void encoder_backward_cpu(float* dwte, float* dwpe,
                          float* dout, int* inp,
                          int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* dwte_ix = dwte + ix * C;
            float* dwpe_t = dwpe + t * C;
            for (int i = 0; i < C; i++) {
                float d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// naive implementation with atomics
void encoder_backward_kernel1(sycl::queue &q, float* dwte, float* dwpe,
                              const float* dout, const int* inp,
                              int B, int T, int C) {
    int N = B * T * C;
    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            int bt = idx[0] / C;
            int b = bt / T;
            int t = bt % T;
            int c = idx[0] % C;

            int ix = inp[b * T + t];

            const float* dout_btc = dout + b * T * C + t * C + c;
            float* dwte_ix = dwte + ix * C + c;
            float* dwpe_tc = dwpe + t * C + c;

            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_dwte(*dwte_ix);
            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_dwpe(*dwpe_tc);
            
            atomic_dwte.fetch_add(*dout_btc);
            atomic_dwpe.fetch_add(*dout_btc);
        });
    }).wait();
}

// naive implementation that parallelizes over C and loops over B,T
// but it gets rid of atomics
void encoder_backward_kernel2(sycl::queue &q, float* dwte, float* dwpe,
                              const float* dout, const int* inp,
                              int B, int T, int C) {
    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(C), [=](sycl::id<1> idx) {
            int c = idx[0];
            int BT = B * T;
            for (int i = 0; i < BT; i++) {
                int t = i % T;
                int ix = inp[i];
                float dout_btc = dout[i * C + c];
                dwte[ix * C + c] += dout_btc;
                dwpe[t * C + c] += dout_btc;
            }
        });
    }).wait();
}

// ----------------------------------------------------------------------------
// kernel launcher

void encoder_backward1(sycl::queue &q, float* dwte, float* dwpe,
                       const float* dout, const int* inp,
                       int B, int T, int C,
                       const int block_size) {
    encoder_backward_kernel1(q, dwte, dwpe, dout, inp, B, T, C);
}

void encoder_backward2(sycl::queue &q, float* dwte, float* dwpe,
                       const float* dout, const int* inp,
                       int B, int T, int C,
                       const int block_size) {
    encoder_backward_kernel2(q, dwte, dwpe, dout, inp, B, T, C);
}

// kernel version dispatch
void encoder_backward(sycl::queue &q, int kernel_num,
                      float* dwte, float* dwpe,
                      const float* dout, const int* inp,
                      int B, int T, int C,
                      const int block_size) {
    switch (kernel_num) {
        case 1:
            encoder_backward1(q, dwte, dwpe, dout, inp, B, T, C, block_size);
            break;
        case 2:
            encoder_backward2(q, dwte, dwpe, dout, inp, B, T, C, block_size);
            break;
        default:
            std::cerr << "Invalid kernel number" << std::endl;
            exit(1);
    }
}

// ----------------------------------------------------------------------------

void validate_results(float* ref, float* res, int size) {
    for (int i = 0; i < size; ++i) {
        if (std::fabs(ref[i] - res[i]) > 1e-5) {
            std::cerr << "Result mismatch at index " << i << ": " << ref[i] << " != " << res[i] << std::endl;
            exit(1);
        }
    }
    std::cout << "Validation passed!" << std::endl;
}

// Function to benchmark kernel execution time
template <typename F>
double benchmark_kernel(F&& f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

int main(int argc, char **argv) {
    int B = 8;
    int T = 1024;
    int C = 768;
    int V = 50257;

    // create host memory of random numbers
    float* dout = make_random_float(B * T * C);
    int* inp = make_random_int(B * T, V);
    float* dwte = make_zeros_float(V * C);
    float* dwpe = make_zeros_float(T * C);

    // select device and create queue
    sycl::queue q;
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // move to GPU
    float* d_dout = sycl::malloc_device<float>(B * T * C, q);
    int* d_inp = sycl::malloc_device<int>(B * T, q);
    float* d_dwte = sycl::malloc_device<float>(V * C, q);
    float* d_dwpe = sycl::malloc_device<float>(T * C, q);

    q.memcpy(d_dout, dout, B * T * C * sizeof(float)).wait();
    q.memcpy(d_inp, inp, B * T * sizeof(int)).wait();
    q.memcpy(d_dwte, dwte, V * C * sizeof(float)).wait();
    q.memcpy(d_dwpe, dwpe, T * C * sizeof(float)).wait();

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = std::atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << std::endl;

    // set up block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    // first check the correctness of the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        std::cout << "Checking block size " << block_size << "." << std::endl;
        encoder_backward_cpu(dwte, dwpe, dout, inp, B, T, C);
        encoder_backward(q, kernel_num, d_dwte, d_dwpe, d_dout, d_inp, B, T, C, block_size);
        q.memcpy(dwte, d_dwte, V * C * sizeof(float)).wait();
        q.memcpy(dwpe, d_dwpe, T * C * sizeof(float)).wait();
        validate_results(dwte, dwte, V * C);
        validate_results(dwpe, dwpe, T * C);
    }
    std::cout << "All results match. Starting benchmarks." << std::endl;

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 1000;
        double elapsed_time = benchmark_kernel([&] {
            for (int i = 0; i < repeat_times; i++) {
                encoder_backward(q, kernel_num, d_dwte, d_dwpe, d_dout, d_inp, B, T, C, block_size);
            }
        });
        std::cout << "block_size " << block_size << " | time " << elapsed_time * 1000 << " ms" << std::endl;
    }

    // free memory
    free(dout);
    free(inp);
    free(dwte);
    free(dwpe);
    sycl::free(d_dout, q);
    sycl::free(d_inp, q);
    sycl::free(d_dwte, q);
    sycl::free(d_dwpe, q);

    return 0;
}

