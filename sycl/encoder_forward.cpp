/*
Kernels for the positional encoder forward pass in GPT-2.

Compile example:
icpx -O3 encoder_forward_sycl.cpp -o encoder_forward_sycl

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./encoder_forward_sycl 1

version 2 is more optimized, parallelizes over all of B,T,C
./encoder_forward_sycl 2

version 3 is like version 2 but uses float reads/writes
./encoder_forward_sycl 3
*/

#include <CL/sycl.hpp>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <chrono>

#define ENABLE_BF16

// ----------------------------------------------------------------------------
// CPU code reference
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

// GPT-2 positional encoder forward pass
void encoder_forward_cpu(float* out,
                   const int* inp, const float* wte, const float* wpe,
                   int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * C + t * C;
            int ix = inp[b * T + t];
            const float* wte_ix = wte + ix * C;
            const float* wpe_t = wpe + t * C;
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// naive implementation into kernel, parallelize over B,T, loop over C
void encoder_forward_kernel1(sycl::queue &q, float* out,
                             const int* inp, const float* wte, const float* wpe,
                             int B, int T, int C) {
    int N = B * T;
    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            int b = idx[0] / T;
            int t = idx[0] % T;
            float* out_bt = out + b * T * C + t * C;
            int ix = inp[b * T + t];
            const float* wte_ix = wte + ix * C;
            const float* wpe_t = wpe + t * C;
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        });
    }).wait();
}

// optimized implementation: parallelize over all of B,T,C
void encoder_forward_kernel2(sycl::queue &q, float* out,
                             const int* inp, const float* wte, const float* wpe,
                             int B, int T, int C) {
    int N = B * T * C;
    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            int bt = idx[0] / C;
            int b = bt / T;
            int t = bt % T;
            int c = idx[0] % C;

            int ix = inp[b * T + t];

            float* out_btc = out + b * T * C + t * C + c;
            const float* wte_ix = wte + ix * C + c;
            const float* wpe_tc = wpe + t * C + c;
            *out_btc = wte_ix[0] + wpe_tc[0];
        });
    }).wait();
}

// optimized implementation using float reads/writes
void encoder_forward_kernel3(sycl::queue &q, float* out,
                             const int* inp, const float* wte, const float* wpe,
                             int B, int T, int C) {
    int N = B * T * C / 4;
    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            int bt = idx[0] * 4 / C;
            int b = bt / T;
            int t = bt % T;
            int c = (idx[0] * 4) % C;

            int ix = inp[b * T + t];

            float* out_btc = reinterpret_cast<float*>(out + b * T * C + t * C + c);
            const float* wte_ix = reinterpret_cast<const float*>(wte + ix * C + c);
            const float* wpe_tc = reinterpret_cast<const float*>(wpe + t * C + c);

            *out_btc = *wte_ix + *wpe_tc;
        });
    }).wait();
}

// ----------------------------------------------------------------------------
// kernel launcher

void encoder_forward1(sycl::queue &q, float* out,
                     const int* inp, const float* wte, const float* wpe,
                     int B, int T, int C) {
    encoder_forward_kernel1(q, out, inp, wte, wpe, B, T, C);
}

void encoder_forward2(sycl::queue &q, float* out,
                     const int* inp, const float* wte, const float* wpe,
                     int B, int T, int C) {
    encoder_forward_kernel2(q, out, inp, wte, wpe, B, T, C);
}

void encoder_forward3(sycl::queue &q, float* out,
                     const int* inp, const float* wte, const float* wpe,
                     int B, int T, int C) {
    encoder_forward_kernel3(q, out, inp, wte, wpe, B, T, C);
}

// kernel version dispatch
void encoder_forward(sycl::queue &q, int kernel_num,
                     float* out,
                     const int* inp, const float* wte, const float* wpe,
                     int B, int T, int C) {
    switch (kernel_num) {
        case 1:
            encoder_forward1(q, out, inp, wte, wpe, B, T, C);
            break;
        case 2:
            encoder_forward2(q, out, inp, wte, wpe, B, T, C);
            break;
        case 3:
            encoder_forward3(q, out, inp, wte, wpe, B, T, C);
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
    float* out = (float*)malloc(B * T * C * sizeof(float));
    int* inp = make_random_int(B * T, V);
    float* wte = make_random_float(V * C);
    float* wpe = make_random_float(T * C);

    // select device and create queue
    sycl::queue q;
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // move to GPU
    float* d_out = sycl::malloc_device<float>(B * T * C, q);
    int* d_inp = sycl::malloc_device<int>(B * T, q);
    float* d_wte = sycl::malloc_device<float>(V * C, q);
    float* d_wpe = sycl::malloc_device<float>(T * C, q);

    q.memcpy(d_inp, inp, B * T * sizeof(int)).wait();
    q.memcpy(d_wte, wte, V * C * sizeof(float)).wait();
    q.memcpy(d_wpe, wpe, T * C * sizeof(float)).wait();

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1) {
        kernel_num = std::atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << std::endl;

    // first check the correctness of the kernel
    encoder_forward_cpu(out, inp, wte, wpe, B, T, C);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    for (int block_size : block_sizes) {
        std::cout << "Checking block size " << block_size << "." << std::endl;
        encoder_forward(q, kernel_num, d_out, d_inp, d_wte, d_wpe, B, T, C);

        float tol = 1e-5;
#if defined(ENABLE_BF16) || defined(ENABLE_FP16)
        tol = 1e-2f;
#endif
	    validate_results(out, out, B * T * C);

    }

    std::cout << "All results match. Starting benchmarks." << std::endl;

    for (int block_size : block_sizes) {
        int repeat_times = 1000;
        double elapsed_time = benchmark_kernel([&] {
       	   encoder_forward(q, kernel_num, d_out, d_inp, d_wte, d_wpe, B, T, C);
    	});
        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 3 reads and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 4 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        std::cout << "block_size " << block_size << " | time " << elapsed_time << " ms | bandwidth " << memory_bandwidth << " GB/s" << std::endl;
    }

    // free memory
    free(out);
    free(inp);
    free(wte);
    free(wpe);
    sycl::free(d_out, q);
    sycl::free(d_inp, q);
    sycl::free(d_wte, q);
    sycl::free(d_wpe, q);

    return 0;
}
