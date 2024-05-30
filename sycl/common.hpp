//
// Created by mspronesti on 30/05/24.
//

#ifndef LLM_SYCL_COMMON_HPP
#define LLM_SYCL_COMMON_HPP

#include <sycl/sycl.hpp>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <chrono>

// ----------------------------------------------------------------------------
// random utils

float* make_random_float_01(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX); // range 0..1
    }
    return arr;
}

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

int* make_random_int(size_t N, int V) {
    int* arr = (int*)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    return arr;
}

float* make_zeros_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float)); // all zero
    return arr;
}

float* make_ones_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    return arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils
template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance=1e-4) {
    sycl::queue q(sycl::gpu_selector_v);

    // Allocate host memory
    D* out_gpu = (D*)malloc(num_elements * sizeof(D));

    // Create a buffer for the device result
    sycl::buffer<D, 1> device_buffer(device_result, sycl::range<1>(num_elements));

    // Copy data from device to host
    q.submit([&](sycl::handler& h) {
        auto acc = device_buffer.template get_access<sycl::access::mode::read>(h);
        h.copy(acc, out_gpu);
    }).wait();

    int nfaults = 0;
#ifndef ENABLE_BF16
    float epsilon = FLT_EPSILON;
#else
    float epsilon = 0.079;
#endif
    for (std::size_t i = 0; i < num_elements; i++) {
        // Skip masked elements
        if(!std::isfinite(cpu_reference[i]))
            continue;

        // print the first few comparisons
        if (i < 5) {
            std::cout << cpu_reference[i] << " " << static_cast<T>(out_gpu[i]) << std::endl;
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        float t_eff = tolerance + std::fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        if (std::fabs(cpu_reference[i] - static_cast<T>(out_gpu[i])) > t_eff) {
            std::cout << "Mismatch of " << name << " at " << i << ": CPU_ref: " << cpu_reference[i] << " vs GPU: " << static_cast<T>(out_gpu[i]) << std::endl;
            nfaults++;
            if (nfaults >= 10) {
                free(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    if (nfaults > 0) {
        free(out_gpu);
        exit(EXIT_FAILURE);
    }

    free(out_gpu);
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    sycl::queue q(sycl::gpu_selector_v);

    // Prepare buffer to scrub L2 cache between benchmarks
    sycl::device device = q.get_device();
    size_t l2_cache_size = device.get_info<sycl::info::device::global_mem_cache_size>();
    std::vector<char> flush_buffer(l2_cache_size);

    float elapsed_time = 0.f;
    for (int i = 0; i < repeats; i++) {
        // Clear L2 cache
        q.submit([&](sycl::handler& h) {
            h.fill(flush_buffer.data(), 0, l2_cache_size);
        }).wait();

        // Start recording the timing of the kernel
        auto start = std::chrono::high_resolution_clock::now();

        kernel(q, std::forward<KernelArgs>(kernel_args)...);

        q.wait();
        auto stop = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float, std::milli> duration = stop - start;
        elapsed_time += duration.count();
    }

    return elapsed_time / repeats;
}


#endif //LLM_SYCL_COMMON_HPP
