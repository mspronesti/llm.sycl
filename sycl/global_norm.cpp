#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include "common.hpp"
// ----------------------------------------------------------------------------
// CPU code reference

float global_norm_cpu(const float* data, size_t count) {
    double acc = 0.0;
    for (size_t i = 0; i < count; ++i) {
        acc += (double)data[i] * (double)data[i];
    }
    return (float)acc;
}

// ----------------------------------------------------------------------------
// SYCL kernels

template<class T>
void norm_kernel1(sycl::queue& q, sycl::buffer<float, 1>& out_buf, sycl::buffer<const T, 1>& data_buf, size_t count, int block_size) {
    q.submit([&](sycl::handler& h) {
        auto out = out_buf.template get_access<sycl::access::mode::read_write>(h);
        auto data = data_buf.template get_access<sycl::access::mode::read>(h);

        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(block_size * 32), sycl::range<1>(block_size)), [=](sycl::nd_item<1> item) {
            size_t index = item.get_global_id(0);
            size_t grid_width = item.get_group_range(0) * item.get_local_range(0);
            float accumulator = 0.f;
            for (size_t i = index; i < count; i += grid_width) {
                accumulator += (float)data[i] * (float)data[i];
            }

            // Reduce within the work-group
            auto wg = item.get_group();
            float wg_sum = sycl::reduce_over_group(wg, accumulator, sycl::plus<>());

            if (item.get_local_id(0) == 0) {
	    	dpct::atomic_fetch_add(&out[0], wg_sum);
            }
        });
    });
}

template<class T>
void norm_kernel2(sycl::queue& q, sycl::buffer<float, 1>& out_buf, sycl::buffer<const T, 1>& data_buf, size_t count, int block_size) {
    q.submit([&](sycl::handler& h) {
        auto out = out_buf.template get_access<sycl::access::mode::read_write>(h);
        auto data = data_buf.template get_access<sycl::access::mode::read>(h);

        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(block_size * 32), sycl::range<1>(block_size)), [=](sycl::nd_item<1> item) {
            size_t index = item.get_global_id(0);
            size_t grid_width = item.get_group_range(0) * item.get_local_range(0);
            float accumulator = 0.f;
            for (size_t i = index; i < count; i += grid_width) {
                accumulator += (float)data[i] * (float)data[i];
            }

            // Reduce within the work-group
            auto wg = item.get_group();
            float wg_sum = sycl::reduce_over_group(wg, accumulator, sycl::plus<>());

            if (item.get_local_id(0) % 32 == 0) {
	    	dpct::atomic_fetch_add(&out[0], wg_sum);
            }
        });
    });
}

// ----------------------------------------------------------------------------
// Kernel launcher

template<typename T>
void global_norm1(sycl::queue& q, float* out, const T* values, size_t count, int block_size) {
    sycl::buffer<float, 1> out_buf(out, sycl::range<1>(1));
    sycl::buffer<const T, 1> values_buf(values, sycl::range<1>(count));
    norm_kernel1(q, out_buf, values_buf, count, block_size);
    q.wait();
}

template<typename T>
void global_norm2(sycl::queue& q, float* out, const T* values, size_t count, int block_size) {
    sycl::buffer<float, 1> out_buf(out, sycl::range<1>(1));
    sycl::buffer<const T, 1> values_buf(values, sycl::range<1>(count));
    norm_kernel2(q, out_buf, values_buf, count, block_size);
    q.wait();
}

void global_norm(int kernel_num, sycl::queue& q, float* out, const float* values, size_t count, int block_size) {
    switch (kernel_num) {
        case 1:
            return global_norm1(q, out, values, count, block_size);
        case 2:
            return global_norm2(q, out, values, count, block_size);
        default:
            std::cerr << "Invalid kernel number" << std::endl;
            exit(1);
    }
}


// ----------------------------------------------------------------------------
// Main function
int main(int argc, char** argv) {
    int C = 768;
    int L = 12;

    size_t num_params = (size_t)(C * 4 * C + C * C) * 2 * L;

    // Create host memory of random numbers
    float* inp = make_random_float(num_params);
    // Scale them down
    for (size_t i = 0; i < num_params; ++i) {
        inp[i] *= 1e-3;
    }

    // Read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = std::atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << std::endl;

    // First check the correctness of the kernel
    float out = global_norm_cpu(inp, num_params);

    // SYCL queue
    sycl::queue q;

    // Time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};
    for (int block_size : block_sizes) {
        std::cout << "Checking block size " << block_size << "." << std::endl;
        float out_result = 0;
        global_norm(kernel_num, q, &out_result, inp, num_params, block_size);
        validate_result(&out_result, &out, "out", 1, 1e-2f);
    }

    std::cout << "All results match. Starting benchmarks.\n\n";

    for (int block_size : block_sizes) {
        int repeat_times = 1000;
        float out_result = 0;
        float elapsed_time = benchmark_kernel(
            repeat_times,
            global_norm, // kernel
            kernel_num, q, &out_result, inp, num_params, block_size // kernel params
        );

        // Napkin math: estimate the memory bandwidth achieved
        size_t memory_ops = num_params * sizeof(float);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        std::cout << "block_size " << block_size << " | time " << elapsed_time << " ms | bandwidth " << memory_bandwidth << " GB/s" << std::endl;
    }

    // Free memory
    delete[] inp;

    return 0;
}

