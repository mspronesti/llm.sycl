#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>

// turn on bf16 as default, done up here for now
#define ENABLE_BF16
#include "common.hpp"

/*
Kernels for a global norm.
Global norm in this context means that we want to calculate a single norm cooperatively using all avalailable SMs, instead
 of multiple norms that can be handled by separate blocks.

Compile example:
nvcc -O3 --use_fast_math global_norm.cu -o global_norm
*/

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
void norm_kernel1(sycl::nd_item<1> id, float* out, const T* data, size_t count,
                  sycl::local_accessor<float> block_result_acc){
    // we want as few atomics as possible, so each block tries to do
    // the maximum amount of work (so no fixed chunk, but instead iterating
    // until we run out of data), and then we reduce inside the block
    // and finally have just one atomic per block.
    sycl::group<1> block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();
    auto warp_thread_rank = warp.get_local_linear_id();

    float* block_result = block_result_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();

    size_t index = id.get_group(0) * id.get_local_range(0) + id.get_local_id(0);
    size_t grid_width = id.get_local_range(0) * id.get_group_range(0);
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }
    // warp-level reduce
    float warp_result = sycl::reduce_over_group(warp, accumulator, sycl::plus<float>());
    block_result[warp.get_group_linear_id()] = warp_result;
    id.barrier();
    if (warp.leader()) {
        float gather = warp_thread_rank < warp.get_group_linear_range() ? block_result[warp_thread_rank] : 0.f;
        float block_sum = sycl::reduce_over_group(warp, gather, sycl::plus<float>{});
        if(warp_thread_rank ==  0) {
            atomicAdd(out, block_sum);
        }
    }
}

template<class T>
void norm_kernel2(sycl::nd_item<1> id, float* out, const T* data, size_t count) {
    // concrete example for an A100 GPU (108 SMs, 2048 max threads each)
    // so there are 2048 * 108 = 221,184 threads total
    // say the block_size is 512, then we would launch 432 blocks in total
    // say num_params is ~100M, each thread will process ~500 elements
    // warps reduce with warp-level reduce, we have 221,184/32 = 6,912 warps
    // and then each warp atomicAdd's to global memory, total of 6,912 atomics

    // no shared memory; but one atomic per warp instead of per block
    sycl::group<1> block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();

    // out will be updated atomically from all thread blocks
    size_t index = id.get_group(0) * id.get_local_range(0) + id.get_local_id(0);
    size_t grid_width = id.get_local_range(0) * id.get_group_range(0);

    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }

    // warp-level reduce
    float warp_result = sycl::reduce_over_group(warp, accumulator, sycl::plus<float>{});
    // and atomic in global buffer
    if(warp.leader()) {
        atomicAdd(out, warp_result);
    }
}

// ----------------------------------------------------------------------------
// Kernel launcher

template<typename T>
void global_norm1(sycl::queue& q, float* out, const T* values, size_t count, int block_size) {
    //const int grid_size = cuda_threads_per_SM * cuda_num_SMs / block_size;
    const int max_num_CUs = q.get_device().get_info<sycl::info::device::max_compute_units>();
    const int max_wgs = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    const int grid_size = max_wgs * max_num_CUs / block_size;
    assert(grid_size > 0);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float> block_result_acc(32, h);
        h.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
            norm_kernel1(id, out, values, count, block_result_acc);
        });
    }).wait();
}

template<typename T>
void global_norm2(sycl::queue &q, float* out, const T* values, size_t count, int block_size) {
    // ditto
    const int max_num_CUs = q.get_device().get_info<sycl::info::device::max_compute_units>();
    const int max_wgs = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    const int grid_size = max_wgs * max_num_CUs / block_size;
    assert(grid_size > 0);      // gives a better error than letting the call below fail
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        norm_kernel2(id, out, values, count);
    }).wait();
}


void global_norm(int kernel_num, sycl::queue& q, float* out, const floatX* values, size_t count, int block_size) {
    switch (kernel_num) {
        case 1:
            global_norm1(q, out, values, count, block_size);
            break;
        case 2:
            global_norm2(q, out, values, count, block_size);
            break;
        default:
            std::cout << "Invalid kernel number" << std::endl;
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
    sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order());

    // move to GPU
    float* d_out = sycl::malloc_device<float>(1024, q);
    floatX* d_inp = sycl::malloc_device<floatX>(num_params, q);
    memcpy_convert(d_inp, inp, num_params, q);


    // Time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};
    for (int block_size : block_sizes) {
        std::cout << "Checking block size " << block_size << "." << std::endl;
        q.memset(d_out, 0, sizeof(float)).wait();
        global_norm(kernel_num, q, d_out, d_inp, num_params, block_size);
        validate_result(d_out, &out, "out", 1, 1e-2f);
    }

    std::cout << "All results match. Starting benchmarks.\n\n";

    for (int block_size : block_sizes) {
        int repeat_times = 1000;
        float out_result = 0;
        float elapsed_time = benchmark_kernel(
            repeat_times,
            global_norm, // kernel
            kernel_num, q, d_out, d_inp, num_params, block_size // kernel params
        );

        // Napkin math: estimate the memory bandwidth achieved
        size_t memory_ops = num_params * sizeof(floatX);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        std::cout << "block_size " << block_size << " | time " << elapsed_time << " ms | bandwidth " << memory_bandwidth << " GB/s" << std::endl;
    }

    // Free memory
    delete[] inp;
    sycl::free(d_out, q);
    sycl::free(d_inp, q);

    return 0;
}

