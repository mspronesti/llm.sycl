#include <sycl/sycl.hpp>
#include <iostream>
#include <cstdlib>
#include "oneapi/mkl/blas.hpp"
#include "mkl.h"
#include <cassert>

#define ENABLE_BF16
#include "common.hpp"

// ----------------------------------------------------------------------------
// utility functions
bool isPowerOfTwo(int n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

int largestPowerOfTwoLessOrEqual(int n) {
    // Return the largest power of 2 less than or equal to n
    if (n < 1) {
        return 0;
    }

    while ((n & (n - 1)) > 0) {
        n = n & (n - 1);
    }

    return n;
}


// ----------------------------------------------------------------------------
// CPU code reference

void matmul_backward_bias_cpu(float* dinp, float* dweight, float* dbias,
                              float* dout, float* inp, float* weight,
                              int B, int T, int C, int OC) {
    for (int o = 0; o < OC; o++) {
        double sum = 0.0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* dout_bt = dout + b * T * OC + t * OC;
                sum += dout_bt[o];
            }
        }
        dbias[o] = sum;
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

void matmul_backward_bias_kernel1(sycl::nd_item<1> id, floatX* dbias, const floatX* dout, int B, int T, int OC) {
    int o = id.get_group(0); // range [0, OC)
    int tid = id.get_local_linear_id(); // range [0, block_size)
    int block_size = id.get_local_range(0);
    const floatX* x = dout + o;
    // thread coarsening
    float sum = 0.0;
    for (int i = tid; i < B * T; i += block_size) {
        sum += x[i * OC];
    }
    sum = sycl::reduce_over_group(id.get_group(), sum, sycl::plus<float>());

    // write the final result (at thread 0) to global memory
    if (id.get_group().leader()) {
        dbias[o] += sum;
    }
}

// cooperative groups solution, one warp per output channel
void matmul_backward_bias_kernel2(sycl::nd_item<1> id, floatX* dbias, const floatX* dout, int B, int T, int OC) {
    // dout is (B, T, OC), dbias is (OC)
    // e.g. if block_size = 128, then we have 4 warps per block, each in charge of one output channel
    sycl::sub_group warp = id.get_sub_group();
    // meta_group_size is the number of warps in a block (e.g. 4), meta_group_rank is the warp index (0,1,2,3)
    int idx = id.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();
    if(idx >= OC) { return; }
    int BT = B * T; // number of elements to reduce in total, per channel
    // first, thread coarsening to sum reduce the problem size from B*T to 32
    float sum = 0.0f;
    for(int i = warp.get_local_linear_id(); i < BT; i += warp.get_max_local_range()[0]) {
        sum += dout[i * OC + idx];
    }
    // now do a warp-level reduce to get the sum across the 32 threads in this warp
    sum = sycl::reduce_over_group(warp, sum, sycl::plus<float>());
    // write the result to output (global memory)
    if(warp.leader()) {
        dbias[idx] += (floatX)sum;
    }
}


// ----------------------------------------------------------------------------
// kernel launcher

// version1: simple cuBLAS calls
void matmul_backward_bias1(sycl::queue &q, floatX* dbias, const floatX* dout,
                           int B, int T, int OC, int block_size) {
    const int block_dim = largestPowerOfTwoLessOrEqual(block_size);
    assert(isPowerOfTwo(block_size));
    const int grid_dim = OC;
    q.parallel_for(sycl::nd_range<1>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<1> id) {
        matmul_backward_bias_kernel1(id, dbias, dout, B, T, OC);
    }).wait();
}

void matmul_backward_bias2(sycl::queue &q, floatX* dbias, const floatX* dout,
                           int B, int T, int OC, int block_size) {
    // block_size 512 seems best
    const int grid_size = ceil_div(OC * 32, block_size);
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        matmul_backward_bias_kernel2(id, dbias, dout, B, T, OC);
    }).wait();
}

void matmul_backward_bias(int kernel_num,
                          sycl::queue &q,
                          floatX* dbias, floatX* dout,
                          int B, int T, int OC, int block_size) {
    switch (kernel_num) {
        case 1:
            matmul_backward_bias1(q, dbias, dout, B, T, OC, block_size);
            break;
        case 2:
            matmul_backward_bias2(q, dbias, dout, B, T, OC, block_size);
            break;
        default:
            std::cout << "Invalid kernel number\n";
            exit(1);
    }
}


// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    int OC = 768 * 4; // expansion of 4, e.g. in the MLP

    // set up the device
    sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order{});

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = std::atoi(argv[1]);
    }
    std::cout << "Using kernel " <<  kernel_num << '\n';

    // create host memory of random numbers
    float* dbias = make_zeros_float(OC);
    float* dout = make_random_float(B * T * OC);

    // move to GPU
    floatX* d_dbias;
    floatX* d_dout;

    d_dbias = sycl::malloc_device<floatX>(OC, q);
    d_dout = sycl::malloc_device<floatX>(B * T * OC, q);

    memcpy_convert(d_dbias, dbias, OC, q);
    memcpy_convert(d_dout, dout, B * T * OC, q);

    int block_sizes[] = {32, 64, 128, 256, 512};

    // calculate the CPU reference
    matmul_backward_bias_cpu(nullptr, nullptr, dbias, dout, nullptr, nullptr, B, T, C, OC);

    for (int block_size: block_sizes) {
        // memset the bias to zero
        q.memset(d_dbias, 0, OC * sizeof(floatX));
        // calculate the GPU version
        matmul_backward_bias(kernel_num, q, d_dbias, d_dout, B, T, OC, block_size);
        // compare
        std::cout << "Checking correctness...\n";
        float tol = std::is_same_v<floatX, float> ? 5e-3f : 1.0f;
        validate_result(d_dbias, dbias, "dbias", OC, tol);
        std::cout << "All results match for block_size " << block_size << '\n';
    }

    // now benchmark the kernel
    for (int block_size: block_sizes) {
        float *d_dinp, *d_dweight, *d_inp, *d_weight, *d_ones;
        int repeat_times = 2000;
        float elapsed_time = benchmark_kernel(repeat_times, matmul_backward_bias,
                                              kernel_num, q,
                                              d_dbias, d_dout, B, T, OC, block_size);
        std::cout << "block_size " << block_size << "time " << elapsed_time << " ms\n";
    }

    // cleanups
    free(dbias);
    free(dout);
    sycl::free(d_dbias, q);
    sycl::free(d_dout, q);

    return 0;
}