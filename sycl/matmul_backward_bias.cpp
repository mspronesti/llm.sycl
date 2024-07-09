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

void matmul_backward_bias_kernel3(sycl::nd_item<1> id, floatX* dbias, const floatX* dout, int B, int T, int OC,
                                  sycl::local_accessor<float> local_acc) {
    // dout is (B, T, OC), dbias is (OC)
    // in this version of the kernel the entire block of block_size is dedicated to one output channel
    sycl::sub_group warp = id.get_sub_group();

    int threadIdx_x = id.get_local_id(0);
    int blockDim_x = id.get_local_range(0);

    float* shared_sum = local_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw(); // block_size max is 1024 = 32 * 32 warps
    int BT = B * T; // number of elements to reduce in total, per channel
    int num_warps = blockDim_x / 32;
    int warp_id = threadIdx_x / 32;
    int lane_id = threadIdx_x % 32;
    int idx = id.get_group(0); // simply one block per row
    // round 1: thread coarsening to reduce the problem size from B*T to block_size
    float thread_sum = 0.0f;
    for(int i = threadIdx_x; i < BT; i += blockDim_x) {
        thread_sum += (float)dout[i * OC + idx];
    }
    // now do a warp-level reduce to get the sum across the 32 threads in each warp
    // reduce the problem size from block_size to block_size/32 i.e. `num_warps`
    float warp_sum = sycl::reduce_over_group(warp, thread_sum, sycl::plus<float>{});
    // store the warp sum in shared memory (we could have lane_id == 0 guard but not needed)
    shared_sum[warp_id] = warp_sum;
    sycl::group_barrier(id.get_group());
    // load results from shared memory to threads, pad with zeros for threads that are out of bounds
    warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    // now reduce the warp-level reductions
    float block_sum = sycl::reduce_over_group(warp, warp_sum, sycl::plus<float>{}); // sum(x)
    // write the result to output (global memory)
    if(threadIdx_x == 0) {
        dbias[idx] = (float)dbias[idx] + block_sum;
    }
}

// this kernel performs a column-wise reduction over dout, in PyTorch equivalent to:
// dbias = dout.sum((0,1))
// the idea is to employ one block to reduce along several columns,
// where each block has a width of 32 columns to ensure coalesced access.
// at the end we accumulate the reductions performed by the warps in each block via shared memory
void matmul_backward_bias_kernel4(sycl::nd_item<1> id, floatX* dbias, const floatX* dout, int B, int T, int OC,
                                  sycl::local_accessor<float> local_acc) {

    // this kernel is launched with 1D grid_dim of OC/32
    // for example let's say block_size is 128
    float* smem = local_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();// of size block_size (128)
    auto warp = id.get_sub_group();
    int warpSize = warp.get_max_local_range()[0];

    const int warp_id = id.get_local_id(0) / warpSize; // warp index in the block, 0,1,2,3
    const int lane_id = id.get_local_id(0) % warpSize; // thread index in the warp, 0,1,2,...,31
    const int tl = id.get_group(0) * warpSize; // pointer to the start column for this block
    const int vstep = id.get_local_range(0) / warpSize; // number of warps in a block, e.g. 4

    // pointer to the start of the column for one lane of threads
    // so e.g. 4 (`vstep`) threads (of the same lane_id) will reduce this one column
    const floatX* dout_col = dout + tl + lane_id;

    // column reductions by looping through the rows
    // each of the 4 threads offsets by its warp_id and then skips by vstep
    // together these 4 threads cover all B*T rows of this (lane_id) column
    // importantly, consecutive threads (in threadId) are processing adjacent columns,
    // leading to a coalesced memory access pattern
    float dout_sum = 0.0f;
    for (int row = warp_id; row < B * T; row += vstep) {
        dout_sum += (float)dout_col[row * OC];
    }
    smem[lane_id + warp_id * warpSize] = dout_sum;
    sycl::group_barrier(id.get_group());

    // warp_id 0 reduces the shared memory column-wise, linearly
    dout_sum = 0.0f;
    if (warp_id == 0) {
        for (int j = 0; j < vstep; j++) {
            dout_sum += smem[lane_id + j * warpSize];
        }
        dbias[tl + lane_id] = (float)dbias[tl + lane_id] + dout_sum;
    }
}

#ifndef ENABLE_BF16
void matmul_backward_bias_kernel5(sycl::nd_item<2> id, floatX* dbias, const floatX* dout, int B, int T, int OC) {
    //int oc = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = id.get_group(1) * id.get_local_range(1) + id.get_local_id(1);
    if(oc >= OC) return;
    float sum = 0.0;
    // grid-wide loop for maximum parallelism
    for (int i = id.get_group(0); i < B * T; i += id.get_group_range(0)) {
        sum += (float)dout[i * OC + oc];
    }
    // and atomically add everything together. atomics within one block are conflict-free!
    atomicAdd(dbias + oc, sum);
}
#endif


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

void matmul_backward_bias3(sycl::queue &q, floatX* dbias, const floatX* dout,
                           int B, int T, int OC, int block_size) {
    // block_size 256 seems best
    //matmul_backward_bias_kernel3<<<OC, block_size>>>(dbias, dout, B, T, OC);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float> local_acc(32, h);
        h.parallel_for(sycl::nd_range<1>(OC * block_size, block_size), [=](sycl::nd_item<1> id) {
            matmul_backward_bias_kernel3(id, dbias, dout, B, T, OC, local_acc);
        });
    }).wait();
}

void matmul_backward_bias4(sycl::queue &q, floatX* dbias, const floatX* dout,
                           int B, int T, int OC, int block_size) {

    assert(OC % 32 == 0); // OC must be divisible by 32 for this kernel
    const int grid_size = OC / 32;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float> local_acc(block_size, h);
        h.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
            matmul_backward_bias_kernel4(id, dbias, dout, B, T, OC, local_acc);
        });
    }).wait();
}

#ifndef ENABLE_BF16
void matmul_backward_bias5(sycl::queue &q, floatX* dbias, const floatX* dout,
                           int B, int T, int OC, int block_size) {

    const int max_CUs = q.get_device().get_info<sycl::info::device::max_compute_units>();
    const int max_wgs = q.get_device().get_info<sycl::info::device::max_work_group_size>();

    const int grid_size_x = ceil_div(OC, block_size);
    const int grid_size_y = std::max(1, max_CUs * max_wgs / block_size);

    sycl::range<2> block_dim(1, block_size);
    sycl::range<2> grid_dim(grid_size_y, grid_size_x);

    q.parallel_for(sycl::nd_range<2>(grid_dim*block_dim, block_dim), [=](sycl::nd_item<2> id) [[sycl::reqd_sub_group_size(32)]] {
        matmul_backward_bias_kernel5(id, dbias, dout, B, T, OC);
    }).wait();
}
#endif


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
        case 3:
            matmul_backward_bias3(q, dbias, dout, B, T, OC, block_size);
            break;
        case 4:
            matmul_backward_bias4(q, dbias, dout, B, T, OC, block_size);
            break;
#ifndef ENABLE_BF16
        case 5:
            matmul_backward_bias5(q, dbias, dout, B, T, OC, block_size);
            break;
#endif
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
        std::cout << "block_size " << block_size << " time " << elapsed_time << " ms\n";
    }

    // cleanups
    free(dbias);
    free(dout);
    sycl::free(d_dbias, q);
    sycl::free(d_dout, q);

    return 0;
}