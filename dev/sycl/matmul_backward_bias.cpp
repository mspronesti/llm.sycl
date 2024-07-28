#include <sycl/sycl.hpp>
#include <iostream>
#include <cstdlib>
#include "oneapi/mkl/blas.hpp"
#include "mkl.h"
#include <cassert>

#define ENABLE_BF16
#include "common.hpp"

namespace oneapi_ext = sycl::ext::oneapi::experimental;

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
float* dbias_buffer;

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

void cast_and_add_kernel(sycl::nd_item<1> id, floatX* dst, const float* src, size_t n) {
    // used only for matmul_backward_bias kernel, a little bit embarassing TODO delete later
    const size_t idx = id.get_global_id(0);
    if (idx < n) { dst[idx] = (floatX)((float)dst[idx] + src[idx]); } // have to += because dbias is a paramater
}

void matmul_backward_bias_kernel7(sycl::nd_item<2> id, float* dbias, const floatX* dout, int B, int T, int OC, const int block_size,
                                  sycl::local_accessor<float> local_acc) {
    // note: this kernel reads in floatX, but it writes to float!
    // this is because we're using atomics, which are super slow in < fp32 precision on < H100 GPUs
    // so the trick is do fp32 atomics to a buffer, and then copy_and_cast the result to floatX
    // (this also results in higher accuracy than doing accumulation directly in floatX)

    // see comments in matmul_backward() for an explanation of block/grid dimensions etc.
    const int block_size_x = 32;
    const int block_size_y = block_size / block_size_x; // 16
    const int OC_per_warp = block_size_x * x128::size;  // 256 at BF16

    int threadIdx_x = id.get_local_id(1);
    int threadIdx_y = id.get_local_id(0);
    int blockIdx_x = id.get_group(1);
    int blockIdx_y = id.get_group(0);

    int local_oc = threadIdx_x * x128::size;
    int global_oc = id.get_group(1) * OC_per_warp + local_oc;
    float accumulators[x128::size];

    float* shared = local_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();

    for (int k = 0; k < x128::size; k++) {
        accumulators[k] = 0.0f;
    }
    int thread_id = threadIdx_y * block_size_x + threadIdx_x;
    for (int idx = thread_id; idx < OC_per_warp; idx += block_size) {
        shared[idx] = 0.0f;
    }
    sycl::group_barrier(id.get_group());
    if(global_oc < OC) {
        for (int idx = blockIdx_y*block_size_y + threadIdx_y; idx < B * T; idx += id.get_group_range(0)*block_size_y) {
            x128 packed_dout = load128(dout + global_oc + idx*OC);
            for (int k = 0; k < x128::size; k++) {
                accumulators[k] += (float)packed_dout[k];
            }
        }
        // we need to avoid shared memory bank conflicts for the atomicAdd to maximise performance,
        // so we accumulate in a conflict-free order, then reorder to match the global memory order
        for (int k = 0; k < x128::size; k++) {
            atomicAdd(shared + threadIdx_x + (k * block_size_x), accumulators[k]);
        }
    }
    if (threadIdx_y >= x128::size) { return; } // only need this many warps to reorder the data
    sycl::group_barrier(id.get_group());
    // read the accumulated values in the conflict-free order
    int i = threadIdx_x + (threadIdx_y * block_size_x);
    float tmp = shared[i];
    sycl::group_barrier(id.get_group());
    // write them back to shared memory in the global memory order
    // 8-way bank conflict for BF16 x128, but only 8x per threadblock (rather than 8x per warp)
    shared[local_oc + threadIdx_y] = tmp;
    sycl::group_barrier(id.get_group());
    // now we do a perfectly coalesced atomic add to global memory (1x 128-byte cacheline per warp)
    if (i + blockIdx_x*OC_per_warp < OC) {
        atomicAdd(dbias + i + blockIdx_x*OC_per_warp, shared[i]);
    }
}

// We want to decrease the amount of channels handled by each block, so that we need fewer across-block reductions.
// We do this by realizing the following: For scalar memory access, we need to read one element per thread in a warp
// to read an entire cacheline, but for vectorized memory access, with 128 bit of data per thread, we only need eight
// threads to fetch a cacheline, which means that we can already operate on a "depth" of four within a single warp.
// => blockDim.x == 4, blockDim.y == 32/4 = 8
//
template<typename OutFloat, bool Atomic>
void matmul_backward_bias_kernel8(sycl::nd_item<3> id, OutFloat* dbias, const floatX* dout, int B, int T, int OC,
                                  std::bool_constant<Atomic>, sycl::local_accessor<float> local_acc) {
    sycl::sub_group warp = id.get_sub_group();
    constexpr const int bdx = 4;
    constexpr const int bdy = 32 / bdx;


    int warp_d = (int)id.get_local_id(2);
    int warp_c = (int)id.get_local_id(1);
    int block_d = (int)id.get_local_id(0);

    const int OC_per_warp = bdy * x128::size;  // 64 at BF16

    int local_oc = warp_c * x128::size;
    int global_oc = id.get_group(2) * OC_per_warp + local_oc;

    int local_bt = warp_d + bdx * block_d;
    int bt_per_block = bdx * id.get_local_range(0);

    float accumulators[x128::size];
    for (int k = 0; k < x128::size; k++) {
        accumulators[k] = 0.0f;
    }

    if(global_oc < OC) {
        // sum up over all bt within registers
        for (int idx = id.get_group(1) * bt_per_block + local_bt; idx < B * T; idx += id.get_group_range(1) * bt_per_block) {
            x128 packed_dout = load128(dout + global_oc + idx*OC);
            for (int k = 0; k < x128::size; k++) {
                accumulators[k] += (float)packed_dout[k];
            }
        }
    }

    float* shared = local_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
    float (*sub_results)[32][bdy] = (float (*)[32][bdy])shared;
    // reference: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_non_uniform_groups.asciidoc#creation-1
    auto group_partition_4 = oneapi_ext::get_fixed_size_group<4>(warp);

    // reduce within-warp results
    for (int k = 0; k < x128::size; k++) {
        float v = accumulators[k];
        v = sycl::reduce_over_group(group_partition_4, v, sycl::plus<float>());
        if(warp_d == 0) {
            sub_results[k][block_d][warp_c] = v;
        }
    }
    sycl::group_barrier(id.get_group());

    // block-wide reductions
    for (int k = block_d; k < x128::size; k += id.get_local_range(0)) {
        float a = 0.f;
        for (int r = warp_d; r < id.get_local_range(0); r += bdx) {
            float v = sub_results[k][r][warp_c];
            v = sycl::reduce_over_group(group_partition_4, v, sycl::plus<float>());
            a += v;
        }
        if(warp_d == 0 && global_oc < OC) {
            // coalesced, but not cacheline-sized
            if constexpr (!Atomic) {
                dbias[global_oc + k] = (OutFloat)(a + (float)dbias[global_oc + k]);
            } else {
                atomicAdd(dbias + global_oc + k, a);
            }
        }
    }
}

// Like kernel 8, but instead of accumulating to the auxiliary buffer, it writes
// multiple values that need to be summed up in a separate kernel call.
// If UseAuxBuffer is false, gridDim.y has to be one, and results are added directly
// to dbias.
template<typename OutFloat, bool UseAuxBuffer>
void matmul_backward_bias_kernel9(sycl::nd_item<3> id, OutFloat* dbias, const floatX* dout, int B, int T, int OC,
                                  std::bool_constant<UseAuxBuffer>, sycl::local_accessor<float> local_acc) {

    sycl::sub_group warp = id.get_sub_group();
    constexpr const int bdx = 4;
    constexpr const int bdy = 32 / bdx;


    int warp_d = (int)id.get_local_id(2);
    int warp_c = (int)id.get_local_id(1);
    int block_d = (int)id.get_local_id(0);

    const int OC_per_warp = bdy * x128::size;  // 64 at BF16

    int local_oc = warp_c * x128::size;
    int global_oc = id.get_group(2) * OC_per_warp + local_oc;

    int local_bt = warp_d + bdx * block_d;
    int bt_per_block = bdx * id.get_local_range(0);

    float accumulators[x128::size];
    for (int k = 0; k < x128::size; k++) {
        accumulators[k] = 0.0f;
    }

    if(global_oc < OC) {
        // sum up over all bt within registers
        for (int idx = id.get_group(1) * bt_per_block + local_bt; idx < B * T; idx += id.get_group_range(1) * bt_per_block) {
            x128 packed_dout = load128(dout + global_oc + idx*OC);
            for (int k = 0; k < x128::size; k++) {
                accumulators[k] += (float)packed_dout[k];
            }
        }
    }

    float* shared = local_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
    float (*sub_results)[32][bdy] = (float (*)[32][bdy])shared;
    // reference: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_non_uniform_groups.asciidoc#creation-1
    auto group_partition_4 = oneapi_ext::get_fixed_size_group<4>(warp);

    // reduce within-warp results
    for (int k = 0; k < x128::size; k++) {
        float v = accumulators[k];
        v = sycl::reduce_over_group(group_partition_4, v, sycl::plus<float>());
        if(warp_d == 0) {
            sub_results[k][block_d][warp_c] = v;
        }
    }
    sycl::group_barrier(id.get_group());

    // block-wide reductions
    for (int k = block_d; k < x128::size; k += id.get_local_range(0)) {
        float a = 0.f;
        for (int r = warp_d; r < id.get_local_range(0); r += bdx) {
            float v = sub_results[k][r][warp_c];
            v = sycl::reduce_over_group(group_partition_4, v, sycl::plus<float>());
            a += v;
        }
        if(warp_d == 0 && global_oc < OC) {
            // coalesced, but not cacheline-sized
            if constexpr (!UseAuxBuffer) {
                dbias[global_oc + k] = (OutFloat)(a + (float)dbias[global_oc + k]);
            } else {
                dbias[global_oc + k + id.get_group(1) * OC] = a;
            }
        }
    }
}

void reduce_add_sum_kernel(sycl::nd_item<1> id, floatX* dst, const float* src, size_t n, size_t m) {
    const size_t idx = id.get_global_id(0) * f128::size;
    assert(n % x128::size == 0);
    if (idx < n) {
        f128 acc;
        for(int k = 0; k < f128::size; ++k) {
            acc[k] = 0.f;
        }

        for(int l = 0; l < m; ++l) {
            f128 s = load128(src + idx + n * l);
            for(int k = 0; k < f128::size; ++k) {
                acc[k] += s[k];
            }
        }
        for(int k = 0; k < f128::size; ++k) {
            dst[idx + k] = (floatX) ((float)dst[idx + k] + acc[k]);
        }
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

void matmul_backward_bias7(sycl::queue &q, floatX* dbias, const floatX* dout,
                           int B, int T, int OC, int block_size) {
    if(block_size < 256) {
        block_size = 256;
    }
    if (block_size > 256) {
        // larger block sizes provokes the early exit problem on Xe1.
        block_size = 256;
    }
    // Each warp is responsible for 32 * "x128::size" = 256 OCs at BF16 (OC must be a multiple of 256!)
    // Block size is 512 threads (16 warps) and we reduce those 16 values into 1 at the end
    // blockDim.x is 32 --> single warp being responsible for those 256 OCs
    // blockDim.y is 16 --> 16 parallel independent warps processing the same OCs for different BTs
    // gridDim.x is OC / 256 --> each block processes 256 OCs
    // grimDim.y is max(1, (cuda_num_SMs * threads_per_SM) / (512 * gridDim.x)); --> fill up the entire GPU!
    const int warp_size = 32;
    const int OC_per_warp = warp_size * x128::size; // 256 at BF16
    const int block_size_x = 32;
    const int block_size_y = block_size / block_size_x; // 16
    const int grid_size_x = ceil_div(OC, OC_per_warp); // e.g. 3 horizontal blocks for 768 OCs at BF16

    int max_CUs = q.get_device().get_info<sycl::info::device::max_compute_units>();
    int max_wgs = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    const int grid_size_y = std::max(1, (max_CUs * max_wgs) / block_size);

    assert(block_size_y >= x128::size);

    sycl::range<2> block_dim(block_size_y, block_size_x);
    sycl::range<2> grid_dim(grid_size_y, grid_size_x);

    // this is needed because dbias_buffer is a non const global variable
    // and can't be passed to the kernel directly in SYCL
    float* dbias_buffer_loc = dbias_buffer;

    q.memset(dbias_buffer_loc, 0, OC * sizeof(float)).wait();
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float> local_acc(OC_per_warp, h);
        h.parallel_for(sycl::nd_range<2>(grid_dim*block_dim, block_dim), [=](sycl::nd_item<2> id) [[sycl::reqd_sub_group_size(32)]]{
            matmul_backward_bias_kernel7(id, dbias_buffer_loc, dout, B, T, OC, block_size, local_acc);
        });
    }).wait();

    q.parallel_for(sycl::nd_range<1>(ceil_div(OC, 256)*256, 256), [=](sycl::nd_item<1> id){
        cast_and_add_kernel(id, dbias, dbias_buffer_loc, OC);
    }).wait();
}

void matmul_backward_bias8(sycl::queue &q, floatX* dbias, const floatX* dout,
                           int B, int T, int OC, int block_size) {
    sycl::range<3> block_dim((unsigned)block_size/32, 8, 4);
    const int OC_per_warp = block_dim[1] * x128::size; // 64 at BF16
    const int grid_size_x = ceil_div(OC, OC_per_warp); // e.g. 12 horizontal blocks for 768 OCs at BF16
    int max_CUs = q.get_device().get_info<sycl::info::device::max_compute_units>();
    int max_wgs = q.get_device().get_info<sycl::info::device::max_work_group_size>();

    const int grid_size_y = std::max(1, max_CUs * max_wgs / (block_size * grid_size_x)); // full GPU!

    sycl::range<3> grid_dim(1, grid_size_y, grid_size_x);

    // If we have enough OC that we don't need cross-block reductions, we can skip the bias_buffer accumulation
    // and write results directly to the output.
    if(grid_size_y == 1) {
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float> local_acc(x128::size*32*8, h);
            h.parallel_for(sycl::nd_range<3>(grid_dim*block_dim, block_dim), [=](sycl::nd_item<3> id) [[sycl::reqd_sub_group_size(32)]] {
                matmul_backward_bias_kernel8(id, dbias, dout, B, T, OC, std::bool_constant<false>{}, local_acc);
            });
        }).wait();
    } else {
        // this is needed because dbias_buffer is a non const global variable
        // and can't be passed to the kernel directly in SYCL
        float* dbias_buffer_loc = dbias_buffer;

        q.memset(dbias_buffer_loc, 0, OC * sizeof(float));
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float> local_acc(x128::size*32*8, h);
            h.parallel_for(sycl::nd_range<3>(grid_dim*block_dim, block_dim), [=](sycl::nd_item<3> id) [[sycl::reqd_sub_group_size(32)]] {
                matmul_backward_bias_kernel8(id, dbias_buffer_loc, dout, B, T, OC, std::bool_constant<true>{}, local_acc);
            });
        });
        q.parallel_for(sycl::nd_range<1>(ceil_div(OC, 256)*256, 256), [=](sycl::nd_item<1> id) {
            cast_and_add_kernel(id, dbias, dbias_buffer_loc, OC);
        });
        q.wait();
    }
}

void matmul_backward_bias9(sycl::queue &q, floatX* dbias, const floatX* dout,
                           int B, int T, int OC, int block_size) {
    sycl::range<3> block_dim((unsigned)block_size/32, 8, 4);
    const int OC_per_warp = block_dim[1] * x128::size; // 64 at BF16
    const int grid_size_x = ceil_div(OC, OC_per_warp); // e.g. 12 horizontal blocks for 768 OCs at BF16
    int max_CUs = q.get_device().get_info<sycl::info::device::max_compute_units>();
    int max_wgs = q.get_device().get_info<sycl::info::device::max_work_group_size>();

    const int grid_size_y = std::max(1, max_CUs * max_wgs / (block_size * grid_size_x)); // full GPU!

    sycl::range<3> grid_dim(1, grid_size_y, grid_size_x);

    // If we have enough OC that we don't need cross-block reductions, we can skip the bias_buffer accumulation
    // and write results directly to the output.
    if(grid_size_y == 1) {
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float> local_acc(x128::size*32*8, h);
            h.parallel_for(sycl::nd_range<3>(grid_dim*block_dim, block_dim), [=](sycl::nd_item<3> id) [[sycl::reqd_sub_group_size(32)]] {
                matmul_backward_bias_kernel9(id, dbias, dout, B, T, OC, std::bool_constant<false>{}, local_acc);
            });
        }).wait();
    } else {
        // kernel 9 overwrites temp buffer, so no need to memset
        float *dbias_buffer_loc = dbias_buffer;
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float> local_acc(x128::size*32*8, h);
            h.parallel_for(sycl::nd_range<3>(grid_dim*block_dim, block_dim), [=](sycl::nd_item<3> id) [[sycl::reqd_sub_group_size(32)]] {
                matmul_backward_bias_kernel9(id, dbias_buffer_loc, dout, B, T, OC, std::bool_constant<true>{}, local_acc);
            });
        });
        q.parallel_for(sycl::nd_range<1>(ceil_div(OC, 256*f128::size)*256, 256), [=](sycl::nd_item<1> id) {
            reduce_add_sum_kernel(id, dbias, dbias_buffer_loc, OC, grid_size_y);
        });
        q.wait();
    }
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
        case 7:
            matmul_backward_bias7(q, dbias, dout, B, T, OC, block_size);
            break;
        case 8:
            matmul_backward_bias8(q, dbias, dout, B, T, OC, block_size);
            break;
        case 9:
            matmul_backward_bias9(q, dbias, dout, B, T, OC, block_size);
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
    dbias_buffer = sycl::malloc_device<float>(OC * 32, q);

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
    sycl::free(dbias_buffer, q);

    return 0;
}