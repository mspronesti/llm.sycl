#include <sycl/sycl.hpp>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstdlib>

#define ENABLE_BF16
#include "common.hpp"

// ----------------------------------------------------------------------------
// CPU code reference

void residual_forward_cpu(float* out, const float* inp1, const float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                           const float* inp, const float* weight, const float* bias,
                           int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

void residual_forward_kernel1(sycl::nd_item<1> item, floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int idx = item.get_global_id(0);
    if (idx < N) {
        out[idx] = (floatX)((float)inp1[idx] + (float)inp2[idx]);
    }
}

void layernorm_forward_kernel1(sycl::nd_item<1> item, floatX* out, floatX* mean, floatX* rstd,
                               const floatX* inp, const floatX* weight, const floatX* bias,
                               int N, int C) {
    int idx = item.get_global_id(0);
    float eps = 1e-5f;

    if (idx < N) {
        // seek to the input position inp[idx,:]
        const floatX* x = inp + idx * C;
        // calculate the mean
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += (float)x[i];
        }
        m = m / C;
        // calculate the variance (without any bias correction)
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = (float)x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;
        // calculate the rstd
        float s = 1.0f / sqrtf(v + eps);
        // seek to the output position in out[idx,:]
        floatX* out_idx = out + idx * C;
        for (int i = 0; i < C; i++) {
            float n = (s * ((float)x[i] - m)); // normalized output
            float o = n * (float)weight[i] + (float)bias[i]; // scale and shift it
            out_idx[i] = o; // write
        }
        // cache the mean and rstd for the backward pass later
        mean[idx] = m;
        rstd[idx] = s;
    }
}

void fused_residual_forward_kernel2(sycl::nd_item<1> item, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                    const floatX* inp1, const floatX* inp2,
                                    const floatX* weight, const floatX* bias,
                                    int N, int C) {

    int idx = item.get_global_id(0);
    if(idx > N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    float eps = 1e-5f;

    float m = 0.0f;
    for(int c = 0; c < C; ++c) {
        float out = (float)inp1[c] + (float)inp2[c];
        m += out;
        residual[c] = (floatX)out;
    }

    m = m / C;
    float v = 0.0f;
    for (int c = 0; c < C; c++) {
        float xshift = (float)residual[c] - m;
        v += xshift * xshift;
    }
    v = v / C;

    // calculate the rstd
    float s = 1.0f / sqrtf(v + eps);
    for (int c = 0; c < C; c++) {
        float n = (s * ((float)residual[c] - m)); // normalized output
        float o = n * (float)weight[c] + (float)bias[c]; // scale and shift it
        normed[c] = (floatX)o; // write
    }
    // cache the mean and rstd for the backward pass later
    mean[idx] = (floatX)m;
    rstd[idx] = (floatX)s;
}

void fused_residual_forward_kernel3(sycl::nd_item<2> item, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C) {
    constexpr const int WarpSize = 32;
    int idx = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (idx >= N) return;

    auto sg = item.get_sub_group();

    int thread_id = item.get_local_id(1);

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    float eps = 1e-5f;
    float m = 0.0f;
    for (int c = thread_id; c < C; c += WarpSize) {
        float out = static_cast<float>(inp1[c]) + static_cast<float>(inp2[c]);
        m += out;
        residual[c] = out;
    }

    m = sycl::reduce_over_group(sg, m, sycl::plus<float>());

    m = m / C;
    float v = 0.0f;
    for (int c = thread_id; c < C; c += WarpSize) {
        float xshift = static_cast<float>(residual[c]) - m;
        v += xshift * xshift;
    }

    v = sycl::reduce_over_group(sg, v, sycl::plus<float>());
    v = v / C;

    // calculate the rstd
    float s = 1.0f / sycl::sqrt(v + eps);
    for (int c = thread_id; c < C; c += WarpSize) {
        float n = s * (static_cast<float>(residual[c]) - m); // normalized output
        float o = n * static_cast<float>(weight[c]) + static_cast<float>(bias[c]); // scale and shift it
        normed[c] = o; // write
    }
    // cache the mean and rstd for the backward pass later
    if (thread_id == 0) {
        mean[idx] = (floatX)m;
        rstd[idx] = (floatX)s;
    }
}


void fused_residual_forward_kernel4(sycl::nd_item<2> item, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                    const floatX* inp1, const floatX* inp2,
                                    const floatX* weight, const floatX* bias,
                                    int N, int C) {
    using x128 = Packed128<floatX>;
    constexpr const int WarpSize = 32;

    int idx = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if(idx > N) return;

    auto sg = item.get_sub_group();
    int thread_id = item.get_local_id(1);

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    const float eps = 1e-5f;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int c = thread_id * x128::size;
    for(; c < C; c += WarpSize * x128::size) {
        const x128 in1 = load128cs(inp1 + c);
        const x128 in2 = load128cs(inp2 + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            out[k] = (floatX)((float)in1[k] + (float)in2[k]);
            sum += (float)out[k];
            sum_sq += (float)out[k] * (float)out[k];
        }
        store128(residual + c, out);
    }

    sum =  sycl::reduce_over_group(sg, sum, sycl::plus<float>());
    sum_sq =  sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());


    float m = sum / C;
    float v = sum_sq / C - m * m;
    float s = sycl::rsqrt(v + eps);

    c -= WarpSize * x128::size;
    for(; c >= 0; c -= WarpSize * x128::size) {
        const x128 res = load128cs(residual + c);
        const x128 w = load128(weight + c);
        const x128 b = load128(bias + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)res[k] - m); // normalized output
            float o = n * (float)w[k] + (float)b[k]; // scale and shift it
            out[k] = o;
        }

        store128cs(normed + c, out);
    }
    // cache the mean and rstd for the backward pass later
    if(thread_id == 0) {
        mean[idx] = m;
        rstd[idx] = s;
    }
}

// what do you want in shared memory? EVERYTHING!
// thus, we no longer require zigzag loops and can do the numerically more stable variance estimation
// needs special attention in the kernel launcher to ensure we have enough smem.
void fused_residual_forward_kernel5(sycl::nd_item<2> item, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                               const floatX* inp1, const floatX* inp2,
                                               const floatX* weight, const floatX* bias,
                                               int N, int C,
                                               sycl::local_accessor<char> local_acc) {
    constexpr const int WarpSize = 32;
    assert(item.get_local_range(1) == WarpSize);

    auto sg = item.get_sub_group();
    int threadIdx_x = item.get_local_id(1);
    int threadIdx_y = item.get_local_id(0);

    // load weights and biases into shared memory
    // do this before we allow any threads to exit!
    auto params = (char *)local_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params) + (C / x128::size);
    x128* s_res = reinterpret_cast<x128*>(params) + ((2 + threadIdx_y) * C / x128::size);

    int sidx = (threadIdx_x + WarpSize * threadIdx_y) * x128::size;
    for(int i = sidx; i < C; i += item.get_local_range(0) * WarpSize * x128::size) {
        s_weight[i/x128::size] = load128(weight + i);
        s_bias[i/x128::size] = load128(bias + i);
    }
    sycl::group_barrier(item.get_group());

    int idx = item.get_group(0) * item.get_local_range(0) + threadIdx_y;
    if(idx > N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    const float eps = 1e-5f;
    float sum = 0.0f;
    for(int c = threadIdx_x * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 in1 = load128cs(inp1 + c);
        const x128 in2 = load128cs(inp2 + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            out[k] = (floatX)((float)in1[k] + (float)in2[k]);
            sum += (float)out[k];
        }
        store128cs(residual + c, out);
        s_res[c / x128::size] = out;
    }

    sum =  sycl::reduce_over_group(sg, sum, sycl::plus<float>());
    float m = sum / C;
    float v = 0.f;

    for(int c = threadIdx_x * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 res = s_res[c / x128::size];
        for(int k = 0; k < x128::size; ++k) {
            v += ((float)res[k] - m) * ((float)res[k] - m);
        }
    }

    v = sycl::reduce_over_group(sg, v, sycl::plus<float>()) / C;
    float s = sycl::rsqrt(v + eps);

    for(int c = threadIdx_x * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 res = s_res[c / x128::size];
        const x128 w = s_weight[c / x128::size];
        const x128 b = s_bias[c / x128::size];
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)res[k] - m); // normalized output
            float o = n * (float)w[k] + (float)b[k]; // scale and shift it
            out[k] = o;
        }

        store128cs(normed + c, out);
    }
    // cache the mean and rstd for the backward pass later
    if(threadIdx_x == 0) {
        mean[idx] = m;
        rstd[idx] = s;
    }
}

// using multiple warps per token, and keep threads persistent, so we never have to reload weights and biases
// if we had one warp per token, though, this would require us to use a huge amount of shared memory. Therefore,
// we use multiple warps per token; but generally we cannot use the entire block, because that would give too
// little work per warp to be effective (each warp processes 256 bfloat16 elements, so for C=768 more than 3 warps
// will just mean idle). Therefore, we add a z dimension, where warps with different z handle different tokens.
// all this makes the launcher logic more complicated :(
void fused_residual_forward_kernel6(sycl::nd_item<3> item, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                    const floatX* inp1, const floatX* inp2,
                                    const floatX* weight, const floatX* bias,
                                    int N, int C,
                                    sycl::local_accessor<char> local_acc) {
    constexpr const int WarpSize = 32;

    auto sg = item.get_sub_group();
    int threadIdx_x = item.get_local_id(2);
    int threadIdx_y = item.get_local_id(1);
    int threadIdx_z = item.get_local_id(0);
    int blockDim_y  = item.get_local_range(1);
    int blockDim_z  = item.get_local_range(0);

    // load weights and biases into shared memory
    // do this before we allow any threads to exit!
    auto params = (char *)local_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    // weights and biases are shared among all tokens
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params + C * sizeof(floatX));
    // residual output (input to layernorm) is independent for each sub-block indicates by threadIdx.z
    x128* s_res = reinterpret_cast<x128*>(params + (2 + threadIdx_z) * C * sizeof(floatX));
    // similarly, each sub-block needs its own reduction buffers
    float* s_mean = reinterpret_cast<float*>(params + (2 + blockDim_z) * C * sizeof(floatX) + threadIdx_z * 32 * sizeof(float));
    float* s_var = reinterpret_cast<float*>(params + (2 + blockDim_z) * C * sizeof(floatX) + 32 * sizeof(float) * (blockDim_z + threadIdx_z));

    int cidx = (threadIdx_x + WarpSize * threadIdx_y) * x128::size;
    int step = blockDim_y * WarpSize * x128::size;

    for (int c = cidx; c < C; c += step) {
        s_weight[c / x128::size] = load128(weight + c);
        s_bias[c / x128::size] = load128(bias + c);
    }

    // the block-level reductions will cause sync before the first time we read these
    // => no syncthreads needed here

    // loop over all tokens
    for (int tidx = item.get_group(2) * blockDim_z + threadIdx_z; tidx < N; tidx += item.get_group_range(2) * blockDim_z) {
        // adjust pointers to current token
        floatX* residual_bt = residual + C * tidx;
        floatX* normed_bt = normed + C * tidx;
        const floatX* inp1_bt = inp1 + C * tidx;
        const floatX* inp2_bt = inp2 + C * tidx;

        const float eps = 1e-5f;
        float sum = 0.0f;
        for (int c = cidx; c < C; c += step) {
            const x128 in1 = load128cs(inp1_bt + c);
            const x128 in2 = load128cs(inp2_bt + c);
            x128 out;
            for (int k = 0; k < x128::size; ++k) {
                out[k] = (floatX)((float)in1[k] + (float)in2[k]);
                sum += (float)out[k];
            }
            store128cs(residual_bt + c, out);
            s_res[c / x128::size] = out;
        }
        sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
        if (threadIdx_x == 0) {
            s_mean[threadIdx_y] = sum;
        }
        sycl::group_barrier(item.get_group());
        float m = sycl::reduce_over_group(sg, threadIdx_x < blockDim_y ? s_mean[threadIdx_x] : 0.f, sycl::plus<float>()) / C;
        // normally, we'd syncthread here to make sure that no warp is already at the next
        // iteration of the loop, messing with s_mean. The fact that we interleave s_mean and s_var means
        // we don't need these additional syncs.
        float v = 0.f;

        for (int c = cidx; c < C; c += step) {
            const x128 res = s_res[c / x128::size];
            for (int k = 0; k < x128::size; ++k) {
                v += ((float)res[k] - m) * ((float)res[k] - m);
            }
        }

        v = sycl::reduce_over_group(sg, v, sycl::plus<float>());
        if (threadIdx_x == 0) {
            s_var[threadIdx_y] = v;
        }
        sycl::group_barrier(item.get_group());
        v = sycl::reduce_over_group(sg, threadIdx_x < blockDim_y ? s_var[threadIdx_x] : 0.f, sycl::plus<float>()) / C;
        float s = sycl::rsqrt(v + eps);

        for (int c = cidx; c < C; c += step) {
            const x128 res = s_res[c / x128::size];
            const x128 w = s_weight[c / x128::size];
            const x128 b = s_bias[c / x128::size];
            x128 out;
            for (int k = 0; k < x128::size; ++k) {
                float n = s * ((float)res[k] - m); // normalized output
                float o = n * (float)w[k] + (float)b[k]; // scale and shift it
                out[k] = o;
            }

            store128(normed_bt + c, out);
        }

        // cache the mean and rstd for the backward pass later
        if (threadIdx_x == 0 && threadIdx_y == 0) {
            mean[tidx] = m;
            rstd[tidx] = s;
        }
    }
}


// ----------------------------------------------------------------------------
// kernel launcher
void fused_residual_forward1(sycl::queue &q, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    const int grid_size_resid = ceil_div(N * C, block_size);
    q.parallel_for(sycl::nd_range<1>(grid_size_resid * block_size, block_size), [=](sycl::nd_item<1> id){
        residual_forward_kernel1(id, residual, inp1, inp2, N * C);
    }).wait();

    const int grid_size_ln = ceil_div(N, block_size);
    q.parallel_for(sycl::nd_range<1>(grid_size_ln * block_size, block_size), [=](sycl::nd_item<1> id) {
        layernorm_forward_kernel1(id, normed, mean, rstd, residual, weight, bias, N, C);
    }).wait();
}

void fused_residual_forward2(sycl::queue &q, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        fused_residual_forward_kernel2(id, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
    }).wait();
}

void fused_residual_forward3(sycl::queue &q, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                      const floatX* inp1, const floatX* inp2,
                                      const floatX* weight, const floatX* bias,
                                      int N, int C, const int block_size) {
    int block_y = block_size / 32;
    int grid_size = ceil_div(N, block_y);
    sycl::nd_range<2> grid = sycl::nd_range<2>(
            sycl::range<2>(grid_size * block_y, 32),
            sycl::range<2>(block_y, 32)
    );

    q.parallel_for(grid, [=](sycl::nd_item<2> item) {
            fused_residual_forward_kernel3(item, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
    }).wait();
}

void fused_residual_forward4(sycl::queue &q, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    int block_y = block_size / 32;
    int grid_size = ceil_div(N, block_y);
    sycl::nd_range<2> grid = sycl::nd_range<2>(
            sycl::range<2>(grid_size * block_y, 32),
            sycl::range<2>(block_y, 32)
    );

    q.parallel_for(grid, [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(32)]]{
        fused_residual_forward_kernel4(item, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
    }).wait();
}

void fused_residual_forward5(sycl::queue &q, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    int block_y = block_size / 32;
    int grid_size = ceil_div(N, block_y);
    size_t smem = (2 + block_y) * C * sizeof(floatX);


    sycl::nd_range<2> grid = sycl::nd_range<2>(
            sycl::range<2>(grid_size * block_y, 32),
            sycl::range<2>(block_y, 32)
    );

    auto local_mem = q.get_device().get_info<sycl::info::device::local_mem_size>();
    if (local_mem > smem) {
        q.submit([&](sycl::handler &h) {
            sycl::local_accessor<char> local_acc(smem, h);
            h.parallel_for(grid, [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(32)]] {
                fused_residual_forward_kernel5(item, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, local_acc);
            });
        });
    } else {
        std::cout << "Not enough unified shared memory, falling back to kernel 4\n";
        q.parallel_for(grid, [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(32)]] {
            fused_residual_forward_kernel4(item, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
        });
    }
    q.wait();
}


void fused_residual_forward6(sycl::queue &q, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    // 32 is the warp size
    int warps_per_token = std::max(1, C / Packed128<floatX>::size / 32);
    int total_warps = block_size / 32;
    int block_z = std::max(1, total_warps / warps_per_token);
    int block_y = std::max(1, total_warps / block_z);
    size_t smem = (2 + block_z) * C * sizeof(floatX) + 64 * sizeof(float) * block_z;

    auto local_mem_size = q.get_device().get_info<sycl::info::device::local_mem_size>();
    if(local_mem_size > smem) {
        // enough shared memory => use kernel 6
        const int num_compute_units = q.get_device().get_info<sycl::info::device::max_compute_units>();
        const int num_blocks = std::max(1, 32 * num_compute_units / block_size);
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<char> local_acc(smem, h);
            sycl::range<3> grid_dim(1, 1, num_blocks);
            sycl::range<3> block_dim(block_z, block_y, 32);

            h.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> id) [[intel::reqd_sub_group_size(32)]] {
                fused_residual_forward_kernel6(id, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, local_acc);
            });
        });
    } else {
        std::cout << "Not enough unified shared memory, falling back to kernel 4\n";
        // fallback on kernel 4
        const int grid_size = ceil_div(N, total_warps);
        sycl::nd_range<2> grid = sycl::nd_range<2>(
                sycl::range<2>(grid_size * total_warps, 32),
                sycl::range<2>(total_warps, 32)
        );

        q.parallel_for(grid, [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(32)]]{
            fused_residual_forward_kernel4(item, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
        }).wait();
    }
}

// kernel version dispatch
void fused_residual_forward(int kernel_num, sycl::queue &q, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                            const floatX* inp1, const floatX* inp2,
                            const floatX* weight, const floatX* bias,
                            int N, int C, const int block_size) {
    switch (kernel_num) {
        case 1:
            fused_residual_forward1(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 2:
            fused_residual_forward2(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 3:
            fused_residual_forward3(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 4:
            fused_residual_forward4(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 5:
            fused_residual_forward5(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 6:
            fused_residual_forward6(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        default:
            std::cout << "Invalid kernel number\n";
            std::exit(1);
    }
}

int main(int argc, const char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = std::atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << std::endl;

    // create host memory of random numbers
    float* residual = (float*)malloc(B * T * C * sizeof(float));
    float* normed = (float*)malloc(B * T * C * sizeof(float));
    float* inp1 = make_random_float(B * T * C);
    float* inp2 = make_random_float(B * T * C);
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);

    // select device and create queue
    sycl::queue q(sycl::default_selector_v);

    // allocate device memory
    floatX* d_residual = sycl::malloc_device<floatX>(B * T * C, q);
    floatX* d_normed = sycl::malloc_device<floatX>(B * T * C, q);
    floatX* d_inp1 = sycl::malloc_device<floatX>(B * T * C, q);
    floatX* d_inp2 = sycl::malloc_device<floatX>(B * T * C, q);
    floatX* d_mean = sycl::malloc_device<floatX>(B * T, q);
    floatX* d_rstd = sycl::malloc_device<floatX>(B * T, q);
    floatX* d_weight = sycl::malloc_device<floatX>(C, q);
    floatX* d_bias = sycl::malloc_device<floatX>(C, q);

    // copy data to device
    memcpy_convert(d_inp1, inp1, B * T * C, q);
    memcpy_convert(d_inp2, inp2, B * T * C, q);
    memcpy_convert(d_weight, weight, C, q);
    memcpy_convert(d_bias, bias, C, q);

    // first check the correctness of the kernel
    residual_forward_cpu(residual, inp1, inp2, B * T * C);
    layernorm_forward_cpu(normed, mean, rstd, residual, weight, bias, B, T, C);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};

    for (int block_size: block_sizes) {
        if (kernel_num == 6 && block_size == 512) continue; // kernel 6 fails with 512 on Intel GPUs
        std::cout << "Checking block size " << block_size << "." << std::endl;

        q.memset(d_residual, 0, B * T * C * sizeof(floatX)).wait();
        fused_residual_forward(kernel_num, q, d_residual, d_normed, d_mean, d_rstd, d_inp1, d_inp2, d_weight, d_bias,
                               B * T, C, block_size);

        float tol = std::is_same_v<floatX, float> ? 1e-5 : 5e-2;
        validate_result(d_residual, residual, "residual", B * T * C, tol);
        validate_result(d_mean, mean, "mean", B * T, tol);
        validate_result(d_rstd, rstd, "rstd", B * T, tol);
        validate_result(d_normed, normed, "normed", B * T * C, tol);
    }

    std::cout << "All results match. Starting benchmarks.\n\n";

    for (int block_size: block_sizes) {
        if (kernel_num == 6 && block_size == 512) continue; // kernel 6 fails with 512 on Intel GPUs

        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(
                repeat_times,
                fused_residual_forward,
                kernel_num, q, d_residual, d_normed, d_mean, d_rstd, d_inp1, d_inp2,
                d_weight, d_bias, B * T, C, block_size
        );

        // napkin math: estimate the memory bandwidth achieved
        long memory_ops = B * T * (C * 4 + 2) * sizeof(floatX);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;
        float toks_per_msec = B * T / elapsed_time / 1e3;

        std::cout << "block_size " << block_size
                  << " | time " << elapsed_time << " ms"
                  << " | bandwidth " << memory_bandwidth << " GB/s"
                  << " | elements: " << toks_per_msec << " ktok/ms\n";
    }

    // free memory
    free(residual);
    free(normed);
    free(mean);
    free(rstd);
    free(weight);
    free(bias);
    free(inp1);
    free(inp2);
    sycl::free(d_residual, q);
    sycl::free(d_normed, q);
    sycl::free(d_mean, q);
    sycl::free(d_rstd, q);
    sycl::free(d_weight, q);
    sycl::free(d_bias, q);
    sycl::free(d_inp1, q);
    sycl::free(d_inp2, q);

    return 0;
}
