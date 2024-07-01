/*
LayerNorm SYCL kernel, and also Residual, because sometimes they are fused

Note in llm.c we try to be clever in the backward pass to conserve memory.
All parameters use a += in the backward pass, so we can do gradient accumulation.
But all activations have = instead of += because these are faster (just read, no write).
This is okay for all activations except for those in the residual stream, where the
gradients have to add. We make sure that we do a += as necessary.
E.g., the layernorms are connected to the residuals so we += in layernorm backward.
*/
#ifndef LLM_SYCL_LAYENORM_HPP
#define LLM_SYCL_LAYENORM_HPP

#include <cassert>
#include "sycl_common.hpp"
#include "sycl_utils.hpp"

// https://github.com/zjin-lcf/HeCBench/blob/8b7575379ffe38b4fcec071e0a9a0affbae07648/src/bh-sycl/main.cpp#L79-L97
inline unsigned int atomicInc(unsigned int *addr, unsigned int operand)
{
    auto atm = sycl::atomic_ref<unsigned int,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>(addr[0]);
    unsigned int old;
    while (true) {
        old = atm.load();
        if (old >= operand) {
            if (atm.compare_exchange_strong(old, 0))
                break;
        } else if (atm.compare_exchange_strong(old, old + 1))
            break;
    }
    return old;
}

// ----------------------------------------------------------------------------
// SYCL kernels

void layernorm_forward_kernel3(sycl::nd_item<1> id, floatX* __restrict__ out, floatX* __restrict__ mean, floatX* __restrict__ rstd,
                               const floatX*  __restrict__ inp, const floatX*  __restrict__ weight,
                               const floatX* __restrict__ bias, int N, int C) {
    sycl::sub_group warp = id.get_sub_group();
    int threadIdx_x = id.get_local_id(0);

    int lane_id = threadIdx_x % WARP_SIZE;
    int warp_id = threadIdx_x / WARP_SIZE;
    int num_warps = id.get_local_range(0) / WARP_SIZE;

    int idx = id.get_group(0) * num_warps + warp_id;
    if(idx >= N) { return; } // guard

    // the row of input that this group of threads is responsible for
    const floatX* x = inp + idx * C;

    // mean
    float sum = 0.0f;
    for (int i = lane_id; i < C; i += WARP_SIZE) {
        sum += (float)x[i];
    }
    sum = warpReduceSum(warp, sum);
    float m = sum / C;
    if(lane_id == 0 && mean != nullptr) {
        mean[idx] = (floatX)m;
    }

    // rstd
    sum = 0.0f;
    for (int i = lane_id; i < C; i += WARP_SIZE) {
        float diff = (float)x[i] - m;
        sum += diff * diff;
    }
    sum = warpReduceSum(warp, sum);
    float s = sycl::rsqrt(sum / C + 1e-5f);
    if(lane_id == 0 && rstd != nullptr) {
        rstd[idx] = (floatX)s;
    }

    // final normalization and scaling by weight/bias
    floatX* o = out + idx * C;
    for (int c = lane_id; c < C; c += WARP_SIZE) {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        float n = s * ((float)x[c] - m);
        o[c] = (floatX)(n * (float)weight[c] + (float)bias[c]);
    }
}

void layernorm_forward_kernel6(sycl::nd_item<2> id, floatX* __restrict__ out, floatX* __restrict__ mean, floatX* __restrict__ rstd,
                                    const floatX*  __restrict__ inp, const floatX*  __restrict__ weight,
                                    const floatX* __restrict__ bias, int N, int C, sycl::local_accessor<char> lmem) {
    assert(id.get_local_range(1) == WARP_SIZE);

    int threadIdx_x = id.get_local_id(1);
    int threadIdx_y = id.get_local_id(0);

    sycl::sub_group warp = id.get_sub_group();

    // load weights and biases into shared memory
    // do this before we allow any threads to exit!
    char* params = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw();
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params) + (C / x128::size);
    x128* s_in = reinterpret_cast<x128*>(params) + ((2 + threadIdx_y) * C / x128::size);

    int sidx = (threadIdx_x + WARP_SIZE * threadIdx_y) * x128::size;
    for(int i = sidx; i < C; i += id.get_local_range(0) * WARP_SIZE * x128::size) {
        s_weight[i/x128::size] = load128(weight + i);
        s_bias[i/x128::size] = load128(bias + i);
    }
    sycl::group_barrier(id.get_group());

    int idx = id.get_group(1) * id.get_local_range(0) + threadIdx_y;
    if(idx > N) return;

    // adjust pointers to current token
    inp += idx * C;
    out += idx * C ;

    const float eps = 1e-5f;
    float sum = 0.0f;
    for(int c = threadIdx_x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = load128cs(inp + c);
        for(int k = 0; k < x128::size; ++k) {
            sum += (float)in_data[k];
        }
        s_in[c / x128::size] = in_data;
    }

    sum = warpReduceSum(warp, sum);
    float m = sum / C;
    float v = 0.f;

    for(int c = threadIdx_x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        for(int k = 0; k < x128::size; ++k) {
            v += ((float)in_data[k] - m) * ((float)in_data[k] - m);
        }
    }

    v = warpReduceSum(warp, v) / C;
    float s = sycl::rsqrt(v + eps);

    for(int c = threadIdx_x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        const x128 w = s_weight[c / x128::size];
        const x128 b = s_bias[c / x128::size];
        x128 out_data;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)in_data[k] - m); // normalized output
            float o = n * (float)w[k] + (float)b[k]; // scale and shift it
            out_data[k] = (floatX)o;
        }

        store128cs(out + c, out_data);
    }
    // cache the mean and rstd for the backward pass later
    if(threadIdx_x == 0 && mean != nullptr) {
        mean[idx] = m;
    }
    // store the rstd, no need to cache it
    if(threadIdx_x == 0 && rstd != nullptr) {
        rstd[idx] = s;
    }
}

void fused_residual_forward_kernel5(sycl::nd_item<2> id, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                    const floatX* inp1, const floatX* inp2,
                                    const floatX* weight, const floatX* bias,
                                    int N, int C, sycl::local_accessor<char> lmem) {
    assert(id.get_local_range(1) == WARP_SIZE);

    int threadIdx_x = id.get_local_id(1);
    int threadIdx_y = id.get_local_id(0);

    sycl::sub_group warp = id.get_sub_group();

    // load weights and biases into shared memory
    // do this before we allow any threads to exit!
    char* params = lmem.get_multi_ptr<sycl::access::decorated::no>().get_raw();
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params) + (C / x128::size);
    x128* s_res = reinterpret_cast<x128*>(params) + ((2 + threadIdx_y) * C / x128::size);

    int sidx = (threadIdx_x + WARP_SIZE * threadIdx_y) * x128::size;
    for(int i = sidx; i < C; i += id.get_local_range(0) * WARP_SIZE * x128::size) {
        s_weight[i/x128::size] = load128(weight + i);
        s_bias[i/x128::size] = load128(bias + i);
    }
    sycl::group_barrier(id.get_group());

    int idx = id.get_group(1) * id.get_local_range(0) + threadIdx_y;
    if(idx > N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    const float eps = 1e-5f;
    float sum = 0.0f;
    for(int c = threadIdx_x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in1 = load128cs(inp1 + c);
        const x128 in2 = load128cs(inp2 + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            out[k] = (float)in1[k] + (float)in2[k];
            sum += (float)out[k];
        }
        store128cs(residual + c, out);
        s_res[c / x128::size] = out;
    }

    sum = warpReduceSum(warp, sum);
    float m = sum / C;
    float v = 0.f;

    for(int c = threadIdx_x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 res = s_res[c / x128::size];
        for(int k = 0; k < x128::size; ++k) {
            v += ((float)res[k] - m) * ((float)res[k] - m);
        }
    }

    v = warpReduceSum(warp, v) / C;
    float s = sycl::rsqrt(v + eps);

    for(int c = threadIdx_x * x128::size; c < C; c += WARP_SIZE * x128::size) {
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


void residual_forward_kernel(sycl::nd_item<1> id, floatX* out, const floatX* inp1, const floatX* inp2) {
    int idx = id.get_global_id(0) * x128::size;

    x128 packed_out;
    x128 packed_inp1 = load128cs(inp1 + idx);
    x128 packed_inp2 = load128cs(inp2 + idx);
    for (int k = 0; k < packed_inp1.size; k++) {
        packed_out[k] = (floatX)((float)packed_inp1[k] + (float)packed_inp2[k]);
    }
    store128(out + idx, packed_out);
}

void  layernorm_backward_kernel10(sycl::nd_item<1> id, floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                            const floatX* dout, const floatX* inp, const floatX* weight,
                            const floatX* mean, const floatX* rstd,
                            int B, int T, int C, sycl::local_accessor<float> local_acc) {
    int threadIdx_x = id.get_local_id(0);
    int blockIdx_x = id.get_group(0);
    int gridDim_x = id.get_group_range(0);

    sycl::group block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();

    int BLOCK_SIZE = id.get_local_range(0);
    int warpsInBlock = BLOCK_SIZE / WARP_SIZE; //number of warps in block
    float* shared = local_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();

    int warpId = threadIdx_x / WARP_SIZE; // warp index within a block
    int baseIdx = blockIdx_x * warpsInBlock + warpId;
    int warpThreadIdx = threadIdx_x % WARP_SIZE; // Thread index within the warp
    int warpsInGrid = gridDim_x * warpsInBlock;
    int C_per_iteration = WARP_SIZE * x128::size;
    int iterations_C = CEIL_DIV(C, C_per_iteration); // + 2;

    // the first half of shared memory is bias, second is weight
    size_t rounded_C = CEIL_DIV(C, (32 * x128::size)) * (32 * x128::size);
    float* dbias_shared = shared;
    float* dweight_shared = shared + rounded_C;
    // warp zero doesn't actually write to the _tmp_shared memory locations, so we don't need to reserve memory
    // the obvious solution is to change the addressing below to use (threadId.x-32) as offset, but that causes
    // register spills, so instead we mess with the base pointer here, which doesn't increase register usage.
    float* dbias_tmp_shared = shared + 2 * rounded_C - WARP_SIZE * f128::size;
    float* dweight_tmp_shared = shared + 2 * rounded_C + f128::size * BLOCK_SIZE - 2 * WARP_SIZE * f128::size;

    // init shared memory to zero
    for(int i = threadIdx_x * f128::size; i < rounded_C; i += BLOCK_SIZE * f128::size) {
        store128(dbias_shared + i, f128::zeros());
        store128(dweight_shared + i, f128::zeros());
    }
    sycl::group_barrier(block);

    for (int bt = baseIdx; bt < B * T; bt += warpsInGrid) {
        const floatX* dout_bt = dout + bt * C;
        const floatX* inp_bt = inp +bt * C;
        floatX* dinp_bt = dinp + bt * C;

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx * x128::size; i < C; i += WARP_SIZE * x128::size) {
            x128 dout128_i   = load128(dout_bt + i);
            x128 inp128_i    = load128(inp_bt  + i);
            x128 weight128_i = load128(weight  + i);
            for (int k = 0; k < x128::size; k++) {
                float dnorm_i = (float)weight128_i[k] * (float)dout128_i[k];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * (float)inp128_i[k];
            }
        }

        const float mean_bt = (float)mean[bt];
        const float rstd_bt = (float)rstd[bt];
        dnorm_mean = warpReduceSum(warp, dnorm_mean) / C;
        dnorm_norm_mean = warpReduceSum(warp, dnorm_norm_mean) / C * rstd_bt - dnorm_mean * mean_bt * rstd_bt;

        for (int c = 0; c < iterations_C; c++) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);

            x128 dout128   = x128::zeros();
            x128 inp128    = x128::zeros();
            x128 dinp128   = x128::zeros();
            x128 weight128 = x128::zeros();

            if(global_index < C) {
                dout128 = load128cs(dout_bt + global_index);
                inp128 = load128cs(inp_bt + global_index);
                dinp128 = load128(dinp_bt + global_index);
                weight128 = load128(weight + global_index);
            }

            for(int o = 0; o < x128::size / f128::size; ++o) {
                f128 dbias_f;
                f128 dweight_f;
                for(int i = 0; i < f128::size; ++i) {
                    int x = o * f128::size + i;
                    float dout_i = (float)dout128[x];
                    float norm_bti = ((float)inp128[x] - mean_bt) * rstd_bt;
                    dbias_f[i] = dout_i;
                    dweight_f[i] = norm_bti * dout_i;

                    float dval = 0.0f;
                    dval += (float) weight128[x] * (float)dout128[x]; // term 1
                    dval -= dnorm_mean; // term 2
                    dval -= norm_bti * dnorm_norm_mean; // term 3
                    dval *= rstd_bt; // final scale
                    dinp128[x] = (floatX) ((float) dinp128[x] + dval);
                }

                if (warpId != 0) {
                    store128(dbias_tmp_shared + threadIdx_x * f128::size, dbias_f);
                    // this seems to generate a 64-bit store, instead of 128-bit.
                    // however, forcing 128-bit (e.g., using inline ptx), results in register
                    // spilling and much worse performance, so we'll keep it like this for now
                    // but ideally, we could reduce the register pressure a little.
                    store128(dweight_tmp_shared + threadIdx_x * f128::size, dweight_f);
                }
                sycl::group_barrier(block);
                if (warpId == 0) {
                    for (int j = 1; j < warpsInBlock; j++) {
                        f128 dbias_tmp = load128(dbias_tmp_shared + f128::size * (threadIdx_x + j * WARP_SIZE));
                        f128 dweight_tmp = load128(dweight_tmp_shared + f128::size * (threadIdx_x + j * WARP_SIZE));
                        for(int i = 0; i < f128::size; ++i) {
                            dbias_f[i] += dbias_tmp[i];
                            dweight_f[i] += dweight_tmp[i];
                        }
                    }
                }
                sycl::group_barrier(block);
                if (warpId == 0) {
                    f128 db_old = load128(dbias_shared + global_index + f128::size * o);
                    f128 dw_old = load128(dweight_shared + global_index + f128::size * o);
                    for(int i = 0; i < f128::size; ++i) {
                        dbias_f[i] += db_old[i];
                        dweight_f[i] += dw_old[i];
                    }
                    store128(dbias_shared + global_index + f128::size * o, dbias_f);
                    store128(dweight_shared + global_index + f128::size * o, dweight_f);
                }
            }
            if(global_index < C) {
                // cache in L2 as this is read by the next kernel, but bypass L1 to minimise thrashing
                store128cg(dinp_bt + global_index, dinp128);
            }
        }
    }
    sycl::group_barrier(block);
    // Each block writes its partial sum to global memory
    // The last block to finish becomes responsible for summing up all the partial sums
    // This is done by atomically incrementing a flag (cleared to 0 before launching the kernel)
    unsigned int* scratchFlag = (unsigned int*)(scratch);
    // Increment scratch pointer by a full cacheline so that everything remains cacheline aligned
    scratch += 32;
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    for(int i = threadIdx_x * f128::size; i < C; i += BLOCK_SIZE * f128::size) {
        // Write to global memory in the same "shared memory banking friendly" order
        store128(scratch_dbias + i + 2*C*blockIdx_x, load128(dbias_shared + i));
        store128(scratch_dweight + i + 2*C*blockIdx_x, load128(dweight_shared + i));
    }
    sycl::group_barrier(block);
    // that portion of shared memory is no longer used, so we can repurpose it for the scratch flag.
    unsigned int *tmp_flag = (unsigned int*)(shared + 2*rounded_C);
    if (threadIdx_x == 0) {
        *tmp_flag = atomicInc(scratchFlag, gridDim_x);
    }
    sycl::group_barrier(block);
    if (*tmp_flag == gridDim_x - 1) {
        // Reduction of the partial sums by the final block
        // todo - there isn't enough parallelism even inside that single SM...
        // ==> so could maybe split into another kernel with YET ANOTHER level of reduction?!
        for(int i = threadIdx_x * f128::size; i < C; i += BLOCK_SIZE * f128::size) {
            f128 dbias_accum = f128::zeros();
            f128 dweight_accum = f128::zeros();

            for (int read_block_idx = 0; read_block_idx < gridDim_x; read_block_idx++) {
                int offset = i + 2*C*read_block_idx;
                f128 dbias128 = load128(scratch_dbias + offset);
                f128 dweight128 = load128(scratch_dweight + offset);
                for(int k = 0; k < f128::size; k++) {
                    dbias_accum[k] += dbias128[k];
                    dweight_accum[k] += dweight128[k];
                }
            }
            store128(dbias_shared + i, dbias_accum);
            store128(dweight_shared + i, dweight_accum);
        }
        sycl::group_barrier(block);

        // convert from float/FP32 to floatX/BF16 for the final write
        // this is separate because it cannot use as many warps as the above (f128 vs x128)
        // todo - if we split this code into another kernel, we could maybe do it at the same time?
        for (int c = warpId; c < iterations_C; c += warpsInBlock) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);
            if (global_index >= C) {
                break;
            }

            x128 dbias128 = load128(dbias + global_index);
            x128 dweight128 = load128(dweight + global_index);
            for(int o = 0; o < x128::size / f128::size; ++o) {
                f128 s_db = load128(dbias_shared + global_index + o * f128::size);
                f128 s_dw = load128(dweight_shared + global_index + o * f128::size);
                for(int i = 0; i < f128::size; ++i) {
                    int x = o * f128::size + i;
                    dbias128[x] = (floatX)(s_db[i] + (float)dbias128[x]);
                    dweight128[x] = (floatX)(s_dw[i] + (float)dweight128[x]);
                }
            }
            store128(dbias + global_index, dbias128);
            store128(dweight + global_index, dweight128);
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

void layernorm_forward(sycl::queue *stream, floatX* out, floatX* mean, floatX* rstd,
                       floatX* inp, const floatX* weight, const floatX* bias,
                       int B, int T, int C) {
    const int block_size = 256;
    int block_y = block_size / WARP_SIZE;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N, block_y);
    size_t smem = (2 + block_y) * C * sizeof(floatX);

    // in order to use more than 48 KiB of smem, need to call cudaFuncSetAttribute
    // this may fail, in which case we fall back to the smem free implementation.
    auto local_mem_size = stream->get_device().get_info<sycl::info::device::local_mem_size>();
    if(local_mem_size > smem) {
        stream->submit([&](sycl::handler& h) {
            sycl::local_accessor<char> local_acc(smem, h);
            sycl::range<2> grid_dim(1, grid_size);
            sycl::range<2> block_dim(block_y, WARP_SIZE);
            h.parallel_for(sycl::nd_range<2>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<2> id)  {
                layernorm_forward_kernel6(id, out, mean, rstd, inp, weight, bias, N, C, local_acc);
            });
        });
    } else {
        // fall back to the version without shared memory
        const int grid_size_fb = CEIL_DIV(N * WARP_SIZE, block_size);
        stream->parallel_for(sycl::nd_range<1>(grid_size*block_size, block_size), [=](sycl::nd_item<1> id)  {
                layernorm_forward_kernel3(id, out, mean, rstd, inp, weight, bias, N, C);
        });
    }
}

void residual_forward(sycl::queue *stream, floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    const int block_size = 256;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    stream->parallel_for(sycl::nd_range<1>(grid_size*block_size, block_size), [=](sycl::nd_item<1> id)  {
            residual_forward_kernel(id, out, inp1, inp2);
    });
}

void fused_residual_forward5(sycl::queue *stream, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C) {
    const int block_size = 256;
    int block_y = block_size / WARP_SIZE;
    const int grid_size = CEIL_DIV(N, block_y);
    size_t smem = (2 + block_y) * C * sizeof(floatX);

    // in order to use more than 48 KiB of smem, need to call cudaFuncSetAttribute
    // this may fail, in which case we fall back to the smem free implementation.
    auto local_mem_size = stream->get_device().get_info<sycl::info::device::local_mem_size>();
    if(local_mem_size > smem) {
        stream->submit([&](sycl::handler& h) {
            sycl::local_accessor<char> local_acc(smem, h);
            sycl::range<2> grid_dim(1, grid_size);
            sycl::range<2> block_dim(block_y, WARP_SIZE);
            h.parallel_for(sycl::nd_range<2>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<2> id) {
                fused_residual_forward_kernel5(id, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, local_acc);
            });
        });
    } else {
        residual_forward(stream, residual, inp1, inp2, N*C);
        layernorm_forward(stream, normed, mean, rstd, residual, weight, bias, N, 1, C);
    }
}


void layernorm_backward(sycl::queue *stream, floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                        const floatX* dout, const floatX* inp, const floatX* weight, const floatX* mean, const floatX* rstd,
                        int B, int T, int C) {
    const int block_size = 512;
    const int blocks_per_sm = 2; // supported on every architecture and less cache thrashing than 3

    int max_cus = stream->get_device().get_info<sycl::info::device::max_compute_units>();
    const int grid_size = blocks_per_sm * max_cus;

    size_t rounded_C = CEIL_DIV(C, (32 * x128::size)) * (32 * x128::size);
    size_t shared_mem_size = (2 * rounded_C + 2 * (block_size - 32) * f128::size) * sizeof(float);

    stream->memset(scratch, 0, 1 * sizeof(float)); // only need to reset the flag to 0
    stream->submit([&](sycl::handler& h) {
        sycl::local_accessor<float> local_acc(shared_mem_size, h);
        h.parallel_for(sycl::nd_range<1>(grid_size*block_size, block_size), [=](sycl::nd_item<1> id)  {
            layernorm_backward_kernel10(id, dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, local_acc);
        });
    });
}

#endif //LLM_SYCL_LAYENORM_HPP
