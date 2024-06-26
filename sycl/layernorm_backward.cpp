#include <sycl/sycl.hpp>
#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <cassert>
#include <chrono>

//#define ENABLE_BF16
#include "common.hpp"

// ----------------------------------------------------------------------------
// CPU code reference

void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                           const float* inp, const float* weight, const float* bias,
                           int B, int T, int C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
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
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalize
                float o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

void layernorm_backward_cpu(float* dinp, float* dweight, float* dbias,
                            const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                            int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout + b * T * C + t * C;
            const float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            const float mean_bt = mean[b * T + t];
            const float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels
template<typename T, sycl::memory_scope MemoryScope = sycl::memory_scope::device>
static inline T atomicAdd(T* val, const T delta)
{
  sycl::atomic_ref<T, sycl::memory_order::relaxed,
     MemoryScope> ref(*val);
  return ref.fetch_add(delta);
}


// super naive kernel that just parallelizes over B,T and loops over C
void layernorm_backward_kernel1(sycl::nd_item<1> id, float* dinp, float* dweight, float* dbias,
                                const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                                int B, int T, int C) {
    int idx = id.get_global_id(0);
    if (idx >= B*T) return;
    int b = idx / T;
    int t = idx % T;

    const float* dout_bt = dout + b * T * C + t * C;
    const float* inp_bt = inp + b * T * C + t * C;
    float* dinp_bt = dinp + b * T * C + t * C;
    const float mean_bt = mean[b * T + t];
    const float rstd_bt = rstd[b * T + t];

    // first: two reduce operations
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // now iterate again and accumulate all the gradients
    for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        // gradient contribution to bias
        atomicAdd(&dbias[i], dout_bt[i]);
        // gradient contribution to weight
        atomicAdd(&dweight[i], norm_bti * dout_bt[i]);
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp_bt[i] += dval;
    }
}

// uses shared memory instead for the reduces
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward_kernel2(sycl::nd_item<1> id, Tdinp* dinp, Tparams* dweight, Tparams* dbias,
                                           const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                                           int B, int T, int C, float* dweight_tmp, float* dbias_tmp, sycl::local_accessor<float> local_acc) {
    auto shared = local_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw(); // size = 2 * C

    auto warp = id.get_sub_group();
    int warp_size = warp.get_max_local_range()[0];

    int idx = id.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();
    int N = B * T;
    if(idx >= N) { return; } // thread guards

    int b = idx / T;
    int t = idx % T;

    const Tdout* dout_bt = dout + b * T * C + t * C;
    const Trest* inp_bt = inp + b * T * C + t * C;
    Tdinp* dinp_bt = dinp + b * T * C + t * C;
    const float mean_bt = (float)mean[b * T + t];
    const float rstd_bt = (float)rstd[b * T + t];

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
#pragma unroll
    for(int i = id.get_local_id(0); i < C; i+= id.get_local_range(0)){
        dbias_shared[i] = 0.0f;
        dweight_shared[i] = 0.0f;
    }
    id.barrier();

    // first: two reduce operations
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = warp.get_local_linear_id(); i < C; i  += warp_size) {
        float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = (float)weight[i] * (float)dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = sycl::reduce_over_group(warp, dnorm_mean, sycl::plus<float>{});
    dnorm_norm_mean = sycl::reduce_over_group(warp, dnorm_norm_mean, sycl::plus<float>{});
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // now iterate again and accumulate all the gradients
    for (int i = warp.get_local_linear_id(); i < C; i += warp_size) {
        float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = (float)weight[i] * (float)dout_bt[i];
        // gradient contribution to bias
        atomicAdd(&dbias_shared[i], (float)dout_bt[i]);
        // gradient contribution to weight
        atomicAdd(&dweight_shared[i], norm_bti * (float)dout_bt[i]);
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
    }
    id.barrier();

    // write to global memory
   for(int i = id.get_local_id(0); i < C; i+= id.get_local_range(0)) {
        atomicAdd(&dbias_tmp[i], dbias_shared[i]);
        atomicAdd(&dweight_tmp[i], dweight_shared[i]);
    }
}

template <typename Tparams>
void copy_to_dweight_dbias(sycl::nd_item<1> id, int C, Tparams* dbias, Tparams* dweight, float* dbias_tmp, float* dweight_tmp) {
    for (int i = id.get_local_id(0); i < C; i += id.get_local_range(0) * id.get_group_range(0)) {
        dbias[i] = (Tparams)dbias_tmp[i];
        dweight[i] = (Tparams)dweight_tmp[i];
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

void layernorm_backward1(sycl::queue &q, float* dinp, float* dweight, float* dbias,
                         const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                         int B, int T, int C, const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        layernorm_backward_kernel1(id, dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
    });
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward2(sycl::queue &q, Tdinp* dinp, Tparams* dweight, Tparams* dbias,
                         const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                         int B, int T, int C, int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(32*N, block_size);
    size_t shared_mem_size = 2 * C * sizeof(float);
    float* dweight_tmp;
    float* dbias_tmp;

    dweight_tmp = sycl::malloc_device<float>(C, q);
    dbias_tmp = sycl::malloc_device<float>(C, q);

    q.memset(dweight_tmp, 0, C * sizeof(float));
    q.memset(dbias_tmp, 0, C * sizeof(float));

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float> local_acc(shared_mem_size, h);
        h.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) [[sycl::reqd_sub_group_size(32)]] {
            layernorm_backward_kernel2(id, dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, dweight_tmp, dbias_tmp, local_acc);
        });
    });

    q.parallel_for(sycl::nd_range<1>(512, 512), [=](sycl::nd_item<1> id) {
        copy_to_dweight_dbias(id, C, dbias, dweight, dbias_tmp, dweight_tmp);
    });

    q.wait();

    sycl::free(dweight_tmp, q);
    sycl::free(dbias_tmp, q);
}

// kernel version dispatch
void layernorm_backward(int kernel_num, sycl::queue &q,
                        floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                        const floatX* dout, const floatX* inp, const floatX* weight, const floatX* mean, const floatX* rstd,
                        int B, int T, int C,
                        const int block_size) {
    switch (kernel_num) {
        case 1:
            layernorm_backward1(q, dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 2:
            layernorm_backward2(q, dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, block_size);
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
    int C = 1600;   // this is the problematic size

    sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order{});
    int num_SMs = q.get_device().get_info<sycl::info::device::max_compute_units>();

    // first do the forward pass in CPU
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);
    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

    // now do the backward pass, again on CPU
    float *dout = make_random_float(B * T * C);
    float *dinp = make_zeros_float(B * T * C);
    float *dweight = make_zeros_float(C);
    float *dbias = make_zeros_float(C);
    layernorm_backward_cpu(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);

    // the above calculations act as the reference
    // now let's do the same on the GPU

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << '\n';

    // move all the variables we need for backward pass onto the GPU
    floatX* d_dinp;
    floatX* d_dweight;
    floatX* d_dbias;
    floatX* d_dout;
    floatX* d_inp;
    floatX* d_weight;
    floatX* d_mean;
    floatX* d_rstd;
    float* d_scratch;
    d_dinp = sycl::malloc_device<floatX>(B * T * C, q);
    d_dweight = sycl::malloc_device<floatX>(C, q);
    d_dbias = sycl::malloc_device<floatX>(C, q);
    d_dout = sycl::malloc_device<floatX>(B * T * C, q);
    d_inp = sycl::malloc_device<floatX>(B * T * C, q);
    d_weight = sycl::malloc_device<floatX>(C, q);
    d_mean = sycl::malloc_device<floatX>(B * T, q);
    d_rstd = sycl::malloc_device<floatX>(B * T, q);
    d_scratch = sycl::malloc_device<float>((1024/32) * num_SMs * (2 * C + 1), q);
    // copy over the "inputs" to the backward call
    memcpy_convert(d_dout, dout, B * T * C, q);
    memcpy_convert(d_inp, inp, B * T * C, q);
    memcpy_convert(d_weight, weight, C, q);
    memcpy_convert(d_mean, mean, B * T, q);
    memcpy_convert(d_rstd, rstd, B * T, q);

    // launch the kernel
    int block_sizes[] = {32, 64, 128, 256, 512};
    for (int block_size: block_sizes) {
        int repeat_times = 100;

        layernorm_backward(kernel_num, q, d_dinp, d_dweight, d_dbias, d_scratch, d_dout, d_inp, d_weight, d_mean, d_rstd,
                           B, T, C, block_size);
        // check the correctness of the kernel
        float error_threshold_dinp = sizeof(floatX) == 4 ? 1e-3f : 1e-1f; // allow larger errors for BF16/FP16
        float error_threshold_dparams = sizeof(floatX) == 4 ? 1e-3f : 5e-1f; // much, much larger...
        std::cout << "Checking correctness...\n";
        std::cout << "dinp:\n";
        validate_result(d_dinp, dinp, "dinp", B * T * C, error_threshold_dinp);
        std::cout << "dweight:\n";
        validate_result(d_dweight, dweight, "dweight", C, error_threshold_dparams);
        std::cout << "dbias:\n";
        validate_result(d_dbias, dbias, "dbias", C, error_threshold_dparams);

        std::cout << "All results match for block_size " << block_size << "\n\n";

        // init the "outputs" of the backward call to zeros
        q.memset(d_dinp, 0, B * T * C * sizeof(floatX));
        q.memset(d_dweight, 0, C * sizeof(floatX));
        q.memset(d_dbias, 0, C * sizeof(floatX));
    }

    // now time the kernel
    for (int block_size: block_sizes) {
        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, layernorm_backward, kernel_num, q,
                                              d_dinp, d_dweight, d_dbias, d_scratch, d_dout, d_inp, d_weight, d_mean, d_rstd,
                                              B, T, C, block_size);
        std::cout << "block_size" << block_size << " time " << elapsed_time << " ms\n";
    }


    // cleanups
    free(out);
    free(mean);
    free(rstd);
    free(inp);
    free(weight);
    free(bias);
    free(dout);
    free(dinp);
    free(dweight);
    free(dbias);
    sycl::free(d_dinp, q);
    sycl::free(d_dweight, q);
    sycl::free(d_dbias, q);
    sycl::free(d_dout, q);
    sycl::free(d_inp, q);
    sycl::free(d_weight, q);
    sycl::free(d_mean, q);
    sycl::free(d_rstd, q);
    sycl::free(d_scratch, q);
    return 0;
}