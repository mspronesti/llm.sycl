#include <sycl/sycl.hpp>
#include <iostream>
#include <cstdlib>
#include <float.h>

#include "common.hpp"


// ----------------------------------------------------------------------------
// CPU code reference

void softmax_forward_cpu(float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    for (int i = 0; i < N; i++) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        for (int j = 0; j < C; j++) {
            out_row[j] /= sum;
        }
    }
}


void crossentropy_forward_cpu(float* losses,
                              const float* probs, const int* targets,
                              int B, int T, int V) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,V) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // loss = -log(probs[target])
            const float* probs_bt = probs + b * T * V + t * V;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

void crossentropy_softmax_backward_cpu(float* dlogits,
                                       const float* dlosses, const float* probs, const int* targets,
                                       int B, int T, int V) {
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * V + t * V;
            const float* probs_bt = probs + b * T * V + t * V;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] = (p - indicator) * dloss;
            }
        }
    }
}
// ----------------------------------------------------------------------------
// GPU kernels

struct SoftmaxParams {
    float Scale;
    float Offset;
};

SoftmaxParams prepare_softmax(sycl::sub_group warp,
                              int idx, const float* inp, int V, int P) {
    // this warp (of 32) threads processes one row of inp, i.e. inp[idx, :] of shape (V,)
    // note that inp is actually (B * T, P) but we only use the first V elements
    // this function then calculates:
    // 1) the max value to subtract for numerical stability and
    // 2) the sum normalization factor
    const float* x = inp + idx * P;
    // thread coarsening loop, where the 32 threads serially process all V elements
    // thread_rank() is in [0, 31], warp.size() is 32
    float maxval = -INFINITY;
    float sumval = 0.0f;
    for (int i = warp.get_local_linear_id(); i < V; i += warp.get_max_local_range()[0]) {
        float v = x[i];
        float old_maxval = maxval;
        // online softmax recurrence from "Online normalizer calculation for softmax" paper
        maxval = sycl::fmax(maxval, v);
        sumval *= sycl::exp((old_maxval - maxval));
        sumval += sycl::exp(v - maxval);
    }
    // warp-level reduction to get the maxval across the 32 threads
    float global_maxval = sycl::reduce_over_group(warp, maxval, sycl::maximum<float>{});
    // all 32 threads do a final shift of the sum considering the global max in this row
    sumval *= sycl::exp((maxval - global_maxval));
    // warp-level reduction to get the sumval across the 32 threads
    float global_sumval = sycl::reduce_over_group(warp, sumval, sycl::plus<float>{});
    // the final normalization factor
    float norm = 1.0f / global_sumval;
    return SoftmaxParams{norm, global_maxval};
}


void fused_classifier_kernel1(sycl::nd_item<1> id, float* dlogits, float* losses,
                              const float* logits, const float* dlosses, const int* targets,
                              int B, int T, int V, int P) {
    sycl::sub_group warp = id.get_sub_group();
    int idx = id.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();
    if (idx >= B * T) {
        return;
    }
    int b = idx / T;
    int t = idx % T;

    // calculate the offset (maxval) and scale (sumval) for the softmax
    SoftmaxParams sp = prepare_softmax(warp, idx, logits, V, P);

    // in each row (handled by one warp), thread 0 calculates the loss
    // calculate the probability needed for the loss and update losses
    if(warp.leader()) {
        int ix = targets[b * T + t];
        float prob = sycl::exp(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[b * T + t] = -sycl::log(prob);
    }

    // finally all threads calculate the gradients
    // prob is only materialized here temporarily and in registers, never
    // as a full tensor that gets written to global memory
    for (int i = warp.get_local_linear_id(); i < V; i += warp.get_max_local_range()[0]) {
        float prob = sycl::exp(logits[idx * P + i] - sp.Offset) * sp.Scale;
        float* dlogits_bt = dlogits + b * T * P + t * P;
        float dloss = dlosses[b * T + t];
        int ix = targets[b * T + t];
        float indicator = i == ix ? 1.0f : 0.0f;
        dlogits_bt[i] = (prob - indicator) * dloss;
    }
}


float vec_at(const sycl::float4& vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}

SoftmaxParams prepare_softmax_blockwide(sycl::nd_item<1> id, sycl::sub_group& warp,
                                        int idx, const float* inp, int V, int P) {
    // one row of inp, i.e. inp[idx, :] of shape (V,)
    // float4 to get 128-bit loads and memory level parallelism
    const sycl::float4* x_vec4 = reinterpret_cast<const sycl::float4*>(inp + idx * P);

    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = ceil_div(V, 4) + id.get_local_linear_id() - id.get_local_range(0); i >= 0; i -= id.get_local_range(0)) {
        sycl::float4 v4 = x_vec4[i];
        #pragma unroll
        for(int k = 0; k < 4; k++) {
            if (i*4+k >= V) {  // bounds checking against real V
                continue;
            }
            float old_maxval = thread_maxval;
            thread_maxval = sycl::fmax(thread_maxval, vec_at(v4, k));
            thread_sumval *= sycl::exp(old_maxval - thread_maxval);
            thread_sumval += sycl::exp(vec_at(v4, k) - thread_maxval);
        }
    }

    float block_maxval = sycl::reduce_over_group(id.get_group(), thread_maxval, sycl::maximum<float>{});
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= sycl::exp(thread_maxval - block_maxval);

    float block_sumval = sycl::reduce_over_group(id.get_group(), thread_sumval, sycl::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// Fused forward and backward pass for classifier including softmax, and logit gradients
// Writes to both probs (only for debugging) and dlogits (only for training) are optional
// N.B.: We may want to reuse the logits memory for dlogits, so they should *not* be __restrict__!
void fused_classifier_kernel2(sycl::nd_item<1> id, float* dlogits, float* losses, float* probs,
                              const float* logits, const float* dlosses, const int* targets,
                              int B, int T, int V, int P) {
    sycl::sub_group warp = id.get_sub_group();
    int idx = id.get_group(0);
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide(id, warp, idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(id.get_group().leader()) {
        float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -logf(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const sycl::float4* logits_vec4 = reinterpret_cast<const sycl::float4*>(logits + idx * P);
    for (int i = id.get_local_linear_id(); i < ceil_div(V, 4); i += id.get_local_range(0)) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // this data will never be needed again, so we reduce cache persistence
        sycl::float4 v4 = logits_vec4[i];

        #pragma unroll
        for(int k = 0; k < 4; ++k) {
            int element = i*4 + k;
            float prob = sycl::exp(vec_at(v4, k) - sp.Offset) * sp.Scale;
            prob = (element < V) ? prob : 0.0f; // bounds checking against real V

            // this kernel is DRAM limited so cost of inner branch is ~zero
            if (probs != nullptr) {
                probs[idx * P + element] = prob;
            }
            if (dlogits != nullptr) {
                float indicator = element == ix ? 1.0f : 0.0f;
                dlogits[idx * P + element] = (prob - indicator) * dloss;
            }
        }
    }
}

SoftmaxParams prepare_softmax_blockwide_nofloat4(sycl::nd_item<1> id, int idx, const float* inp, int V, int P) {
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const float* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = V + id.get_local_linear_id() - id.get_local_range(0); i >= 0; i -= id.get_local_range(0)) {
        float v = x[i];
        float old_maxval = thread_maxval;
        thread_maxval = sycl::fmax(thread_maxval, v);
        thread_sumval *= sycl::exp((old_maxval - thread_maxval));
        thread_sumval += sycl::exp(v - thread_maxval);
    }

    float block_maxval = sycl::reduce_over_group(id.get_group(), thread_maxval, -FLT_MAX, sycl::maximum<float>{});
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= sycl::exp(thread_maxval - block_maxval);
    float block_sumval = sycl::reduce_over_group(id.get_group(), thread_sumval, 0.0f, sycl::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// same as 2 but not using float4
void fused_classifier_kernel3(sycl::nd_item<1> id, float* dlogits, float* losses, float* probs,
                              const float* logits, const float* dlosses, const int* targets,
                              int B, int T, int V, int P) {
    int idx = id.get_group(0);
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide_nofloat4(id, idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(id.get_group().leader()) {
        float prob = sycl::exp(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -sycl::log(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != nullptr ? dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const float* logits_vec = logits + idx * P;
    for (int i = id.get_local_linear_id(); i < V; i += id.get_local_range(0)) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // this data will never be needed again, so we reduce cache persistence
        float v = logits_vec[i];
        float prob = sycl::exp(v - sp.Offset) * sp.Scale;
        if (probs != nullptr) {
            probs[idx * P + i] = prob;
        }
        if (dlogits != nullptr) {
            float indicator = (i == ix) ? 1.0f : 0.0f;
            dlogits[idx * P + i] = (prob - indicator) * dloss;
        }
    }
}

SoftmaxParams prepare_softmax_blockwide2(sycl::nd_item<1> id, sycl::sub_group& warp,
                                        int idx, const float* inp, int V, int P) {
    // one row of inp, i.e. inp[idx, :] of shape (V,)
    const floatX* x = inp + idx * P;

    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = ceil_div(V, x128::size) + id.get_local_linear_id() - id.get_local_range(0); i >= 0; i -= id.get_local_range(0)) {
        x128 packed_x = load128cs(x + i * x128::size); // load and do not keep in cache
        for(int k = 0; k < packed_x.size; ++k) {
            if (i*x128::size+k >= V) {  // bounds checking against real V
                continue;
            }
            float v = (float)packed_x[k];
            float old_maxval = thread_maxval;
            thread_maxval = sycl::fmax(thread_maxval, v);
            thread_sumval *= sycl::exp(old_maxval - thread_maxval);
            thread_sumval += sycl::exp(v - thread_maxval);
        }
    }

    float block_maxval = sycl::reduce_over_group(id.get_group(), thread_maxval, sycl::maximum<float>{});
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= sycl::exp(thread_maxval - block_maxval);

    float block_sumval = sycl::reduce_over_group(id.get_group(), thread_sumval, sycl::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// same as 2 but using x128
void fused_classifier_kernel4(sycl::nd_item<1> id, float* dlogits, float* losses, float* probs,
                              const float* logits, const float* dlosses, const int* targets,
                              int B, int T, int V, int P) {
    sycl::sub_group warp = id.get_sub_group();
    int idx = id.get_group(0);
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide2(id, warp, idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(id.get_group().leader()) {
        float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -logf(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != nullptr ? (float)dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const floatX* logits_vec = logits + idx * P;
    for (int i = id.get_local_linear_id(); i < ceil_div(V, x128::size); i += id.get_local_range(0)) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // this data will never be needed again, so we reduce cache persistence
        x128 packed_logits_vec = load128cs(logits_vec + i * x128::size); // load and do not keep in cache
        x128 packed_probs;
        x128 packed_dlogits;
        for (int k = 0; k < packed_logits_vec.size; ++k) {
            int element = i * packed_logits_vec.size + k;
            if (element >= V) {  // bounds checking against real V
                continue;
            }
            float v = packed_logits_vec[k];
            float prob = expf(v - sp.Offset) * sp.Scale;
            packed_probs[k] = prob;
            float indicator = (element == ix) ? 1.0f : 0.0f;
            packed_dlogits[k] = (prob - indicator) * dloss;
        }
        // Note: missing .cs hint hurts our performance due to cache thrashing, fixed in kernel5
        store128(dlogits + idx * P + i * packed_logits_vec.size, packed_dlogits);
        if (probs != nullptr) {
            store128(probs + idx * P + i * packed_logits_vec.size, packed_probs);
        }
    }
}

SoftmaxParams prepare_softmax_blockwide3(sycl::nd_item<1> id, sycl::sub_group& warp,
                                         int idx, const float* inp, int V, int P) {
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)
    const floatX* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    int i = (V+x128::size-1)/x128::size + id.get_local_linear_id() - id.get_local_range(0);

    // special-case loop to handle the unaligned elements at the end of the array
    // this lets us skip the bounds check in the main loop below, which improves performance
    while ((i+1)*x128::size > V) {
        for(int k = 0; k < x128::size; ++k) {
            if (i*x128::size+k >= V) {
                break; // bounds checking against real V (rather than padded P)
            }
            float v = (float)x[i*x128::size+k];
            float old_maxval = thread_maxval;
            thread_maxval = sycl::fmax(thread_maxval, v);
            thread_sumval *= sycl::exp((old_maxval - thread_maxval));
            thread_sumval += sycl::exp(v - thread_maxval);
        }
        i -= id.get_local_range(0);
    }

    // main loop for the bulk of the iterations (no bounds checking required!)
    for (; i >= 0; i -= id.get_local_range(0)) {
        x128 packed_x = load128(x + i * x128::size); // load and keep in cache until fused_classifier loop
        for(int k = 0; k < x128::size; ++k) {
            float v = (float)packed_x[k];
            float old_maxval = thread_maxval;
            thread_maxval = sycl::fmax(thread_maxval, v);
            thread_sumval *= sycl::exp((old_maxval - thread_maxval));
            thread_sumval += sycl::exp(v - thread_maxval);
        }
    }

    // Block Max Reduction -> Maths -> Block Sum Reduction
    float block_maxval = sycl::reduce_over_group(id.get_group(), thread_maxval, sycl::maximum<float>{});
    thread_sumval *= sycl::exp(thread_maxval - block_maxval);
    float block_sumval = sycl::reduce_over_group(id.get_group(), thread_sumval, sycl::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// will _update_ logits to logit gradients
// uses template to decide whether to write logits and probs
// split both loops in "multiple-of-x128-size" and "bounds-checked remainder" parts
template <bool WriteLogits = true, bool WriteProbs = false>
void fused_classifier_kernel5(sycl::nd_item<1> id, floatX* dlogits, floatX* losses, floatX* probs,
                              const floatX* logits, const floatX* dlosses, const int* targets,
                              int B, int T, int V, int P){
    sycl::sub_group warp = id.get_sub_group();
    int idx = id.get_group(0);
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide3(id, warp, idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(id.get_group().leader()) {
        float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = (floatX)(-logf(prob));
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != nullptr ? dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const float* logits_vec = logits + idx * P;
    for (int i = id.get_local_linear_id(); i < V/x128::size; i += id.get_local_range(0)) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // it will be overwritten by the logits gradients which is when we reduce cache persistence
        x128 packed_logits_vec = load128(logits_vec + i * x128::size); // rely on cs of store128cs
        x128 packed_probs;
        for(int k = 0; k < x128::size; ++k) {
            int element = i*x128::size + k;
            float prob = expf((float)packed_logits_vec[k] - sp.Offset) * sp.Scale;
            packed_probs[k] = (floatX)prob;
            float indicator = (element == ix) ? 1.0f : 0.0f;
            packed_logits_vec[k] = (floatX)((prob - indicator) * dloss);
        }
        if (WriteLogits){
            // reduce cache persistence for the overwritten logits
            // to maximise probability that logits remain in cache between prepare_softmax and here
            store128cs(dlogits + idx * P + i * x128::size, packed_logits_vec);
        }
        if (WriteProbs) {
            store128(probs + idx * P + i * x128::size, packed_probs);
        }
    }

    // handle remaining elements after the last multiple of x128::size
    // e.g. if V = 8003, and x128::size = 8, we need to handle the last 3 elements
    int unaligned_start = V & ~(x128::size - 1); // round down to multiple of x128::size
    for (int i = id.get_local_linear_id() + unaligned_start; i < V; i++) {
        float prob = expf((float)logits_vec[i] - sp.Offset) * sp.Scale;
        float indicator = (i == ix) ? 1.0f : 0.0f;
        float dlogit = (prob - indicator) * dloss;
        if (WriteLogits){
            // __stcs(dlogits + idx * P + i, (floatX)dlogit);
            dlogits[idx * P + i] = (floatX)dlogit;
        }
        if (WriteProbs) {
            probs[idx * P + i] = (floatX)prob;
        }
    }
}
// ----------------------------------------------------------------------------
// kernel launcher

void fused_classifier1(sycl::queue &q, float* dlogits, float* losses,
                       const float* logits, const float* dlosses, const int* targets,
                       int B, int T, int V, int P, int block_size) {
    const int N = B * T; // total number of rows in the input
    // how many rows of the input can each block of threads process?
    // e.g. in block_size=128, 4 rows get handled by 4 warps (of 32 threads each)
    const int rows_per_block = block_size / 32;
    const int grid_size = N / rows_per_block; // total number of blocks needed

    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) [[sycl::reqd_sub_group_size(32)]] {
        fused_classifier_kernel1(id, dlogits, losses, logits, dlosses, targets, B, T, V, P);
    }).wait();
}

void fused_classifier2(sycl::queue &q, float* dlogits, float* losses,
                       const float* logits, const float* dlosses, const int* targets,
                       int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N; // total number of blocks needed

    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) [[sycl::reqd_sub_group_size(32)]] {
        fused_classifier_kernel2(id, dlogits, losses, nullptr, logits, dlosses, targets, B, T, V, P);
    }).wait();
}


void fused_classifier3(sycl::queue &q, float* dlogits, float* losses,
                       const float* logits, const float* dlosses, const int* targets,
                       int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N; // total number of blocks needed

    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) [[sycl::reqd_sub_group_size(32)]] {
        fused_classifier_kernel3(id, dlogits, losses, nullptr, logits, dlosses, targets, B, T, V, P);
    }).wait();
}

void fused_classifier4(sycl::queue &q, float* dlogits, float* losses,
                       const float* logits, const float* dlosses, const int* targets,
                       int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N; // total number of blocks needed

    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) [[sycl::reqd_sub_group_size(32)]] {
        fused_classifier_kernel4(id, (floatX*)dlogits, (floatX*)losses, nullptr, (floatX*)logits, (floatX*)dlosses, targets, B, T, V, P);
    }).wait();
}

void fused_classifier5(sycl::queue &q, float* dlogits, float* losses,
                       const float* logits, const float* dlosses, const int* targets,
                       int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;

    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=]
            // this tries to port __launch_bounds__(1024, MAX_1024_THREADS_BLOCKS)
            [[intel::max_work_group_size(1, 1, 1024),
              intel::min_work_groups_per_cu(2)]](sycl::nd_item<1> id){
        fused_classifier_kernel5<true, false>(id, (floatX*)dlogits, (floatX*)losses, nullptr, (floatX*)logits, (floatX*)dlosses, targets, B, T, V, P);
    }).wait();
}


void fused_classifier(int kernel_num, sycl::queue &q, float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    switch (kernel_num) {
        case 1:
            fused_classifier1(q, dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size);
            break;
        case 2:
            fused_classifier2(q, dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size);
            break;
        case 3:
            fused_classifier3(q, dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size);
            break;
        case 4:
            fused_classifier4(q, dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size);
            break;
        case 5:
            fused_classifier5(q, dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size);
            break;
        default:
            std::cout << "Invalid kernel number\n";
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;              // batch size
    int T = 1024;           // sequence length
    int V = 50257;          // vocab size
    int P = (V + 63) & ~63; // padded vocab size, up to nearest multiple of 64

    sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order{});

    // create host memory of random numbers
    float* logits = make_random_float_01(B * T * V);
    float* probs = make_random_float_01(B * T * V);
    float* dlogits = (float*)malloc(B * T * V * sizeof(float));
    float* losses = (float*)malloc(B * T * sizeof(float));
    float* dlosses = make_random_float(B * T);
    int* targets = make_random_int(B * T, V);
    // make the input less uniformly random: Otherwise, all probabilities will be basically zero,
    // and the tests are not actually meaningful.
    int* outliers = make_random_int(B * T * 3, V);
    for(int k = 0; k < 3; ++k) {
        for(int j = 0; j < B * T; ++j) {
            logits[j * V +  outliers[j*3 + k]] *= 20;
        }
    }

    // move to GPU
    float *d_dlogits = sycl::malloc_device<float>(B * T * P, q);
    float *d_logits = sycl::malloc_device<float>(B * T * P, q);
    float *d_dlogits_no_pad = sycl::malloc_device<float>(B * T * V, q);
    int *d_targets = sycl::malloc_device<int>(B * T, q);
    float *d_losses = sycl::malloc_device<float>(B * T, q);
    float *d_dlosses = sycl::malloc_device<float>(B * T, q);

    q.memset(d_dlogits, 0xff, B * T * P * sizeof(float)).wait();
    q.ext_oneapi_memcpy2d(d_logits, P * sizeof(float), logits, V * sizeof(float), V * sizeof(float), B * T).wait();
    q.memcpy(d_dlosses, dlosses, B * T * sizeof(float)).wait();
    q.memcpy(d_targets, targets, B * T * sizeof(int)).wait();

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << '\n';

    // define block sizes we'll use in correctness and timing
    int block_sizes[] = {32, 64, 128, 256};

    // first check the correctness of the kernel
    softmax_forward_cpu(probs, logits, B * T, V);
    crossentropy_forward_cpu(losses, probs, targets, B, T, V);
    crossentropy_softmax_backward_cpu(dlogits, dlosses, probs, targets, B, T, V);

    float tolerance = 1e-3f;  // the original version uses 1e-4f, which doesn't work on Intel GPUs :(
    // time the kernel at different block sizes
    for (int block_size : block_sizes) {
        std::cout << "Checking block size " << block_size << '\n';
        fused_classifier(kernel_num, q, d_dlogits, d_losses, d_logits, d_dlosses, d_targets, B, T, V, P, block_size);
        validate_result(d_losses, losses, "losses", B * T, tolerance);
        // undo the padding before we can check for correctness
        q.ext_oneapi_memcpy2d(d_dlogits_no_pad, V * sizeof(float), d_dlogits, P * sizeof(float), V * sizeof(float), B * T);
        q.wait();
        validate_result(d_dlogits_no_pad, dlogits, "dlogits", B * T * V, tolerance);
    }
    std::cout << "All results match. Starting benchmarks.\n\n";

    for (int block_size : block_sizes) {
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, fused_classifier,
                                              kernel_num, q, d_dlogits, d_losses, d_logits, d_dlosses, d_targets,
                                              B, T, V, P, block_size);
        std::cout << "block_size " << block_size << " | time " << elapsed_time << " ms\n";
    }

    // free memory
    free(logits);
    free(probs);
    free(dlogits);
    free(losses);
    free(dlosses);
    free(targets);
    free(outliers);
    sycl::free(d_dlogits, q);
    sycl::free(d_losses, q);
    sycl::free(d_logits, q);
    sycl::free(d_dlosses, q);
    sycl::free(d_targets, q);

    return 0;
}