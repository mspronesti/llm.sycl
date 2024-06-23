#ifndef LLM_SYCL_FUSED_CLASSIFIER_HPP
#define LLM_SYCL_FUSED_CLASSIFIER_HPP

/*
Fused Classifier:
- Forwards the Cross Entropy Loss
- Never materializes the full normalized logits, only at the target label
- (fusion) Also kicks off the backward pass, because everything is already loaded
*/

#include "sycl_common.h"
#include "sycl_utils.hpp"

struct SoftmaxParams {
    float Scale;
    float Offset;
};

SoftmaxParams prepare_softmax_blockwide3(sycl::nd_item<1> id, int64_t idx, const floatX* inp, int V, int P) {
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)
    int blockDim_x = id.get_local_range(0);

    const floatX* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    int i = (V+x128::size-1)/x128::size + id.get_local_id(0) - id.get_group_range(0);

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
        i -= blockDim_x;
    }

    // main loop for the bulk of the iterations (no bounds checking required!)
    for (; i >= 0; i -= blockDim_x) {
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
    float block_maxval = sycl::reduce_over_group(id.get_group(), thread_maxval, -INFINITY, sycl::maximum<float>());
    thread_sumval *= sycl::exp(thread_maxval - block_maxval);
    float block_sumval = sycl::reduce_over_group(id.get_group(), thread_sumval, sycl::plus<float>());

    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// will _update_ logits to logit gradients
// uses template to decide whether to write logits and probs
// split both loops in "multiple-of-x128-size" and "bounds-checked remainder" parts
template <bool WriteLogits = true, bool WriteProbs = false>
void
fused_classifier_kernel5(sycl::nd_item<1> id, floatX* logits, floatX* losses, floatX* probs,
                         const float dloss, const int* targets,
                         int B, int T, int V, int P) {
    int threadIdx_x = id.get_local_id(0);
    int blockDim_x = id.get_local_range(0);
    int blockIdx_x = id.get_group(0);
    int gridDim_x = id.get_group_range(0);
    // note: idx is small enough that it easily fits into 32 bit;
    // by making it a long here, we ensure that any offsets calculated with it (e.g., idx * P)
    // are done is 64 bit
    int64_t idx = gridDim_x - (blockIdx_x+1); // reverse order for cache hits on matmul data
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide3(id, idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx_x == 0) {
        float prob = sycl::exp((float)logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = (floatX)(-sycl::log(prob));
    }

    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const floatX* logits_vec = logits + idx * P;
    for (int i = threadIdx_x; i < V/x128::size; i += blockDim_x) {
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
            store128cs(logits + idx * P + i * x128::size, packed_logits_vec);
        }
        if (WriteProbs) {
            store128(probs + idx * P + i * x128::size, packed_probs);
        }
    }

    // handle remaining elements after the last multiple of x128::size
    // e.g. if V = 8003, and x128::size = 8, we need to handle the last 3 elements
    int unaligned_start = V & ~(x128::size - 1); // round down to multiple of x128::size
    for (int i = threadIdx_x + unaligned_start; i < V; i++) {
        float prob = expf((float)logits_vec[i] - sp.Offset) * sp.Scale;
        float indicator = (i == ix) ? 1.0f : 0.0f;
        float dlogit = (prob - indicator) * dloss;
        if (WriteLogits){
            logits[idx * P + i] = (floatX)dlogit;
        }
        if (WriteProbs) {
            probs[idx * P + i] = (floatX)prob;
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

// replaces logits with logit gradients
template <typename Type>
void fused_classifier(sycl::queue &q, Type* logits, Type* losses,
                      const float dloss, const int* targets,
                      int B, int T, int V, int P) {
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = N;
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        fused_classifier_kernel5(id, logits, losses, (floatX*)nullptr, dloss, targets, B, T, V, P);
    });
}

#endif //LLM_SYCL_FUSED_CLASSIFIER_HPP
