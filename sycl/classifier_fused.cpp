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
    // this function tehen calculates:
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


void fused_classifier(int kernel_num, sycl::queue &q, float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    switch (kernel_num) {
        case 1:
            fused_classifier1(q, dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size);
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
    float* probs = (float*)malloc(B * T * V * sizeof(float));
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

    // time the kernel at different block sizes
    for (int block_size : block_sizes) {
        std::cout << "Checking block size " << block_size << '\n';
        fused_classifier(kernel_num, q, d_dlogits, d_losses, d_logits, d_dlosses, d_targets, B, T, V, P, block_size);
        validate_result(d_losses, losses, "losses", B * T, 1e-3f);
        // undo the padding before we can check for correctness
        q.ext_oneapi_memcpy2d(d_dlogits_no_pad, V * sizeof(float), d_dlogits, P * sizeof(float), V * sizeof(float), B * T);
        q.wait();
        validate_result(d_dlogits_no_pad, dlogits, "dlogits", B * T * V, 1e-3f);
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
    sycl::free(d_dlogits, q);
    sycl::free(d_losses, q);
    sycl::free(d_logits, q);
    sycl::free(d_dlosses, q);
    sycl::free(d_targets, q);

    return 0;
}