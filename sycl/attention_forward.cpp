#include <sycl/sycl.hpp>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include "common.hpp"

// ----------------------------------------------------------------------------
// CPU code reference

void attention_forward_cpu(float* out, float* preatt, float* att,
                           const float* inp,
                           int B, int T, int C, int NH) {
    // input is (B, T, 3C) Q,K,V
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }
                // pad with -INFINITY outside of autoregressive region for debugging comparisons
                for (int t2 = t+1; t2 < T; t2++) {
                    preatt_bth[t2] = -INFINITY;
                }

                // pass 2: calculate the exp and keep track of sum
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

// ---------------------------------------
// GPU kernels
void attention_query_key_kernel1(sycl::queue& q, float* preatt, const float* inp, int B, int T, int C, int NH) {
    q.parallel_for(sycl::range<1>(B * NH * T * T), [=](sycl::id<1> idx) {
        int index = idx[0];
        if (index < B * NH * T * T) {
            int t2 = index % T;
            int t = (index / T) % T;
            if (t2 > t) {
                // autoregressive mask
                preatt[index] = -INFINITY;
                return;
            }
            int h = (index / (T * T)) % NH;
            int b = index / (NH * T * T);

            int C3 = C * 3;
            int hs = C / NH; // head size
            const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
            const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

            // (query_t) dot (key_t2)
            float val = 0.0f;
            for (int i = 0; i < hs; i++) {
                val += query_t[i] * key_t2[i];
            }
            val *= 1.0 / sqrtf(hs);

            preatt[index] = val;
        }
    }).wait();
}

void attention_softmax_kernel1(sycl::queue& q, float* att, const float* preatt, int B, int T, int NH) {
    q.parallel_for(sycl::range<1>(B * T * NH), [=](sycl::id<1> idx) {
        int index = idx[0];
        if (index < B * T * NH) {
            int h = index % NH;
            int t = (index / NH) % T;
            int b = index / (NH * T);

            const float* preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
            float* att_bth = att + b * NH * T * T + h * T * T + t * T;

            // find maxval
            float maxval = -10000.0f; // TODO something better
            for (int t2 = 0; t2 <= t; t2++) {
                if (preatt_bth[t2] > maxval) {
                    maxval = preatt_bth[t2];
                }
            }

            // calculate the exp and keep track of sum
            float expsum = 0.0f;
            for (int t2 = 0; t2 <= t; t2++) {
                float expv = sycl::exp(preatt_bth[t2] - maxval);
                expsum += expv;
                att_bth[t2] = expv;
            }
            float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

            // normalize to get the softmax
            for (int t2 = 0; t2 < T; t2++) {
                if (t2 <= t) {
                    att_bth[t2] *= expsum_inv;
                } else {
                    // causal attention mask. not strictly necessary to set to zero here
                    // only doing this explicitly for debugging and checking to PyTorch
                    att_bth[t2] = 0.0f;
                }
            }
        }
    }).wait();
}

void attention_value_kernel1(sycl::queue& q, float* out, const float* att, const float* inp, int B, int T, int C, int NH) {
    q.parallel_for(sycl::range<1>(B * T * NH), [=](sycl::id<1> idx) {
        int index = idx[0];
        if (index < B * T * NH) {
            int h = index % NH;
            int t = (index / NH) % T;
            int b = index / (NH * T);

            int C3 = C * 3;
            int hs = C / NH; // head size

            float* out_bth = out + b * T * C + t * C + h * hs;
            const float* att_bth = att + b * NH * T * T + h * T * T + t * T;

            for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
            for (int t2 = 0; t2 <= t; t2++) {
                const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
                float att_btht2 = att_bth[t2];
                for (int i = 0; i < hs; i++) {
                    out_bth[i] += att_btht2 * value_t2[i];
                }
            }
        }
    }).wait();
}

void attention_forward1(sycl::queue& q, float* out, float* preatt, float* att, const float* inp, int B, int T, int C, int NH, const int block_size) {
    // attention calculation
    int total_threads = B * NH * T * T;
    int num_blocks = ceil_div(total_threads, block_size);
    attention_query_key_kernel1(q, preatt, inp, B, T, C, NH);

    // softmax and value accumulation
    total_threads = B * T * NH;
    num_blocks = ceil_div(total_threads, block_size);
    attention_softmax_kernel1(q, att, preatt, B, T, NH);
    attention_value_kernel1(q, out, att, inp, B, T, C, NH);
}

int main(int argc, char **argv) {
    sycl::queue q(sycl::default_selector_v);
    int B = 8;
    int T = 1024;
    int C = 768;
    int NH = 12;

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* preatt = (float*)malloc(B * NH * T * T * sizeof(float));
    float* att = (float*)malloc(B * NH * T * T * sizeof(float));
    float* inp = make_random_float(B * T * 3 * C);

    // move to GPU
    float* d_out = sycl::malloc_device<float>(B * T * C, q);
    float* d_preatt = sycl::malloc_device<float>(B * NH * T * T, q);
    float* d_att = sycl::malloc_device<float>(B * NH * T * T, q);
    float* d_inp = sycl::malloc_device<float>(B * T * 3 * C, q);
    q.memcpy(d_inp, inp, B * T * 3 * C * sizeof(float)).wait();

    // read kernel_num from command line
    /*int kernel_num = 1;
    if (argc > 1) {
        kernel_num = std::atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << std::endl;*/
    int block_sizes[] = {32, 64, 128, 256, 512};

    // Lower accuracy requirements for FP16 (1e-4f also too much for TF32 on kernels 3 & 4)
    float accuracy_threshold = 1e-3f;

    // first check the correctness of the kernel
    attention_forward_cpu(out, preatt, att, inp, B, T, C, NH);
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        std::cout << "Checking block size " << block_size << "." << std::endl;
        attention_forward1(q, d_out, d_preatt, d_att, d_inp, B, T, C, NH, block_size);
        q.memcpy(out, d_out, B * T * C * sizeof(float)).wait();
        q.memcpy(att, d_att, B * NH * T * T * sizeof(float)).wait();
        q.memcpy(preatt, d_preatt, B * NH * T * T * sizeof(float)).wait();
        validate_result(d_out, out, "out", B * T * C, accuracy_threshold);
        validate_result(d_att, att, "att", B * NH * T * T, accuracy_threshold);
        validate_result(d_preatt, preatt, "preatt", B * NH * T * T, accuracy_threshold);

    }
    std::cout << "All results match. Starting benchmarks." << std::endl;

    // benchmark speed of the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 100;

        float elapsed_time = benchmark_kernel(
            repeat_times,
            attention_forward1,
            q, d_out, d_preatt, d_att, d_inp, B, T, C, NH, block_size
        );

        std::cout << "block_size " << block_size << " | time " << elapsed_time << " ms" << std::endl;
    }

    // free memory
    free(out);
    free(preatt);
    free(att);
    free(inp);
    sycl::free(d_out, q);
    sycl::free(d_preatt, q);
    sycl::free(d_att, q);
    sycl::free(d_inp, q);

    return 0;
}