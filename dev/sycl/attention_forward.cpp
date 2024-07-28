#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <iostream>
#include <cmath>
#include <cstdlib>

#define ENABLE_BF16
#include "common.hpp"

auto MKL_OP_T = oneapi::mkl::transpose::trans;
auto MKL_OP_N = oneapi::mkl::transpose::nontrans;

static bool first_run_validation = true; // always run e.g. permute on 1st run

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

void attention_query_key_kernel1(sycl::nd_item<1> id, float* preatt, const float* inp,
                                 int B, int T, int C, int NH) {
    int idx = id.get_global_id(0);
    int total_threads = B * NH * T * T;

    if (idx < total_threads) {
        int t2 = idx % T;
        int t = (idx / T) % T;
        if (t2 > t) {
            // autoregressive mask
            preatt[idx] = -INFINITY;
            return;
        }
        int h = (idx / (T * T)) % NH;
        int b = idx / (NH * T * T);

        int C3 = C*3;
        int hs = C / NH; // head size
        const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
        const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

        // (query_t) dot (key_t2)
        float val = 0.0f;
        for (int i = 0; i < hs; i++) {
            val += query_t[i] * key_t2[i];
        }

        val *= 1.0f / sycl::sqrt(static_cast<float>(hs));

        preatt[idx] = val;
    }
}

void attention_softmax_kernel1(sycl::nd_item<1> id, float* att, const float* preatt,
                               int B, int T, int NH) {
    int idx = id.get_global_id(0);
    int total_threads = B * T * NH;

    if (idx < total_threads) {
        int h = idx % NH;
        int t = (idx / NH) % T;
        int b = idx / (NH * T);

        const float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
        float* att_bth = att + b*NH*T*T + h*T*T + t*T;

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
}

void attention_value_kernel1(sycl::nd_item<1> id, float* out, const float* att, const float* inp,
                             int B, int T, int C, int NH) {
    int idx = id.get_global_id(0);
    int total_threads = B * T * NH;

    if (idx < total_threads) {
        int h = idx % NH;
        int t = (idx / NH) % T;
        int b = idx / (NH * T);

        int C3 = C*3;
        int hs = C / NH; // head size

        float* out_bth = out + b * T * C + t * C + h * hs;
        const float* att_bth = att + b*NH*T*T + h*T*T + t*T;

        for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
        for (int t2 = 0; t2 <= t; t2++) {
            const  float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
            float att_btht2 = att_bth[t2];
            for (int i = 0; i < hs; i++) {
                out_bth[i] += att_btht2 * value_t2[i];
            }
        }
    }
}

void permute_kernel(sycl::nd_item<1> id, float* q, float* k, float* v,
                    const float* inp,
                    int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = id.get_global_id(0);

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]

    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = \
            (b * N * 3 * NH * d)
            +   (n * 3 * NH * d)
            +       (0 * NH * d)
            +          (nh_ * d)
            +                d_;

        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + NH * d];
        v[idx] = inp[inp_idx + 2 * (NH * d)];
    }
}

void unpermute_kernel(sycl::nd_item<1> id, const float* inp, float *out, int B, int N, int NH, int d) {
    // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = id.get_global_id(0);

    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = inp[idx];
    }
}

void scale_kernel(sycl::nd_item<1> id, float* inp, float scale, int B, int NH, int T) {
    // scales the pre-softmax attention scores by scale
    // and sets the autoregressive locations to -INFINITY
    int idx = id.get_global_id(0);
    if (idx < B * NH * T * T) {
        int rest = idx % (NH * T * T);
        rest = rest % (T * T);
        int t2 = rest / T;
        int t = rest % T;
        if (t > t2) {
            inp[idx] = -INFINITY;
        } else {
            inp[idx] *= scale;
        }
    }
}

void attention_forward_kernel2(
        sycl::nd_item<2> id,
        const float* Q,
        const float* K,
        const float* V,
        const int N,
        const int d,
        const int Tc,
        const int Tr,
        const int Bc,
        const int Br,
        const float softmax_scale,
        float* l,
        float* m,
        float* O,
        sycl::local_accessor<float> sram_acc
) {
    int tx = id.get_local_id(1);
    int bx = id.get_group(1); int by = id.get_group(0);  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * id.get_group_range(0) * N * d) + (by * N * d);
    int lm_offset = (bx * id.get_group_range(0) * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* sram = sram_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        sycl::group_barrier(id.get_group());  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {
            // if past the end of the sequence, break
            if (i * Br + tx >= N) {
                break;
            }

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            // S[tx][y] = Sum_{x = 0}^{d-1} {Qi[tx][x] * Kj[y][x]}
            // row_m = Max_{y = 0}^{Bc-1} S[tx][y]
            // with causal masking
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                if (j * Bc + y >= N) {
                    break;
                }
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                if (i * Br + tx < j * Bc + y)
                    sum = -INFINITY;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // implement softmax with causal masking
            // P = exp(S - row_m), row_l = rowsum(P)
            // P[tx][y] = exp(S[tx][y] - row_m)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                if (j * Bc + y >= N) {
                    break;
                }
                if (i * Br + tx < j * Bc + y)
                    S[(Bc * tx) + y] = 0;
                else
                    S[(Bc * tx) + y] = sycl::native::exp(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = sycl::max(row_m_prev, row_m);
            float row_l_new = (sycl::native::exp(row_m_prev - row_m_new) * row_l_prev) + (sycl::native::exp(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    if (j * Bc + y >= N) {
                        break;
                    }
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * sycl::native::exp(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (sycl::native::exp(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        sycl::group_barrier(id.get_group());  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

void softmax_forward_kernel4(sycl::nd_item<1> id, float* out, const float* inp, int N, int C) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel3, but can handle any block size (multiple of 32)
    // each row of C elements is handled by block_size threads
    // furthermore, each block_size threads get executed in warps of 32 threads

    int idx = id.get_group(0);
    int tid = id.get_local_linear_id();

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    const float* x = inp + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += id.get_local_range(0)) {
        maxval = sycl::fmax(maxval, x[i]);
    }
    maxval = sycl::reduce_over_group(id.get_group(), maxval, sycl::maximum<float>());

    // broadcast the max to all threads
    float offset = maxval;

    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += id.get_local_range(0)) {
        out[idx * C + i] = sycl::exp(x[i] - offset);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += id.get_local_range(0)) {
        sumval += x[i];
    }
    sumval = sycl::reduce_over_group(id.get_group(), sumval, sycl::plus<float>());

    // broadcast the sum to all threads
    float sum = sumval;

    // divide the whole row by the sum
    for (int i = tid; i < C; i += id.get_local_range(0)) {
        out[idx * C + i] = x[i] / sum;
    }
}

float& vec_at(sycl::float4& vec, int index) {
    return reinterpret_cast<float*>(&vec)[index];
}

float vec_at(const sycl::float4& vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}


void softmax_forward_kernel5(sycl::nd_item<1> id, float* out, float inv_temperature, const float* inp, int N, int T) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(T % 4  == 0);
    sycl::sub_group warp = id.get_sub_group();
    int idx = id.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const float* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    const sycl::float4* x_vec = reinterpret_cast<const sycl::float4*>(x);
    for (int i = warp.get_local_linear_id(); i < pos_by_4; i += warp.get_max_local_range()[0]) {
        sycl::float4 v = x_vec[i];
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = sycl::fmax(maxval, vec_at(v, k));
        }
        sumval *= sycl::exp(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += sycl::exp(inv_temperature * (vec_at(v, k) - maxval));
        }
    }

    if(4*pos_by_4 + warp.get_local_linear_id() <= own_pos) {
        float old_maxval = maxval;
        maxval = sycl::fmax(maxval, x[4*pos_by_4 + warp.get_local_linear_id()]);
        sumval *= sycl::exp(inv_temperature * (old_maxval - maxval));
        sumval += sycl::exp(inv_temperature * (x[4*pos_by_4 + warp.get_local_linear_id()] - maxval));
    }

    float global_maxval = sycl::reduce_over_group(warp, maxval, sycl::maximum<float>{});
    sumval *= sycl::exp(inv_temperature * (maxval - global_maxval));

    float sum = sycl::reduce_over_group(warp, sumval, sycl::plus<float>{});
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = warp.get_local_linear_id(); i <= own_pos; i += warp.get_max_local_range()[0]) {
        // recalculation is faster than doing the round-trip through memory.
        float ev = sycl::exp(inv_temperature * (x[i] - global_maxval));
        out[idx * T + i] = ev * norm;
    }
}

// --- Forward 5 & 6 kernels ---
void softmax_forward_kernel5_lowp(sycl::nd_item<1> id, floatX* out, float inv_temperature,
                                  const floatX* inp, int N, int T) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(T % 4  == 0);
    sycl::sub_group warp = id.get_sub_group();
    int idx = id.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const floatX* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    // Same thing but without float4, one at a time
    for (int i = warp.get_local_linear_id(); i < pos_by_4; i += warp.get_max_local_range()[0]) {
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = sycl::fmax(maxval, (float)x[4*i + k]);
        }
        sumval *= sycl::exp(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += sycl::exp(inv_temperature * ((float)x[4*i + k] - maxval));
        }
    }

    if(4*pos_by_4 + warp.get_local_linear_id() <= own_pos) {
        float old_maxval = maxval;
        maxval = sycl::fmax(maxval, (float)x[4*pos_by_4 + warp.get_local_linear_id()]);
        sumval *= sycl::exp(inv_temperature * (old_maxval - maxval));
        sumval += sycl::exp(inv_temperature * ((float)x[4*pos_by_4 + warp.get_local_linear_id()] - maxval));
    }

    float global_maxval = sycl::reduce_over_group(warp, maxval, sycl::maximum<float>());
    sumval *= sycl::exp(inv_temperature * (maxval - global_maxval));

    float sum = sycl::reduce_over_group(warp, sumval, sycl::plus<float>());
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = warp.get_local_linear_id(); i <= own_pos; i += warp.get_max_local_range()[0]) {
        // recalculation is faster than doing the round-trip through memory.
        float ev = sycl::exp(inv_temperature * ((float)x[i] - global_maxval));
        out[idx * T + i] = (floatX)(ev * norm);
    }
}

void permute_kernel_lowp(sycl::nd_item<1> id, floatX* q, floatX* k, floatX* v,
                         const float* inp,
                         int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = id.get_global_id(0);

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = \
            (b * N * 3 * NH * d)
            +   (n * 3 * NH * d)
            +       (0 * NH * d)
            +          (nh_ * d)
            +                d_;

        q[idx] = (floatX)inp[inp_idx];
        k[idx] = (floatX)inp[inp_idx + NH * d];
        v[idx] = (floatX)inp[inp_idx + 2 * (NH * d)];
    }
}

void unpermute_kernel_lowp(sycl::nd_item<1> id, const floatX* inp, float *out, int B, int N, int NH, int d) {
    // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = id.get_global_id(0);

    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = (float)inp[idx];
    }
}


// ----------------------------------------------------------------------------
// kernel launcher

void attention_forward1(sycl::queue& q, float* out, float* preatt, float* att,
                        const float* inp,
                        int B, int T, int C, int NH,
                        const int block_size) {
    // attention calculation
    int total_threads = B * NH * T * T;
    int num_blocks = ceil_div(total_threads, block_size);
    q.parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        attention_query_key_kernel1(id, preatt, inp, B, T, C, NH);
    });

    // softmax and value accumulation
    total_threads = B * T * NH;
    num_blocks = ceil_div(total_threads, block_size);
    q.parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        attention_softmax_kernel1(id, att, preatt, B, T, NH);
    });
    q.parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        attention_value_kernel1(id, out, att, inp, B, T, C, NH);
    });
    q.wait();
}


void attention_forward2(sycl::queue &queue, float* out,
                        const float* inp,
                        int B, int T, int C, int NH,
                        const int block_size) {
    // TODO there should be no mallocs inside any of these functions!
    // not fixing this because we don't intend to use attention_forward2,
    // it seems to be way too slow as is

    // these are hardcoded to 32 for now
    const int Bc = 32;
    const int Br = 32;
    // renaming these to be consistent with the kernel
    // const int B = B;
    const int nh = NH;
    const int N = T;
    const int d = C / NH;
    // more
    const int Tc = ceil((float) N / Bc);
    const int Tr = ceil((float) N / Br);
    // Use a float literal because Intel client GPUs do not support fp64
    const float softmax_scale = 1.0f / sqrt(d);
    // create some temporary memory
    float* l;
    float* m;
    l = sycl::malloc_device<float>(B * nh * N, queue);
    m = sycl::malloc_device<float>(B * nh * N, queue);

    queue.memset(l, 0, B * nh * N * sizeof(float));
    queue.memset(m, -10000.0f, B * nh * N * sizeof(float));

    // calculate SRAM size needed per block, ensure we have enough shared memory
    int col_tile_size = Bc * d;  // size of Kj, Vj
    int row_tile_size = Br * d;  // size of Qi
    const int sram_size =
            (2 * col_tile_size * sizeof(float))  // SRAM size for Kj, Vj
            + (row_tile_size * sizeof(float))  // SRAM size for Qi
            + (Bc * Br * sizeof(float));  // SRAM size for S
    int max_sram_size;
    max_sram_size = queue.get_device().get_info<sycl::info::device::local_mem_size>();

    if (sram_size > max_sram_size) {
        printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);
        printf("SRAM size exceeds maximum shared memory per block\n");
        printf("Try decreasing col_tile_size or row_tile_size further\n");
        exit(1);
    }

    // grid and block dims
    sycl::range<2> grid_dim(nh, B);  // batch_size x num_heads
    sycl::range<2> block_dim(1, Br);  // Br threads per block

    // okay so now, this kernel wants Q,K,V to all be of shape (B, nh, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, nh, d)
    // so we have to permute the tensor using a kernel with block_size
    float *q, *k, *v;
    q = sycl::malloc_device<float>(B * T * C, queue);
    k = sycl::malloc_device<float>(B * T * C, queue);
    v = sycl::malloc_device<float>(B * T * C, queue);

    int total_threads = B * N * nh * d;
    int num_blocks = ceil_div(total_threads, block_size);
    queue.parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        permute_kernel(id, q, k, v, inp, B, N, nh, d);
    });


    // now actually call the flash attention kernel
    queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<float> sram_acc(ceil_div(sram_size, (int) sizeof(float)), h);
        h.parallel_for(sycl::nd_range<2>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<2> id) {
            attention_forward_kernel2(id, q, k, v, N, d, Tc, Tr, Bc, Br, softmax_scale, l, m, out, sram_acc);
        });
    });

    // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    queue.parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        unpermute_kernel(id, out, q, B, N, nh, d);
    });
    queue.memcpy(out, q, B * T * C * sizeof(float));

    queue.wait();

    // free memory
    sycl::free(l, queue);
    sycl::free(m, queue);
    sycl::free(q, queue);
    sycl::free(k, queue);
    sycl::free(v, queue);
}

void attention_forward3(sycl::queue &queue, float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                        const float* inp,
                        int B, int T, int C, int NH,
                        const int block_size) {
    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = ceil_div(total_threads, block_size);
    queue.parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        permute_kernel(id, q, k, v, inp, B, T, NH, HS);
    }).wait();

    // batched matrix multiply with oneMKL blas
    const float alpha = 1.0f;
    const float beta = 0.0f;
    oneapi::mkl::blas::column_major::gemm_batch(queue,
                                                MKL_OP_T, MKL_OP_N,
                                                T, T, HS,
                                                &alpha,
                                                k, HS, T * HS,
                                                q, HS, T * HS,
                                                &beta,
                                                preatt, T, T * T,
                                                B * NH
    );

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0f / sqrtf(HS);
    total_threads = B * NH * T * T;
    num_blocks = ceil_div(total_threads, block_size);
    queue.parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        scale_kernel(id, preatt, scale, B, NH, T);
    }).wait();

    // softmax. preatt is (B, NH, T, T) but we view it as (B * NH * T, T) and use the softmax kernel
    int softmax_block_size = 256;
    int grid_size = B * NH * T;
    queue.parallel_for(sycl::nd_range<1>(grid_size * softmax_block_size, softmax_block_size), [=](sycl::nd_item<1> id) {
        softmax_forward_kernel4(id, att, preatt, B * NH * T, T);
    }).wait();

    // new approach: first cuBLAS another batched matmul
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    oneapi::mkl::blas::column_major::gemm_batch(queue,
                                                MKL_OP_N, MKL_OP_N,
                                                HS, T, T,
                                                &alpha,
                                                v, HS, T * HS,
                                                att, T, T * T,
                                                &beta,
                                                vaccum, HS, T * HS,
                                                B * NH
    );
    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    queue.parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        unpermute_kernel(id, vaccum, out, B, T, NH, HS);
    }).wait();
}

void attention_forward4(sycl::queue &queue, float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                        const float* inp,
                        int B, int T, int C, int NH,
                        const int block_size) {
    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = ceil_div(total_threads, block_size);
    queue.parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        permute_kernel(id, q, k, v, inp, B, T, NH, HS);
    }).wait();

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    oneapi::mkl::blas::column_major::gemm_batch(queue,
                                                MKL_OP_T, MKL_OP_N,
                                                T, T, HS,
                                                &alpha,
                                                k, HS, T * HS,
                                                q, HS, T * HS,
                                                &beta,
                                                preatt, T, T * T,
                                                B * NH
    );

    // multiply all elements of preatt elementwise by scale
    // Use a float literal because Intel client GPUs do not support fp64
    float scale = 1.0f / sqrtf(HS);
    int softmax_block_size = 256;
    int grid_size = ceil_div(B * NH * T * 32, softmax_block_size);
    queue.parallel_for(sycl::nd_range<1>(grid_size * softmax_block_size, softmax_block_size), [=](sycl::nd_item<1> id) {
        softmax_forward_kernel5(id, att, scale, preatt, B * NH, T);
    }).wait();

    // new approach: first cuBLAS another batched matmul
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    oneapi::mkl::blas::column_major::gemm_batch(queue,
                                                MKL_OP_N, MKL_OP_N,
                                                HS, T, T,
                                                &alpha,
                                                v, HS, T * HS,
                                                att, T, T * T,
                                                &beta,
                                                vaccum, HS, T * HS,
                                                B * NH
    );
    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    queue.parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        unpermute_kernel(id, vaccum, out, B, T, NH, HS);
    }).wait();
}

void attention_forward5(sycl::queue &queue, float* out, floatX* vaccum, floatX* qkvr, floatX* preatt, floatX* att,
                        const float* inp,
                        int B, int T, int C, int NH,
                        const int block_size, bool skip_permute=false){
    // FP16 version of kernel 4 (with permute/unpermute doing FP32<->FP16)
    // That permute can be skipped on perf runs to analyse its performance impact
    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    int HS = C / NH; // head size
    floatX *q = qkvr + 0 * B * T * C;
    floatX *k = qkvr + 1 * B * T * C;
    floatX* v = qkvr + 2 * B * T * C;

    int total_threads = B * NH * T * HS;
    int num_blocks = ceil_div(total_threads, block_size);
    if (!skip_permute || first_run_validation) {
        queue.parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
            permute_kernel_lowp(id, q, k, v, inp, B, T, NH, HS);
        }).wait();
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const floatX alpha_lowp = (floatX)alpha;
    const floatX beta_lowp = (floatX)beta;

    // batched matrix multiply with cuBLAS
    oneapi::mkl::blas::column_major::gemm_batch(queue,
                                                MKL_OP_T, MKL_OP_N,
                                                T, T, HS,
                                                alpha_lowp,
                                                k, HS, T * HS,
                                                q, HS, T * HS,
                                                beta_lowp,
                                                preatt, T, T * T,
                                                B * NH
    );

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0f / sqrtf(HS);
    int softmax_block_size = 256;
    int grid_size = ceil_div(B * NH * T * 32, softmax_block_size);
    queue.parallel_for(sycl::nd_range<1>(grid_size * softmax_block_size, softmax_block_size), [=](sycl::nd_item<1> id) {
        softmax_forward_kernel5_lowp(id, att, scale, preatt, B * NH, T);
    }).wait();

    // new approach: first cuBLAS another batched matmul
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    oneapi::mkl::blas::column_major::gemm_batch(queue,
                                                MKL_OP_N, MKL_OP_N,
                                                HS, T, T,
                                                alpha_lowp,
                                                v, HS, T * HS,
                                                att, T, T * T,
                                                beta_lowp,
                                                vaccum, HS, T * HS,
                                                B * NH
    );

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    if(!skip_permute || first_run_validation) {
        queue.parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
            unpermute_kernel_lowp(id, vaccum, out, B, T, NH, HS);
        }).wait();
    }
}


// kernel version dispatch
void attention_forward(int kernel_num,
                       sycl::queue &q,
                       float* out, float* vaccum,
                       float* qkvr, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    switch (kernel_num) {
        case 1:
            attention_forward1(q, out, preatt, att, inp, B, T, C, NH, block_size);
            break;
        case 2:
            attention_forward2(q, out, inp, B, T, C, NH, block_size);
            break;
        case 3:
            attention_forward3(q, out, vaccum, qkvr, preatt, att, inp, B, T, C, NH, block_size);
            break;
        case 4:
            attention_forward4(q, out, vaccum, qkvr, preatt, att, inp, B, T, C, NH, block_size);
            break;
        case 5:
            attention_forward5(q, out, (floatX*)vaccum, (floatX*)qkvr,
                               (floatX*)preatt, (floatX*)att,
                               inp, B, T, C, NH, block_size, false);
            break;
        case 6: // skip permutes for perf passes (to analyse perf as if in/out were truly 16-bit)
            attention_forward5(q, out, (floatX*)vaccum, (floatX*)qkvr,
                               (floatX*)preatt, (floatX*)att,
                               inp, B, T, C, NH, block_size, true);
            break;
        default:
            std::cout << "Invalid kernel number\n";
            std::exit(1);
    }
}

int main(int argc, char **argv) {
    sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order{});
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
    float* d_vaccum = sycl::malloc_device<float>(B * T * C, q);
    float* d_qkvr = sycl::malloc_device<float>(B * T * 3 * C, q);
    float* d_preatt = sycl::malloc_device<float>(B * NH * T * T, q);
    float* d_att = sycl::malloc_device<float>(B * NH * T * T, q);
    float* d_inp = sycl::malloc_device<float>(B * T * 3 * C, q);
    q.memcpy(d_inp, inp, B * T * 3 * C * sizeof(float)).wait();

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = std::atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << std::endl;
    int block_sizes[] = {32, 64, 128, 256, 512};

    // Lower accuracy requirements for FP16 (1e-4f also too much for TF32 on kernels 3 & 4)
    float accuracy_threshold = (kernel_num <= 4) ? 1e-3f : 1e-2f;

    // first check the correctness of the kernel
    attention_forward_cpu(out, preatt, att, inp, B, T, C, NH);
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        std::cout << "Checking block size " << block_size << "." << std::endl;
        attention_forward(kernel_num, q, d_out, d_vaccum, d_qkvr, d_preatt, d_att, d_inp, B, T, C, NH, block_size);
        // all kernels should produce the correct output out
        validate_result(d_out, out, "out", B * T * C, accuracy_threshold);
        // but as for preatt and att, things get a bit more complicated:
        if (kernel_num != 2 && kernel_num < 5) {
            // kernel 2 (knowingly) fails att/preatt because it uses a different algorithm
            // that estimates the softmax online and never materializes preatt/att
            validate_result(d_att, att, "att", B * NH * T * T, accuracy_threshold);
        }
        if (kernel_num != 2 && kernel_num < 4) {
            // kernel 4 (knowingly) fails preatt because it fuses the scale normalization
            // into the softmax, so preatt is off by 1.0f / sqrt(HS)
            // but att and out (checked below) should match.
            validate_result(d_preatt, preatt, "preatt", B * NH * T * T, accuracy_threshold);
        }
    }
    std::cout << "All results match. Starting benchmarks." << std::endl;
    first_run_validation = false;

    // benchmark speed of the kernel
    for (int block_size: block_sizes) {
        int repeat_times = 100;

        float elapsed_time = benchmark_kernel(
            repeat_times,
            attention_forward,
            kernel_num, q, d_out, d_vaccum, d_qkvr, d_preatt, d_att, d_inp,
            B, T, C, NH, block_size
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
    sycl::free(d_qkvr, q);
    sycl::free(d_vaccum, q);


    return 0;
}