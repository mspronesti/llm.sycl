/*
Attention, as a fallback when we do not use the Flash Attention from cuDNN
*/
#ifndef LLM_SYCL_ATTENTION_HPP
#define LLM_SYCL_ATTENTION_HPP

#include <cassert>
#include <oneapi/mkl.hpp>

#include "sycl_utils.hpp"
#include "sycl_common.hpp"

// ----------------------------------------------------------------------------
// SYCL kernels

void permute_kernel(sycl::nd_item<1> id, floatX* q, floatX* k, floatX* v,
                    const floatX* inp,
                    int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = id.get_global_id(0);
    if (idx >= B * NH * N * d) { return; }

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
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

void permute_kernel_backward(sycl::nd_item<1> id,
                             floatX* dinp,
                             const floatX* dq, const floatX* dk, const floatX* dv,
                             int B, int N, int NH, int d) {
    int idx = id.get_global_id(0);
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;

    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
    dinp[inp_idx] += dq[idx];
    dinp[inp_idx + NH * d] += dk[idx];
    dinp[inp_idx + 2 * (NH * d)] += dv[idx];
}

void unpermute_kernel(sycl::nd_item<1> id, const floatX* inp, floatX *out, int B, int N, int NH, int d) {
    // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = id.get_group(0) * id.get_local_range(0) + id.get_local_id(0);
    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;

    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    out[other_idx] = inp[idx];
}

void unpermute_kernel_backward(sycl::nd_item<1> id, floatX* dinp, const floatX* dout, int B, int N, int NH, int d) {
    int idx = id.get_group(0) * id.get_local_range(0) + id.get_local_id(0);
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;

    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    dinp[idx] += dout[other_idx];
}

void softmax_forward_kernel5(sycl::nd_item<1> id, floatX* out, float inv_temperature, const floatX* inp, int N, int T) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(T % 4  == 0);

    int lane_id = id.get_local_id(0) % WARP_SIZE;
    int warp_id = id.get_local_id(0) / WARP_SIZE;
    int num_warps = id.get_local_range(0) / WARP_SIZE;

    sycl::sub_group warp = id.get_sub_group();

    // micro-optimization: we iterate backwards so that
    // after the softmax backward operation completes, the cache retains the
    // part of the matrix close to the upper left corner, which benefits the
    // matmul operation that immediately follows.
    int idx = (id.get_group_range(0) - id.get_group(0) - 1) * num_warps + warp_id; // backward order
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const floatX* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    const float flt_max = 340282346638528859811704183484516925440.0f; // to avoid including float.h
    float maxval = -flt_max;
    float sumval = 0.0f;

    const floatX* x_aligned = reinterpret_cast<const floatX*>(__builtin_assume_aligned(x, 16));
    for (int i = lane_id; i < pos_by_4; i += WARP_SIZE) {
        float regarray[4];
        for (int k = 0; k < 4; ++k) {
            regarray[k] = (float)x_aligned[4*i + k];
        }
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = sycl::fmax(maxval, regarray[k]);
        }
        sumval *= sycl::exp(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += sycl::exp(inv_temperature * (regarray[k] - maxval));
        }
    }

    if(4*pos_by_4 + lane_id <= own_pos) {
        float old_maxval = maxval;
        maxval = sycl::fmax(maxval, (float)x[4*pos_by_4 + lane_id]);
        sumval *= sycl::exp(inv_temperature * (old_maxval - maxval));
        sumval += sycl::exp(inv_temperature * ((float)x[4*pos_by_4 + lane_id] - maxval));
    }

    float global_maxval = warpReduceMax(warp, maxval);
    sumval *= sycl::exp(inv_temperature * (maxval - global_maxval));

    float sum = warpReduceSum(warp, sumval);
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = lane_id; i <= own_pos; i += WARP_SIZE) {
        // recalculation is faster than doing the round-trip through memory.
        float ev = sycl::exp(inv_temperature * ((float)x[i] - global_maxval));
        out[idx * T + i] = (floatX)ev * norm;
    }
}

void softmax_autoregressive_backward_inplace_kernel(sycl::nd_item<2> id, floatX* datt, const floatX* att,
                                                    int B, int T, int C, float scale) {
    constexpr const int BlockSize = 256;
    constexpr int T_per_block = 4;

    // go through blocks in reverse order, so the slowest block starts first
    int t0 = T - 1 - T_per_block * id.get_group(1);
    int idx = id.get_group(0);
    sycl::group block = id.get_group();

    att += idx * T * T;
    datt += idx * T * T;

    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;
        if(t < 0) return;
        const floatX* att_bth = att + t * T;
        const floatX* datt_bth = datt + t * T;
        floatX* dpreatt_bth = datt + t * T;

        float local_sum = 0;
        for (int t2 = id.get_local_id(1); t2 <= t; t2 += BlockSize) {
            local_sum += (float)att_bth[t2] * (float)datt_bth[t2];
        }

        local_sum = sycl::reduce_over_group(block, local_sum, sycl::plus<float>());

        for (int t3 = id.get_local_id(1); t3 < T; t3 += BlockSize) {
            // don't touch the cache. Some parts will still be here from the previous loop, and
            // we want to exploit those.
            if(t3 <= t) {
                float acc = (float)att_bth[t3] * ((float)datt_bth[t3] - local_sum);
                dpreatt_bth[t3] = (floatX) (scale * acc);
            } else {
                // explicitly set non-causal elements to zero
                dpreatt_bth[t3] = (floatX)0.f;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

void attention_forward(sycl::queue *stream, floatX* out, floatX* qkvr, floatX* att,
                       floatX* inp,
                       int B, int T, int C, int NH) {
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    // Its contents will be overwritten by this function.
    const int block_size = 256;
    const float alpha = 1.0f, beta = 0.0f;

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    stream->parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        permute_kernel(id, q, k, v, inp, B, T, NH, HS);
    }).wait();

    floatX* preatt = inp;

    auto trans = oneapi::mkl::transpose::trans;
    auto no_trans = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::blas::column_major::gemm_batch(*stream,
                                                trans, no_trans,
                                                T, T, HS,
                                                &alpha,
                                                k, HS, T * HS,
                                                q, HS, T * HS,
                                                &beta,
                                                preatt, T, T * T,
                                                B * NH,
                                                oneapi::mkl::blas::compute_mode::float_to_bf16
    );

    // multiply all elements of preatt elementwise by scale
    // Use a float literal because Intel client GPUs do not support fp64
    float scale = 1.0f / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * T * WARP_SIZE, block_size);
    stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        softmax_forward_kernel5(id, att, scale, preatt, B * NH, T);
    }).wait();

    // new approach: first cuBLAS another batched matmul
    floatX* vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    oneapi::mkl::blas::column_major::gemm_batch(*stream,
                                                no_trans, no_trans,
                                                HS, T, T,
                                                &alpha,
                                                v, HS, T * HS,
                                                att, T, T * T,
                                                &beta,
                                                vaccum, HS, T * HS,
                                                B * NH,
                                                oneapi::mkl::blas::compute_mode::float_to_bf16
    );
    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV(B * T * C, block_size);
    stream->parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        unpermute_kernel(id, vaccum, out, B, T, NH, HS);
    }).wait();
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
void attention_backward(sycl::queue *stream, floatX* dinp, floatX* dqkvr, floatX* datt, floatX* scratch,
                        const floatX* dout,
                        const floatX* qkvr, const floatX* att,
                        int B, int T, int C, int NH) {
    const int block_size = 256;
    int HS = C / NH; // head size
    const float alpha = 1.0f, beta = 0.0f;

    // unpack convenience pointers into q, k, v
    const floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    floatX *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;

    // backward through the unpermute operation
    int num_blocks = CEIL_DIV(B * T * C, block_size);
    stream->parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        unpermute_kernel_backward(id, scratch, dout, B, T, NH, HS);
    }).wait();

    auto trans = oneapi::mkl::transpose::trans;
    auto no_trans = oneapi::mkl::transpose::nontrans;
    // backward into datt
    oneapi::mkl::blas::column_major::gemm_batch(
            *stream,
            trans, no_trans,
            T, T, HS,
            &alpha,
            v, HS, T * HS,
            scratch, HS, T * HS,
            &beta,
            datt, T, T * T,
            B * NH
    );

    // backward into dv
    oneapi::mkl::blas::column_major::gemm_batch(
            *stream,
            no_trans, trans,
            HS, T, T,
            &alpha,
            scratch, HS, T * HS,
            att, T, T * T,
            &beta,
            dv, HS, T * HS,
            B * NH
    );

    const float scale = 1.0f / sqrtf((float)HS);
    // backward into preatt. this is an in-place operation; datt turns into dpreatt here
    stream->parallel_for(sycl::nd_range<2>(sycl::range<2>(B * NH, (T / 4) * 256),
                                   sycl::range<2>(1, 256)), [=](sycl::nd_item<2> id) {
        softmax_autoregressive_backward_inplace_kernel(id, datt, att, B, T, C, scale);
    }).wait();

    floatX* dpreatt = datt;
    // backward into q
    oneapi::mkl::blas::column_major::gemm_batch(
            *stream,
            no_trans, no_trans,
            HS, T, T,
            &alpha,
            k, HS, T * HS,
            dpreatt, T, T * T,
            &beta,
            dq, HS, T * HS,
            B * NH
    );
    // backward into k
    oneapi::mkl::blas::column_major::gemm_batch(
            *stream,
            no_trans, trans,
            HS, T, T,
            &alpha,
            q, HS, T * HS,
            dpreatt, T, T * T,
            &beta,
            dk, HS, T * HS,
            B * NH
    );

    // backward into inp
    num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    stream->parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        permute_kernel_backward(id, dinp, dq, dk, dv, B, T, NH, HS);
    }).wait();
}

#endif //LLM_SYCL_ATTENTION_HPP
