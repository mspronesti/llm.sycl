#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include <iostream>
#include <cmath>
#include <cstdlib>
#include "common.hpp"

// ----------------------------------------------------------------------------
// CPU code reference

/*
NOTE:
This version of attention_forward is modified to be consistent with the
attention_forward GPU kernel in the following way small but important way:
- preatt is only QUERY @ KEY, without the scale
- the scale instead moved and fused into the softmax
- the full preatt matrix is materialized, even the parts that get masked out
    - this doesn't actually change anything due to masking, but it lets us
      easily compare to the GPU version, which also does the full, dense sgemm
In this way we'll be able to make sure that preatt and att agree CPU vs GPU
*/
void attention_forward_cpu(float* out, float* preatt, float* att,
                           float* inp,
                           int B, int T, int C, int NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sycl::sqrt((float)hs);

#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 < T; t2++) { // used to be t2 <= t
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(scale * (preatt_bth[t2] - maxval));
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
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

// NOTE: Also contains the re-shuffling of the exact position of "scale"
// and when it is applied (after preatt, not "during" preatt)
// also, full matrices are materialized, even the parts that get masked out
void attention_backward_cpu(float* dinp, float* dpreatt, float* datt,
                            float* dout, float* inp, float* att,
                            int B, int T, int C, int NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                float* dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 < T; t2++) { // ADJUSTED! this was t2 <= t (see note on function)
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += scale * local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += query_t[i] * key_t2[i]
                        // so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2];
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2];
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels
// the forward pass that is the sequence [permute, sgemm, softmax, sgemm, unpermute]

void permute_kernel(sycl::nd_item<1> id,
                    float* q, float* k, float* v,
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

        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + NH * d];
        v[idx] = inp[inp_idx + 2 * (NH * d)];
    }
}

void permute_kernel_backward(sycl::nd_item<1> id,
                             float* dinp,
                             const float* dq, const float* dk, const float* dv,
                             int B, int N, int NH, int d) {
    int idx = id.get_global_id(0);
    if (idx < B * NH * N * d) {
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

void unpermute_kernel_backward(sycl::nd_item<1> id, float* dinp, const float *dout, int B, int N, int NH, int d) {
    int idx = id.get_global_id(0);
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        dinp[idx] += dout[other_idx];
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

// --- Softmax Autoregressive Backward kernels ---

// naive kernel to backward through an autoregressive softmax, just to get correctness
void softmax_autoregressive_backward_kernel1(sycl::nd_item<1> id,
                                             float* dpreatt, const float* datt, const float* att,
                                             int B, int T, int C, int NH) {
    // dpreatt, datt, att are all (B, NH, T, T)
    int t3 = id.get_global_id(0);
    if (t3 < T) {
        int hs = C / NH; // head size
        float scale = 1.0f / sycl::sqrt(static_cast<float>(hs));
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < NH; h++) {
                for (int t = t3; t < T; t++) {
                    const float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                    const float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                    float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                    float accum = 0.0f;
                    for (int t2 = 0; t2 <= t; t2++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        accum +=  scale * local_derivative * datt_bth[t2];
                    }
                    dpreatt_bth[t3] = accum;
                }
            }
        }
    }
}

// parallelize across t,b,h
void softmax_autoregressive_backward_kernel2(sycl::nd_item<2> id,
                                             float* dpreatt, const float* datt, const float* att,
                                             int B, int T, int C, int NH) {
    int t3 = id.get_global_id(1);
    int idx = id.get_group(0) * T * T;
    if (t3 >= T) { return; }

    int hs = C / NH; // head size
    float scale = 1.0f / sycl::sqrt(static_cast<float>(hs));
    for (int t = t3; t < T; t++) {
        float result = 0.0f;
        const float* att_bth = att + idx + t*T;
        const float* datt_bth = datt + idx + t*T;
        float* dpreatt_bth = dpreatt + idx + t*T;

        for (int t2 = 0; t2 <= t; t2++) {
            float indicator = t2 == t3 ? 1.0f : 0.0f;
            float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
            result += scale * local_derivative * datt_bth[t2];
        }

        dpreatt_bth[t3] = result;
    }
}

// parallelize across t,b,h
void softmax_autoregressive_backward_kernel3(sycl::nd_item<2> id,
                                             float* dpreatt, const float* datt, const float* att,
                                             int B, int T, int C, int NH) {
    sycl::sub_group warp = id.get_sub_group();
    int t3 = id.get_group(1) * warp.get_group_linear_range() + warp.get_group_linear_id();

    int idx = id.get_group(0) * T * T;
    if (t3 >= T) { return; }

    int hs = C / NH; // head size
    float scale = 1.0f / sycl::sqrt(static_cast<float>(hs));
    for (int t = t3; t < T; t++) {
        float result = 0.0f;
        const float* att_bth = att + idx + t*T;
        const float* datt_bth = datt + idx + t*T;
        float* dpreatt_bth = dpreatt + idx + t*T;
        const float att_at_t3 = att_bth[t3];

        for (int t2 = warp.get_local_linear_id(); t2 <= t; t2 += warp.get_max_local_range()[0]) {
            float indicator = t2 == t3 ? 1.0f : 0.0f;
            float local_derivative = att_bth[t2] * (indicator - att_at_t3);
            result += local_derivative * datt_bth[t2];
        }

        result = sycl::reduce_over_group(warp, result, sycl::plus<float>());
        if(warp.leader()) {
            dpreatt_bth[t3] = scale * result;
        }
    }
}

void softmax_autoregressive_backward_kernel4(sycl::nd_item<2> id,
                                             float* __restrict__ dpreatt, const float* __restrict__ datt,
                                             const float* __restrict__ att,
                                             int B, int T, int C, int NH) {
    constexpr int UNROLL = 8;
    sycl::sub_group warp = id.get_sub_group();
    int t3 = UNROLL * (id.get_group(1) * warp.get_group_linear_range() + warp.get_group_linear_id());

    int idx = id.get_group(0) * T * T;
    if (t3 >= T) { return; }

    int hs = C / NH; // head size
    float scale = 1.0f / sycl::sqrt(static_cast<float>(hs));

    // the innermost loop combines different values of t2 with different values of t.
    // by handling [t3, t3 + UNROLL) in one thread, we get much better memory reuse:
    // any t3/t-dependent value can be loaded once before the t2 loop.
    // within the t2 loop, we can combine each loaded value with each of the UNROLL
    // pre-loaded values, thus cutting memory ready by a factor of ~UNROLL.

    // one iteration of this loop has to handle the cases
    // this may lead to some invalid indices; therefore, we have several
    // early-outs in the iteration over k below.
    for (int t = t3; t < T; t++) {
        float result[UNROLL] = {};
        const float* att_bth = att + idx + t * T;
        const float* datt_bth = datt + idx + t * T;
        float* dpreatt_bth = dpreatt + idx + t * T;

        float att_at_t3[UNROLL];
        for(int k = 0; k < UNROLL; ++k) {
            if (t < t3 + k) continue;
            att_at_t3[k] = att_bth[t3 + k];
        }

        for (int t2 = warp.get_local_linear_id(); t2 <= t; t2 += warp.get_max_local_range()[0]) {
            float att_t2 = att_bth[t2];
            float datt_t2 = datt_bth[t2];
            for(int k = 0; k < UNROLL; ++k) {
                if (t < t3 + k) continue;
                float indicator = t2 == (t3 + k) ? 1.0f : 0.0f;
                float local_derivative = att_t2 * (indicator - att_at_t3[k]);
                result[k] += local_derivative * datt_t2;
            }
        }

        for(int k = 0; k < UNROLL; ++k) {
            result[k] = sycl::reduce_over_group(warp, result[k], sycl::plus<float>());
        }
        if (warp.get_local_linear_id() < UNROLL) {
            dpreatt_bth[t3 + warp.get_local_linear_id()] = scale * result[warp.get_local_linear_id()];
        }
    }
}

void softmax_autoregressive_backward_kernel5(sycl::nd_item<2> id,
                                             float* __restrict__ dpreatt, const float* __restrict__ datt,
                                             const float* __restrict__ att,
                                             int B, int T, int C, int NH) {
    constexpr int UNROLL = 8;
    sycl::sub_group warp = id.get_sub_group();
    int t3 = UNROLL * (id.get_group(1) * warp.get_group_linear_range() + warp.get_group_linear_id());

    int idx = id.get_group(0) * T * T;
    if (t3 >= T) { return; }

    int hs = C / NH; // head size
    float scale = 1.0f / sycl::sqrt(static_cast<float>(hs));
    for (int t = t3; t < T; t++) {
        float result[UNROLL] = {};
        const float* att_bth = att + idx + t * T;
        const float* datt_bth = datt + idx + t * T;
        float* dpreatt_bth = dpreatt + idx + t * T;

        float att_at_t3[UNROLL];
        for(int k = 0; k < UNROLL; ++k) {
            // if t < t3+k, we're out of bounds.
            // in that case, we don't care what we read, because later on,
            // we won't write the corresponding result. So just clip to
            // make sure this is a valid (in-bounds) memory access.
            att_at_t3[k] = att_bth[sycl::min(t, t3 + k)];
        }

        // the code below is actually just a for loop; except,
        // we have to do something special in one iteration in
        // the middle, and an if turned out to have significant
        // performance impact.
        // so we split the loop in three parts. Ugly, but effective.

        // the beginning/end loop does the same thing, so we write the code
        // just once in a lambda. In this step, we're guaranteed that
        // indicator == 0
        auto loop_step = [&](int t2){
            float p = att_bth[t2] * datt_bth[t2];
            for (int k = 0; k < UNROLL; ++k) {
                result[k] -= p * att_at_t3[k];
            }
        };

        // Now the actual loop.
        {
            // declare the loop iterator. Needs to be kept across the
            // three different parts, so it's not a local variable in
            // the for loop.
            int t2 = warp.get_local_linear_id();

            // first part, as long as t2 < t3, indicator == 0
            for (; t2 < t3; t2 += warp.get_max_local_range()[0]) {
                loop_step(t2);
            }

            // because k <= warp.size() (==32), the event that t3+k == t2
            // has to happen at this particular step.
            static_assert(UNROLL <= 32, "UNROLL is too large, this won't produce correct results.");
            if (t2 <= t) {
                float att_t2 = att_bth[t2];
                float datt_t2 = datt_bth[t2];
                float p = att_t2 * datt_t2;
                for (int k = 0; k < UNROLL; ++k) {
                    float indicator = t2 == (t3 + k) ? 1.0f : 0.0f;
                    result[k] += p * (indicator - att_at_t3[k]);
                }
                t2 += warp.get_max_local_range()[0];
            }

            // rest of the loop, indicator == 0 again
            for (; t2 <= t; t2 += warp.get_max_local_range()[0]) {
                loop_step(t2);
            }
        }

        for(int k = 0; k < UNROLL; ++k) {
            result[k] = sycl::reduce_over_group(warp, result[k], sycl::plus<float>());
        }

        // when storing, we need to check that this is actually a valid result.
        // here, warp.thread_rank() corresponds to `k` in the previous loops.
        if (warp.get_local_linear_id() < UNROLL && t >= t3 + warp.get_local_linear_id()) {
            dpreatt_bth[t3 + warp.get_local_linear_id()] = scale * result[warp.get_local_linear_id()];
        }
    }
}

// I want `BlockSize` to be statically known to the compiler, thus we get a template here.
// This kernel takes a step back, and looks at the original CPU code again. We have some simple outer loops
// That are independent, (b, t, h), and then the inner loops over (t2, t3) where we're combining elements -- this is
// where we can reuse data and be more efficient
// => handle b, t, h  through block indices; each block does all the work for the (t2, t3) loop cooperatively.
// Now we have two nested loops, and in the inner instruction, we combine indexing from both => this calls for
// loop tiling, and lifting some of the memory ops out of the loop.
// We're in luck here;  if we tile so that t3 is the outer loop, we can get a sinlge write op per result, AND also cache
// the t2-indexed part of the computation, which is the problematic one because it contains a multiplication that now we
// do not have to repeat over and over.
// => do an outer t3 loop where each thread gets one t3 index. Then, do an outer t2 loop in steps of BlockSize, and
// prepare BlockSize many elements for the inner loop. Here, each thread calculates one element and stores it in shmem.
// Then, in the inner t2 loop, each thread reads *all* the elements previously stored and does its computations.
// This way, we do 3*BlockSize loads, but BlockSize^2 computation steps => This kernel is now entirely compute bound.
// To fix up the compute issues, as above, we replace ifs in memory reading with min, and also split the inner loop
// into a large region where we don't have to calculate the indicator, and a small, costly region where we do.
template<int BlockSize>
void softmax_autoregressive_backward_kernel6(sycl::nd_item<2> id,
                                             float* att_bth_s,
                                             float* dpreatt, const float* datt, const float* att,
                                             int B, int T, int C, int NH) {
    sycl::group block = id.get_group();

    int idx = id.get_group(0);
    int t = id.get_group(1);

    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    int hs = C / NH; // head size
    float scale = 1.0f / sycl::sqrt(static_cast<float>(hs));
    const float* att_bth = att + t * T;
    const float* datt_bth = datt + t * T;
    float* dpreatt_bth = dpreatt + t * T;

    int block_steps = ceil_div(t+1, BlockSize);
    // very important: This loop condition needs to be the same for all threads.
    // even if a thread later on is not going to do any work, it needs to participate in the
    // data loading process!
    for (int t3f = 0; t3f < block_steps; ++t3f) {
        int t3 = t3f * BlockSize + block.get_local_linear_id();
        float acc = 0.f;
        float at3 = att_bth[t3];
        for (int t2b = 0; t2b <= t; t2b += BlockSize) {
            int end = sycl::min(t + 1 - t2b, BlockSize);
            sycl::group_barrier(block);
            {
                int t2i = block.get_local_linear_id();
                int t2 = sycl::min(t, t2b + t2i);
                att_bth_s[t2i] = att_bth[t2] * datt_bth[t2];
            }

            sycl::group_barrier(block);
            if(t3f * BlockSize == t2b) {
                for (int t2i = 0; t2i < end; t2i++) {
                    int t2 = t2b + t2i;
                    float indicator = t2 == t3 ? 1.0f : 0.0f;
                    acc += att_bth_s[t2i] * (indicator - at3);
                }
            } else {
                for (int t2i = 0; t2i < end; t2i++) {
                    acc +=  att_bth_s[t2i] * (0.f - at3);
                }
            }
        }
        dpreatt_bth[t3] = scale * acc;
    }
}


// Actually disentangling the loops and simplifying the resulting math gives us this pretty nice kernel.
template<int BlockSize>
void softmax_autoregressive_backward_kernel7(sycl::nd_item<2> id,
                                             float* dpreatt, const float* datt, const float* att,
                                             int B, int T, int C, float scale) {
    sycl::group block = id.get_group();
    int idx = id.get_group(0);
    int t = id.get_group(1);

    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    const float* att_bth = att + t * T;
    const float* datt_bth = datt + t * T;
    float* dpreatt_bth = dpreatt + t * T;

    float local_sum = 0;
    for(int t2 = block.get_local_linear_id(); t2 <= t; t2 += BlockSize) {
        local_sum += att_bth[t2] * datt_bth[t2];
    }

    local_sum = sycl::reduce_over_group(block, local_sum, sycl::plus<float>());

    for (int t3 = block.get_local_linear_id(); t3 <= t; t3 += BlockSize) {
        float acc = att_bth[t3] * (datt_bth[t3] - local_sum);
        dpreatt_bth[t3] = scale * acc;
    }
}

// The slightly less pretty version of kernel 7. Adding in all the dirty tricks that can give us a few more percent
//  - streaming memory access instructions
//  - reordering blocks to prevent tail effect
//  - multiple values of T per block
template<int BlockSize>
void softmax_autoregressive_backward_kernel8(sycl::nd_item<2> id,
                                             float* block_acc,
                                             float* dpreatt, const float* datt, const float* att,
                                             int B, int T, int C, float scale) {
    constexpr int T_per_block = 4;
    sycl::group block = id.get_group();
    sycl::sub_group warp = id.get_sub_group();

    int idx = id.get_group(0);
    // go through blocks in reverse order, so the slowest block starts first
    int t0 = T - 1 - T_per_block*id.get_group(1);

    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    if (warp.get_group_linear_id() == 0) {
        block_acc[warp.get_local_linear_id()] = 0;
    }

    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;
        if(t < 0) return;
        const float* att_bth = att + t * T;
        const float* datt_bth = datt + t * T;
        float* dpreatt_bth = dpreatt + t * T;

        float local_sum = 0;
        for (int t2 = block.get_local_linear_id(); t2 <= t; t2 += BlockSize) {
            local_sum += att_bth[t2] * datt_bth[t2];
        }

        local_sum = sycl::reduce_over_group(block, local_sum, sycl::plus<float>{});

        for (int t3 = block.get_local_linear_id(); t3 <= t; t3 += BlockSize) {
            // don't touch the cache. Some parts will still be here from the previous loop, and
            // we want to exploit those.
            float acc = att_bth[t3] * (datt_bth[t3] - local_sum);
            dpreatt_bth[t3] = scale * acc;
        }
    }
}


// ----------------------------------------------------------------------------
// kernel launchers
void attention_forward(sycl::queue &queue, float* out, float* vaccum, float* qkvr, float* preatt, float* att,
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

    auto trans = oneapi::mkl::transpose::trans;
    auto no_trans = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::blas::column_major::gemm_batch(
            queue,
            trans, no_trans,
            T, T, HS,
            alpha,
            k, HS, T * HS,
            q, HS, T * HS,
            beta,
            preatt, T, T * T,
            B * NH
    );

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);
    int softmax_block_size = 256;
    int grid_size = ceil_div(B * NH * T * 32, softmax_block_size);

    queue.parallel_for(sycl::nd_range<1>(grid_size * softmax_block_size, softmax_block_size), [=](sycl::nd_item<1> id) {
        softmax_forward_kernel5(id, att, scale, preatt, B * NH, T);
    }).wait();

    // new approach: first cuBLAS another batched matmul
    // vaccum = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    oneapi::mkl::blas::column_major::gemm_batch(
            queue,
            no_trans, no_trans,
            HS, T, T,
            alpha,
            v, HS, T * HS,
            att, T, T * T,
            beta,
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

void launch_softmax_1(sycl::queue &q, float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(T, block_size);
    sycl::range<2> block_size_range(1, block_size);
    sycl::range<2> num_blocks_range(B*NH, num_blocks);

    q.parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        softmax_autoregressive_backward_kernel1(id, dpreatt, datt, att, B, T, C, NH);
    }).wait();
}

void launch_softmax_2(sycl::queue &q, float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(T, block_size);
    sycl::range<2> block_size_range(1, block_size);
    sycl::range<2> num_blocks_range(B*NH, num_blocks);

    q.parallel_for(sycl::nd_range<2>(block_size_range * num_blocks_range, block_size_range), [=](sycl::nd_item<2> id) {
        softmax_autoregressive_backward_kernel2(id, dpreatt, datt, att, B, T, C, NH);
    });
}

void launch_softmax_3(sycl::queue &q, float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(32*T, block_size);
    sycl::range<2> block_size_range(1, block_size);
    sycl::range<2> num_blocks_range(B*NH, num_blocks);

    q.parallel_for(sycl::nd_range<2>(block_size_range * num_blocks_range, block_size_range), [=](sycl::nd_item<2> id) {
        softmax_autoregressive_backward_kernel3(id, dpreatt, datt, att, B, T, C, NH);
    });
}

void launch_softmax_4(sycl::queue &q, float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(32/8*T, block_size);
    sycl::range<2> block_size_range(1, block_size);
    sycl::range<2> num_blocks_range(B*NH, num_blocks);

    q.parallel_for(sycl::nd_range<2>(block_size_range * num_blocks_range, block_size_range), [=](sycl::nd_item<2> id) {
        softmax_autoregressive_backward_kernel4(id, dpreatt, datt, att, B, T, C, NH);
    });
}

void launch_softmax_5(sycl::queue &q, float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(32/8*T, block_size);
    sycl::range<2> block_size_range(1, block_size);
    sycl::range<2> num_blocks_range(B*NH, num_blocks);

    q.parallel_for(sycl::nd_range<2>(block_size_range * num_blocks_range, block_size_range), [=](sycl::nd_item<2> id) {
        softmax_autoregressive_backward_kernel5(id, dpreatt, datt, att, B, T, C, NH);
    });
}

template<class Launcher>
void dispatch_launch(Launcher&& launch, int block_size) {
    switch(block_size) {
        case 32:
            return launch(std::integral_constant<int, 32>{});
        case 64:
            return launch(std::integral_constant<int, 64>{});
        case 128:
            return launch(std::integral_constant<int, 128>{});
        case 256:
            return launch(std::integral_constant<int, 256>{});
        case 512:
            return launch(std::integral_constant<int, 512>{});
        case 1024:
            return launch(std::integral_constant<int, 1024>{});
        default:
            assert(false && "Invalid block size");
    }
}

void launch_softmax_6(sycl::queue &q, float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    auto launch = [&](auto int_const) {
        constexpr int block_size = int_const.value;

        sycl::range<2> block_size_range(1, block_size);
        sycl::range<2> num_blocks_range(B * NH, T);
        sycl::nd_range<2> grid(num_blocks_range * block_size_range, block_size_range);
        q.parallel_for(grid, [=](sycl::nd_item<2> id) [[sycl::reqd_work_group_size(1, block_size)]]{
            auto l_mem_ptr = (float*)sycl::ext::oneapi::group_local_memory_for_overwrite<float[block_size]>(
                    id.get_group()).get_raw();
            softmax_autoregressive_backward_kernel6<block_size>(id, l_mem_ptr, dpreatt, datt, att, B, T, C, NH);
        });
    };
    dispatch_launch(launch, block_size);
}

void launch_softmax_7(sycl::queue &q, float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    auto launch = [&](auto int_const) {
        constexpr int block_size = int_const.value;
        sycl::range<2> block_size_range(1, int_const.value);
        sycl::range<2> num_blocks_range(B * NH, T);
        sycl::nd_range<2> grid(num_blocks_range * block_size_range, block_size_range);
        q.parallel_for(grid, [=](sycl::nd_item<2> id) {
            softmax_autoregressive_backward_kernel7<block_size>(id, dpreatt, datt, att, B, T, C, scale);
        });
    };
    dispatch_launch(launch, block_size);
}

void launch_softmax_8(sycl::queue &q, float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    auto launch = [&](auto int_const) {
        constexpr int block_size = int_const.value;
        sycl::range<2> block_size_range(1, int_const.value);
        sycl::range<2> num_blocks_range(B * NH, T / 4);
        sycl::nd_range<2> grid(num_blocks_range * block_size_range, block_size_range);
        q.parallel_for(grid, [=](sycl::nd_item<2> id) {
            auto l_mem_ptr = (float*)sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
                    id.get_group()).get_raw();
            softmax_autoregressive_backward_kernel8<block_size>(id, l_mem_ptr, dpreatt, datt, att, B, T, C, scale);
        });
    };
    dispatch_launch(launch, block_size);
}




// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
template<class SoftmaxKernel>
void attention_backward1(sycl::queue &queue, float* dinp, float* dqkvr, float* dpreatt, float* datt, float* dvaccum,
                         const float* dout,
                         const float* inp, const float* qkvr, const float* preatt, const float* att, const float* vaccum,
                         int B, int T, int C, int NH,
                         SoftmaxKernel softmax_autoregressive_backward,
                         const int block_size) {
    int HS = C / NH; // head size
    const float alpha = 1.0f;
    const float beta = 1.0f; // note beta = 1.0f so that we accumulate gradients (+=)
    // unpack convenience pointers into q, k, v
    const float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    float *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;

    // backward through the unpermute operation
    int num_blocks = ceil_div(B * T * C, block_size);
    queue.parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        unpermute_kernel_backward(id, dvaccum, dout, B, T, NH, HS);
    }).wait();

    auto trans = oneapi::mkl::transpose::trans;
    auto no_trans = oneapi::mkl::transpose::nontrans;
    // backward into datt
    oneapi::mkl::blas::column_major::gemm_batch(
            queue,
            trans, no_trans,
            T, T, HS,
            &alpha,
            v, HS, T * HS,
            dvaccum, HS, T * HS,
            &beta,
            datt, T, T * T,
            B * NH
    );

    // backward into dv
    oneapi::mkl::blas::column_major::gemm_batch(
            queue,
            no_trans, trans,
            HS, T, T,
            &alpha,
            dvaccum, HS, T * HS,
            att, T, T * T,
            &beta,
            dv, HS, T * HS,
            B * NH
    );

    // backward into preatt
    softmax_autoregressive_backward(queue, dpreatt, datt, att, B, T, C, NH, block_size);
    // backward into q
    oneapi::mkl::blas::column_major::gemm_batch(
            queue,
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
            queue,
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
    num_blocks = ceil_div(B * NH * T * HS, block_size);
    queue.parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> id) {
        permute_kernel_backward(id, dinp, dq, dk, dv, B, T, NH, HS);
    }).wait();
}


// kernel version dispatch
void attention_backward(int kernel_num,
                        sycl::queue &q,
                        float* dinp, float* dqkvr, float* dpreatt, float* datt, float* dvaccum,
                        const float* dout,
                        const float* inp, const float* qkvr, const float* preatt, const float* att, const float* vaccum,
                        int B, int T, int C, int NH,
                        const int block_size) {
    switch (kernel_num) {
        case 1:
            attention_backward1(q, dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_1, block_size);
            break;
        case 2:
            attention_backward1(q, dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_2, block_size);
            break;
        case 3:
            attention_backward1(q, dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_3, block_size);
            break;
        case 4:
            attention_backward1(q, dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_4, block_size);
            break;
        case 5:
            attention_backward1(q, dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_5, block_size);
            break;
        case 6:
            attention_backward1(q, dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_6, block_size);
            break;
        case 7:
            attention_backward1(q, dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_7, block_size);
            break;
        case 8:
            attention_backward1(q, dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_8, block_size);
            break;
        default:
            std::cout << "Invalid kernel number\n";
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    // hyperparameters
    int B = 4;
    int T = 1024;
    int C = 768;
    int NH = 12;

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << '\n';

    sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order{});

    // create the host memory for the forward pass
    float* inp = make_random_float(B * T * 3 * C);
    float* qkvr = (float*)malloc(B * T * 3 * C * sizeof(float));
    float* preatt = (float*)malloc(B * NH * T * T * sizeof(float));
    float* att = (float*)malloc(B * NH * T * T * sizeof(float));
    float* vaccum = (float*)malloc(B * T * C * sizeof(float));
    float* out = (float*)malloc(B * T * C * sizeof(float));

    // execute the forward pass on the CPU
    attention_forward_cpu(out, preatt, att, inp, B, T, C, NH);

    // create device memory for the forward pass
    float *d_inp, *d_qkvr, *d_preatt, *d_att, *d_vaccum, *d_out;

    d_inp = sycl::malloc_device<float>(B * T * 3 * C, q);
    d_qkvr = sycl::malloc_device<float>(B * T * 3 * C, q);
    d_preatt = sycl::malloc_device<float>(B * NH * T * T, q);
    d_att = sycl::malloc_device<float>(B * NH * T * T, q);
    d_vaccum = sycl::malloc_device<float>(B * T * C, q);
    d_out = sycl::malloc_device<float>(B * T * C, q);

    // copy over the input
    q.memcpy(d_inp, inp, B * T * 3 * C * sizeof(float)).wait();

    // execute the forward pass on the GPU
    const int block_size = 256;
    attention_forward(q, d_out, d_vaccum, d_qkvr, d_preatt, d_att, d_inp, B, T, C, NH, block_size);

    // check that preatt, att, and out match between the CPU and GPU versions
    std::cout << "Checking the forward pass CPU <-> GPU...\n";
    std::cout << "[preatt]\n"; validate_result(d_preatt, preatt, "preatt", B * T * C, 5e-3f);
    std::cout << "[att]\n";    validate_result(d_att, att, "att", B * T * C, 1e-3f);
    std::cout << "[out]\n";    validate_result(d_out, out, "out", B * T * C, 1e-3f);

    // set up the memory for the backward pass
    float* dout = make_random_float(B * T * C); // the gradients on the output
    float* dinp = make_zeros_float(B * T * 3 * C); // zeros for all else, to += into
    float* dpreatt = make_zeros_float(B * NH * T * T);
    float* datt = make_zeros_float(B * NH * T * T);

    // call backward() on the CPU to get our reference gradients
    attention_backward_cpu(dinp, dpreatt, datt, dout, inp, att, B, T, C, NH);

    // create device memory for the backward pass
    float *d_dinp, *d_dqkvr, *d_dpreatt, *d_datt, *d_dvaccum, *d_dout;
    d_dinp = sycl::malloc_device<float>(B * T * 3 * C, q);
    d_dqkvr = sycl::malloc_device<float>(B * T * 3 * C, q);
    d_dpreatt = sycl::malloc_device<float>(B * NH * T * T, q);
    d_datt = sycl::malloc_device<float>(B * NH * T * T, q);
    d_dvaccum = sycl::malloc_device<float>(B * T * C, q);
    d_dout = sycl::malloc_device<float>(B * T * C, q);

    // copy over the dout gradients that starts the backprop chain
    q.memcpy(d_dout, dout, B * T * C * sizeof(float)).wait();
    // memset all the other memory to zeros, to += into
    q.memset(d_dinp, 0, B * T * 3 * C * sizeof(float)).wait();
    q.memset(d_dqkvr, 0, B * T * 3 * C * sizeof(float)).wait();
    q.memset(d_dpreatt, 0, B * NH * T * T * sizeof(float)).wait();
    q.memset(d_datt, 0, B * NH * T * T * sizeof(float)).wait();
    q.memset(d_dvaccum, 0, B * T * C * sizeof(float)).wait();

    // call backward() on the GPU
    attention_backward(kernel_num, q, d_dinp, d_dqkvr, d_dpreatt, d_datt, d_dvaccum,
                       d_dout, d_inp, d_qkvr, d_preatt, d_att, d_vaccum,
                       B, T, C, NH, block_size);

    // check that the gradients match between the CPU and GPU versions
    // note that we will only check the correctness at [att, preatt, inp]
    // the gradients at qkvr and vaccum will remain unchecked, but are
    // assumed to be correct if the other gradients are correct
    std::cout << "Checking the backward pass CPU <-> GPU...\n";
    std::cout << "[datt]\n";    validate_result(d_datt, datt, "datt", B * NH * T * T, 5e-3f);
    std::cout << "[dpreatt]\n"; validate_result(d_dpreatt, dpreatt, "dpreatt", B * NH * T * T, 1e-3f);
    std::cout << "[dinp]\n";    validate_result(d_dinp, dinp, "dinp", B * T * 3 * C, 1e-3f);

    // also let's manually step through the gradients here
    float* h_dinp = (float*)malloc(B * T * 3 * C * sizeof(float));
    q.memcpy(h_dinp, d_dinp, B * T * 3 * C * sizeof(float)).wait();
    int num_match = 0;
    int num_no_match = 0;
    int num_zero_grad = 0;
    int HS = C / NH;
    for (int i = 0; i < B * T * 3 * C; i++) {

        // the dimensions of inp are (B, T, 3, NH, HS)
        // where B = batch, T = time, 3 = qkv, NH = num heads, HS = head size
        // unpack the individual b,t,qkvix,h,c indices
        int ix = i;
        int c = ix % HS;
        ix /= HS;
        int h = ix % NH;
        ix /= NH;
        int qkvix = ix % 3;
        ix /= 3;
        int t = ix % T;
        ix /= T;
        int b = ix;

        float diff = fabs(dinp[i] - h_dinp[i]);

        // attempt to index at random
        if (b == 1 && t == 5 && c == 23 && h == 2) {
            printf("ix %5d [b=%4d, t=%4d, qkv=%4d, nh=%4d, hs=%4d]: ref: %f gpu: %f\n", i, b, t, qkvix, h, c, dinp[i], h_dinp[i]);
        }

        if (diff > 1e-4f) {
            num_no_match++;
        } else {
            num_match++;
        }

        if (dinp[i] == 0.0f) {
            num_zero_grad++;
        }
    }
    printf("Number of matching gradients: %d (%.2f%% of total)\n", num_match, 100*(float)num_match / (B * T * 3 * C));
    printf("Number of non-matching gradients: %d (%.2f%% of total)\n", num_no_match, 100*(float)num_no_match / (B * T * 3 * C));
    printf("Number of gradients that are exactly zero: %d (%.2f%% of total)\n", num_zero_grad, 100*(float)num_zero_grad / (B * T * 3 * C));

    // final verdict
    printf("All results match. Starting benchmarks.\n\n");

    // benchmark speed of the kernel
    int block_sizes[] = {32, 64, 128, 256, 512};
    for (int block_size: block_sizes) {
        int repeat_times = 10;
        float elapsed_time = benchmark_kernel(repeat_times, attention_backward,
                                              kernel_num, q, d_dinp, d_dqkvr, d_dpreatt, d_datt, d_dvaccum,
                                              d_dout, d_inp, d_qkvr, d_preatt, d_att, d_vaccum,
                                              B, T, C, NH, block_size);

        printf("block_size %4d | time %f ms\n", block_size, elapsed_time);
    }

    // free memory
    free(inp);
    free(qkvr);
    free(preatt);
    free(att);
    free(vaccum);
    free(out);
    free(dout);
    free(dinp);
    free(dpreatt);
    free(datt);
    sycl::free(d_inp, q);
    sycl::free(d_qkvr, q);
    sycl::free(d_preatt, q);
    sycl::free(d_att, q);
    sycl::free(d_vaccum, q);
    sycl::free(d_out, q);
    sycl::free(d_dinp, q);
    sycl::free(d_dqkvr, q);
    sycl::free(d_dpreatt, q);
    sycl::free(d_datt, q);
    sycl::free(d_dvaccum, q);
    sycl::free(d_dout, q);

    return 0;
}