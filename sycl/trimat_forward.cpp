/*
Triangular matrix multiplication as in autoregressive attention. A short story.

*/

#include <sycl/sycl.hpp>
#include <iostream>
#include <cmath>
#include <cassert>

#define ENABLE_ONEDNN
#include "common.hpp"

#ifdef ENABLE_ONEDNN
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#else
#include <oneapi/mkl.hpp>
auto MKL_OP_T = oneapi::mkl::transpose::trans;
auto MKL_OP_N = oneapi::mkl::transpose::nontrans;
#endif

static float* d_qkvr;   // scratch for the onemkl blas kernel


// taken from then attention forward pass
void trimul_cpu(float* out, const float* inp,
                int B, int T, int C, int NH) {
    // inp shape: (B, T, 3, NH, HS)
    // out shape: (B, NH, T, T)
    int C3 = C*3;
    int HS = C / NH; // head size
    float scale = 1.0 / sqrtf(HS);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int nh = 0; nh < NH; nh++) {
                // Q[b][nh][t][:] = inp[b][t][0][nh][:] (where : is the slice operator for hs)
                const float* query_t = inp + b * T * C3 + t * C3 + nh * HS;
                // out[b][nh][t][:]
                float* out_bth = out + b * NH * T * T + nh * T * T + t * T;

                // pass 1: calculate query dot key and maxval
                for (int t2 = 0; t2 <= t; t2++) {
                    // K[b][nh][t2][:] = inp[b][t2][1][nh][:]
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + nh * HS + C; // +C because it's key

                    // Q[b][nh][t][:] dot K[b][nh][t2][:]
                    float val = 0.0f;
                    for (int i = 0; i < HS; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;

                    // out[b][nh][t][t2] = val
                    out_bth[t2] = val;
                }
                for(int t2 = t + 1; t2 < T; ++t2) {
                    // causal mask, using NAN to supress warnings -> it could be -inf
                    // but it doesn't matter because in validate_result we ignore infinities/NANs
                    out_bth[t2] = NAN;
                }
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

// baseline implementation
void matmul_tri_naive(sycl::nd_item<3> id, float* p, int ps, const float* k, int ks, const float* q, int qs, int T, int hs, float alpha) {
    // get coordinates of our block
    int i_base = 128 * id.get_group(2) + 8 * id.get_local_id(2);
    int j_base = 128 * id.get_group(1) + 8 * id.get_local_id(1);

    // one more check to skip the upper diagonal in blocks that are on the diagonal.
    if(j_base > i_base)
        return;

    // Simple nested loop that calculates 8x8 results in one thread.
    for(int io = 0; io < 8; ++io) {
        int i = i_base + io;
        for(int jo = 0; jo < 8; ++jo) {
            int j = j_base + jo;
            float val = 0;
            for (int s = 0; s < hs; ++s) {
                val += q[i * ks + s] * k[j * qs + s];
            }
            p[i * ps + j] = val * alpha;
        }
    }
}


// reorganize loops to enable data reuse
void matmul_tri_registers(sycl::nd_item<3> id, float* p, int PS, const float* k, int KS, const float* q, int QS, int T, int HS, float alpha) {
    int i_base = 128 * id.get_group(2) + 8 * id.get_local_id(2);
    int j_base = 128 * id.get_group(1) + 8 * id.get_local_id(1);

    if (j_base > i_base)
        return;

    // shift our pointers to the sub-block this thread is responsible for
    q += i_base * QS;
    k += j_base * KS;
    p += i_base * PS + j_base;

    float vals[8][8] = {};
    for (int hs = 0; hs < HS; ++hs) {
        float lhs[8];
        float rhs[8];
        for (int u = 0; u < 8; ++u) {
            lhs[u] = q[u * QS + hs];
            rhs[u] = k[u * KS + hs];
        }

        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                vals[i][j] += lhs[i] * rhs[j];
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            p[i * PS + j] = vals[i][j] * alpha;
        }
    }
}

// convenient helper functions to make the code below more readable
sycl::float4 ld_vec(const float* address) {
    return *reinterpret_cast<const sycl::float4*>(address);
}

void st_vec(float* address, sycl::float4 val) {
    *reinterpret_cast<sycl::float4*>(address) = val;
}

// vector instructions for coalesced memory access: 1.7 ms
void matmul_tri3(sycl::nd_item<3> id, float* p, int PS, const float* k, int KS, const float* q, int QS, int T, int HS, float alpha) {
    int i_base = 128 * id.get_group(2) + 8 * id.get_local_id(2);
    int j_base = 128 * id.get_group(1) + 8 * id.get_local_id(1);

    if (j_base > i_base)
        return;

    // shift our pointers to the sub-block this thread is responsible for
    q += i_base * QS;
    k += j_base * KS;
    p += i_base * PS + j_base;

    float vals[8][8] = {};
    for (int hs = 0; hs < HS; hs += 4) {
        // load in float4 to improve coalescing
        sycl::float4 rhs[8];
        for (int u = 0; u < 8; ++u) {
            rhs[u] = ld_vec(k + u * KS + hs);
        }

        for (int i = 0; i < 8; ++i) {
            // no need to keep lhs around for the i loop, it's only reused in the j loop anyway.
            sycl::float4 lhs = ld_vec(q + i * QS + hs);
            for (int j = 0; j < 8; ++j) {
                vals[i][j] += lhs.x() * rhs[j].x();
                vals[i][j] += lhs.y() * rhs[j].y();
                vals[i][j] += lhs.z() * rhs[j].z();
                vals[i][j] += lhs.w() * rhs[j].w();
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; j += 4) {
            sycl::float4 result;
            result.x() = vals[i][j + 0] * alpha;
            result.y() = vals[i][j + 1] * alpha;
            result.z() = vals[i][j + 2] * alpha;
            result.w() = vals[i][j + 3] * alpha;
            st_vec(p + i * PS + j, result);
        }
    }
}

void matmul_tri4(sycl::nd_item<3> id, float* p, int PS, const float* k, int KS, const float* q, int QS, int T, int HS, float alpha,
                 sycl::multi_ptr<float[128][32][2], sycl::access::address_space::local_space> local_acc) {

    // thread indices in "cuda style"
    int threadIdx_x = id.get_local_id(2);
    int threadIdx_y = id.get_local_id(1);
    int blockIdx_x = id.get_group(2);
    int blockIdx_y = id.get_group(1);

    int i_base = 128 * blockIdx_x + 8 * threadIdx_x;
    int j_base = 128 * blockIdx_y + 8 * threadIdx_y;

    // we need all threads for loading data, so none of them can chicken out early, even
    // if they are not responsible for any useful result.
    if (blockIdx_y > blockIdx_x)
        return;

    q += 128 * blockIdx_x * QS;
    k += 128 * blockIdx_y * KS;

    float *shared = (float*) local_acc.get_raw();
    float (*lhs_s)[32] = (float (*)[32]) shared;
    float (*rhs_s)[32] = (float (*)[32]) (shared + 128 * 32);

    float vals[8][8] = {};
    for (int so = 0; so < HS; so += 32) {
        // Read a large slice of the input, worked on together by all threads.
        // They are organized differently for this part. We want to ensure
        // fully coalesced loads, so we let a single warp handle consecutive
        // addresses, which means we need to combine two threadIdx.y values
        // in one read operation.
        // note: threads may read data here that they don't need themselves.
        //       this really is a block-level operation.
        // note2: 16x16 threads (i.e. the block) will, through this for loop, fetch 32 dims from 128 keys and 128 queries
        // i.e. from Q/K, of shape (T, HS) take q[:128, so*32:(so+1)*32] and k[:128, so*32:(so+1)*32]
        sycl::group_barrier(id.get_group());
        for(int y = threadIdx_y / 2; y < 128; y += 8) {
            int xo = (threadIdx_y % 2) * 16;
            lhs_s[y][threadIdx_x + xo] = q[y * QS + so + threadIdx_x + xo];
            rhs_s[y][threadIdx_x + xo] = k[y * KS + so + threadIdx_x + xo];
        }
        sycl::group_barrier(id.get_group());

        // Now we compute a partial dot product (only 32 dims) for all combinations of keys and queries (128x128).
        // Each thread does 8x8 of these partial dot products.
        // E.g. thread (0,0) covers queries 0-7 and keys 0-7. More generally first row of threads
        // (0,:) covers queries 0-7 with keys 0-127 and so on.
        // In the next iterations of the outer (`so`) loop we'll be accumulating values to `vals` until we
        // get the full dot product. We then later deposit it into the output matrix for all 8x8 blocks
        // that are below the diagonal.
        for (int si = 0; si < 32; ++si) {
            float rhs[8];
            for (int u = 0; u < 8; ++u) {
                rhs[u] = rhs_s[u + 8 * threadIdx_y][(si + threadIdx_x) % 32];
            }

            for (int ii = 0; ii < 8; ++ii) {
                float lhs = lhs_s[ii + 8 * threadIdx_x][(si + threadIdx_x) % 32];
                for (int ji = 0; ji < 8; ++ji) {
                    vals[ii][ji] += lhs * rhs[ji];
                }
            }
        }
    }

    // don't write above the diagonal
    if (j_base > i_base)
        return;

    for (int ii = 0; ii < 8; ++ii) {
        for (int ji = 0; ji < 8; ji += 4) {
            int i = i_base + ii;
            int j = j_base + ji;
            sycl::float4 result;
            result.x() = vals[ii][ji + 0] * alpha;
            result.y() = vals[ii][ji + 1] * alpha;
            result.z() = vals[ii][ji + 2] * alpha;
            result.w() = vals[ii][ji + 3] * alpha;
            st_vec(p + i * PS + j, result);
        }
    }
}

// oneDNN or oneMKL blas version
void trimul_blas(sycl::queue &queue, float* preatt,
                   const float* inp,
                   int B, int T, int C, int NH) {
    int HS = C / NH; // head size
    int total_threads = B * NH * T * HS;

    // Permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float* q = d_qkvr + 0 * B * T * C;
    float* k = d_qkvr + 1 * B * T * C;
    float* v = d_qkvr + 2 * B * T * C;

    int num_blocks = ceil_div(total_threads, 256);
    // Launch SYCL kernel for permutation
    queue.parallel_for(sycl::nd_range<1>(num_blocks * 256, 256), [=](sycl::nd_item<1> id) {
        permute_kernel(id, q, k, v, inp, B, T, NH, HS);
    });


    const float alpha = 1.0f / std::sqrt(HS);
    const float beta = 0.0f;
    // This schedules in parallel B*NH matmuls of shape q@k^t = (T, HS) @ (HS, T) = (T, T).
    // IMPORTANT NOTE: Cublas uses a column-major (and we use row-major in our codebase) representation,
    // so this call might look confusing to you if you look at the `cublasSgemmStridedBatched` signature.
    //
    // In order to avoid having to do an additional transpose operation after this func call,
    // we need to pass in K as the first argument and Q as the second argument, which might make you think we're computing K^T @ Q.
    // That combined with the shapes we got after the permute kernel - (B, NH, T, HS) (I'll omit B, NH for brevity going forward)
    // and you might think we end up with (HS, T) @ (T, HS) = (HS, HS).
    // This is not the case. :)
    //
    // Cublas sees our row-major matrix (T, HS) as (HS, T), hence we set the lead dimensions to HS (see function signature).
    // We transpose K and end up computing K^T @ Q = (T, HS) @ (HS, T) = (T, T).
    // If you were to interpret the above formula K^T @ Q you might think we end up with:
    // -----------------------------------
    // k1.dot(q1) k1.dot(q2) ... k1.dot(qT)
    // k2.dot(q1) k2.dot(q2) ... k2.dot(qT)
    // ...
    // kT.dot(q1) kT.dot(q2) ... kT.dot(qT)
    // -----------------------------------
    // But as I mentioned, Cublas is column-major!
    // So given that the dot product is symmetric we can write k1.dot(q1) as q1.dot(k1) and transposing the above
    // representation we can see what we actually end up with in the row-major format:
    // -----------------------------------
    // q1.dot(k1) q1.dot(k2) ... q1.dot(kT)
    // q2.dot(k1) q2.dot(k2) ... q2.dot(kT)
    // ...
    // qT.dot(k1) qT.dot(k2) ... qT.dot(kT)
    // -----------------------------------
    // which is exactly what we wanted! :)
    // this takes far more than expected though ...
#ifdef ENABLE_ONEDNN
    auto engine = dnnl::sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto stream = dnnl::sycl_interop::make_stream(engine, queue);

    // Memory descriptors
    auto q_md = dnnl::memory::desc({B * NH, T, HS}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc);
    auto k_md = dnnl::memory::desc({B * NH, HS, T}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::acb);
    auto preatt_md = dnnl::memory::desc({B * NH, T, T}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc);

    // Memory objects
    auto q_mem = dnnl::sycl_interop::make_memory(q_md, engine, dnnl::sycl_interop::memory_kind::usm, q);
    auto k_mem = dnnl::sycl_interop::make_memory(k_md, engine, dnnl::sycl_interop::memory_kind::usm, k);
    auto preatt_mem = dnnl::sycl_interop::make_memory(preatt_md, engine, dnnl::sycl_interop::memory_kind::usm, preatt);

    dnnl::primitive_attr matmul_attr;

    // Primitive descriptor
    auto matmul_pd = dnnl::matmul::primitive_desc(engine, q_md, k_md, preatt_md, matmul_attr);

    // Matmul primitive
    auto matmul_prim = dnnl::matmul(matmul_pd);

    // Run matmul
    matmul_prim.execute(stream, {
        {DNNL_ARG_SRC, q_mem},
        {DNNL_ARG_WEIGHTS, k_mem},
        {DNNL_ARG_DST, preatt_mem}
    });

    stream.wait();
#else
    oneapi::mkl::blas::column_major::gemm_batch(queue,
                                             MKL_OP_T, MKL_OP_N,
                                             T, T, HS,
                                             alpha,
                                             k, HS, T * HS,
                                             q, HS, T * HS,
                                             beta,
                                             preatt, T, T * T,
                                             B * NH);
#endif
}


// ----------------------------------------------------------------------------
// Generalized kernel launcher

// using creates an alias for a function pointer
using matmul_fn_ptr = void(*)(sycl::nd_item<3> id, float* p, int PS, const float* k, int KS, const float* q, int QS, int T, int HS, float alpha);

template<matmul_fn_ptr matmul_tri>
void trimul_launcher(sycl::queue &queue, float* out, const float* inp, int B, int T, int C, int NH) {
    // we assume nice shapes here. Let's not make the code a mess by supporting weird shapes that you
    // wouldn't want to use anyway.
    assert(T % 128 == 0);
    // No need to ceil_div, if it's not a multiple of 128, we would get wrong results anyway.
    sycl::range<3> grid_dim(NH * B, T / 128, T / 128);
    sycl::range<3> block_dim(1, 16, 16);

    queue.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> id) {
        // skip above the diagonal
        if (id.get_group(1) > id.get_group(2))
            return;

        // set up indices
        int C3 = C * 3;
        int HS = C / NH; // head size
        float scale = 1.0f / sycl::sqrt(static_cast<float>(HS));

        // we put the "batch x head" dimension into the z block index.
        int b = id.get_group(0) / NH;
        int nh = id.get_group(0) % NH;

        // Get the base address for the current batch and head
        // shapes -> inp (B, T, 3, NH, HS), Q (B, NH, T, HS), K (B, NH, T, HS)
        const float* q = inp + b * T * C3 + nh * HS;  // Q[b][nh][:][:] = inp[b][:][0][nh][:]
        const float* k = inp + b * T * C3 + nh * HS + C;  // K[b][nh][:][:] = inp[b][:][1][nh][:]
        float* r = out + (b*NH + nh)*T*T;  // out[b][nh][:][:]

        matmul_tri(id, r, T, k, C3, q, C3, T, HS, scale);
    }).wait();
}

// Identical to the above `trimul_launcher`, but copes with the different
// prototype of the `matmul_tri4` function.
// TODO: find a way to avoid this code duplication.
void trimul_launcher_matmul_tri4(sycl::queue &q, float* out, const float* inp, int B, int T, int C, int NH) {
    // we assume nice shapes here. Let's not make the code a mess by supporting weird shapes that you
    // wouldn't want to use anyway.
    assert(T % 128 == 0);
    // No need to ceil_div, if it's not a multiple of 128, we would get wrong results anyway.
    sycl::range<3> grid_dim(NH * B, T / 128, T / 128);
    sycl::range<3> block_dim(1, 16, 16);
    q.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim), [=](sycl::nd_item<3> id) {
        // skip above the diagonal
        if (id.get_group(1) > id.get_group(2))
            return;

       // set up indices
       int C3 = C*3;
       int HS = C / NH; // head size
       float scale = 1.0f / sycl::sqrt(static_cast<float>(HS));

        // we put the "batch x head" dimension into the z block index.
        int b = id.get_group(0) / NH;
        int nh = id.get_group(0) % NH;

        // Get the base address for the current batch and head
        // shapes -> inp (B, T, 3, NH, HS), Q (B, NH, T, HS), K (B, NH, T, HS)
        const float* q = inp + b * T * C3 + nh * HS;  // Q[b][nh][:][:] = inp[b][:][0][nh][:]
        const float* k = inp + b * T * C3 + nh * HS + C;  // K[b][nh][:][:] = inp[b][:][1][nh][:]
        float* r = out + (b*NH + nh)*T*T;  // out[b][nh][:][:]

       auto local_acc = sycl::ext::oneapi::group_local_memory_for_overwrite<float[128][32][2]>(
               id.get_group());
       matmul_tri4(id, r, T, k, C3, q, C3, T, HS, scale, local_acc);
   }).wait();
}

// ----------------------------------------------------------------------------
// Dispatcher
void trimul_gpu(int kernel_num,
                sycl::queue &q,
                float* out,  const float* inp,
                int B, int T, int C, int NH) {
    switch (kernel_num) {
        case 0:
            trimul_blas(q, out, inp, B, T, C, NH);
        case 1:
            trimul_launcher<matmul_tri_naive>(q, out, inp, B, T, C, NH);
            break;
        case 2:
            trimul_launcher<matmul_tri_registers>(q, out, inp, B, T, C, NH);
            break;
        case 3:
            trimul_launcher<matmul_tri3>(q, out, inp, B, T, C, NH);
            break;
        case 4:
            trimul_launcher_matmul_tri4(q, out, inp, B, T, C, NH);
            break;
        default:
            std::cout << "Invalid kernel number\n";
            exit(1);
    }
}

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    int NH = 12;

    // set up the device
    sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order{});

    // create host memory of random numbers
    float* out = (float*)malloc(B * NH * T * T * sizeof(float));
    float* inp = make_random_float(B * T * 3 * C);

    // move to GPU
    float* d_out;
    float* d_inp;

    d_qkvr = sycl::malloc_device<float>(B * T * 3 * C, q);

    d_out = sycl::malloc_device<float>(B * NH * T * T, q);
    d_inp = sycl::malloc_device<float>(B * T * 3 * C, q);

    q.memcpy(d_inp, inp, B * T * 3 * C * sizeof(float)).wait();

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << '\n';

    // first check the correctness of the kernel
    trimul_cpu(out, inp, B, T, C, NH);
    trimul_gpu(kernel_num, q, d_out, d_inp, B, T, C, NH);
    validate_result(d_out, out, "out", B * NH * T * T, 1e-4f);

    std::cout << "All results match. Starting benchmarks.\n";

    // benchmark speed of the kernel
    int repeat_times = 100;

    float elapsed_time = benchmark_kernel(
            repeat_times,
            trimul_gpu,
            kernel_num, q, d_out, d_inp, B, T, C, NH
    );

    float blas_time = benchmark_kernel(
        repeat_times,
        trimul_gpu,
        0, // kernel 0 == oneMKL blas kernel
        q, d_out, d_inp, B, T, C, NH
    );

    std::cout << "time " << elapsed_time << " ms vs " << blas_time << " ms for blas\n";

    // free memory
    free(out);
    free(inp);
    sycl::free(d_out, q);
    sycl::free(d_inp, q);
    sycl::free(d_qkvr, q);

    return 0;
}