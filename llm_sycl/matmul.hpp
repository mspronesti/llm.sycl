/*
Matrix Multiplication, with help from cuBLASLt
*/
#ifndef LLM_SYCL_MATMUL_HPP
#define LLM_SYCL_MATMUL_HPP

#include <cassert>
#include <type_traits>      // std::bool_constant

#include "sycl_common.hpp"
#include "sycl_utils.hpp"

// oneDNN
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

// ----------------------------------------------------------------------------
// SYCL kernels

template<typename OutFloat, bool UseAuxBuffer>
void matmul_backward_bias_kernel9(sycl::nd_item<3> id, OutFloat* dbias, const floatX* dout, int B, int T, int OC,
                                 std::bool_constant<UseAuxBuffer>, sycl::local_accessor<float> local_acc){
    constexpr const int bdx = 4;
    constexpr const int bdy = WARP_SIZE / bdx;

    assert(id.get_local_range(0) == bdx);
    assert(id.get_local_range(1) == bdy);

    sycl::sub_group warp = id.get_sub_group();

    int warp_d = (int) id.get_local_id(0);
    int warp_c = (int) id.get_local_id(1);
    int block_d = (int) id.get_local_id(2);


    const int OC_per_warp = bdy * x128::size;  // 64 at BF16

    int local_oc = warp_c * x128::size;
    int global_oc = id.get_group(2) * OC_per_warp + local_oc;

    int local_bt = warp_d + bdx * block_d;
    int bt_per_block = bdx * id.get_local_range(2);

    float accumulators[x128::size];
    for (int k = 0; k < x128::size; k++) {
        accumulators[k] = 0.0f;
    }

    if (global_oc < OC) {
        // sum up over all bt within registers
        for (int idx = id.get_group(1) * bt_per_block + local_bt;
             idx < B * T; idx += id.get_group_range(1) * bt_per_block) {
            x128 packed_dout = load128(dout + global_oc + idx * OC);
            for (int k = 0; k < x128::size; k++) {
                accumulators[k] += (float) packed_dout[k];
            }
        }
    }

    float* shared = local_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
    float (*sub_results)[WARP_SIZE][bdy] = (float (*)[WARP_SIZE][bdy])shared;

    // reduce within-warp results
    for (int k = 0; k < x128::size; k++) {
        float v = accumulators[k];
        v += shuffle_down(warp, v, 1, 4);
        v += shuffle_down(warp, v, 2, 4);
        if(warp_d == 0) {
            sub_results[k][block_d][warp_c] = v;
        }
    }
    sycl::group_barrier(id.get_group());

    // block-wide reductions
    for (int k = block_d; k < x128::size; k += id.get_local_range(0)) {
        float a = 0.f;
        for (int r = warp_d; r < id.get_local_range(0); r += bdx) {
            float v = sub_results[k][r][warp_c];
            v += shuffle_down(warp, v, 1, 4);
            v += shuffle_down(warp, v, 2, 4);
            a += v;
        }
        if(warp_d == 0 && global_oc < OC) {
            if constexpr (!UseAuxBuffer) {
                dbias[global_oc + k] = (OutFloat)(a + (float)dbias[global_oc + k]);
            } else {
                dbias[global_oc + k + id.get_group(1) * OC] = a;
            }
        }
    }
}


void reduce_add_sum_kernel(sycl::nd_item<1> id, floatX* dst, const float* src, size_t n, size_t m) {
    const size_t idx = id.get_global_id(0) * f128::size;
    assert(n % x128::size == 0);
    if (idx < n) {
        f128 acc;
        for(int k = 0; k < f128::size; ++k) {
            acc[k] = 0.f;
        }

        for(int l = 0; l < m; ++l) {
            f128 s = load128(src + idx + n * l);
            for(int k = 0; k < f128::size; ++k) {
                acc[k] += s[k];
            }
        }
        for(int k = 0; k < f128::size; ++k) {
            dst[idx + k] = (floatX) ((float)dst[idx + k] + acc[k]);
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers


// converted with copilot -- double check this
void matmul_forward_dnnl(sycl::queue *stream, floatX* out, floatX* inp, floatX* weight, floatX* bias,
                           int B, int T, int C, int OC) {
    bool has_bias = (bias != nullptr);

    // Create memory descriptors
    dnnl::memory::data_type elt_type = dnnl::memory::data_type::f32;
    switch (PRECISION_MODE) {
        case PrecisionMode::PRECISION_FP32:
            elt_type = dnnl::memory::data_type::f32;
            break;
        case PrecisionMode::PRECISION_FP16:
            elt_type = dnnl::memory::data_type::f16;
            break;
        case PrecisionMode::PRECISION_BF16:
            elt_type = dnnl::memory::data_type::bf16;
            break;
        default:
            std::cout << "Unsupported precision mode\n";
            exit(EXIT_FAILURE);
    }

    auto engine = dnnl::sycl_interop::make_engine(stream->get_device(), stream->get_context());
    auto onednn_stream = dnnl::sycl_interop::make_stream(engine, *stream);

    auto inp_md = dnnl::memory::desc({B*T, C}, elt_type, dnnl::memory::format_tag::ab);
    auto weight_md = dnnl::memory::desc({C, OC}, elt_type, dnnl::memory::format_tag::ba);
    auto out_md = dnnl::memory::desc({B*T, OC}, elt_type, dnnl::memory::format_tag::ab);

    // Create memory objects
    auto inp_mem = dnnl::sycl_interop::make_memory(inp_md, engine, dnnl::sycl_interop::memory_kind::usm, inp);
    auto weight_mem = dnnl::sycl_interop::make_memory(weight_md, engine, dnnl::sycl_interop::memory_kind::usm, weight);
    auto out_mem = dnnl::sycl_interop::make_memory(out_md, engine, dnnl::sycl_interop::memory_kind::usm, out);

    if (has_bias) {
        auto bias_md = dnnl::memory::desc({1, OC}, elt_type, dnnl::memory::format_tag::ab);
        auto bias_mem = dnnl::sycl_interop::make_memory(bias_md, engine, dnnl::sycl_interop::memory_kind::usm, bias);

        // Create primitive descriptor
        auto matmul_pd = dnnl::matmul::primitive_desc(engine, inp_md, weight_md, bias_md, out_md);

        // Create primitive
        auto matmul_prim = dnnl::matmul(matmul_pd);

        // Set arguments and execute
        matmul_prim.execute(onednn_stream, {
                {DNNL_ARG_SRC, inp_mem},
                {DNNL_ARG_WEIGHTS, weight_mem},
                {DNNL_ARG_BIAS, bias_mem},
                {DNNL_ARG_DST, out_mem}
        });
    } else {
        // Create primitive descriptor
        auto matmul_pd = dnnl::matmul::primitive_desc(engine, inp_md, weight_md, out_md);

        // Create primitive
        auto matmul_prim = dnnl::matmul(matmul_pd);

        // Set arguments and execute
        matmul_prim.execute(onednn_stream, {
                {DNNL_ARG_SRC, inp_mem},
                {DNNL_ARG_WEIGHTS, weight_mem},
                {DNNL_ARG_DST, out_mem}
        });
    }
}


void matmul_backward(sycl::queue *stream, floatX* dinp, floatX* dweight, floatX* dbias,
                     floatX* dout, floatX* inp, floatX* weight,
                     float* dbias_buffer,
                     int B, int T, int C, int OC) {
    float one = 1.0f, zero = 0.0f;

    // backward to bias, if given, does a +=
    if (dbias != nullptr) {
        // Each warp is responsible for 8 * "x128::size" = 64 OCs at BF16 (OC must be a multiple of 64!)
        // Block size is 1024 | 768 threads (32|24 warps) and we reduce those values into 1 at the end

        int max_cus = stream->get_device().get_info<sycl::info::device::max_compute_units>();

        const int block_size = 256;

        sycl::range<3> block_dim((unsigned)block_size/WARP_SIZE, 8, 4);
        const int OC_per_warp = block_dim[1] * x128::size; // 64 at BF16
        const int grid_size_x = CEIL_DIV(OC, OC_per_warp); // e.g. 12 horizontal blocks for 768 OCs at BF16
        const int grid_size_y =
                std::max(1, max_cus * (int)WARP_SIZE /
                            (block_size * grid_size_x)); // full GPU!

        sycl::range<3> grid_dim(1, grid_size_y, grid_size_x);
        // If we have enough OC that we don't need cross-block reductions, we can skip the bias_buffer accumulation
        // and write results directly to the output.
        if(grid_size_y == 1) {
            stream->submit([&](sycl::handler& h) {
                sycl::local_accessor<float> lmem(x128::size*32*8, h);
                h.parallel_for(sycl::nd_range<3>(grid_dim*block_dim, block_dim), [=](sycl::nd_item<3> id) [[intel::reqd_sub_group_size(32)]] {
                    matmul_backward_bias_kernel9(id, dbias, dout, B, T, OC, std::bool_constant<false>{}, lmem);
                });
            }).wait();
        } else {
            // kernel 9 overwrites temp buffer, so no need to memset
            stream->submit([&](sycl::handler& h) {
                sycl::local_accessor<float> lmem(x128::size*32*8, h);
                h.parallel_for(sycl::nd_range<3>(grid_dim*block_dim, block_dim), [=](sycl::nd_item<3> id) [[intel::reqd_sub_group_size(32)]] {
                    matmul_backward_bias_kernel9(id, dbias_buffer, dout, B, T, OC, std::bool_constant<true>{}, lmem);
                });
            }).wait();
            stream->parallel_for(sycl::nd_range<1>(CEIL_DIV(OC, 256*f128::size)*256, 256), [=](sycl::nd_item<1> id) {
                reduce_add_sum_kernel(id, dbias, dbias_buffer, OC, grid_size_y);
            }).wait();
        }
    }

    // backward to input, uses = in the backward pass (set the gradient)
    auto trans = oneapi::mkl::transpose::trans;
    auto no_trans = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::blas::column_major::gemm(*stream,
                                          no_trans, no_trans,
                                          C, B * T, OC,
                                          &one,
                                          weight, C,
                                          dout, OC,
                                          &zero,
                                          dinp, C
    );
    // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
    oneapi::mkl::blas::column_major::gemm(*stream,
                                          no_trans, trans,
                                          C, OC, B * T,
                                          &one,
                                          inp, C,
                                          dout, OC,
                                          &one,
                                          dweight, C
    );
}

#endif //LLM_SYCL_MATMUL_HPP
