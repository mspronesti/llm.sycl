#include <iostream>
#include <sycl/sycl.hpp>
#include <omp.h>
#include <oneapi/mkl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include "common.hpp"

auto MKL_OP_T = oneapi::mkl::transpose::trans;
auto MKL_OP_N = oneapi::mkl::transpose::nontrans;

// ----------------------------------------------------------------------------
// CPU code reference

void matmul_forward_cpu(float* out,
                        const float* inp, const float* weight, const float* bias,
                        int B, int T, int C, int OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            const float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                const float* wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// kernel 1: naive kernel, every thread handles one output element, direct global memory access
void matmul_forward_kernel1(sycl::nd_item<2> id, float* out,
                            const float* inp, const float* weight, const float* bias,
                            int BT, int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // in the naive kernel, every thread handles one element of out
    int bt = id.get_global_id(1);
    int oc = id.get_global_id(0);
    if (bt < BT && oc < OC) {
        int b = bt / BT;
        int t = bt % BT;
        float val = (bias != NULL) ? bias[oc] : 0.0f;
        const float* wrow = weight + oc*C;
        const float* inp_bt = inp + b * BT * C + t * C;
        for (int i = 0; i < C; i++) {
            val += inp_bt[i] * wrow[i];
        }
        out[bt * OC + oc] = val;
    }
}

// is there no better way other than just adding bias with a whole separate kernel?
// this is a highly memory-bound operation, should be fused into the matmul kernel
// but I can't seem to find a cuBLAS function that does this
void add_bias(sycl::nd_item<1> id, float* out, const float* bias, int B, int T, int OC) {
    int idx = id.get_global_id(0);
    int stride = id.get_global_range(0);
    for (int i = idx; i < B * T * OC; i += stride) {
        int col = i % OC;
        out[i] += bias[col];
    }
}

// kernel 4: semi-efficient handwritten kernel
// see trimat_forward.cu for some intermediate development steps
sycl::float4 ld_vec(const float* address) {
    return *reinterpret_cast<const sycl::float4*>(address);
}

void st_vec(float* address, sycl::float4 val) {
    *reinterpret_cast<sycl::float4*>(address) = val;
}

void matmul_forward_kernel4(sycl::nd_item<2> id, float* out, const float* inp, const float* weight, const float* bias,
                                                     int C, int OC,
                            sycl::multi_ptr<float[128][32][2], sycl::access::address_space::local_space> local_mem_ptr) {
    int blockIdx_x = id.get_group(1);
    int blockIdx_y = id.get_group(0);

    int threadIdx_x = id.get_local_id(1);
    int threadIdx_y = id.get_local_id(0);

    int blockDim_y = id.get_local_range(0);

    float *shared = (float*) local_mem_ptr.get_raw();
    float (*lhs_s)[32] = (float (*)[32]) shared;
    float (*rhs_s)[32] = (float (*)[32]) (shared + 128 * 32);

    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // each thread handles 8x8 elements; each block 128 by 128 elements.
    int oc = 8*(blockIdx_y * blockDim_y + threadIdx_y);

    // adjust our pointers for the current block
    inp += 128 * blockIdx_x * C;
    weight += 128 * blockIdx_y * C;
    out += 128 * blockIdx_x * OC + 128 * blockIdx_y;

    float vals[8][8] = {};
    if(bias != nullptr) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j += 4) {
                sycl::float4 b = ld_vec(bias + oc + j);
                vals[i][j+0] = b.x();
                vals[i][j+1] = b.y();
                vals[i][j+2] = b.z();
                vals[i][j+3] = b.w();
            }
        }
    }

    int si_start = 4*(16 * threadIdx_y + threadIdx_x);
    for (int so = 0; so < C; so += 32) {
        id.barrier();
        int xmod8 = threadIdx_x % 8;
        int xby8 = threadIdx_x / 8;
        int xo = 4 * xmod8;
        for(int y = 2 * threadIdx_y + xby8; y < 128; y += 32) {
            st_vec(&lhs_s[y][xo], ld_vec(inp + y * C + so + xo));
            st_vec(&rhs_s[y][xo], ld_vec(weight + y * C + so + xo));
        }
        id.barrier();

        for (int si = si_start; si < si_start + 32; si += 4) {
            sycl::float4 rhs[8];
            for (int u = 0; u < 8; ++u) {
                rhs[u] = ld_vec(&rhs_s[u + 8 * threadIdx_y][si % 32]);
            }

            for (int ii = 0; ii < 8; ++ii) {
                sycl::float4 lhs = ld_vec(&lhs_s[ii + 8 * threadIdx_x][si % 32]);
                for (int ji = 0; ji < 8; ++ji) {
                    vals[ii][ji] += lhs.x() * rhs[ji].x();
                    vals[ii][ji] += lhs.y() * rhs[ji].y();
                    vals[ii][ji] += lhs.z() * rhs[ji].z();
                    vals[ii][ji] += lhs.w() * rhs[ji].w();
                }
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; j += 4) {
            sycl::float4 result;
            result.x() = vals[i][j + 0];
            result.y() = vals[i][j + 1];
            result.z() = vals[i][j + 2];
            result.w() = vals[i][j + 3];
            st_vec(out + (8*threadIdx_x+i) * OC + 8*threadIdx_y + j, result);
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

// kernel 1 is the most naive matmul kernel
void matmul_forward1(sycl::queue& q, float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     const int sqrt_block_size) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    sycl::nd_range<2> grid = sycl::nd_range<2>(sycl::range<2>(ceil_div(OC, sqrt_block_size) * sqrt_block_size,
                                                              ceil_div(B*T, sqrt_block_size) * sqrt_block_size),
                                               sycl::range<2>(sqrt_block_size, sqrt_block_size));
    q.parallel_for(grid, [=](sycl::nd_item<2> id) {
        matmul_forward_kernel1(id, out, inp, weight, bias, B*T, C, OC);
    }).wait();
}

// kernel 2 calls cuBLAS (oneMKL actually), which should be very efficient
void matmul_forward2(sycl::queue &q, float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     const int sqrt_block_size) {
    // for reference API is:
    // cublasStatus_t cublasSgemm(cublasHandle_t handle,
    //                        cublasOperation_t transa, cublasOperation_t transb,
    //                        int m, int n, int k,
    //                        const float           *alpha,
    //                        const float           *A, int lda,
    //                        const float           *B, int ldb,
    //                        const float           *beta,
    //                        float           *C, int ldc)
    // for us, inp is (B*T, C), weight is (OC, C), out is (B*T, OC)
    // cuBLAS does C = alpha * A * B + beta * C
    // where A is mxk, B is kxn, C is mxn
    // now, because we use row-major storage, cuBLAS (which is column-major) sees our matrices transposed.
    // algorithmically / in e.g. PyTorch we want to do: out = inp @ weight.T
    // but because cuBLAS is column-major, we actually want to get it to calculate out.T . Mathematically, this is:
    // out.T = weight @ inp.T
    // but again, our variables look transposed, so using the actual weight/inp we have here in this function, this becomes
    // out.T = weight.T @ inp
    // so we need to get cuBLAS to calculate weight.T @ inp (the variables here are the actual ones in this function)
    // => need to call cuBLAS with A = weight, B = inp
    // => need to call cuBLAS with transa = CUBLAS_OP_T, transb = CUBLAS_OP_N

    const float alpha = 1.0f;
    const float beta = 0.0f;

    oneapi::mkl::blas::column_major::gemm(q,
                                          MKL_OP_T, MKL_OP_N,
                                          OC, B*T, C,
                                          &alpha,
                                          weight, C,
                                          inp, C,
                                          &beta,
                                          out, OC
    );
    // and now we still have to add the bias... (ew)
    if (bias != nullptr) {
        int block_size = sqrt_block_size * sqrt_block_size;
        int grid_size = ceil_div(OC * B * T, block_size);

        q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
            add_bias(id, out, bias, B, T, OC);
        }).wait();
    }
}

// uses cublasLt to fuse the bias and gelu
// https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtSgemm/sample_cublasLt_LtSgemm.cu
void matmul_forward3(sycl::queue& queue, float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC) {
    bool has_bias = (bias != nullptr);
    bool has_gelu = false;

    // Setup engine and stream
    auto engine = dnnl::sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto stream = dnnl::sycl_interop::make_stream(engine, queue);

    // Create memory descriptors
    auto inp_md = dnnl::memory::desc({B*T, C}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
    auto weight_md = dnnl::memory::desc({C, OC}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ba);
    auto out_md = dnnl::memory::desc({B*T, OC}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
    auto bias_md = dnnl::memory::desc({1, OC}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

    // Create memory objects
    auto inp_mem = dnnl::sycl_interop::make_memory(inp_md, engine, dnnl::sycl_interop::memory_kind::usm, const_cast<float *>(inp));
    auto weight_mem = dnnl::sycl_interop::make_memory(weight_md, engine, dnnl::sycl_interop::memory_kind::usm, const_cast<float *>(weight));
    auto out_mem = dnnl::sycl_interop::make_memory(out_md, engine, dnnl::sycl_interop::memory_kind::usm, out);
    auto bias_mem = dnnl::sycl_interop::make_memory(bias_md, engine, dnnl::sycl_interop::memory_kind::usm, const_cast<float *>(bias));

    // Create primitive attributes
    dnnl::primitive_attr matmul_attr;
    if (has_gelu) {
        dnnl::post_ops po;
        po.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 1.0f, 0.0f);
        matmul_attr.set_post_ops(po);
    }

    // Create primitive descriptor
    dnnl::matmul::primitive_desc matmul_pd;
    if (has_bias) {
        matmul_pd = dnnl::matmul::primitive_desc(engine, inp_md, weight_md, bias_md, out_md, matmul_attr);
    }
    else {
        matmul_pd = dnnl::matmul::primitive_desc(engine, inp_md, weight_md, out_md, matmul_attr);
    }

    // Create primitive
    auto matmul_prim = dnnl::matmul(matmul_pd);

    // Set arguments and execute
    if (has_bias) {
        matmul_prim.execute(stream, {
                {DNNL_ARG_SRC, inp_mem},
                {DNNL_ARG_WEIGHTS, weight_mem},
                {DNNL_ARG_BIAS, bias_mem},
                {DNNL_ARG_DST, out_mem}
        });
    }
    else {
        matmul_prim.execute(stream, {
                {DNNL_ARG_SRC, inp_mem},
                {DNNL_ARG_WEIGHTS, weight_mem},
                {DNNL_ARG_DST, out_mem}
        });
    }
    stream.wait();
}

// handwritten, relatively efficient non-tensorcore matmul kernel
void matmul_forward4(sycl::queue &q, float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     int sqrt_block_size) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    sqrt_block_size = 16;

    sycl::nd_range<2> grid = sycl::nd_range<2>(sycl::range<2>(ceil_div(OC, 8*sqrt_block_size) * sqrt_block_size,
                                                              ceil_div(B*T, 8*sqrt_block_size) * sqrt_block_size),
                                               sycl::range<2>(sqrt_block_size, sqrt_block_size));

    q.parallel_for(grid, [=](sycl::nd_item<2> id) {
        auto local_mem_ptr = sycl::ext::oneapi::group_local_memory_for_overwrite<float[128][32][2]>(
                id.get_group());
        matmul_forward_kernel4(id, out, inp, weight, bias, C, OC, local_mem_ptr);
    }).wait();
}

// kernel version dispatch
void matmul_forward(int kernel_num,
                    sycl::queue& q,
                    float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC,
                    const int sqrt_block_size) {
    switch (kernel_num) {
        case 1:
            matmul_forward1(q, out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
            break;
        case 2:
            matmul_forward2(q, out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
            break;
        case 3:
            matmul_forward3(q, out, inp, weight, bias, B, T, C, OC);
            break;
        case 4:
            matmul_forward4(q, out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    int OC = 768 * 4; // expansion of 4, e.g. in the MLP

    sycl::queue q(sycl::default_selector_v,
                             sycl::property::queue::in_order{});

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * OC * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(OC * C);
    float* bias = make_random_float(OC);

    // move to GPU
    float* d_out;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    d_out = sycl::malloc_device<float>(B * T * OC, q);
    d_inp = sycl::malloc_device<float>(B * T * C, q);
    d_weight = sycl::malloc_device<float>(OC * C, q);
    d_bias = sycl::malloc_device<float>(OC, q);

    q.memcpy(d_inp, inp, B * T * C * sizeof(float));
    q.memcpy(d_weight, weight, OC * C * sizeof(float));
    q.memcpy(d_bias, bias, OC * sizeof(float));


    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    // time the kernel at different block sizes
    int sqrt_block_sizes[] = {4, 8, 16};

    for (int sqrt_block_size: sqrt_block_sizes){
        printf("Checking block size %d x %d.\n", sqrt_block_size, sqrt_block_size);
        matmul_forward(kernel_num, q, d_out, d_inp, d_weight, d_bias, B, T, C, OC, sqrt_block_size);
        validate_result(d_out, out, "out", B * T * OC, 1e-1f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int sqrt_block_size: sqrt_block_sizes) {
        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, matmul_forward,
                                              kernel_num, q, d_out, d_inp, d_weight, d_bias,
                                              B, T, C, OC, sqrt_block_size);

        // napkin math: estimate the flops achieved
        // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
        float tflops = (float)B * T * C * OC * 2 / elapsed_time * 1e3f / 1e12f;
        printf("sqrt_block_size %4d | time %.4f ms | tflops %.2f\n", sqrt_block_size, elapsed_time, tflops);
    }

    // free memory
    free(out);
    free(inp);
    free(weight);
    free(bias);

    sycl::free(d_out, q);
    sycl::free(d_inp, q);
    sycl::free(d_weight, q);
    sycl::free(d_bias, q);

    return 0;
}