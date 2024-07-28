#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#define ENABLE_BF16
#include "common.hpp"

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

// ----------------------------------------------------------------------------
// CPU code reference

void gelu_backward_cpu(float* dinp, const float* inp, const float* dout, const int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = local_grad * dout[i];
    }
}

// ----------------------------------------------------------------------------
// SYCL kernels

void gelu_backward_kernel1(sycl::nd_item<1> id, floatX* dinp, const floatX* inp, const floatX* dout, int N) {
    int i = id.get_global_id(0);
    if (i < N) {
        float x = (float)inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = sycl::tanh(tanh_arg);
        float coshf_out = sycl::cosh(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = (floatX)(local_grad * (float)dout[i]);
    }
}

void gelu_backward_kernel2(sycl::nd_item<1> id,  floatX* dinp, const floatX* inp, const floatX* dout, const int N) {
    int i = id.get_global_id(0) * x128::size;
    if (i < N) {
        x128 packed_dinp;
        x128 packed_inp = load128cs(inp + i);
        x128 packed_dout = load128cs(dout + i);
        for (int k = 0; k < packed_inp.size; ++k) {
            float x = (float)packed_inp[k];
            float cube = 0.044715f * x * x * x;
            float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            float tanh_out = tanhf(tanh_arg);
            float coshf_out = coshf(tanh_arg);
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            packed_dinp[k] = (floatX)(local_grad * (float)packed_dout[k]);
        }

        store128(dinp + i, packed_dinp);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void gelu_backward1(sycl::queue &q, floatX* dinp, const floatX* inp, const floatX* dout, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        gelu_backward_kernel1(id, dinp, inp, dout, N);
    }).wait();
}

void gelu_backward2(sycl::queue &q, floatX* dinp, const floatX* inp, const floatX* dout, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size * x128::size);
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        gelu_backward_kernel2(id, dinp, inp, dout, N);
    }).wait();
}

// kernel version dispatch
void gelu_backward(int kernel_num,
                   sycl::queue &q,
                   floatX* dinp,
                   const floatX* inp,
                   const floatX* dout,
                   int B, int T, int C,
                   int block_size) {
    switch (kernel_num) {
        case 1:
            gelu_backward1(q, dinp, inp, dout, B * T * C, block_size);
            break;
        case 2:
            gelu_backward2(q, dinp, inp, dout, B * T * C, block_size);
            break;
        default:
            std::cout << "Invalid kernel number\n";
            std::exit(1);
    }
}


// ----------------------------------------------------------------------------
// Main function

int main(int argc, char** argv) {
    srand(0);
    int B = 8;
    int T = 1024;
    int C = 768;

    // Create host memory of random numbers
    float* dinp = new float[B * T * C];
    float* inp = make_random_float(B * T * C);
    float* dout = make_random_float(B * T * C);


    // Read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = std::atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << std::endl;

    // First check the correctness of the kernel
    gelu_backward_cpu(dinp, inp, dout, B * T * C);

    // move to GPU
    sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order());
    auto d_dinp = sycl::malloc_device<floatX>(B * T * C, q);
    auto d_inp = sycl::malloc_device<floatX>(B * T * C, q);
    auto d_dout = sycl::malloc_device<floatX>(B * T * C, q);

    memcpy_convert(d_inp, inp, B * T * C, q);
    memcpy_convert(d_dout, dout, B * T * C, q);

    // Time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};
    for (int block_size : block_sizes) {
        std::cout << "Checking block size " << block_size << "." << std::endl;
        gelu_backward(kernel_num, q, d_dinp, d_inp, d_dout, B, T, C, block_size);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_dinp, dinp, "dinp", B * T * C, tol);
    }

    std::cout << "All results match. Starting benchmarks.\n\n";

    for (int block_size : block_sizes) {
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(
                repeat_times,
                gelu_backward, // kernel
                kernel_num, q, d_dinp, d_inp, d_dout, B, T, C, block_size // params
        );

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 1 read and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 2 * 4;

        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        std::cout << "block_size " << block_size << " | time " << elapsed_time << " ms | bandwidth " << memory_bandwidth << " GB/s" << std::endl;
    }

    // Free memory
    delete[] dinp;
    delete[] inp;
    delete[] dout;

    sycl::free(d_dinp, q);
    sycl::free(d_inp, q);
    sycl::free(d_dout, q);

    return 0;
}

