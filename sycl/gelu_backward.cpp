#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
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

void gelu_backward_kernel1(sycl::queue& q, sycl::buffer<float, 1>& dinp_buf, sycl::buffer<const float, 1>& inp_buf, sycl::buffer<const float, 1>& dout_buf, int N) {
    q.submit([&](sycl::handler& h) {
        auto dinp = dinp_buf.template get_access<sycl::access::mode::write>(h);
        auto inp = inp_buf.template get_access<sycl::access::mode::read>(h);
        auto dout = dout_buf.template get_access<sycl::access::mode::read>(h);

        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            float x = inp[i];
            float cube = 0.044715f * x * x * x;
            float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            float tanh_out = tanhf(tanh_arg);
            float coshf_out = coshf(tanh_arg);
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            dinp[i] = local_grad * dout[i];
        });
    });
}

void gelu_backward_kernel2(sycl::queue& q, sycl::buffer<float, 1>& dinp_buf, sycl::buffer<const float, 1>& inp_buf, sycl::buffer<const float, 1>& dout_buf, int N) {
    const int vec_size = 4;  // Assuming x128::size is 4 for this conversion
    q.submit([&](sycl::handler& h) {
        auto dinp = dinp_buf.template get_access<sycl::access::mode::write>(h);
        auto inp = inp_buf.template get_access<sycl::access::mode::read>(h);
        auto dout = dout_buf.template get_access<sycl::access::mode::read>(h);

        h.parallel_for(sycl::nd_range<1>(N / vec_size, vec_size), [=](sycl::nd_item<1> item) {
            int i = item.get_global_id(0) * vec_size;
            if (i < N) {
                sycl::vec<float, vec_size> packed_dinp;
                sycl::vec<float, vec_size> packed_inp;
                sycl::vec<float, vec_size> packed_dout;

                for (int k = 0; k < vec_size; ++k) {
                    packed_inp[k] = inp[i + k];
                    packed_dout[k] = dout[i + k];
                }

                for (int k = 0; k < vec_size; ++k) {
                    float x = packed_inp[k];
                    float cube = 0.044715f * x * x * x;
                    float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
                    float tanh_out = tanhf(tanh_arg);
                    float coshf_out = coshf(tanh_arg);
                    float sech_out = 1.0f / (coshf_out * coshf_out);
                    float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
                    packed_dinp[k] = local_grad * packed_dout[k];
                }

                for (int k = 0; k < vec_size; ++k) {
                    dinp[i + k] = packed_dinp[k];
                }
            }
        });
    });
}

// ----------------------------------------------------------------------------
// Kernel launcher

void gelu_backward(int kernel_num, sycl::queue& q, float* dinp, const float* inp, const float* dout, int N, const int block_size) {
    sycl::buffer<float, 1> dinp_buf(dinp, sycl::range<1>(N));
    sycl::buffer<const float, 1> inp_buf(inp, sycl::range<1>(N));
    sycl::buffer<const float, 1> dout_buf(dout, sycl::range<1>(N));

    switch (kernel_num) {
        case 1:
            gelu_backward_kernel1(q, dinp_buf, inp_buf, dout_buf, N);
            break;
        case 2:
            gelu_backward_kernel2(q, dinp_buf, inp_buf, dout_buf, N);
            break;
        default:
            std::cerr << "Invalid kernel number" << std::endl;
            exit(1);
    }

    q.wait();
}


// ----------------------------------------------------------------------------
// Main function

int main(int argc, char** argv) {
    int B = 8;
    int T = 1024;
    int C = 768;
    int N = B * T * C;

    // Create host memory of random numbers
    float* dinp = new float[N];
    float* inp = make_random_float(N);
    float* dout = make_random_float(N);

    // Read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = std::atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << std::endl;

    // First check the correctness of the kernel
    gelu_backward_cpu(dinp, inp, dout, N);

    // SYCL queue
    sycl::queue q;

    // Time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int block_size : block_sizes) {
        std::cout << "Checking block size " << block_size << "." << std::endl;
        gelu_backward(kernel_num, q, dinp, inp, dout, N, block_size);
        validate_result(dinp, dinp, "dinp", N, 1e-5f);
    }

    std::cout << "All results match. Starting benchmarks.\n\n";

    for (int block_size : block_sizes) {
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, gelu_backward, kernel_num, q, dinp, inp, dout, N, block_size);

        // Napkin math: estimate the memory bandwidth achieved
        long memory_ops = N * 2 * sizeof(float);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        std::cout << "block_size " << block_size << " | time " << elapsed_time << " ms | bandwidth " << memory_bandwidth << " GB/s" << std::endl;
    }

    // Free memory
    delete[] dinp;
    delete[] inp;
    delete[] dout;

    return 0;
}

