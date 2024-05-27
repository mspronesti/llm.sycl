/*
Kernels for layernorm forward pass.

Compile example:
icpx -O3 -fsycl layernorm_forward.cpp -o layernorm_forward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./layernorm_forward 1

version 2 parallelizes over all of B,T,C
./layernorm_forward 2

version 3 uses cooperative groups to parallelize over all of B,T,C
./layernorm_forward 3

version 4 uses a more clever way to estimate variance, var(x) = mean(x**2) - mean(x)**2
          (allowing us to do a single pass over x on load)
./layernorm_forward 4

version 5 allocates blocks per row instead of warps per row, same alg as 4 otherwise
./layernorm_forward 5
*/

#include <CL/sycl.hpp>
#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <cassert>
#include <chrono>

// Function to generate random floats
void make_random_float(float* ptr, int size) {
    for (int i = 0; i < size; ++i) {
        ptr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

// Function to validate results between reference and computed arrays
void validate_results(float* ref, float* res, int size) {
    for (int i = 0; i < size; ++i) {
        if (fabs(ref[i] - res[i]) > 1e-5) {
            std::cerr << "Result mismatch at index " << i << ": " << ref[i] << " != " << res[i] << std::endl;
            exit(1);
        }
    }
    std::cout << "Validation passed!" << std::endl;
}

// Function to benchmark kernel execution time
template <typename F>
double benchmark_kernel(F&& f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 layernorm forward pass
void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m / C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v / C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// naive drag and drop implementation into kernel, parallelize over B,T, loop over C
void layernorm_forward_kernel1(sycl::queue &q, float* out, float* mean, float* rstd,
                               const float* inp, const float* weight, const float* bias,
                               int N, int C) {
    float eps = 1e-5f;

    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
        if (idx[0] < N) {
            const float* x = inp + idx[0] * C;
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m / C;
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v / C;
            float s = 1.0f / sycl::sqrt(v + eps);
            float* out_idx = out + idx[0] * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m));
                float o = n * weight[i] + bias[i];
                out_idx[i] = o;
            }
            mean[idx[0]] = m;
            rstd[idx[0]] = s;
        }
    }).wait();
}

// parallelize over all of B, T, C
void layernorm_forward_kernel2(sycl::queue &q, float* out, float* mean, float* rstd,
                               const float* inp, const float* weight, const float* bias,
                               int N, int C) {
    float eps = 1e-5f;

    q.parallel_for(sycl::range<2>(N, C), [=](sycl::id<2> idx) {
        int n = idx[0];
        int c = idx[1];
        const float* x = inp + n * C;
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += x[i];
        }
        m = m / C;
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;
        float s = 1.0f / sycl::sqrt(v + eps);
        float n_val = (s * (x[c] - m));
        out[n * C + c] = n_val * weight[c] + bias[c];
        if (c == 0) {
            mean[n] = m;
            rstd[n] = s;
        }
    }).wait();
}

// other kernel implementations remain the same
void layernorm_forward_kernel3(sycl::queue &q, float* out, float* mean, float* rstd,
                               const float* inp, const float* weight, const float* bias, int N, int C) {
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(N * 32), sycl::range<1>(32)), [=](sycl::nd_item<1> item) {
        auto sg = item.get_sub_group();
        int idx = item.get_group(0) * sg.get_group_range().size() + sg.get_group_id();
        if (idx >= N) return;
        const float* x = inp + idx * C;
        float sum = 0.0f;
        for (int i = sg.get_local_id(); i < C; i += sg.get_local_range().size()) {
            sum += x[i];
        }
        sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
        float m = sum / C;
        if (sg.get_local_id() == 0 && mean != nullptr) {
            mean[idx] = m;
        }
        sum = 0.0f;
        for (int i = sg.get_local_id(); i < C; i += sg.get_local_range().size()) {
            float diff = x[i] - m;
            sum += diff * diff;
        }
        sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
        float s = sycl::rsqrt(sum / C + 1e-5f);
        if (sg.get_local_id() == 0 && rstd != nullptr) {
            rstd[idx] = s;
        }
        float* o = out + idx * C;
        for (int i = sg.get_local_id(); i < C; i += sg.get_local_range().size()) {
            float n = s * (x[i] - m);
            o[i] = n * weight[i] + bias[i];
        }
    }).wait();
}

void layernorm_forward_kernel4(sycl::queue &q, float* out, float* mean, float* rstd,
                               const float* inp, const float* weight, const float* bias, int N, int C) {
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(N * 32), sycl::range<1>(32)), [=](sycl::nd_item<1> item) {
        auto sg = item.get_sub_group();
        int idx = item.get_group(0) * sg.get_group_range().size() + sg.get_group_id();
        if (idx >= N) return;
        const float* x = inp + idx * C;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        for (int i = sg.get_local_id(); i < C; i += sg.get_local_range().size()) {
            float xi = x[i];
            sum1 += xi;
            sum2 += xi * xi;
        }
        sum1 = sycl::reduce_over_group(sg, sum1, sycl::plus<float>());
        sum2 = sycl::reduce_over_group(sg, sum2, sycl::plus<float>());
        float m = sum1 / C;
        if (sg.get_local_id() == 0 && mean != nullptr) {
            mean[idx] = m;
        }
        float v = sum2 / C - m * m;
        float s = sycl::rsqrt(v + 1e-5f);
        if (sg.get_local_id() == 0 && rstd != nullptr) {
            rstd[idx] = s;
        }
        float* o = out + idx * C;
        for (int c = sg.get_local_id(); c < C; c += sg.get_local_range().size()) {
            float n = s * (x[c] - m);
            o[c] = n * weight[c] + bias[c];
        }
    }).wait();
}

void layernorm_forward_kernel5(sycl::queue &q, float* out, float* mean, float* rstd,
                               const float* inp, const float* weight, const float* bias, int N, int C) {
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(N * 128), sycl::range<1>(128)), [=](sycl::nd_item<1> item) {
        auto sg = item.get_sub_group();
        int idx = item.get_group(0);
        int tid = item.get_local_id(0);
        const float* x = inp + idx * C;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        for (int i = tid; i < C; i += 128) {
            float xi = x[i];
            sum1 += xi;
            sum2 += xi * xi;
        }
        sum1 = sycl::reduce_over_group(sg, sum1, sycl::plus<float>());
        sum2 = sycl::reduce_over_group(sg, sum2, sycl::plus<float>());
        float m = sum1 / C;
        if (tid == 0 && mean != nullptr) {
            mean[idx] = m;
        }
        float v = sum2 / C - m * m;
        float s = sycl::rsqrt(v + 1e-5f);
        if (tid == 0 && rstd != nullptr) {
            rstd[idx] = s;
        }
        float* o = out + idx * C;
        for (int i = tid; i < C; i += 128) {
            float n = s * (x[i] - m);
            o[i] = n * weight[i] + bias[i];
        }
    }).wait();
}

// ----------------------------------------------------------------------------
// Main

int main(int argc, char** argv) {
    int version = 1;
    if (argc == 2) version = atoi(argv[1]);
    std::cout << "version: " << version << std::endl;

    // (over)allocate memory, we don't use all of it necessarily
    int B = 32; // batch size
    int T = 128; // sequence length
    int C = 768; // embedding size
    int N = B * T;

    // initialize inputs
    float* x = (float*)malloc(B * T * C * sizeof(float));
    float* w = (float*)malloc(C * sizeof(float));
    float* b = (float*)malloc(C * sizeof(float));
    make_random_float(x, B * T * C);
    make_random_float(w, C);
    make_random_float(b, C);

    // initialize outputs
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));

    // Create a SYCL queue
    sycl::queue q;

    // Device memory allocation
    float* d_x = sycl::malloc_device<float>(B * T * C, q);
    float* d_w = sycl::malloc_device<float>(C, q);
    float* d_b = sycl::malloc_device<float>(C, q);
    float* d_out = sycl::malloc_device<float>(B * T * C, q);
    float* d_mean = sycl::malloc_device<float>(B * T, q);
    float* d_rstd = sycl::malloc_device<float>(B * T, q);

    // Copy data to device
    q.memcpy(d_x, x, B * T * C * sizeof(float)).wait();
    q.memcpy(d_w, w, C * sizeof(float)).wait();
    q.memcpy(d_b, b, C * sizeof(float)).wait();

    // Choose kernel version
    double elapsed_time;
    switch(version){
	case 1:
          elapsed_time = benchmark_kernel([&]() {
              layernorm_forward_kernel1(q, d_out, d_mean, d_rstd, d_x, d_w, d_b, N, C);
          });
          std::cout << "Kernel 1 execution time: " << elapsed_time << " seconds\n";
	  break;
	case 2:
 	  elapsed_time = benchmark_kernel([&]() {
              layernorm_forward_kernel2(q, d_out, d_mean, d_rstd, d_x, d_w, d_b, N, C);
          });
          std::cout << "Kernel 2 execution time: " << elapsed_time << " seconds\n";
	  break;
	case 3:
	  elapsed_time = benchmark_kernel([&]() {
            layernorm_forward_kernel3(q, d_out, d_mean, d_rstd, d_x, d_w, d_b, N, C);
          });
          std::cout << "Kernel 3 execution time: " << elapsed_time << " seconds\n";
	  break;
	case 4:
	  elapsed_time = benchmark_kernel([&]() {
            layernorm_forward_kernel4(q, d_out, d_mean, d_rstd, d_x, d_w, d_b, N, C);
          });
          std::cout << "Kernel 4 execution time: " << elapsed_time << " seconds\n";
	  break;
	case 5:
	  elapsed_time = benchmark_kernel([&]() {
            layernorm_forward_kernel5(q, d_out, d_mean, d_rstd, d_x, d_w, d_b, N, C);
          });
          std::cout << "Kernel 5 execution time: " << elapsed_time << " seconds\n";
	  break;
	default:
	  std::cout << "Invalid kernel number\n";
          exit(1);
    }

    // Copy results back to host
    q.memcpy(out, d_out, B * T * C * sizeof(float)).wait();
    q.memcpy(mean, d_mean, B * T * sizeof(float)).wait();
    q.memcpy(rstd, d_rstd, B * T * sizeof(float)).wait();

    // Run the CPU version for validation
    float* cpu_out = (float*)malloc(B * T * C * sizeof(float));
    float* cpu_mean = (float*)malloc(B * T * sizeof(float));
    float* cpu_rstd = (float*)malloc(B * T * sizeof(float));
    layernorm_forward_cpu(cpu_out, cpu_mean, cpu_rstd, x, w, b, B, T, C);

    // Validate the results
    validate_results(cpu_out, out, B * T * C);
    validate_results(cpu_mean, mean, B * T);
    validate_results(cpu_rstd, rstd, B * T);

    // Free device memory
    sycl::free(d_x, q);
    sycl::free(d_w, q);
    sycl::free(d_b, q);
    sycl::free(d_out, q);
    sycl::free(d_mean, q);
    sycl::free(d_rstd, q);

    // Free host memory
    free(x);
    free(w);
    free(b);
    free(out);
    free(mean);
    free(rstd);
    free(cpu_out);
    free(cpu_mean);
    free(cpu_rstd);

    return 0;
}

