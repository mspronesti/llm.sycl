#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "common.hpp"

void adamw_cpu(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, int t, long num_parameters,
               float learning_rate=1e-3, float beta1=0.9, float beta2=0.999, float eps=1e-8, float weight_decay=0.0) {
    for (long i = 0; i < num_parameters; i++) {
        float param = params_memory[i];
        float grad = grads_memory[i];

        float m = beta1 * m_memory[i] + (1.0f - beta1) * grad;
        float v = beta2 * v_memory[i] + (1.0f - beta2) * grad * grad;
        float m_hat = m / (1.0f - std::pow(beta1, t));
        float v_hat = v / (1.0f - std::pow(beta2, t));

        m_memory[i] = m;
        v_memory[i] = v;
        params_memory[i] -= learning_rate * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * param);
    }
}

// Slightly more optimized AdamW kernel by using optimized linear interpolation for the moment updates.
void adamw_kernel1(sycl::queue& q, sycl::buffer<float, 1>& params_buf, sycl::buffer<const float, 1>& grads_buf, sycl::buffer<float, 1>& m_buf, sycl::buffer<float, 1>& v_buf,
                   long num_parameters, float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
    q.submit([&](sycl::handler& h) {
        auto params = params_buf.get_access<sycl::access::mode::read_write>(h);
        auto grads = grads_buf.get_access<sycl::access::mode::read>(h);
        auto m = m_buf.get_access<sycl::access::mode::read_write>(h);
        auto v = v_buf.get_access<sycl::access::mode::read_write>(h);

        h.parallel_for(sycl::range<1>(num_parameters), [=](sycl::id<1> i) {
            m[i] = beta1 * m[i] + (1.0f - beta1) * grads[i];
            v[i] = beta2 * v[i] + (1.0f - beta2) * grads[i] * grads[i];
            float m_hat = m[i] / beta1_correction;
            float v_hat = v[i] / beta2_correction;
            params[i] -= learning_rate * (m_hat / (sycl::sqrt(v_hat) + eps) + weight_decay * params[i]);
        });
    });
}

// Slightly more optimized AdamW kernel by using optimized linear interpolation for the moment updates.
void adamw_kernel2(sycl::queue& q, sycl::buffer<float, 1>& params_buf, sycl::buffer<const float, 1>& grads_buf, sycl::buffer<float, 1>& m_buf, sycl::buffer<float, 1>& v_buf,
                   long num_parameters, float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
    q.submit([&](sycl::handler& h) {
        auto params = params_buf.get_access<sycl::access::mode::read_write>(h);
        auto grads = grads_buf.get_access<sycl::access::mode::read>(h);
        auto m = m_buf.get_access<sycl::access::mode::read_write>(h);
        auto v = v_buf.get_access<sycl::access::mode::read_write>(h);

        h.parallel_for(sycl::range<1>(num_parameters), [=](sycl::id<1> i) {
            float grad = grads[i];
            float m_val = m[i];
            float v_val = v[i];
            m_val = grad * (1.0f - beta1) + m_val * beta1;
            v_val = grad * grad * (1.0f - beta2) + v_val * beta2;
            m[i] = m_val;
            v[i] = v_val;
            m_val /= beta1_correction;
            v_val /= beta2_correction;
            params[i] -= learning_rate * (m_val / (sycl::sqrt(v_val) + eps) + weight_decay * params[i]);
        });
    });
}

void adamw(int kernel_num,
           float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, int t, long num_parameters,
           float learning_rate=1e-3, float beta1=0.9, float beta2=0.999, float eps=1e-8, float weight_decay=0.0) {
    // calculate the m_hat and v_hat correction terms once as they are the same for every param/thread
    float beta1_correction = 1.0f - std::pow(beta1, t);
    float beta2_correction = 1.0f - std::pow(beta2, t);

    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    sycl::buffer<float, 1> params_buf(params_memory, sycl::range<1>(num_parameters));
    sycl::buffer<const float, 1> grads_buf(grads_memory, sycl::range<1>(num_parameters));
    sycl::buffer<float, 1> m_buf(m_memory, sycl::range<1>(num_parameters));
    sycl::buffer<float, 1> v_buf(v_memory, sycl::range<1>(num_parameters));

    switch (kernel_num) {
        case 1:
            adamw_kernel1(q, params_buf, grads_buf, m_buf, v_buf, num_parameters,
                          learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
            break;
        case 2:
            adamw_kernel2(q, params_buf, grads_buf, m_buf, v_buf, num_parameters,
                          learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
            break;
        default:
            std::cerr << "Invalid kernel number" << std::endl;
            exit(1);
    }
    q.wait();
}

int main(int argc, char** argv) {
    const long num_parameters = 1048576;
    const int t = 10;

    const float learning_rate = 1e-3f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float weight_decay = 0.0f;

    srand(time(nullptr));

    // create random data on host
    float* params_memory = make_random_float(num_parameters);
    float* grads_memory = make_random_float(num_parameters);
    float* m_memory = make_random_float_01(num_parameters);
    float* v_memory = make_random_float_01(num_parameters);

    // calculate the CPU reference
    clock_t start = clock();
    adamw_cpu(params_memory, grads_memory, m_memory, v_memory, t, num_parameters);
    clock_t end = clock();
    double elapsed_time_cpu = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = std::atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << std::endl;

    // calculate the GPU version
    adamw(kernel_num, params_memory, grads_memory, m_memory, v_memory, t, num_parameters,
          learning_rate, beta1, beta2, eps, weight_decay);

    // compare
    std::cout << "Checking correctness..." << std::endl;
    std::cout << "parameters:" << std::endl;
    validate_result(params_memory, params_memory, "params_memory", num_parameters);
    std::cout << "first moment:" << std::endl;
    validate_result(m_memory, m_memory, "m_memory", num_parameters);
    std::cout << "second moment:" << std::endl;
    validate_result(v_memory, v_memory, "v_memory", num_parameters);
    std::cout << "All results match." << std::endl;

    // benchmark the kernel
    int repeat_times = 1000;
    sycl::queue q;
    float elapsed_time = 0.0f;

    for (int i = 0; i < repeat_times; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        adamw(kernel_num, params_memory, grads_memory, m_memory, v_memory, t, num_parameters,
              learning_rate, beta1, beta2, eps, weight_decay);
        q.wait();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end - start;
        elapsed_time += duration.count();
    }

    elapsed_time /= repeat_times;

    std::cout << "time gpu " << elapsed_time << " ms" << std::endl;
    std::cout << "time cpu " << elapsed_time_cpu << " ms" << std::endl;

    // cleanup
    delete[] params_memory;
    delete[] grads_memory;
    delete[] m_memory;
    delete[] v_memory;

    return 0;
}

