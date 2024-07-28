#ifndef LLM_SYCL_COMMON_HPP
#define LLM_SYCL_COMMON_HPP

#include <sycl/sycl.hpp>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <chrono>

//  SYCL equivalent of CUDA atomics builtins
//  ------------------------------------------------------------
template<typename T, sycl::memory_scope MemoryScope = sycl::memory_scope::device>
static inline T atomicAdd(T* val, const T delta)
{
    sycl::atomic_ref<T, sycl::memory_order::relaxed,
    MemoryScope> ref(*val);
    return ref.fetch_add(delta);
}

// https://github.com/zjin-lcf/HeCBench/blob/aad74973f5ceb1e059829e46725f8dbd573cd546/src/bh-sycl/main.cpp#L79-L97
template<typename T, sycl::memory_scope MemoryScope = sycl::memory_scope::device>
T atomicInc(T* addr, unsigned int val) {
    auto atm = sycl::atomic_ref<T,
            sycl::memory_order::relaxed,
            MemoryScope
    >(addr[0]);
    T old;
    while (true) {
        old = atm.load();
        if (old >= val) {
            if (atm.compare_exchange_strong(old, 0))
                break;
        } else if (atm.compare_exchange_strong(old, old + 1))
            break;
    }
    return old;
}

//https://github.com/zjin-lcf/HeCBench/blob/aad74973f5ceb1e059829e46725f8dbd573cd546/src/cc-sycl/main.cpp#L56-L65
template<typename T, sycl::memory_scope MemoryScope = sycl::memory_scope::device>
inline T atomicCAS(T* addr, T expected, T desired)
{
    T expected_value = expected;
    auto atm = sycl::atomic_ref<T,
            sycl::memory_order::relaxed,
            MemoryScope
    >(*addr);
    atm.compare_exchange_strong(expected_value, desired);
    return expected_value;
}

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

template<class ElementType>
struct alignas(16) Packed128 {
    Packed128() = default;

    explicit Packed128(sycl::int4 bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        *reinterpret_cast<sycl::int4*>(payload) = bits;
    }

    static Packed128 constant(ElementType value) {
        Packed128 result;
        for(int k = 0; k < size; ++k) {
            result.payload[k] = value;
        }
        return result;
    }

    static Packed128 zeros() {
        return constant(0);
    }

    static Packed128 ones() {
        return constant(1);
    }

    ElementType& operator[](int index) {
        return payload[index];
    }

    const ElementType& operator[](int index) const {
        return payload[index];
    }

    sycl::int4 get_bits() const {
        sycl::int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        bits = *reinterpret_cast<const sycl::int4*>(payload);
        return bits;
    }

    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    static constexpr const int size = sizeof(sycl::int4) / sizeof(ElementType);
    ElementType payload[size];
};

// short-form typedef
typedef Packed128<float> f128;

// load a Packed128 from an aligned memory address
template<class ElementType>
Packed128<ElementType> load128(const ElementType* address) {
    return Packed128<ElementType>{*reinterpret_cast<const sycl::int4*>(address)};
}

// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
Packed128<ElementType> load128cs(const ElementType* address) {
    return Packed128<ElementType>{*reinterpret_cast<const sycl::int4*>(address)};
}

// store a Packed128 to an aligned memory address
template<class ElementType>
void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<sycl::int4*>(target) = value.get_bits();
}

// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
void store128cs(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<sycl::int4*>(target) = value.get_bits();
}

// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
void store128cg(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<sycl::int4*>(target) = value.get_bits();
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#if defined(ENABLE_BF16)
typedef sycl::ext::oneapi::bfloat16 floatX;
typedef sycl::ext::oneapi::bfloat16 floatN;
#elif defined(ENABLE_FP16)
typedef sycl::half floatX;
typedef sycl::half floatN;
#else
typedef float floatX;
typedef float floatN;
#endif

typedef Packed128<floatX> x128;

// ----------------------------------------------------------------------------
// random utils

float* make_random_float_01(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX); // range 0..1
    }
    return arr;
}

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

int* make_random_int(size_t N, int V) {
    int* arr = (int*)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    return arr;
}

float* make_zeros_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float)); // all zero
    return arr;
}

float* make_ones_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    return arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

template<class TargetType>
void memcpy_convert(TargetType* d_ptr, float* h_ptr, size_t count, sycl::queue &q) {
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*)malloc(count * sizeof(TargetType));
    for (int i = 0; i < count; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }
    q.memcpy(d_ptr, converted, count * sizeof(TargetType)).wait();
    free(converted);
    return;
}


template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance = 1e-4) {
    sycl::queue q(sycl::default_selector_v);

    D* out_gpu = (D*)malloc(num_elements * sizeof(D));

    // Copy results from device to host
    q.memcpy(out_gpu, device_result, num_elements * sizeof(D)).wait();

    int nfaults = 0;
#ifndef ENABLE_BF16
    float epsilon = FLT_EPSILON;
#else
    float epsilon = 0.079f;
#endif

    for (std::size_t i = 0; i < num_elements; ++i) {
        // Skip masked elements
        if (!std::isfinite(cpu_reference[i])) {
            continue;
        }

        // Print the first few comparisons
        if (i < 5) {
            std::cout << cpu_reference[i] << " " << static_cast<T>(out_gpu[i]) << std::endl;
        }

        // Effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        float t_eff = tolerance + std::fabs(cpu_reference[i]) * epsilon;

        // Ensure correctness for all elements
        if (std::fabs(cpu_reference[i] - static_cast<T>(out_gpu[i])) > t_eff) {
            std::cerr << "Mismatch of " << name << " at " << i << ": CPU_ref: " << cpu_reference[i] << " vs GPU: " << static_cast<T>(out_gpu[i]) << std::endl;
            nfaults++;
            if (nfaults >= 10) {
                free(out_gpu);
                std::exit(EXIT_FAILURE);
            }
        }
    }

    if (nfaults > 0) {
        free(out_gpu);
        std::exit(EXIT_FAILURE);
    }

    free(out_gpu);
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    float elapsed_time = 0.f;
    for (int i = 0; i < 1; i++) {
        // Start recording the timing of the kernel
        auto start = std::chrono::high_resolution_clock::now();

        kernel(std::forward<KernelArgs>(kernel_args)...);

        auto stop = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float, std::milli> duration = stop - start;
        elapsed_time += duration.count();
    }

    return elapsed_time / repeats;
}

//
template<class T>
T ceil_div(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}
#endif //LLM_SYCL_COMMON_HPP
