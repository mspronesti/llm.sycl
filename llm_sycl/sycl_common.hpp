#ifndef LLM_SYCL_COMMON_HPP
#define LLM_SYCL_COMMON_HPP

#include <sycl/sycl.hpp>
#include <cfloat>
#include <cmath>
#include <cstdlib>

// WarpSize is not a compile time constant
// Defining here like this possibly allows the compiler to optimize better
#define WARP_SIZE 32U

// convenience macro for calculating grid/block dimensions for kernels
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// ----------------------------------------------------------------------------
// SYCL Precision settings and defines

enum PrecisionMode {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16
};

// Specific configurations based on the enabled precision
#if defined(ENABLE_FP32)
typedef float floatX;
#define PRECISION_MODE PRECISION_FP32
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#elif defined(ENABLE_FP16)
typedef sycl::half floatX;
#define PRECISION_MODE PRECISION_FP16
#else // Default to bfloat16
typedef sycl::ext::oneapi::bfloat16 floatX;
#define PRECISION_MODE PRECISION_BF16
#endif


#endif //LLM_SYCL_COMMON_HPP
