#ifndef LLM_SYCL_SYCL_UTILS_HPP
#define LLM_SYCL_SYCL_UTILS_HPP

#include <sycl_common.hpp>

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


// short-form typedef
typedef Packed128<float> f128;
typedef Packed128<floatX> x128;

// ----------------------------------------------------------------------------
// Copy, cast functions

// device functions and the kernel to cast data between types
template<typename Td, typename Ts>
Td cast_value(Ts val);

template<>
float cast_value<float, float>(float val) {
    return val;
}

template<>
float cast_value<float, sycl::half>(sycl::half val) {
    return (float)val;
}

template<>
float cast_value<float, sycl::ext::oneapi::bfloat16>(sycl::ext::oneapi::bfloat16 val) {
    return (float)val;
}

template<typename Td, typename Ts>
void copy_and_cast_kernel(sycl::nd_item<2> id, Td* dst, const Ts* src, size_t n, ptrdiff_t stride_dst, ptrdiff_t stride_src) {
    int idx = id.get_group(1) * id.get_local_range(1) + id.get_local_id(1);
    // need to try grid stride looping for more perf later
    if (idx < n) {
        dst[idx + stride_dst * id.get_group(0)] = cast_value<Td, Ts>(src[idx + stride_src * id.get_group(0)]);
    }
}

// ----------------------------------------------------------------------------
// Warp/Block communication primitives

// warp-level reduction for summing values
inline float warpReduceSum(sycl::sub_group warp, float val) {
    return sycl::reduce_over_group(warp, val, sycl::plus<float>{});
}
// warp-level reduction for finding the maximum value
inline float warpReduceMax(sycl::sub_group warp, float val) {
    return sycl::reduce_over_group(warp, val, sycl::maximum<float>{});
}


// ----------------------------------------------------------------------------
// Random Number Generation used in Stochastic Rounding

// SquirrelNoise5 - Squirrel's Raw Noise utilities (version 5)
// This gives us a random number from threadIdx/blockIdx + a single seed for the entire GPU
// todo - possibly overkill and we don't need such high quality random numbers? (tbd)
// http://eiserloh.net/noise/SquirrelNoise5.hpp
constexpr unsigned int SquirrelNoise5(int positionX, unsigned int seed)
{
    constexpr unsigned int SQ5_BIT_NOISE1 = 0xd2a80a3f;	// 11010010101010000000101000111111
    constexpr unsigned int SQ5_BIT_NOISE2 = 0xa884f197;	// 10101000100001001111000110010111
    constexpr unsigned int SQ5_BIT_NOISE3 = 0x6C736F4B; // 01101100011100110110111101001011
    constexpr unsigned int SQ5_BIT_NOISE4 = 0xB79F3ABB;	// 10110111100111110011101010111011
    constexpr unsigned int SQ5_BIT_NOISE5 = 0x1b56c4f5;	// 00011011010101101100010011110101
    unsigned int mangledBits = (unsigned int) positionX;
    mangledBits *= SQ5_BIT_NOISE1;
    mangledBits += seed;
    mangledBits ^= (mangledBits >> 9);
    mangledBits += SQ5_BIT_NOISE2;
    mangledBits ^= (mangledBits >> 11);
    mangledBits *= SQ5_BIT_NOISE3;
    mangledBits ^= (mangledBits >> 13);
    mangledBits += SQ5_BIT_NOISE4;
    mangledBits ^= (mangledBits >> 15);
    mangledBits *= SQ5_BIT_NOISE5;
    mangledBits ^= (mangledBits >> 17);
    return mangledBits;
}
constexpr unsigned int Get2dNoiseUint(int indexX, int indexY, unsigned int seed)
{
    constexpr int PRIME_NUMBER = 198491317; // Large prime number with non-boring bits
    return SquirrelNoise5(indexX + (PRIME_NUMBER * indexY), seed);
}

// stochastic rounding built on top of Squirel Noise above (with seed updated per step via xorshift)
inline void stochastic_rounding(sycl::nd_item<2> id, float in, sycl::ext::oneapi::bfloat16 *out, unsigned int seed) {
    // todo - is this stochastic rounding *too good*? can we cut any corners?
    unsigned int random = Get2dNoiseUint(threadIdx_x(id), blockIdx_x(id) * blockDim_x(id) + blockIdx_y(id), seed);
    unsigned int threshold = random & 0xFFFF;
    unsigned int float_bits =  *reinterpret_cast<unsigned int*>(&in);
    unsigned int rounded_bits = float_bits & 0x0000FFFF;
    float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFF) : (float_bits  & ~0xFFFF);
    *out = sycl::ext::oneapi::bfloat16(*reinterpret_cast<float*>(&float_bits));
}
inline void stochastic_rounding(sycl::nd_item<2> id, float in, sycl::half *out, unsigned int random) {
    *out = (float)in; // todo - implement this...
}
inline void stochastic_rounding(sycl::nd_item<2> id, float in, float *out, unsigned int random) {
    *out = in; // dummy function for when floatX is float (FP32 mode)
}


#endif //LLM_SYCL_SYCL_UTILS_HPP
