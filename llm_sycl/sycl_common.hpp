#ifndef LLM_SYCL_COMMON_HPP
#define LLM_SYCL_COMMON_HPP

#include <sycl/sycl.hpp>
#include <cfloat>
#include <math.h>
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

// ----------------------------------------------------------------------------
// Utilities to Read & Write between CUDA memory <-> files

// copy num_bytes from device pointer src into file dest, using double buffering running on the given stream.
inline void device_to_file(FILE* dest, void* src, size_t num_bytes, size_t buffer_size, sycl::queue *stream) {
    // allocate pinned buffer for faster, async transfer
    char* buffer_space;
    buffer_space = sycl::malloc_host<char>(2*buffer_size, *stream);
    // split allocation in two
    void* read_buffer = buffer_space;
    void* write_buffer = buffer_space + buffer_size;

    // prime the read buffer; first copy means we have to wait
    char* gpu_read_ptr = (char*)src;
    size_t copy_amount = std::min(buffer_size, num_bytes);
    stream->memcpy(read_buffer, gpu_read_ptr, copy_amount);
    stream->wait();
    size_t rest_bytes = num_bytes - copy_amount;
    size_t write_buffer_size = copy_amount;
    gpu_read_ptr += copy_amount;

    std::swap(read_buffer, write_buffer);
    // now the main loop; as long as there are bytes left
    while(rest_bytes > 0) {
        // initiate next read
        copy_amount = std::min(buffer_size, rest_bytes);
        stream->memcpy(read_buffer, gpu_read_ptr, copy_amount);
        // while this is going on, transfer the write buffer to disk
        fwriteCheck(write_buffer, 1, write_buffer_size, dest);
        stream->wait();    // wait for both buffers to be ready.

        std::swap(read_buffer, write_buffer);
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
        gpu_read_ptr += copy_amount;
    }

    // make sure to write the last remaining write buffer
    fwriteCheck(write_buffer, 1, write_buffer_size, dest);
    sycl::free(buffer_space, *stream);
}

// copy num_bytes from file src into device pointer dest, using double buffering running on the given stream.
inline void file_to_device(void* dest, FILE* src, size_t num_bytes, size_t buffer_size, sycl::queue *stream) {
    // allocate pinned buffer for faster, async transfer
    // from the docs (https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__HIGHLEVEL_ge439496de696b166ba457dab5dd4f356.html)
    // WC memory is a good option for buffers that will be written by the CPU and read by the device via mapped pinned memory or host->device transfers.
    char* buffer_space;
    buffer_space = sycl::malloc_host<char>(2*buffer_size, *stream);
    // split allocation in two
    void* read_buffer = buffer_space;
    void* write_buffer = buffer_space + buffer_size;

    // prime the read buffer;
    char* gpu_write_ptr = (char*)dest;
    size_t copy_amount = std::min(buffer_size, num_bytes);
    freadCheck(read_buffer, 1, copy_amount, src);

    size_t rest_bytes = num_bytes - copy_amount;
    size_t write_buffer_size = copy_amount;
    std::swap(read_buffer, write_buffer);

    // now the main loop; as long as there are bytes left
    while(rest_bytes > 0) {
        // initiate next read
        copy_amount = std::min(buffer_size, rest_bytes);
        stream->memcpy(gpu_write_ptr, write_buffer, write_buffer_size);
        gpu_write_ptr += write_buffer_size;
        // while this is going on, read from disk
        freadCheck(read_buffer, 1, copy_amount, src);
        stream->wait();    // wait for both buffers to be ready.

        std::swap(read_buffer, write_buffer);
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
    }

    // copy the last remaining write buffer to gpu
    stream->memcpy(gpu_write_ptr, write_buffer, write_buffer_size);
    stream->wait();
    sycl::free(buffer_space, *stream);
}

#endif //LLM_SYCL_COMMON_HPP
