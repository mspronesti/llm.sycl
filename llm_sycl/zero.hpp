/*
Utilities for ZeRO sharding
*/
#ifndef LLM_SYCL_ZERO_HPP
#define LLM_SYCL_ZERO_HPP

#include <sycl/sycl.hpp>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>



// ----------------------------------------------------------------------------
// Multi-GPU related


// ----------------------------------------------------------------------------
// MPI / multi-processing setup

// Parameters specific to training on multiple GPUs.
typedef struct {
    int process_rank;      // Rank of this process among all MPI processes. 0 if no multi-GPU.
    int num_processes;     // Total number of processes. 1 if no multi-GPU.
    int local_device_idx;  // This process GPU index on current machine. 0 if no multi-GPU.

    // Zero Redundancy Optimizer stage - https://fairscale.readthedocs.io/en/stable/deep_dive/oss_sdp_fsdp.html
    // 0-Disabled
    // 1-Optimizer State Sharding (OSS)
    // 2-Optimizer + Gradient State Sharding (SDP)
    // 3-Optimizer + Gradient + Horizontal Model Sharding (FSDP)
    int zero_stage;
    size_t shard_num_parameters;
} MultiGpuConfig;


MultiGpuConfig multi_gpu_config_init(int *argc, char ***argv) {
    printf("Multi-GPU support is disabled. Using a single GPU.\n");

    MultiGpuConfig result;
    result.process_rank = 0;
    result.num_processes = 1;
    result.local_device_idx = 0;
    return result;
}

void multi_gpu_config_free(MultiGpuConfig* multi_gpu_config) {}

void multi_gpu_barrier(const MultiGpuConfig* multi_gpu_config) {}

// Offset and size of a tensor shard
typedef struct {
    ptrdiff_t offset;
    size_t size;
} ShardInfo;

// Get info about sharding for a tensor of elements many numbers
ShardInfo multi_gpu_get_shard_offset(size_t elements, const MultiGpuConfig* multi_gpu_config, int shard_at_stage) {
    const int nproc = multi_gpu_config->num_processes;
    if(multi_gpu_config->zero_stage >= shard_at_stage) {
        if (elements % nproc != 0) {
            fprintf(stderr, "Number of elements %zu must be a multiple of the number of processes %d\n", elements, nproc);
            exit(EXIT_FAILURE);
        }
        return {(ptrdiff_t) (multi_gpu_config->process_rank * (elements / nproc)), elements / nproc};
    } else {
        return {0, elements};
    }
}

// Block NCCL stream until computations on compute_stream are done, then aggregate multiple pointers in an NCCL group.
// This can work either as an all-reduce (i.e., no ZeRo), or a reduce-scatter (ZeRO 1).
// The awkward `(&pointers)[N]` syntax ensures we are capturing the parameters as sized arrays, so that it becomes impossible
// to call this function if pointers and pointers_sizes do not match.
template<int N>
void multi_gpu_async_reduce_gradient(
        sycl::queue *stream,
        floatX* const (&pointers)[N], const size_t (&pointers_sizes)[N],
        MultiGpuConfig* multi_gpu_config) {
    if (multi_gpu_config->num_processes == 1) {
        return; // no multi-GPU, just exit.
    }
}

#endif //LLM_SYCL_ZERO_HPP
