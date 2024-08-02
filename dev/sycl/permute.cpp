/*
Kernels to demonstrate permute operation.

Compile example:
nvcc -O3 permute.cu -o permute

The goal is to permute a 4D matrix from its original shape (dim1, dim2, dim3, dim4) to a new shape (dim4, dim3, dim1, dim2).

Before permutation, we need to understand how to access elements in a flattened (linear) form of the matrix.

Given:

dim1 = size of the 1st dimension
dim2 = size of the 2nd dimension
dim3 = size of the 3rd dimension
dim4 = size of the 4th dimension

For any element in a 4D matrix at position (i1, i2, i3, i4), where:

i1 is the index in dimension 1
i2 is the index in dimension 2
i3 is the index in dimension 3
i4 is the index in dimension 4

If you find it challenging to calculate the indices i1, i2, i3, and i4, observe the pattern in the index calculations.
Initially, it might take some time to grasp, but with practice, you'll develop a mental model for it.

To calculate the indices, use the following formulas:

i1 = (idx / (dim2 * dim3 * dim4)) % dim1;
i2 = (idx / (dim3 * dim4)) % dim2;
i3 = (idx / dim4) % dim3;
i4 = idx % dim4;

Pattern Explanation:
To find the index for any dimension, divide the thread ID (idx) by the product of all subsequent dimensions.
Then, perform modulo operation with the current dimension.



The linear index in a flattened 1D array is calculated as:
linear_idx = i1 × ( dim2 × dim3 × dim4 ) + i2 × ( dim3 × dim4 ) + i3 × dim4 + i4
This linear index uniquely identifies the position of the element in the 1D array.

To permute the matrix, we need to rearrange the indices according to the new shape.
In this case, we are permuting from (dim1, dim2, dim3, dim4) to (dim4, dim3, dim1, dim2).

The new dimension post permutation will be as follows:

dim1 becomes the new 3rd dimension.
dim2 becomes the new 4th dimension.
dim3 becomes the new 2nd dimension.
dim4 becomes the new 1st dimension.

permuted_idx = i4 * (dim3 * dim1 * dim2) + i3 * (dim1 * dim2) + i1 * dim2 + i2;

Here's how this works:

i4 * (dim3 * dim1 * dim2): This accounts for how many complete dim3 × dim1 × dim2 blocks fit before the current i4 block.
i3 * (dim1 * dim2): This accounts for the offset within the current i4 block, specifying which i3 block we are in.
i1 * dim2: This accounts for the offset within the current i3 block, specifying which i1 block we are in.
i2: This gives the offset within the current i1 block.

Lastly at the end we store the current value at idx index of the original value to the permuted index in the permuted_matrix.


--------------------------------------------------------------------------------------------------------------------------------------------------------

Similarly we can follow the above approach to permute matrices of any dimensions.

*/


#include <sycl/sycl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include "common.hpp"

// CPU function to permute a 4D matrix
void permute_cpu(const float* matrix, float* out_matrix, int dim1, int dim2, int dim3, int dim4) {
    int total_threads = dim1 * dim2 * dim3 * dim4;

    for (int idx = 0; idx < total_threads; idx++) {
        // Calculate the 4D indices from the linear index
        int i1 = (idx / (dim2 * dim3 * dim4)) % dim1;
        int i2 = (idx / (dim3 * dim4)) % dim2;
        int i3 = (idx / dim4) % dim3;
        int i4 = idx % dim4;

        // Compute the new index for the permuted matrix
        // Transpose from (dim1, dim2, dim3, dim4) to (dim4, dim3, dim1, dim2)
        int permuted_idx = i4 * (dim3 * dim1 * dim2) + i3 * (dim1 * dim2) + i1 * dim2 + i2;
        out_matrix[permuted_idx] = matrix[idx];
    }
}

// SYCL kernel to permute a 4D matrix
void permute_kernel(sycl::nd_item<1> id, const float* matrix, float* out_matrix, int dim1, int dim2, int dim3, int dim4) {
    int idx = id.get_global_id(0);

    // Ensure index is within bounds
    if (idx < dim1 * dim2 * dim3 * dim4) {
        // Calculate the 4D indices from the linear index
        int i1 = (idx / (dim2 * dim3 * dim4)) % dim1;
        int i2 = (idx / (dim3 * dim4)) % dim2;
        int i3 = (idx / dim4) % dim3;
        int i4 = idx % dim4;

        // Compute the new index for the permuted matrix
        // Transpose from (dim1, dim2, dim3, dim4) to (dim4, dim3, dim1, dim2)
        int permuted_idx = i4 * (dim3 * dim1 * dim2) + i3 * (dim1 * dim2) + i1 * dim2 + i2;
        out_matrix[permuted_idx] = matrix[idx];
    }
}

void launch_permute_kernel(sycl::queue &q, const float* d_matrix, float* d_out_matrix,
                           int dim1, int dim2, int dim3, int dim4, int block_size) {
    // Define block and grid sizes
    int total_threads = dim1 * dim2 * dim3 * dim4;
    int grid_size = ceil_div(total_threads, block_size); // Compute grid size

    // Launch SYCL kernel to perform permutation
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        permute_kernel(id, d_matrix, d_out_matrix, dim1, dim2, dim3, dim4);
    }).wait();
}


int main() {
    int dim_1 = 24;
    int dim_2 = 42;
    int dim_3 = 20;
    int dim_4 = 32;

    // Set up the device
    sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order());
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << '\n';

    // Allocate host memory
    float* matrix = make_random_float(dim_1 * dim_2 * dim_3 * dim_4);
    float* permuted_matrix = new float[dim_1 * dim_2 * dim_3 * dim_4];

    // Initialize the matrix with random values

    // Allocate device memory
    float *d_matrix, *d_permuted_matrix;
    d_matrix = sycl::malloc_device<float>(dim_1 * dim_2 * dim_3 * dim_4, q);
    d_permuted_matrix = sycl::malloc_device<float>(dim_1 * dim_2 * dim_3 * dim_4, q);

    // Copy matrix from host to device
    q.memcpy(d_matrix, matrix, dim_1 * dim_2 * dim_3 * dim_4 * sizeof(float)).wait();

    // Perform permutation on CPU
    clock_t start = clock();
    permute_cpu(matrix, permuted_matrix, dim_1, dim_2, dim_3, dim_4);
    clock_t end = clock();
    double elapsed_time_cpu = (double)(end - start) / CLOCKS_PER_SEC;

    // Verify results
    std::cout << "Checking correctness...\n";
    int block_sizes[] = {32, 64, 128, 256, 512};
    for (int block_size : block_sizes) {
        std::cout << "Checking block size " << block_size << "." << std::endl;
        launch_permute_kernel(q, d_matrix, d_permuted_matrix, dim_1, dim_2, dim_3, dim_4, block_size);
        validate_result(d_permuted_matrix, permuted_matrix,
                        "permuted_matrix", dim_1 * dim_2 * dim_3 * dim_4, 1e-5f);
    }
    std::cout << "All results match.\n\n";
    // benchmark kernel
    int repeat_times = 1000;
    for (int block_size : block_sizes) {
        float elapsed_time = benchmark_kernel(
                repeat_times,
                launch_permute_kernel,
                // params
                q, d_matrix, d_permuted_matrix, dim_1, dim_2, dim_3, dim_4, block_size
        );

        std::cout << "block size: " << block_size << "\t| time: " << elapsed_time << " ms\n";
    }
    std::cout << "time cpu: " << elapsed_time_cpu << " ms\n";

    // Free allocated memory
    delete[] matrix;
    delete[] permuted_matrix;

    sycl::free(d_matrix, q);
    sycl::free(d_permuted_matrix, q);

    return 0;
}