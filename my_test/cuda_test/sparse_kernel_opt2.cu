#include <torch/extension.h>
#include <cuda_fp16.h>

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void sparse_scatter_kernel_optimized(
    const int* __restrict__ indices,  // [4, N]
    const double* __restrict__ values, // [N]
    const double* __restrict__ B,      // [B0, B1, B2]
    double* __restrict__ output,       // [B0, B1, 16,16,16]
    int num_nonzeros,
    int B0, int B1, int B2
) {
    __shared__ int sh_indices[4];  // Cache indices for current non-zero element
    __shared__ double sh_values;   // Cache value for current non-zero element
    
    // Each block handles one non-zero element (n)
    int n = blockIdx.z;
    if (n >= num_nonzeros) return;
    
    // First thread in block loads indices and value
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sh_indices[0] = indices[n];
        sh_indices[1] = indices[n + num_nonzeros];
        sh_indices[2] = indices[n + 2 * num_nonzeros];
        sh_indices[3] = indices[n + 3 * num_nonzeros];
        sh_values = values[n];
    }
    __syncthreads();
    
    // Process B matrix in tiles
    int i = blockIdx.x * TILE_DIM + threadIdx.x;
    int j = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load B values in coalesced manner
    double b_val = 0.0;
    if (i < B0 && j < B1) {
        b_val = B[i * B1 * B2 + j * B2 + sh_indices[3]];
    }
    
    // Compute the product
    double product = sh_values * b_val;
    
    // Compute output index
    if (i < B0 && j < B1) {
        int out_idx = i * B1 * 16 * 16 * 16 + j * 16 * 16 * 16 
                    + sh_indices[0] * 16 * 16 + sh_indices[1] * 16 + sh_indices[2];
        atomicAdd(&output[out_idx], product);
    }
}

// Warp-level optimization version using vectorized memory access
__global__ void sparse_scatter_kernel_vectorized(
    const int* __restrict__ indices,  // [4, N]
    const double* __restrict__ values, // [N]
    const double* __restrict__ B,      // [B0, B1, B2]
    double* __restrict__ output,       // [B0, B1, 16,16,16]
    int num_nonzeros,
    int B0, int B1, int B2
) {
    // Each warp handles one non-zero element
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    if (n >= num_nonzeros) return;
    
    // Load indices and value once per warp
    int i0 = indices[n];
    int i1 = indices[n + num_nonzeros];
    int i2 = indices[n + 2 * num_nonzeros];
    int i3 = indices[n + 3 * num_nonzeros];
    double a_val = values[n];
    
    // Process B matrix in vectorized manner
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j_start = blockIdx.y * blockDim.y * 4;  // Process 4 elements per thread
    
    if (i < B0) {
        // Vectorized load and compute
        for (int k = 0; k < 4; k++) {
            int j = j_start + k * blockDim.y + threadIdx.y;
            if (j < B1) {
                double b_val = B[i * B1 * B2 + j * B2 + i3];
                double product = a_val * b_val;
                
                int out_idx = i * B1 * 16 * 16 * 16 + j * 16 * 16 * 16 
                            + i0 * 16 * 16 + i1 * 16 + i2;
                atomicAdd(&output[out_idx], product);
            }
        }
    }
}

void sparse_scatter_mul_add_optimized(
    at::Tensor indices,
    at::Tensor values,
    at::Tensor B,
    at::Tensor output
) {
    int num_nonzeros = values.size(0);
    int B0 = B.size(0);
    int B1 = B.size(1);
    int B2 = B.size(2);

    // Choose between two kernel implementations based on problem size
    if (B0 * B1 > 1000000) {  // Large problem - use vectorized kernel
        dim3 threads(32, 8, 4);  // x, y, z
        dim3 blocks(
            (B0 + threads.x - 1) / threads.x,
            (B1 + threads.y * 4 - 1) / (threads.y * 4),
            (num_nonzeros + threads.z - 1) / threads.z
        );
        
        sparse_scatter_kernel_vectorized<<<blocks, threads>>>(
            indices.data_ptr<int>(),
            values.data_ptr<double>(),
            B.data_ptr<double>(),
            output.data_ptr<double>(),
            num_nonzeros, B0, B1, B2
        );
    } else {  // Smaller problem - use shared memory kernel
        dim3 threads(TILE_DIM, BLOCK_ROWS);
        dim3 blocks(
            (B0 + TILE_DIM - 1) / TILE_DIM,
            (B1 + TILE_DIM - 1) / TILE_DIM,
            num_nonzeros
        );
        
        sparse_scatter_kernel_optimized<<<blocks, threads>>>(
            indices.data_ptr<int>(),
            values.data_ptr<double>(),
            B.data_ptr<double>(),
            output.data_ptr<double>(),
            num_nonzeros, B0, B1, B2
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_scatter_mul_add_optimized", &sparse_scatter_mul_add_optimized, "Optimized sparse scatter multiply-add");
}
