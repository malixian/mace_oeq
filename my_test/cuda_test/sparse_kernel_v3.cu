#include <torch/extension.h>

__global__ void sparse_scatter_kernel_block_accum(
    const int* __restrict__ indices,  // [4, N]
    const double* __restrict__ values, // [N]
    const double* __restrict__ B,      // [B0, B1, B2]
    double* __restrict__ output,       // [B0, B1, 16, 16, 16]
    int num_nonzeros,
    int B0, int B1, int B2
) {
    // Block (i, j)
    int i = blockIdx.x;
    int j = blockIdx.y;
    int tid = threadIdx.x;

    // Allocate shared memory for the output sub-block
    __shared__ double accum[16][16][16]; // shape of output[i,j,:,:,:]
    for (int a = tid; a < 16*16*16; a += blockDim.x) {
        int z = a % 16;
        int y = (a / 16) % 16;
        int x = a / 256;
        accum[x][y][z] = 0.0;
    }
    __syncthreads();

    int total_threads = blockDim.x;

    // Flattened thread block processes all n in [0, num_nonzeros)
    for (int n = tid; n < num_nonzeros; n += total_threads) {
        int i0 = indices[n];
        int i1 = indices[n + num_nonzeros];
        int i2 = indices[n + 2 * num_nonzeros];
        int i3 = indices[n + 3 * num_nonzeros];

        double a_val = values[n];

        double b_val = B[((i * B1) + j) * B2 + i3];

        // Accumulate to shared memory
        atomicAdd(&accum[i0][i1][i2], a_val * b_val);
    }

    __syncthreads();

    // Write back to global memory (with atomicAdd to avoid block overlap)
    for (int a = tid; a < 16 * 16 * 16; a += blockDim.x) {
        int z = a % 16;
        int y = (a / 16) % 16;
        int x = a / 256;

        size_t out_idx = (((i * B1 + j) * 16 + x) * 16 + y) * 16 + z;
        double v = accum[x][y][z];
        if (v != 0.0) {
            atomicAdd(&output[out_idx], v);
        }
    }
}

void sparse_scatter_mul_add_v3(
    at::Tensor indices,
    at::Tensor values,
    at::Tensor B,
    at::Tensor output
) {
    int num_nonzeros = values.size(0);
    int B0 = B.size(0);
    int B1 = B.size(1);
    int B2 = B.size(2);

    dim3 blocks(B0, B1);         // Each block processes one output[i,j]
    int threads = 256;           // Enough to cover accumulation

    sparse_scatter_kernel_block_accum<<<blocks, threads>>>(
        indices.data_ptr<int>(),
        values.data_ptr<double>(),
        B.data_ptr<double>(),
        output.data_ptr<double>(),
        num_nonzeros, B0, B1, B2
    );
}


//#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_scatter_mul_add_v3", &sparse_scatter_mul_add_v3, "Optimized sparse scatter multiply-add");
}
