#include <torch/extension.h>

__global__ void sparse_scatter_kernel_improved(
    const int* __restrict__ indices,    // [4, N]
    const double* __restrict__ values,  // [N]
    const double* __restrict__ B,       // [B0, B1, B2]
    double* __restrict__ output,        // [B0, B1, 16,16,16]
    int num_nonzeros,
    int B0, int B1, int B2
) {
    int n = blockIdx.z;
    int i = blockIdx.x * blockDim.x + threadIdx.x; // B0
    int j = blockIdx.y * blockDim.y + threadIdx.y; // B1

    if (n >= num_nonzeros || i >= B0 || j >= B1) return;

    int i0 = indices[n];
    int i1 = indices[n + num_nonzeros];
    int i2 = indices[n + 2 * num_nonzeros];
    int i3 = indices[n + 3 * num_nonzeros];
    double a_val = values[n];

    double b_val = B[i * B1 * B2 + j * B2 + i3];
    double prod = a_val * b_val;

    int out_idx = i * B1 * 16 * 16 * 16 + j * 16 * 16 * 16 + i0 * 16 * 16 + i1 * 16 + i2;

    atomicAdd(&output[out_idx], prod);
}

void sparse_scatter_mul_add_improved(
    at::Tensor indices,
    at::Tensor values,
    at::Tensor B,
    at::Tensor output
) {
    int num_nonzeros = values.size(0);
    int B0 = B.size(0);
    int B1 = B.size(1);
    int B2 = B.size(2);

    dim3 threads(32, 8);
    dim3 blocks(
        (B0 + threads.x - 1) / threads.x,
        (B1 + threads.y - 1) / threads.y,
        num_nonzeros
    );

    sparse_scatter_kernel_improved<<<blocks, threads>>>(
        indices.data_ptr<int>(),
        values.data_ptr<double>(),
        B.data_ptr<double>(),
        output.data_ptr<double>(),
        num_nonzeros, B0, B1, B2
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_scatter_mul_add_optimized", &sparse_scatter_mul_add_improved, "Optimized sparse scatter multiply-add");
}
