#include <torch/extension.h>

__global__ void sparse_scatter_kernel_v2(
    const int* __restrict__ indices,  // [4, N]
    const double* __restrict__ values, // [N]
    const double* __restrict__ B,      // [B0, B1, B2]
    double* __restrict__ output,       // [B0, B1, 16, 16, 16]
    int num_nonzeros,
    int B0, int B1, int B2
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    int total_work = num_nonzeros * B0 * B1;

    for (int global_idx = tid; global_idx < total_work; global_idx += total_threads) {
        int n = global_idx / (B0 * B1);
        int rem = global_idx % (B0 * B1);
        int i = rem / B1;
        int j = rem % B1;

        if (n >= num_nonzeros || i >= B0 || j >= B1) continue;

        int i0 = indices[n];
        int i1 = indices[n + num_nonzeros];
        int i2 = indices[n + 2 * num_nonzeros];
        int i3 = indices[n + 3 * num_nonzeros];
        double a_val = values[n];

        double b_val = B[i * B1 * B2 + j * B2 + i3];

        int out_idx = i * B1 * 16 * 16 * 16 + j * 16 * 16 * 16 + i0 * 16 * 16 + i1 * 16 + i2;

        atomicAdd(&output[out_idx], a_val * b_val);
    }
}

void sparse_scatter_mul_add_v2(
    at::Tensor indices,
    at::Tensor values,
    at::Tensor B,
    at::Tensor output
) {
    int num_nonzeros = values.size(0);
    int B0 = B.size(0);
    int B1 = B.size(1);
    int B2 = B.size(2);

    int total_work = num_nonzeros * B0 * B1;

    int threads = 256;
    int blocks = (total_work + threads - 1) / threads;

    sparse_scatter_kernel_v2<<<blocks, threads>>>(
        indices.data_ptr<int>(),
        values.data_ptr<double>(),
        B.data_ptr<double>(),
        output.data_ptr<double>(),
        num_nonzeros, B0, B1, B2
    );
}

//#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_scatter_mul_add_v2", &sparse_scatter_mul_add_v2, "Optimized sparse scatter multiply-add");
}
