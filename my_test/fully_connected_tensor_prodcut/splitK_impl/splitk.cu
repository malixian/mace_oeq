#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

#define U 96
#define V 10
#define W 96
#define SPLIT_K_THREADS 32  // warp-level并行


extern "C" __global__
void fused_tensor_outermatmul_warp(
    const double* __restrict__ x,    // [B, U]
    const double* __restrict__ y,    // [B, V]
    const double* __restrict__ w,    // [B, U, V, W]
    double* __restrict__ out,        // [B, W]
    int B
) {
    int b = blockIdx.x;      // batch index
    int k_out = blockIdx.y;  // output index
    int tid = threadIdx.x;   // warp thread index

    if (b >= B || k_out >= W) return;

    double acc = 0.0;

    for (int idx = tid; idx < U * V; idx += blockDim.x) {
        int i = idx / V;
        int j = idx % V;

        double x_val = x[b * U + i];                        // x[b, i]
        double y_val = y[b * V + j];                        // y[b, j]
        double w_val = w[((b * U + i) * V + j) * W + k_out]; // w[b, i, j, k]

        acc += x_val * y_val * w_val;
    }

    for (int offset = 16; offset > 0; offset /= 2)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    if (tid == 0) {
        out[b * W + k_out] = acc;  // out[b, k]
    }
}

void fused_tensor_outermatmul_warp_launcher(
    torch::Tensor x,   // [B, 96]
    torch::Tensor y,   // [B, 10]
    torch::Tensor w,   // [B, 96, 10, 96]
    torch::Tensor out  // [B, 96]
) {
    int B = x.size(0);

    dim3 grid(B, 96);      // B × output_dim(W)
    dim3 block(32);        // warp-size threads

    fused_tensor_outermatmul_warp<<<grid, block>>>(
        x.data_ptr<double>(),
        y.data_ptr<double>(),
        w.data_ptr<double>(),
        out.data_ptr<double>(),
        B
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tensor_product", &fused_tensor_outermatmul_warp_launcher, "Fused Tensor Product with Warp (CUDA)");
}
