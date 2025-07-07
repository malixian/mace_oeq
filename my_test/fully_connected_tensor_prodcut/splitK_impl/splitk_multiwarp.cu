#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


#define U 96
#define V 10
#define W 96
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

extern "C" __global__
void fused_tensor_outermatmul_multiwarp(
    const double* __restrict__ x,   // [B, U]
    const double* __restrict__ y,   // [B, V]
    const double* __restrict__ w,   // [B, U, V, W]
    double* __restrict__ out,       // [B, W]
    int B
) {
    int b = blockIdx.x;                 // batch index
    int k_base = blockIdx.y * WARPS_PER_BLOCK;  // 每 block 起始输出通道索引
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    int k_out = k_base + warp_id;
    if (b >= B || k_out >= W) return;

    double acc = 0.0;

    for (int idx = lane_id; idx < U * V; idx += WARP_SIZE) {
        int i = idx / V;
        int j = idx % V;

        double x_val = x[b * U + i];                            // x[b, i]
        double y_val = y[b * V + j];                            // y[b, j]
        double w_val = w[((b * U + i) * V + j) * W + k_out];    // w[b, i, j, k_out]

        acc += x_val * y_val * w_val;
    }

    // warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    // 只有每 warp 的第一个线程写入结果
    if (lane_id == 0) {
        out[b * W + k_out] = acc;
    }
}

void fused_tensor_outermatmul_multiwarp_launcher(
    torch::Tensor x,   // [B, 96]
    torch::Tensor y,   // [B, 10]
    torch::Tensor w,   // [B, 96, 10, 96]
    torch::Tensor out  // [B, 96]
) {
    int B = x.size(0);

    dim3 grid(B, (W + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK); // 每 block 处理 8 个输出通道
    dim3 block(THREADS_PER_BLOCK);  // 256 threads = 8 warps

    fused_tensor_outermatmul_multiwarp<<<grid, block>>>(
        x.data_ptr<double>(),
        y.data_ptr<double>(),
        w.data_ptr<double>(),
        out.data_ptr<double>(),
        B
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tensor_product", &fused_tensor_outermatmul_multiwarp_launcher, "Fused Tensor Product with Warp (CUDA)");
}
