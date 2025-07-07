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
void fused_tensor_outermatmul_shm(
    const double* __restrict__ x,    // [B, U]
    const double* __restrict__ y,    // [B, V]
    const double* __restrict__ w,    // [B, U, V, W]
    double* __restrict__ out,        // [B, W]
    int B
) {
    int b = blockIdx.x;                      // batch index
    int k_base = blockIdx.y * WARPS_PER_BLOCK;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int k_out = k_base + warp_id;

    if (b >= B || k_out >= W) return;

    // Shared memory for x[b], y[b]
    __shared__ double x_shared[U];
    __shared__ double y_shared[V];

    // Only first warp loads shared x and y
    if (threadIdx.x < U) {
        x_shared[threadIdx.x] = x[b * U + threadIdx.x];  // x[b, i]
    }
    if (threadIdx.x < V) {
        y_shared[threadIdx.x] = y[b * V + threadIdx.x];  // y[b, j]
    }
    __syncthreads();  // sync all 256 threads

    double acc = 0.0;

    for (int idx = lane_id; idx < U * V; idx += WARP_SIZE) {
        int i = idx / V;
        int j = idx % V;

        double x_val = x_shared[i];
        double y_val = y_shared[j];
        double w_val = w[((b * U + i) * V + j) * W + k_out];  // w[b, i, j, k]

        acc += x_val * y_val * w_val;
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // write result
    if (lane_id == 0) {
        out[b * W + k_out] = acc;
    }
}


void fused_tensor_outermatmul_shm_launcher(
    torch::Tensor x,   // [B, 96]
    torch::Tensor y,   // [B, 10]
    torch::Tensor w,   // [B, 96, 10, 96]
    torch::Tensor out  // [B, 96]
) {
    int B = x.size(0);

    dim3 grid(B, (W + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    dim3 block(THREADS_PER_BLOCK);  // 256 threads (8 warp)

    fused_tensor_outermatmul_shm<<<grid, block>>>(
        x.data_ptr<double>(),
        y.data_ptr<double>(),
        w.data_ptr<double>(),
        out.data_ptr<double>(),
        B
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tensor_product", &fused_tensor_outermatmul_shm_launcher, "Fused Tensor Product with Warp (CUDA)");
}
