#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define U 96
#define V 10
#define W 96
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

__global__ void fused_single_path_debug_kernel(
    const double* __restrict__ x,   // [B, U]
    const double* __restrict__ y,   // [B, V]
    const double* __restrict__ w,   // [B, U, V, W]
    double* __restrict__ out,       // [B, W]
    int B
) {
    int b = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int k = blockIdx.y * WARPS_PER_BLOCK + warp_id;

    if (b >= B || k >= W) return;

    __shared__ double x_shared[WARPS_PER_BLOCK][U];
    __shared__ double y_shared[V];

    // 每个 block 只读一次 y[b]
    if (threadIdx.x < V) {
        y_shared[threadIdx.x] = y[b * V + threadIdx.x];
    }

    // 每个 warp 分别加载 x[b]
    for (int u = lane_id; u < U; u += WARP_SIZE) {
        x_shared[warp_id][u] = x[b * U + u];
    }

    __syncthreads();

    double acc = 0.0;

    for (int idx = lane_id; idx < U * V; idx += WARP_SIZE) {
        int uu = idx / V;
        int vv = idx % V;

        double x_val = x_shared[warp_id][uu];
        double y_val = y_shared[vv];

        int w_idx = ((b * U + uu) * V + vv) * W + k;
        double w_val = w[w_idx];

        acc += x_val * y_val * w_val;
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane_id == 0) {
        out[b * W + k] = acc;
    }
}

void fused_single_path_debug_launcher(
    torch::Tensor x,   // [B, U]
    torch::Tensor y,   // [B, V]
    torch::Tensor w,   // [B, U, V, W]
    torch::Tensor out  // [B, W]
) {
    int B = x.size(0);
    dim3 grid(B, (W + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    dim3 block(THREADS_PER_BLOCK);

    fused_single_path_debug_kernel<<<grid, block>>>(
        x.data_ptr<double>(),
        y.data_ptr<double>(),
        w.data_ptr<double>(),
        out.data_ptr<double>(),
        B
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_single_path_debug", &fused_single_path_debug_launcher,
          "Fused Single Path Debug (CUDA)");
}

