#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define U 96
#define V 10
#define W 96
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 16
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

__global__ void fused_multi_path_kernel(
    const double* __restrict__ x,     // [B, total_i * U]
    const double* __restrict__ y,     // [B, V]
    const double* __restrict__ w,     // [B, num_paths * U * V * W]
    double* __restrict__ out,         // [B, total_k * W]
    const int* __restrict__ i_dims,   // [num_paths]
    const int* __restrict__ k_dims,   // [num_paths]
    int num_paths,
    int total_i,
    int total_k,
    int B,
    double val
) {
    int b = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int k = blockIdx.y * WARPS_PER_BLOCK + warp_id;

    if (b >= B || k >= W) return;

    __shared__ double y_shared[V];
    if (threadIdx.x < V) {
        y_shared[threadIdx.x] = y[b * V + threadIdx.x];
    }

    __syncthreads();

    int i_offset = 0;
    int k_offset = 0;

    for (int pid = 0; pid < num_paths; ++pid) {
        int i = i_dims[pid];
        int k_len = k_dims[pid];

        for (int local_i = 0; local_i < i; ++local_i) {
            // Load x[b, i_offset + local_i] into shared memory
            __shared__ double x_shared[WARPS_PER_BLOCK][U];
            int x_base = b * total_i * U + (i_offset + local_i) * U;

            for (int u = lane_id; u < U; u += WARP_SIZE) {
                x_shared[warp_id][u] = x[x_base + u];
            }

            __syncthreads();

            double acc = 0.0;

            // W[b, pid, U, V, W]
            int w_base = b * num_paths * U * V * W + pid * U * V * W;

            
	    for (int idx = lane_id; idx < U * V; idx += WARP_SIZE) {
                int uu = idx / V;
                int vv = idx % V;

                double x_val = x_shared[warp_id][uu];
                double y_val = y_shared[vv];
                int w_idx = ((uu * V + vv) * W) + k;
                double w_val = w[w_base + w_idx];

                acc += x_val * y_val * w_val;
            }
	    

            // warp reduce
            for (int offset = 16; offset > 0; offset /= 2) {
                acc += __shfl_down_sync(0xffffffff, acc, offset);
            }

            if (lane_id == 0) {
                int out_idx = (k_offset + local_i) * W + k;
                atomicAdd(&out[b * total_k * W + out_idx], val * acc);
            }

            __syncthreads();
        }

        i_offset += i;
        k_offset += k_len;
    }
}

void fused_multi_path_launcher(
    torch::Tensor x,       // [B, total_i * U]
    torch::Tensor y,       // [B, V]
    torch::Tensor w,       // [B, num_paths * U * V * W]
    torch::Tensor output,  // [B, total_k * W]
    std::vector<int> i_dims_vec,
    std::vector<int> k_dims_vec,
    double val
) {
    TORCH_CHECK(x.dtype() == torch::kFloat64, "X must be float64");
    TORCH_CHECK(y.dtype() == torch::kFloat64, "Y must be float64");
    TORCH_CHECK(w.dtype() == torch::kFloat64, "W must be float64");
    TORCH_CHECK(output.dtype() == torch::kFloat64, "output must be float64");

    int B = x.size(0);
    int num_paths = i_dims_vec.size();
    int total_i = 0, total_k = 0;

    for (int i : i_dims_vec) total_i += i;
    for (int k : k_dims_vec) total_k += k;

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto i_tensor = torch::tensor(i_dims_vec, options);
    auto k_tensor = torch::tensor(k_dims_vec, options);

    dim3 grid(B, (W + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    dim3 block(THREADS_PER_BLOCK);

    fused_multi_path_kernel<<<grid, block>>>(
        x.data_ptr<double>(),
        y.data_ptr<double>(),
        w.data_ptr<double>(),
        output.data_ptr<double>(),
        i_tensor.data_ptr<int>(),
        k_tensor.data_ptr<int>(),
        num_paths,
        total_i,
        total_k,
        B,
        val
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_multi_path", &fused_multi_path_launcher,
          "Fused Multi-Path Tensor Outer Product (CUDA)");
}

