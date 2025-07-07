#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <torch/extension.h>


__global__ void outer_kernel(const double* x, const double* y, double* z, int B, int U, int V) {
    int b = blockIdx.x;
    int tid = threadIdx.x + blockIdx.y * blockDim.x;
    int K = U * V;
    if (tid >= K) return;

    int i = tid / V;
    int j = tid % V;
    z[b * K + i * V + j] = x[b * U + i] * y[b * V + j];
}

using ElementInputA = double;
using ElementInputB = double;
using ElementOutput = double;
using ElementAccumulator = double;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
using WarpShape        = cutlass::gemm::GemmShape<32, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutA,
    ElementInputB, LayoutB,
    ElementOutput, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm90,
    ThreadblockShape,
    WarpShape,
    InstructionShape
>;

torch::Tensor einsum_cutlass(torch::Tensor x, torch::Tensor y, torch::Tensor w) {
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2 && w.dim() == 3, "Invalid dimensions");
    int B = x.size(0);
    int U = x.size(1);
    int V = y.size(1);
    int W = w.size(2);
    int K = U * V;

    auto z = torch::empty({B, K}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto out = torch::zeros({B, W}, torch::dtype(torch::kFloat64).device(torch::kCUDA));

    // z[b, i*V + j] = x[b, i] * y[b, j]
    dim3 block(32);
    dim3 grid(B, (K + 31) / 32);
    auto x_ptr = x.data_ptr<double>();
    auto y_ptr = y.data_ptr<double>();
    auto z_ptr = z.data_ptr<double>();

    // Launch outer product kernel
    outer_kernel<<<grid, block>>>(x_ptr, y_ptr, z_ptr, B, U, V);

    // Prepare GEMM: [B, 1×K] @ [B, K×W] = [B, 1×W]
    for (int b = 0; b < B; ++b) {
        Gemm gemm_op;
        Gemm::Arguments args(
            {1, W, K},
            {z_ptr + b * K, K},
            {w.data_ptr<double>() + b * K * W, W},
            {out.data_ptr<double>() + b * W, W},
            {out.data_ptr<double>() + b * W, W},
            {1.0, 0.0}
        );
        cutlass::Status status = gemm_op(args);
        TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM failed");
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("einsum_cutlass", &einsum_cutlass, "Einsum optimized with CUTLASS and WG-MMA (CUDA)");
}
