import torch
import os, time
from torch.utils.cpp_extension import load

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 编译 CUDA kernel
fused_tp = load(
    name="fused_tensor_product",
    sources=["fused_single_path.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True
)

B = 4
u, v, w = 96, 10, 96

x = torch.randn(B, u, dtype=torch.float64, device='cuda')
y = torch.randn(B, v, dtype=torch.float64, device='cuda')
W = torch.randn(B, u, v, w, dtype=torch.float64, device='cuda')
out = torch.zeros(B, w, dtype=torch.float64, device='cuda')

# 调用 CUDA kernel
fused_tp.fused_single_path_debug(x, y, W, out)

torch.cuda.synchronize()
start_time = time.perf_counter() * 1000
fused_tp.fused_single_path_debug(x, y, W, out)
torch.cuda.synchronize()
end_time = time.perf_counter() * 1000
execution_time_ms = (end_time - start_time)
print(f"========= einsum impl cost: {execution_time_ms:.3f} ms ========")
# 参考实现
ref = torch.einsum("bu,bv,buvw->bw", x, y, W)

max_error = (out - ref).abs().max().item()
print("Max Error:", max_error)
print("Reference norm:", ref.norm().item())
print("Output norm:", out.norm().item())

