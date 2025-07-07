import torch
import os, time
from torch.utils.cpp_extension import load


os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

fused_tp = load(
    name="fused_tensor_product",
    sources=["splitk_shm.cu"],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '-gencode=arch=compute_90,code=sm_90', '-DCUTLASS_ENABLE_WGMMA=1'],
)

B, U, V, W = 5888, 96, 10, 96
x = torch.randn(B, U, dtype=torch.float64, device='cuda')
y = torch.randn(B, V, dtype=torch.float64, device='cuda')
w = torch.randn(B, U, V, W, dtype=torch.float64, device='cuda')
out = torch.zeros(B, W, dtype=torch.float64, device='cuda')
w_flat = w.view(B, U * V, W).contiguous()
retry = 10

# 调用 CUDA 实现
torch.cuda.synchronize()
start_time = time.perf_counter() * 1000
for i in range(retry):
    fused_tp.fused_tensor_product(x.contiguous(), y.contiguous(), w.contiguous(), out)
    #fused_tp.einsum_cutlass(x.contiguous(), y.contiguous(), w_flat.contiguous())
torch.cuda.synchronize()
end_time = time.perf_counter() * 1000
execution_time_ms = (end_time - start_time) / retry
print(f"========= cuda impl cost: {execution_time_ms:.3f} ms ========")

# 参考验证
z = torch.einsum("bi,bj->bij", x, y)    # [B,1,96,10]
ref_1 = torch.einsum("bij,bijk->bk", z, w) # [B,1,96]


torch.cuda.synchronize()
start_time = time.perf_counter() * 1000
for _ in range(retry):
    ref_2 = torch.einsum("bi,bj,bijk->bk", x, y, w)
torch.cuda.synchronize()
end_time = time.perf_counter() * 1000
execution_time_ms = (end_time - start_time) / retry
print(f"========= einsum impl cost: {execution_time_ms:.3f} ms ========")

#print("ref_1 and out Max Error:", (ref_1 - out).abs().max().item())
print("ref_1 and ref_2 Max Error:", (ref_1 - ref_2).abs().max().item())
