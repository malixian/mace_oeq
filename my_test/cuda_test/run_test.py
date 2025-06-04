import torch
import time
import os
from torch.utils.cpp_extension import load

# 设置 CUDA 调试同步
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda")

# 编译并加载 CUDA 扩展
sparse_kernel = load(
    name="sparse_kernel",
    sources=["sparse_kernel.cu"],
    verbose=True,
    extra_cuda_cflags=[
        "-gencode=arch=compute_90,code=sm_90"
    ]
)

sparse_kernel_opt = load(
    name="sparse_kernel_opt",
    sources=["sparse_kernel_opt2.cu"],
    verbose=True,
    extra_cuda_cflags=[
        "-gencode=arch=compute_90,code=sm_90"
    ]
)

# 初始化数据
shape_A = (16, 16, 16, 23)
num_nonzeros = 353

indices = torch.stack([
    torch.randint(0, shape_A[0], (num_nonzeros,), device=device),
    torch.randint(0, shape_A[1], (num_nonzeros,), device=device),
    torch.randint(0, shape_A[2], (num_nonzeros,), device=device),
    torch.randint(0, shape_A[3], (num_nonzeros,), device=device),
])

print(indices.shape)

values = torch.randn(num_nonzeros, dtype=torch.float64, device=device)

B = torch.randn(5152, 96, 23, dtype=torch.float64, device=device)

out_sparse_cuda = torch.zeros(5152, 96, 16, 16, 16, dtype=torch.float64, device=device)

retry = 1

# 调用 CUDA kernel
torch.cuda.synchronize()
start_time = time.perf_counter() * 1000

for i in range(0, retry):
    sparse_kernel.sparse_scatter_mul_add(
        indices.int(), values, B, out_sparse_cuda
    )

torch.cuda.synchronize()
end_time = time.perf_counter() * 1000
execution_time_ms = (end_time - start_time) / retry
print(f"CUDA kernel 时间: {execution_time_ms:.3f} ms")


'''
torch.cuda.synchronize()
start_time = time.perf_counter() * 1000

for i in range(0, retry):
    sparse_kernel_opt.sparse_scatter_mul_add_optimized(
        indices.int(), values, B, out_sparse_cuda
    )

torch.cuda.synchronize()
end_time = time.perf_counter() * 1000
execution_time_ms = (end_time - start_time) / retry
print(f"Opt CUDA kernel 时间: {execution_time_ms:.3f} ms")
'''
# 验证正确性（使用原生 PyTorch 循环版本）
out_sparse_ref = torch.zeros_like(out_sparse_cuda)
for n in range(num_nonzeros):
    a_val = values[n]
    d = indices[3, n].item()
    out_sparse_ref[:, :, indices[0, n], indices[1, n], indices[2, n]] += a_val * B[:, :, d]

print("最大误差:", (out_sparse_ref - out_sparse_cuda).abs().max().item())
