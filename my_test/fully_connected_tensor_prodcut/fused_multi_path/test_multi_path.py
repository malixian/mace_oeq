import torch
import os, time
from torch.utils.cpp_extension import load

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 编译 CUDA kernel
fused_tp = load(
    name="fused_tensor_product",
    sources=["fused_multi_path_optv2.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--ptxas-options=-v"],
    verbose=True
)

# 参数
B = 5152
u, v, w = 96, 10, 96
i_dims = [1, 3, 5, 7]
k_dims = [1, 3, 5, 7]
#i_dims = [1]
#k_dims = [1]
val = 0.03227486121839514
num_paths = len(i_dims)

# 展开总维度
total_i = sum(i_dims)
total_k = sum(k_dims)

# 构造输入张量
X = torch.randn(B, total_i * u, dtype=torch.float64, device='cuda')  # [B, total_i * u]
Y = torch.randn(B, v, dtype=torch.float64, device='cuda')            # [B, v]
W = torch.randn(B, num_paths * u * v * w, dtype=torch.float64, device='cuda')  # [B, num_paths * u*v*w]
output = torch.zeros(B, total_k * w, dtype=torch.float64, device='cuda')      # [B, total_k * w]

# CUDA kernel 调用
fused_tp.fused_multi_path(X, Y, W, output, i_dims, k_dims, val)

# PyTorch 参考实现
torch_ref = torch.zeros_like(output)
i_offset = 0
k_offset = 0


torch.cuda.synchronize()
start_time = time.perf_counter() * 1000
for pid, (i, k) in enumerate(zip(i_dims, k_dims)):
    x_reshape = X[:, i_offset * u : (i_offset + i) * u].reshape(B, i, u)   # [B, i, u]
    y_reshape = Y[:, :v].reshape(B, 1, v)                                  # [B, 1, v]
    W_seg = W[:, pid * u * v * w : (pid + 1) * u * v * w].reshape(B, u, v, w)  # [B, u, v, w]
    
    for idx in range(i):
        x_i = x_reshape[:, idx, :]           # [B, u]
        y_j = y_reshape[:, 0, :]             # [B, v]
        z = torch.einsum("bu,bv,buvw->bw", x_i, y_j, W_seg)
        torch_ref[:, (k_offset + idx) * w : (k_offset + idx + 1) * w] += val * z

    i_offset += i
    k_offset += k

torch.cuda.synchronize()
end_time = time.perf_counter() * 1000
execution_time_ms = (end_time - start_time)
print(f"========= torch impl cost: {execution_time_ms:.3f} ms ========")


# 验证误差
max_error = (output - torch_ref).abs().max().item()
print(f"Max Error: {max_error:.6e}")
#assert max_error < 1e-6, "CUDA kernel result does not match PyTorch reference"

output = torch.zeros(B, total_k * w, dtype=torch.float64, device='cuda')
torch.cuda.synchronize()
start_time = time.perf_counter() * 1000
retry = 3
for _ in range(0, retry):
    fused_tp.fused_multi_path(X, Y, W, output, i_dims, k_dims, val)
torch.cuda.synchronize()
end_time = time.perf_counter() * 1000
execution_time_ms = (end_time - start_time) / retry
print(f"========= cuda impl cost: {execution_time_ms:.3f} ms ========")
