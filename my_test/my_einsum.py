import torch
import time

device = "cuda"
# 定义输入张量（随机初始化）
B = torch.randn(10, 96, 16, 16, 16, dtype=torch.float64).to(device)
A = torch.randn(5152, 10, 96, 16, dtype=torch.float64).to(device)

# 步骤 1: 将张量重塑为适合 batch GEMM 的形式
# A: [5152, 10, 96, 16] -> [96, 5152, 10*16] (batch=96, m=5152, k=10*16)
A_reshaped = A.permute(2, 0, 1, 3).reshape(96, 5152, 10 * 16)

# B: [10, 96, 16, 16, 16] -> [96, 10*16, 16*16] (batch=96, k=10*16, n=16*16)
B_reshaped = B.permute(1, 0, 2, 3, 4).reshape(96, 10 * 16, 16 * 16)

# 步骤 2: 执行批量矩阵乘法 [96, 5152, 10*16] @ [96, 10*16, 256] -> [96, 5152, 256]
output = torch.bmm(A_reshaped, B_reshaped)

# 步骤 3: 调整输出形状 [96, 5152, 256] -> [5152, 96, 256]
output_bmm = output.permute(1, 0, 2)

# 验证形状
print(output.shape)  # 应输出 torch.Size([5152, 96, 256])

output_einsum_reshaped = torch.einsum('acbd,cbefd->abef', A, B).reshape(5152, 96, 256)

diff = torch.abs(output_bmm - output_einsum_reshaped)
print(f"最大绝对误差: {diff.max().item()}")  # 期望接近0
print(f"平均绝对误差: {diff.mean().item()}")  # 期望接近0


retry = 5
start = time.time()
for _ in range(retry):
    #torch.einsum('acbd,cbefd->abef', A, B).reshape(5152, 96, 256)
    A_reshaped = A.permute(2, 0, 1, 3).reshape(96, 5152, 10 * 16)
    B_reshaped = B.permute(1, 0, 2, 3, 4).reshape(96, 10 * 16, 16 * 16)
    torch.bmm(A_reshaped, B_reshaped)
torch.cuda.synchronize()
end = time.time()
avg_time_ms = (end - start) * 1000 / retry
print(f"Average time per run on bmm: {avg_time_ms:.3f} ms")
