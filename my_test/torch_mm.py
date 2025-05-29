import torch
import time

# 设置设备为 CUDA（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建随机输入张量

A = torch.randn(5152 * 96, 16, dtype=torch.float64, device=device)
B = torch.randn(16, 5152 * 96 * 16, dtype=torch.float64, device=device)

#A = torch.randn(5888 * 96, 23, dtype=torch.float64, device=device)
#B = torch.randn(23, 16 * 16 * 16, dtype=torch.float64, device=device)

# CUDA 预热（热身几次，避免首次调用的初始化开销影响计时）
for _ in range(5):
    _ = torch.mm(A, B)

# 确保所有 CUDA 操作完成再开始计时
torch.cuda.synchronize()
start_time = time.time()

# 执行 batch GEMM
C = torch.mm(A, B)

# 再次同步以确保操作完成后计时
torch.cuda.synchronize()
end_time = time.time()

# 打印输出形状和耗时
print(f"C shape: {C.shape}")
print(f"Time elapsed: {(end_time - start_time) * 1000:.3f} ms")

