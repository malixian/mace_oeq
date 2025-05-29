import torch
import time

# 设置设备为 GPU（若有）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置张量维度
W = X = 16
K = 4
B = 128
C = 96

# 构造对角矩阵张量 A: [W, X, K]
A = torch.zeros(W, X, K, device=device)
for k in range(K):
    A[:, :, k] = torch.diag(torch.randn(W, device=device))

# 构造输入张量 B: [B, K, C]
B_tensor = torch.randn(B, K, C, device=device)

# ---------------------
# 原始方法（einsum）
# ---------------------
torch.cuda.synchronize()
start = time.time()
C_einsum = torch.einsum("wxk,bkc->bcwx", A, B_tensor)
torch.cuda.synchronize()
end = time.time()
print(f"原始方法耗时（GPU）: {end - start:.6f} 秒")

# ---------------------
# 对角优化方法
# ---------------------
# 提取对角线元素: A_diag[k, w]
A_diag = torch.stack([torch.diagonal(A[:, :, k]) for k in range(K)], dim=0)  # [K, W]

torch.cuda.synchronize()
start = time.time()
C_diag = torch.einsum("kw,bkc->bcw", A_diag, B_tensor)  # [B, C, W]

# 构造全输出张量并写入对角
C_opt = torch.zeros((B, C, W, X), device=device)
w_idx = torch.arange(W, device=device)
C_opt[:, :, w_idx, w_idx] = C_diag
torch.cuda.synchronize()
end = time.time()
print(f"对角优化方法耗时（GPU）: {end - start:.6f} 秒")

# ---------------------
# 误差检查
# ---------------------
max_diff = (C_opt - C_einsum).abs().max().item()
print(f"最大误差: {max_diff:.6e}")

