import torch
import time

#corelation=2时候的计算逻辑
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

B, C, W, K = 5152, 96, 16, 4  
X = W


# 构造对角矩阵张量 A: [W, X, K]
A = torch.zeros(W, X, K, dtype=torch.float64, device=device)
for k in range(K):
    A[:, :, k] = torch.diag(torch.randn(W, dtype=torch.float64, device=device))

# 构造稀疏对角结构的 A_diag: [K, W]
A_diag = torch.zeros(K, W, dtype=torch.float64, device=device)
A_diag[0, 0] = torch.randn(1, dtype=torch.float64, device=device)
A_diag[1, 1:4] = torch.randn(3, dtype=torch.float64, device=device)
A_diag[2, 4:9] = torch.randn(5, dtype=torch.float64, device=device)
A_diag[3, 9:16] = torch.randn(7, dtype=torch.float64, device=device)

# 构造输入张量 yw: [B, K, C]
B_tensor = torch.randn(B, K, C, dtype=torch.float64, device=device)

# 构造输入张量 x: [B, C, W]
X_tensor = torch.randn(B, C, W, dtype=torch.float64, device=device)

# ====================
# 步骤1: out = einsum(yw, u)
# ====================

# ---------------------
# 原始方法（einsum）
# ---------------------
torch.cuda.synchronize()
start = time.perf_counter() * 1000
C_einsum = torch.einsum("wxk,bkc->bcwx", A, B_tensor)
torch.cuda.synchronize()
end = time.perf_counter() * 1000
print(f"步骤1原始方法耗时（GPU）: {end - start:.6f} ms")

# ------------------------
# 方法1：手动稀疏迭代计算
# ------------------------
nonzero_map = {
    0: [0],
    1: [1, 2, 3],
    2: [4, 5, 6, 7, 8],
    3: [9, 10, 11, 12, 13, 14, 15]
}

torch.cuda.synchronize()
start_time = time.perf_counter() * 1000

C_manual = torch.zeros(B, C, W, X, dtype=torch.float64, device=device)
for k, ws in nonzero_map.items():
    for w in ws:
        C_manual[:, :, w, w] += A_diag[k, w] * B_tensor[:, k, :]

torch.cuda.synchronize()
end_time = time.perf_counter() * 1000
execution_time_ms = end_time - start_time
print(f"步骤1手动稀疏计算耗时: {execution_time_ms:.3f} ms")

# ====================
# 步骤2: out = einsum(c, x)
# ====================
torch.cuda.synchronize()
start = time.perf_counter() * 1000
print(C_manual.shape, X_tensor.shape)
out = torch.einsum("bcwi,bci->bcw", C_manual, X_tensor)
torch.cuda.synchronize()
end = time.perf_counter() * 1000
print(f"步骤2einsum方法耗时（GPU）: {end - start:.6f} ms")

torch.cuda.synchronize()
start = time.perf_counter() * 1000
C_reshape = C_manual.reshape(C_manual.shape[0], C_manual.shape[1]*C_manual.shape[2], -1)
X_permute = X_tensor.permute(0,2,1)
torch.bmm(C_reshape, X_permute)
torch.cuda.synchronize()
end = time.perf_counter() * 1000
print(f"步骤2bmm方法耗时（GPU）: {end - start:.6f} ms")
