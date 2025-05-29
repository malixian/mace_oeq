import torch

# 假设输入维度
i, j, m, c = 2, 3, 4, 5
k, a, b = 6, 7, 8

device = "cuda"
# 初始化张量 A(ijmc) 和 B(mkab)
A = torch.randn(i, j, m, c, dtype=torch.float64).to(device)  # 形状 (i, j, m, c)
B = torch.randn(m, k, a, b, dtype=torch.float64).to(device)  # 形状 (m, k, a, b)

# 步骤1：重组维度，将 c 和 k 作为批处理维度
# A 重组为 (c, i*j, m)
A_reshaped = A.permute(3, 0, 1, 2).reshape(c, i*j, m)  # 形状 (c, i*j, m)

# B 重组为 (k, m, a*b)
B_reshaped = B.permute(1, 0, 2, 3).reshape(k, m, a*b)  # 形状 (k, m, a*b)

# 步骤2：扩展维度以对齐批处理维度（利用广播）
# A_reshaped 扩展为 (c, 1, i*j, m)
A_expanded = A_reshaped.unsqueeze(1)  # 形状 (c, 1, i*j, m)

# B_reshaped 扩展为 (1, k, m, a*b)
B_expanded = B_reshaped.unsqueeze(0)  # 形状 (1, k, m, a*b)

# 步骤3：执行批处理矩阵乘法
C = torch.matmul(A_expanded, B_expanded)  # 输出形状 (c, k, i*j, a*b)

# 步骤4：调整形状和维度顺序得到最终结果
# 重塑为 (c, k, i, j, a, b)
C_reshaped = C.view(c, k, i, j, a, b)

# 转置维度到目标顺序 Cabcijk
C_final = C_reshaped.permute(0, 4, 5, 2, 3, 1)  # 形状 (c, a, b, i, j, k)

# 验证结果形状
print("C_final shape:", C_final.shape)  # 应输出 (5, 7, 8, 2, 3, 6)
