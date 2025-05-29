import torch

# 假设输入张量
A = torch.zeros(4, 16)
B = torch.randn(5888, 4, 96)

# 设置 A 的非零对角元素（只保留非零对角，压缩为 [4, 16]）
A[0, 0] = 1.0
A[1, 1:4] = torch.randn(3)
A[2, 4:9] = torch.randn(5)
A[3, 9:15] = torch.randn(6)

# 对应非零索引
nonzero_blocks = [
    [0],
    [1, 2, 3],
    [4, 5, 6, 7, 8],
    [9, 10, 11, 12, 13, 14]
]

# 输出： [5888, 4, 16, 6] 假设每组输出是 6 维
out = torch.zeros(5888, 4, 16, 6, device=B.device, dtype=B.dtype)

for i in range(4):
    indices = torch.tensor(nonzero_blocks[i], device=B.device)
    d = len(indices)
    
    # A_diag: [d]，B_slice: [5888, d, 6]
    A_diag = A[i, indices]  # [d]
    
    # reshape B：把 [5888, 4, 96] → [5888, 4, 16, 6]
    B_reshaped = B.view(5888, 4, 16, 6)
    B_block = B_reshaped[:, i, indices, :]  # [5888, d, 6]
    
    # 执行按元素乘法（扩展 A_diag）
    result = A_diag[None, :, None] * B_block  # [5888, d, 6]

    # scatter 到输出：在 dim=2 上 scatter
    out[:, i].scatter_(dim=1, index=indices[None, :, None].expand(5888, d, 6), src=result)

