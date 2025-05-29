import torch

# 假设这两个张量已初始化：
# sparse_tensor: torch.sparse_coo_tensor，shape [a, b, e, f, d]
# dense_tensor: torch.Tensor，shape [a, b, d]

# 示例初始化（你应替换为你已有的实际张量，并确保它们已在 CUDA 上）
# 以下只是举例：
a, b, e, f, d = 2, 3, 4, 5, 6
nnz = 100

# 稀疏张量初始化（示例）
indices = torch.stack([
    torch.randint(0, a, (nnz,)),
    torch.randint(0, b, (nnz,)),
    torch.randint(0, e, (nnz,)),
    torch.randint(0, f, (nnz,)),
    torch.randint(0, d, (nnz,))
], dim=0)

values = torch.randn(nnz)

sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(a, b, e, f, d)).coalesce()

# 稠密张量初始化（示例）
dense_tensor = torch.randn(a, b, d)

# ------------------------------
# 将所有张量移动到 GPU（cuda:0）
device = torch.device("cuda:0")

sparse_tensor = sparse_tensor.to(device)
dense_tensor = dense_tensor.to(device)
# ------------------------------

# 提取 indices/values
indices = sparse_tensor._indices()
values = sparse_tensor._values()

# 生成新 indices：用于合并 batch 维并展平 (e, f) → e*f
batch_indices = indices[0] * b + indices[1]        # [nnz]
row_indices = indices[2] * f + indices[3]          # [nnz]
col_indices = indices[4]                           # [nnz]

new_indices = torch.stack([batch_indices, row_indices, col_indices], dim=0)  # [3, nnz]

# 构造新的稀疏张量 [a*b, e*f, d]
sparse_reshaped = torch.sparse_coo_tensor(
    new_indices, values, size=(a * b, e * f, d), device=device
).coalesce()

# reshape 稠密张量为 [a*b, d, 1]
dense_reshaped = dense_tensor.reshape(a * b, d, 1).to(device)

# 执行 batched 稀疏乘稠密
output = torch.sparse.bmm(sparse_reshaped, dense_reshaped)  # shape: [a*b, e*f, 1]

# reshape 回 [a, b, e, f]
output = output.reshape(a, b, e, f)

print("输出 shape:", output.shape)
print("是否在 CUDA 上:", output.is_cuda)

