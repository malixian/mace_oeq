import torch
import cupy as cp
from cupyx.scipy import sparse
from cupyx.cusparse import coo2csr, spgemm

# 示例：创建一个稠密 PyTorch 张量
dense_tensor = torch.tensor([[0, 0, 3], [4, 0, 0], [0, 5, 0]], device="cuda")

# 转换为 PyTorch 稀疏 COO 张量
sparse_tensor = dense_tensor.to_sparse()

# 转换为 CuPy 稀疏 COO 矩阵
indices_cp = cp.asarray(sparse_tensor.indices())
values_cp = cp.asarray(sparse_tensor.values())
rows, cols = indices_cp[0], indices_cp[1]

coo_matrix = sparse.coo_matrix((values_cp, (rows, cols)), shape=sparse_tensor.shape, dtype="float64")
csr_matrix = coo2csr(coo_matrix)

a, b = csr_matrix, csr_matrix
print("matrix shape:", a.shape)
out = spgemm(a, b)

dense_array = out.toarray()
torch_tensor = torch.as_tensor(dense_array, device='cuda')

print(torch_tensor)
