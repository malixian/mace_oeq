import torch
import cupy
import cupyx.scipy.sparse as cusparse
from torch.utils.dlpack import to_dlpack

# 1. 创建 PyTorch CUDA Tensor
dense_tensor = torch.tensor(
    [[1, 0, 0, 5],
     [8, 0, 0, 0],
     [0, 0, 3, 0],
     [0, 6, 0, 0]],
    dtype=torch.float32,
    device='cuda'
)

# 2. 转为 CuPy 数组（零拷贝）
dense_cupy = cupy.fromDlpack(to_dlpack(dense_tensor))

# 3. 转为 CuPy CSR 稀疏矩阵
csr_matrix = cusparse.csr_matrix(dense_cupy)

# 验证
print("PyTorch Tensor:")
print(dense_tensor.cpu().numpy())  # 转到 CPU 打印
print("\nCuPy CSR Matrix (as dense):")
print(csr_matrix.toarray())
