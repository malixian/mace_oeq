import cupy as cp
from cupyx.scipy.sparse import random
from cupyx.scipy.sparse import csr_matrix

# 设置矩阵参数
n_rows = 1000  # 矩阵行数
n_cols = 1000  # 矩阵列数
density = 0.1  # 密度为10%（稀疏度为90%）

# 使用CuPy的随机稀疏矩阵生成器
sparse_matrix = random(n_rows, n_cols, density=density, format='csr', dtype=cp.float32)

# 转换为CSR矩阵（确保格式）
csr_sparse_matrix = csr_matrix(sparse_matrix)


# 打印矩阵信息
print(f"矩阵形状: {csr_sparse_matrix.shape}")
print(f"非零元素数量: {csr_sparse_matrix.nnz}")
print(f"实际稀疏度: {1 - csr_sparse_matrix.nnz / (n_rows * n_cols):.2%}")

# 如果需要将矩阵传输到GPU设备（默认已在GPU上）
# 可以这样访问数据:
# data = csr_sparse_matrix.data
# indices = csr_sparse_matrix.indices
# indptr = csr_sparse_matrix.indptr
