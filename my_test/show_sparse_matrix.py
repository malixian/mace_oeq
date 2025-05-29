import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random

# 创建稀疏矩阵
sparse_matrix = random(100, 100, density=0.01, format='coo')

# 创建可视化
plt.figure(figsize=(10, 10))
plt.spy(sparse_matrix, markersize=5)
plt.title("Sparse Matrix Visualization")

# 保存图像
plt.savefig('sparse_matrix.png')  # 默认保存为PNG格式
plt.close()  # 关闭图形，释放内存
