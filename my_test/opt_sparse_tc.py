import torch
import time

# 设置 CUDA
device = torch.device("cuda")

# 原始稠密张量 A 的维度
shape_A = (16, 16, 16, 23)
num_nonzeros = 353

# 随机生成稀疏张量 A（COO 格式）
indices = torch.stack([
    torch.randint(0, shape_A[0], (num_nonzeros,), device=device),
    torch.randint(0, shape_A[1], (num_nonzeros,), device=device),
    torch.randint(0, shape_A[2], (num_nonzeros,), device=device),
    torch.randint(0, shape_A[3], (num_nonzeros,), device=device),
])
values = torch.randn(num_nonzeros, device=device)

A_sparse = torch.sparse_coo_tensor(indices, values, size=shape_A, device=device)

# 稠密张量 B
B = torch.randn(5152, 96, 23, device=device)

# ==== baseline (未优化) ====
# 转为稠密再收缩（会造成大量浪费）
A_dense = A_sparse.to_dense()

retry = 10

torch.cuda.synchronize()
start = time.perf_counter() * 1000
for i in (0, retry):
    out_naive = torch.einsum('abcd,xyd->xyabc', A_dense, B)
torch.cuda.synchronize()
time_naive = (time.perf_counter() * 1000 - start) / retry

# ==== 优化版本 ====
# 只对非零元素参与收缩
# indices: [4, N], values: [N]
i0, i1, i2, i3 = indices

# 获取对应的 B 值：B[x, y, i3] => [5152, 96]
torch.cuda.synchronize()
start = time.perf_counter() * 1000

# 初始化结果张量
out_sparse = torch.zeros(5152, 96, 16, 16, 16, device=device)

for i in range(0, retry):
    for n in range(num_nonzeros):
        a_val = values[n]
        d = i3[n].item()
        b_slice = B[:, :, d]  # shape [5152, 96]
        # 累加到 out_sparse[:, :, i0[n], i1[n], i2[n]]
        out_sparse[:, :, i0[n], i1[n], i2[n]] += a_val * b_slice

torch.cuda.synchronize()
time_sparse = (time.perf_counter()*1000 - start) / retry

# ==== 比较 ====
#error = (out_sparse - out_naive).abs().max().item()

print(f"未优化时间: {time_naive:.4f} ms")
print(f"稀疏优化时间: {time_sparse:.4f} ms")
#print(f"最大误差: {error:.6e}")

