import torch
import triton
import triton.language as tl
import time

# 缩小张量尺寸，适合本环境执行
B, C, W, K = 5888, 96, 16, 4
X = W
device = torch.device("cuda")

# 构造稀疏对角矩阵 A_full: [W, W, K]
A_full = torch.zeros(W, W, K, device=device)
A_full[0, 0, 0] = torch.randn(1, device=device)
A_full[1, 1, 1] = torch.randn(1, device=device)
A_full[2, 2, 1] = torch.randn(1, device=device)
A_full[3, 3, 1] = torch.randn(1, device=device)
A_full[4, 4, 2] = torch.randn(1, device=device)
A_full[5, 5, 2] = torch.randn(1, device=device)
A_full[6, 6, 2] = torch.randn(1, device=device)
A_full[7, 7, 2] = torch.randn(1, device=device)
A_full[8, 8, 2] = torch.randn(1, device=device)
A_full[9, 9, 3] = torch.randn(1, device=device)
A_full[10, 10, 3] = torch.randn(1, device=device)
A_full[11, 11, 3] = torch.randn(1, device=device)
A_full[12, 12, 3] = torch.randn(1, device=device)
A_full[13, 13, 3] = torch.randn(1, device=device)
A_full[14, 14, 3] = torch.randn(1, device=device)

# 提取对角元素: A_diag[k, w]
A_diag = torch.stack([torch.diagonal(A_full[:, :, k]) for k in range(K)], dim=0)  # [K, W]

# 输入张量 B: [B, K, C]
B_tensor = torch.randn(B, K, C, device=device)

# 稀疏非零位置映射
nonzero_map = {
    0: [0],
    1: [1, 2, 3],
    2: [4, 5, 6, 7, 8],
    3: [9, 10, 11, 12, 13, 14]
}

# Triton 稀疏 kernel
@triton.jit
def triton_sparse_diag_kernel(B_ptr, A_ptr, C_ptr,
                               B_stride_bk, B_stride_kc,
                               A_stride_kw, C_stride_bcw,
                               B, C, W,
                               nonzero_ws_ptr, nonzero_ws_len,
                               k_idx: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C
    if b >= B or c >= C:
        return

    b_val = tl.load(B_ptr + b * B_stride_bk + k_idx * B_stride_kc + c)
    for i in range(nonzero_ws_len):
        w = tl.load(nonzero_ws_ptr + i)
        a_val = tl.load(A_ptr + k_idx * A_stride_kw + w)
        acc = a_val * b_val
        tl.store(C_ptr + b * C_stride_bcw + c * W + w, acc)

# 准备 Triton 输入
C_triton = torch.zeros(B, C, W, device=device)
nonzero_ws_all = {
    k: torch.tensor(w_list, dtype=torch.int32, device=device)
    for k, w_list in nonzero_map.items()
}

# Triton 执行
grid = lambda META: (B * C,)
torch.cuda.synchronize()
start = time.time()
for k in range(K):
    nonzero_ws = nonzero_ws_all[k]
    triton_sparse_diag_kernel[grid](
        B_tensor, A_diag, C_triton,
        B_tensor.stride(0), B_tensor.stride(1),
        A_diag.stride(1), C_triton.stride(1),
        B, C, W,
        nonzero_ws, len(nonzero_ws),
        k_idx=k
    )
torch.cuda.synchronize()
triton_time = time.time() - start

print("triton_time:", triton_time)

# 转为 full 矩阵形式 [B, C, W, W]
C_triton_full = torch.zeros(B, C, W, W, device=device)
idx = torch.arange(W, device=device)
C_triton_full[:, :, idx, idx] = C_triton

# 优化 einsum 方法
torch.cuda.synchronize()
start = time.time()
C_diag_ein = torch.einsum("kw,bkc->bcw", A_diag, B_tensor)
C_ein_full = torch.zeros(B, C, W, W, device=device)
C_ein_full[:, :, idx, idx] = C_diag_ein
torch.cuda.synchronize()
einsum_time = time.time() - start
print("einsum_time:", einsum_time)

# 对比结果
max_diff = (C_triton_full - C_ein_full).abs().max().item()
print("max_diff:", max_diff)


