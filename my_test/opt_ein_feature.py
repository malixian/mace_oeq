import torch
import time

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N, C, H, W = 5888, 96, 16, 16  # A 的形状
print(f"Using device: {device}")

# 模拟输入数据（在对角线位置放置随机值，其他为零）
A = torch.zeros(N, C, H, W, device=device)
w_indices = torch.arange(W, device=device)
A[:, :, w_indices, w_indices] = torch.randn(N, C, W, device=device)

B = torch.randn(N, C, W, device=device)

######################################
# 方法1：原始实现（低效版本）
######################################

def original_method(A, B):
    # 扩展 B 的维度使其与 A 广播匹配
    B_expanded = B.unsqueeze(-2)  # [N, C, 1, W]
    result = torch.sum(A * B_expanded, dim=-1)  # [N, C, H]
    return result

######################################
# 方法2：优化实现（只处理对角线）
######################################

def optimized_method(A, B):
    A_diag = A.diagonal(offset=0, dim1=2, dim2=3)  # [N, C, W]
    result = A_diag * B  # [N, C, W]
    return result

######################################
# 性能测试函数
######################################

def benchmark(func, name, *args, warmup=3, iters=10):
    # GPU warm-up
    for _ in range(warmup):
        func(*args)

    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    for _ in range(iters):
        output = func(*args)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.perf_counter()
    
    avg_time_ms = (end - start) * 1000 / iters
    print(f"{name:<20} | Avg time: {avg_time_ms:.3f} ms")
    return output

######################################
# 执行对比
######################################

# 输出和性能比较
output_orig = benchmark(original_method, "Original method", A, B)
output_opt = benchmark(optimized_method, "Optimized method", A, B)

# 验证两者的结果是否一致
max_diff = (output_orig - output_opt).abs().max().item()
print(f"\nMax difference between outputs: {max_diff:.3e}")
assert max_diff < 1e-4, "Mismatch between original and optimized outputs!"

