import torch
import time

def diagonal_add_to_full(A: torch.Tensor, B: torch.Tensor):
    """
    对张量 A 和 B 的主对角线位置执行加法，其它位置为 0，返回完整形状的张量。
    """
    assert A.shape == B.shape
    N, C, W, _ = A.shape
    device = A.device
    idx = torch.arange(W, device=device)

    A_diag = A[:, :, idx, idx]  # [N, C, W]
    B_diag = B[:, :, idx, idx]
    diag_sum = A_diag + B_diag

    result = torch.zeros_like(A)
    result[:, :, idx, idx] = diag_sum
    return result


def generate_diagonal_tensor(shape, device):
    """
    生成一个只在对角线有值的稀疏张量，其他位置为 0。
    """
    N, C, W, _ = shape
    idx = torch.arange(W, device=device)
    diag_vals = torch.randn(N, C, W, device=device)
    result = torch.zeros(shape, device=device)
    result[:, :, idx, idx] = diag_vals
    return result


def benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    shape = (5152, 96, 16, 16)

    # 构造稀疏对角张量 A 和 B
    A = generate_diagonal_tensor(shape, device)
    B = generate_diagonal_tensor(shape, device)

    # --- 方法 1：常规加法 ---
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    C1 = A + B  # 常规加法（包含无用运算）
    end.record()
    torch.cuda.synchronize()
    time_full = start.elapsed_time(end)  # 毫秒

    # --- 方法 2：对角优化加法 ---
    start.record()
    C2 = diagonal_add_to_full(A, B)
    end.record()
    torch.cuda.synchronize()
    time_diag = start.elapsed_time(end)  # 毫秒

    # --- 验证正确性 ---
    is_equal = torch.allclose(C1, C2)

    print(f"常规加法时间:     {time_full:.3f} ms")
    print(f"对角优化加法时间: {time_diag:.3f} ms")
    print(f"结果是否一致:     {is_equal}")

if __name__ == "__main__":
    benchmark()

