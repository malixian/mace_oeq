import torch
from torch.utils.cpp_extension import load
import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.manual_seed(23)


fused_tp = load(
    name="fused_tensor_product",
    sources=["fused_tp_path.cu"],
    extra_cuda_cflags=["--use_fast_math"],
    extra_cflags=["-O3"],
    verbose=True
)


# 输入张量
B = 5888
i_len, j_len, k_len = 16, 1, 16
u, v, w = 96, 10, 96
path_num = 4

x = torch.randn(B, i_len, u, dtype=torch.float64, device="cuda")        # shape: (B, 1536)
y = torch.randn(B, j_len, v, dtype=torch.float64, device="cuda")          # shape: (B, 10)
W = torch.randn(B, path_num, u, v, w, dtype=torch.float64, device="cuda")      # shape: (B, 368640)，等于 4 * (96*10*96)

# CG 系数（稀疏对角结构，用稠密张量模拟）
cg_coeffs = [
    torch.tensor([[[0.03227486]]],  dtype=torch.float64, device="cuda"),  # shape: (i, j, k)

    torch.diag(torch.tensor([0.03227486] * 3,  dtype=torch.float64, device="cuda")).reshape(3, 3),

    torch.diag(torch.tensor([0.03227486] * 5,  dtype=torch.float64, device="cuda")).reshape(5, 5),

    torch.diag(torch.tensor([0.03227486] * 7,  dtype=torch.float64, device="cuda")).reshape(7, 7),
]

path_segments = [
    (slice(0, 1), slice(0, 1), slice(0, 1)),     # i=1, j=1, k=1
    (slice(1, 4), slice(0, 1), slice(1, 4)), # i=3, j=1, k=3
    (slice(4, 9), slice(0, 1), slice(4, 9)),# i=5, j=1, k=5
    (slice(9, 16), slice(0, 1), slice(9, 16)) # i=7, j=1, k=7
]


def fused_tensor_product(x, y, W, cg_coeffs, path_segments):
    path_segments = [
        (0, 0, 0),
        (1, 0, 1),
        (4, 0, 4),
        (9, 0, 9)
    ]

    # 构造融合后的非零索引表和权重列表
    meta_list = []
    coeff_list = []

    for path_id, (i_off, j_off, k_off) in enumerate(path_segments):
        cg = cg_coeffs[path_id]
        nonzero = (cg != 0).nonzero(as_tuple=False)
        print("nonzero:", nonzero)
        for i, j, k in nonzero:
            meta_list.append([i.item(), j.item(), k.item(), i_off, k_off, path_id])
            coeff_list.append(cg[i, j, k].item())

    meta = torch.tensor(meta_list, dtype=torch.int32, device="cuda").flatten()
    coeffs = torch.tensor(coeff_list, dtype=torch.float64, device="cuda")

    print(meta)
    
    x = x.contiguous()
    y = y.contiguous()
    W = W.contiguous()
    coeffs = coeffs.contiguous()
    meta = meta.contiguous()

    out = fused_tp.fused_tensor_product(
        x, y, W,
        coeffs, meta, B, u, v, w
    )
    return out

def tensor_product():
    z = torch.zeros(B, k_len, w, dtype=torch.float64, device=x.device)
    for path_id, (i_slice, j_slice, k_slice) in enumerate(path_segments):
        x_seg = x[:, i_slice]          # shape: (B, i, u)
        y_seg = y[:, j_slice]          # shape: (B, j, v)
        cg = cg_coeffs[path_id]
        print("x_seg, y_seg, cg, shape:", x_seg.shape, y_seg.shape, cg.shape)
        # Tensor Product: einsum over i,j with CG -> (B, u, v, k)
        z_uvk = torch.einsum("biu, bjv, ijk -> buvk", x_seg, y_seg, cg)
        W_seg = W[:, path_id]
        z[:, k_slice] = torch.einsum("buvk, buvw -> bkw", z_uvk, W_seg)
    return z.view(B, -1)

fused_out = fused_tensor_product(x, y, W, cg_coeffs, path_segments)
torch.cuda.synchronize()
#ref = tensor_product()
#print("ref.shape, ref[0]", ref.shape, ref[0])
print("fused_out.shape", fused_out.shape)
