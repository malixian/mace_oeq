import torch

# 输入张量
B = 5888
x = torch.randn(B, 1536, dtype=torch.float64, device="cuda")        # shape: (B, 1536)
y = torch.randn(B, 10, dtype=torch.float64, device="cuda")          # shape: (B, 10)
W = torch.randn(B, 368640, dtype=torch.float64, device="cuda")      # shape: (B, 368640)，等于 4 * (96*10*96)

# CG 系数（稀疏对角结构，用稠密张量模拟）
cg_coeffs = [
    torch.tensor([[[0.03227486]]],  dtype=torch.float64, device="cuda"),  # shape: (i, j, k)

    torch.diag(torch.tensor([0.03227486] * 3,  dtype=torch.float64, device="cuda")).reshape(3, 1, 3),  

    torch.diag(torch.tensor([0.03227486] * 5,  dtype=torch.float64, device="cuda")).reshape(5, 1, 5),  

    torch.diag(torch.tensor([0.03227486] * 7,  dtype=torch.float64, device="cuda")).reshape(7, 1, 7), 
]

paths = [(0, 0, 0, 0), (1, 1, 0, 1), (2, 2, 0, 2), (3, 3, 0, 3)]

path_segments = [
    (slice(0, 1), slice(0, 1), slice(0, 1)),     # i=1, j=1, k=1
    (slice(1, 4), slice(0, 1), slice(1, 4)), # i=3, j=1, k=3
    (slice(4, 9), slice(0, 1), slice(4, 9)),# i=5, j=1, k=5
    (slice(9, 16), slice(0, 1), slice(9, 16)) # i=7, j=1, k=7
]

# 初始化输出
u = 96
v = 10
w = 96
z = torch.zeros_like(x)  # shape: (B, 1536)

x = x.reshape(B, -1, u)
y = y.reshape(B, -1, v)
W = W.reshape(B, -1, u, v, w)
z = z.reshape(B, -1, w)

def tensor_product():
    for path_id, (i_slice, j_slice, k_slice) in enumerate(path_segments):
        x_seg = x[:, i_slice]          # shape: (B, i, u)
        y_seg = y[:, j_slice]          # shape: (B, j, v)
        cg = cg_coeffs[path_id]
    
        # Tensor Product: einsum over i,j with CG -> (B, u, v, k)
        z_uvk = torch.einsum("biu, bjv, ijk -> buvk", x_seg, y_seg, cg)
        print("baseline: z_uvk shape:", z_uvk.shape)
        W_seg = W[:, path_id]
        z[:, k_slice] = torch.einsum("buvk, buvw -> bkw", z_uvk, W_seg)
    return z

def tensor_product_opt():
    out = torch.zeros_like(x)
    print("init out size:", out.shape)

    for path_id, (i_slice, j_slice, k_slice) in enumerate(path_segments):
        cg = cg_coeffs[path_id]        # shape: (k, i, j)
        x_seg = x[:, i_slice]          # shape: (B, i, u)
        y_seg = y[:, j_slice]          # shape: (B, j, v)
        cg = cg_coeffs[path_id]
    
        i_len, j_len, k_len = cg.shape

        # 初始化 (B, u, v, k)
        z_uvk = torch.zeros(B, u, v, k_len, dtype=x.dtype, device=x.device)

        # 编译式路径索引 (i, j, k)
        nonzeros = (cg != 0).nonzero(as_tuple=False)  # (N, 3)
        for idx in range(nonzeros.shape[0]):
            i, j, k = nonzeros[idx]
            print("i,j,k:", i,j,k)
            coeff = cg[i, j, k].item()
            z_uvk[:, :, :, k] += coeff * x_seg[:, i].unsqueeze(2) * y_seg[:, j].unsqueeze(1)
        
        print("opt shape: z_uvk shape:", z_uvk.shape)
        W_seg = W[:, path_id]
        out[:, k_slice] = torch.einsum("buvk, buvw -> bkw", z_uvk, W_seg)

    return out


ref = tensor_product()
out = tensor_product_opt()
for path_id, (i_slice, j_slice, k_slice) in enumerate(path_segments):
    err = (out[:, k_slice] - ref[:, k_slice]).abs().max().item()
    print(f"最大误差 = {err:.2e}")
