import torch

batch = 5152
num_feature = 96
num_ell = 16
num_element = 10
num_param = 4

device = "cuda"
X = torch.randn(batch, num_feature, num_ell, dtype=torch.float64, device=device)
# TODO: 构造onehot 张量
random_y = torch.randn(batch, num_element, dtype=torch.float64, device=device)
y_idx = random_y.argmax(dim=1)
Y = torch.zeros(batch, num_element, device=device)
Y.scatter_(1, y_idx.unsqueeze(1), 1)

# TODO: 手动构造U 张量
U = torch.zeros(num_ell, num_ell, num_param, device=device)
nonzero_map = {
    0: [0],
    1: [1, 2, 3],
    2: [4, 5, 6, 7, 8],
    3: [9, 10, 11, 12, 13, 14,15]
}
nonzero_list = [1, 0.5774, 0.4472, 0.3780]
for k, ws in nonzero_map.items():
    for w in ws:
        U[w, w, k] = nonzero_list[k]

W = torch.randn(num_element, num_param, num_feature, dtype=torch.float64, device=device)

def manual_weight_compute(A_tensor, B_tensor):
    C_manual_full = torch.zeros(batch, num_feature, num_ell, num_ell, device=device)
    for k, ws in nonzero_map.items():
        for w in ws:
            C_manual_full[:, :, w, w] += A_tensor[w, w, k] * B_tensor[:, k, :]

    return C_manual_full


def opt_tc_cl2(weight, U_tensor, x, y):
    # contract weight
    w_selected = weight[y_idx]
    c_tensor = manual_weight_compute(U_tensor, w_selected)
    
    # tenosr add 
    '''
    idx = torch.arange(c_tensor.shape[-1], device=device)
    c_diag = c_tensor[:, :, idx, idx]  # [N, C, W]
    out_diag = out[:, :, idx, idx]
    diag_sum = c_diag + out_diag
    out[:, :, idx, idx] = diag_sum
    '''
    # contract features
    c_diag = c_tensor.diagonal(offset=0, dim1=2, dim2=3)  # [N, C, W]
    c_diag = c_diag.permute(0,2,1)
    x = x.permute(0,2,1)
    out = c_diag * x

if __name__ == "__main__":
    opt_tc_cl2(W, U, X, Y)
