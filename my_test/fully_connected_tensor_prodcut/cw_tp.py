import torch
import e3nn.o3 as o3

# ==================== e3nn ===========================

gen = torch.Generator(device='cuda')

batch_size = 107560

X_ir, Y_ir, Z_ir = o3.Irreps("96x0e"), o3.Irreps("1x0e+1x1o+1x2e+1x3o"), o3.Irreps("96x0e+96x1o+96x2e+96x3o") 
X = torch.rand(batch_size, X_ir.dim, device='cuda', generator=gen)
Y = torch.rand(batch_size, Y_ir.dim, device='cuda', generator=gen)

instructions=[(0, 0, 0, "uvu", True), (0, 1, 1, "uvu", True), (0, 2, 2, "uvu", True), (0, 3, 3, "uvu", True)]

tp_e3nn = o3.TensorProduct(X_ir, Y_ir, Z_ir, instructions,
        shared_weights=False, internal_weights=False).to('cuda')
W = torch.rand(batch_size, tp_e3nn.weight_numel, device='cuda', generator=gen)

Z = tp_e3nn(X, Y, W)
print(torch.norm(Z))


# ===================== cueq ===================

import openequivariance as oeq

problem = oeq.TPProblem(X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False)
tp_fast = oeq.TensorProduct(problem, torch_op=True)

Z = tp_fast(X, Y, W) # Reuse X, Y, W from earlier
print(torch.norm(Z))
