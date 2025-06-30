import torch
import e3nn.o3 as o3
import time

# ==================== e3nn ===========================

gen = torch.Generator(device='cuda')

batch_size = 5888

X_ir, Y_ir, Z_ir = o3.Irreps("96x0e+96x1o+96x2e+96x3o"), o3.Irreps("10x0e"), o3.Irreps("96x0e+96x1o+96x2e+96x3o") 
X = torch.rand(batch_size, X_ir.dim, device='cuda', generator=gen)
Y = torch.rand(batch_size, Y_ir.dim, device='cuda', generator=gen)

instructions=[(0, 0, 0, "uvw", True), (1, 0, 1, "uvw", True), (2, 0, 2, "uvw", True), (3, 0, 3, "uvw", True)]

tp_e3nn = o3.TensorProduct(X_ir, Y_ir, Z_ir, instructions,
        shared_weights=False, internal_weights=False).to('cuda')
W = torch.rand(batch_size, tp_e3nn.weight_numel, device='cuda', generator=gen)

print("X, Y, W shape:", X.shape, Y.shape, W.shape)

retry = 1

torch.cuda.synchronize()
start_time = time.perf_counter() * 1000

for _ in range(0, retry):
    Z = tp_e3nn(X, Y, W)

torch.cuda.synchronize()
end_time = time.perf_counter() * 1000
execution_time_ms = (end_time - start_time) / retry
print(f"========= e3nn cost: {execution_time_ms:.3f} ms ========")

print(torch.norm(Z))


# ===================== cueq ==================

import cuequivariance as cue
import cuequivariance_torch as cuet

cu_X_ir, cu_Y_ir, cu_Z_ir = cue.Irreps("O3", "96x0e+96x1o+96x2e+96x3o"),  cue.Irreps("O3", "10x0e"),  cue.Irreps("O3", "96x0e+96x1o+96x2e+96x3o")

tp_cueq = cuet.FullyConnectedTensorProduct(cu_X_ir, cu_Y_ir, cu_Z_ir, layout=cue.ir_mul, internal_weights=False, device="cuda")

torch.cuda.synchronize()
start_time = time.perf_counter() * 1000
for _ in range(0, retry):
    Z = tp_cueq(X, Y, W) # Reuse X, Y, W from earlier
torch.cuda.synchronize()
end_time = time.perf_counter() * 1000
execution_time_ms = (end_time - start_time) / retry
print(f"========= cueq cost: {execution_time_ms:.3f} ms ========")

print(torch.norm(Z))

# ===================== oeq ===================

import openequivariance as oeq

problem = oeq.TPProblem(X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False)
tp_fast = oeq.TensorProduct(problem, torch_op=True)

torch.cuda.synchronize()
start_time = time.perf_counter() * 1000
for _ in range(0, retry):
    Z = tp_fast(X, Y, W) # Reuse X, Y, W from earlier
torch.cuda.synchronize()
end_time = time.perf_counter() * 1000
execution_time_ms = (end_time - start_time) / retry
print(f"========= oeq cost: {execution_time_ms:.3f} ms ========")
print("output Z shape:", Z.shape)
print(torch.norm(Z))
