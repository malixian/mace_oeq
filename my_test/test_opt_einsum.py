import opt_einsum_fx
import torch
import torch.fx
from torch.profiler import profile, record_function, ProfilerActivity

def einmatvecmul(utensors, weights, x, y):
    """Batched matrix-matrix-vector product using einsum"""
    return torch.einsum("wxik,ekc,bci,be->bcwx", utensors, weights, x, y)

def einfeatures(c, x):
    return torch.einsum("bcwi,bci->bcw")

# 将模型和输入数据移动到CUDA设备
device = torch.device('cuda')

graph_mod = torch.fx.symbolic_trace(einmatvecmul)
print("# Original code:\n", graph_mod.code)
graph_opt = opt_einsum_fx.optimize_einsums_full(
    model=graph_mod,
    example_inputs=(
        torch.randn(16, 16, 16, 23),
        torch.randn(10, 23, 96),
        torch.randn(5152, 96, 16),
        torch.randn(5152, 10)
    )
)
print("# Optimized code:\n", graph_opt.code)


uw = torch.randn(10, 96, 16, 16, 16, dtype=torch.float64).to(device)
xy = torch.randn(5152, 10, 96, 16, dtype=torch.float64).to(device)
def mace_tc(uw, xy):
    return torch.einsum("acbd,cbefd->abef", uw, xy)

mace_tc_mod = torch.fx.symbolic_trace(mace_tc)
mace_graph_opt = opt_einsum_fx.optimize_einsums_full(
    model=mace_tc_mod,
    example_inputs=(
        torch.randn(5152, 10, 96, 16),
        torch.randn(10, 96, 16, 16, 16)
    )
)
print("# MACE TC Optimized code:\n", mace_graph_opt.code)



from torch.utils.benchmark import Timer

#batch = 1000
# 创建CUDA张量

utensors = torch.randn(16, 16, 16, 23, dtype=torch.float64).to(device)
weights = torch.randn(10, 23, 96, dtype=torch.float64).to(device)
x = torch.randn(5152, 96, 16, dtype=torch.float64).to(device)
y = torch.randn(5152, 10, dtype=torch.float64).to(device)


activities = [ProfilerActivity.CUDA, ProfilerActivity.CPU]
sort_by_keyword = "self_cuda_time_total"

with profile(activities=activities, with_modules=True, with_stack=True) as prof:
    g = {"f": graph_mod, "utensors": utensors, "weights": weights, "x": x, "y": y}
    t_orig = Timer(
        "f(utensors, weights, x, y); torch.cuda.synchronize()",  # 添加同步以确保准确计时
        globals=g,
        label="Original CUDA",
        description="Original einsum on CUDA"
    )
    print(t_orig.timeit(1_0))
print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("opt-einsum-trace.json")

with profile(activities=activities, with_modules=True, with_stack=True) as prof:
    g["f"] = graph_opt
    t_opt = Timer(
        "f(utensors, weights, x, y); torch.cuda.synchronize()",  # 添加同步以确保准确计时
        globals=g,
        label="Optimized CUDA",
        description="Optimized einsum on CUDA"
    )
print(t_opt.timeit(1_0))
print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=10))

