import torch
from torch.profiler import profile, record_function, ProfilerActivity

from ase import units
import ase.io
import numpy as np
import time, os
from tqdm import tqdm

from mace.calculators import MACECalculator


if torch.cuda.is_available():
    device = 'cuda'
elif torch.xpu.is_available():
    device = 'xpu'
else:
    print('Neither CUDA nor XPU devices are available to demonstrate profiling on acceleration devices')
    import sys
    sys.exit(0)


calculator = MACECalculator(model_paths='../../mace_bench/models/MACE-OFF23_small.model', device=device, compile_mode=None, enable_cueq=False, enable_oeq=True, batch_size=32)
atoms_list = []
data_path = "../../mace_bench/test_data"
# args.configs now should be a directory. 
for file in tqdm(os.listdir(data_path), desc="Reading files", unit="file"):
    if file.endswith(".cif"):
        atoms_list.append(ase.io.read(os.path.join(data_path, file), index=0)) 


'''
activities = [ProfilerActivity.CUDA, ProfilerActivity.CPU]
sort_by_keyword = "self_" + device + "_time_total"
#warmup
for step in range(0, 5):
    calculator.batch_calculate(atoms_list=atoms_list)

with profile(activities=activities, with_modules=True, with_stack=True) as prof:
    calculator.batch_calculate(atoms_list=atoms_list)
print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=10))


prof.export_chrome_trace("e3nn-oeq-trace.json")
'''


'''
with torch.profiler.profile(
        activities = [ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=3, repeat=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/mace'),
        #record_shapes=True,
        #profile_memory=True,
        with_modules=True,
) as prof:
    for step in range(0, 10):
        prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
        calculator.batch_calculate(atoms_list=atoms_list)
'''

for i in range(0, 5):
    print("<<<<<<<<<<<<<<<<<<<<<<<<<< warmup %d >>>>>>>>>>>>>>>>>>>>>>" % i)
    calculator.batch_calculate(atoms_list=atoms_list)
