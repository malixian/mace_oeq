import torch


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


calculator = MACECalculator(model_paths='../../mace_bench/models/MACE-OFF23_small.model', device=device, compile_mode=None, enable_cueq=True, use_batch_size=1)
data_path = "../../mace_bench/test_data"
atoms_list = []
start_time = time.perf_counter()
for file in tqdm(os.listdir(data_path), desc="Reading files", unit="file"):
    if file.endswith(".cif"):
        atoms_list.append(ase.io.read(os.path.join(data_path, file), index=0)) 
end_time = time.perf_counter()
print("Load data cost %.2f" % (end_time - start_time))

""" # warm up
warm_up = 5
for i in range(0, warm_up):
    calculator.calculate(atoms=atoms_list[0]) """

for atoms in tqdm(atoms_list, desc="Inference"):
    calculator.calculate(atoms=atoms)
