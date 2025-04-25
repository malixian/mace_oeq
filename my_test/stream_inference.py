import torch


from ase import units
import ase.io
import numpy as np
import time, os, math
from tqdm import tqdm

from mace.calculators import MACECalculator

from multiprocessing import Pool


if torch.cuda.is_available():
    device = 'cuda'
elif torch.xpu.is_available():
    device = 'xpu'
else:
    print('Neither CUDA nor XPU devices are available to demonstrate profiling on acceleration devices')
    import sys
    sys.exit(0)


calculator = MACECalculator(model_paths='../../mace-bench/models/mace-large-density-agnesi-stress.model', device=device, compile_mode="max-autotune", use_batch_size=1)
models = calculator.models
atoms_list = []
data_path = "../../mace-bench/data/data_raw"
# args.configs now should be a directory. 
for file in tqdm(os.listdir(data_path), desc="Reading files", unit="file"):
    if file.endswith(".cif"):
        atoms_list.append(ase.io.read(os.path.join(data_path, file), index=0)) 

# warm up
warm_up = 5
for i in range(0, warm_up):
    calculator.calculate(atoms=atoms_list[0])

num_processes = 4

start_time = time.perf_counter()
#atoms = ase.io.read("./dataset/BOTNet-datasets/dataset_3BPA/test_300K.xyz")

chunk_size = math.ceil(len(atoms_list) / num_processes) 
atmos_chunk = []
start_index = 0

for i in range(num_processes):
    end_index = start_index + chunk_size
    data_chunks.append(atoms_list[start_index:end_index])
    start_index = end_index

with Pool(num_processes) as pool:
    pool.map(calculator.batch_calculate, atmos_chunk)

end_time = time.perf_counter()
during_time = end_time - start_time
print("during time:", during_time)
