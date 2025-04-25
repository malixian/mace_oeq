import torch


from ase import units
import ase.io
import numpy as np
import time, os, math
from tqdm import tqdm

from mace.calculators import MACECalculator

import multiprocessing as mp

total_file_num = 2099
world_size = 2

def mace_infer(rank_id):

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.xpu.is_available():
        device = 'xpu'
    else:
        print('Neither CUDA nor XPU devices are available to demonstrate profiling on acceleration devices')
        import sys
        sys.exit(0)


    calculator = MACECalculator(model_paths='../../mace-bench/models/MACE-OFF23_small.model', device=device, compile_mode=None, use_batch_size=32)
    models = calculator.models
    atoms_list = []
    data_path = "../../mace-bench/data/data_raw"
    # args.configs now should be a directory.
    chunk_size = math.ceil(total_file_num / world_size)
    start_idx = chunk_size * rank_id
    end_idx = (start_idx + chunk_size) if (start_idx + chunk_size) < total_file_num else total_file_num
    
    start_time = time.perf_counter()
    print(rank_id, start_idx, end_idx) 
    for file in tqdm(os.listdir(data_path)[start_idx:end_idx], desc="Reading files", unit="file"):
        if file.endswith(".cif"):
            atoms_list.append(ase.io.read(os.path.join(data_path, file), index=0)) 
    end_time = time.perf_counter()
    print("Load data cost %.2f" % (end_time - start_time))
    
    start_time = time.perf_counter()
    print("timestamp start calculate:", start_time)
    calculator.batch_calculate(atoms_list=atoms_list)
    end_time = time.perf_counter()
    print("Compute cost %.2f" % (end_time - start_time))
    print("timestamp end calculate:", start_time)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    processes = []
    
    start_time = time.perf_counter()
    
    for rank in range(world_size):
        p = mp.Process(target=mace_infer, args=(rank,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    end_time = time.perf_counter()
    during_time = end_time - start_time
    print("during time:", during_time)
