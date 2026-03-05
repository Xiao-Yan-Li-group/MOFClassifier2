from ase.io import read
import pandas as pd
import json
import os
from tqdm import tqdm
import numpy as np

dataset = pd.read_csv("../dataset/label/UNIQUE_errors.csv")["filename"]
import warnings
warnings.filterwarnings("ignore")

cif_path = "/hpctmp/guobinzhao/Project/NCR-LLM/dataset/cifs/"
save_path = "/hpctmp/guobinzhao/Project/NCR-LLM/dataset/n_atoms/"

atoms_data = {}
for cif in tqdm(dataset[:]):
  try:
    atoms = read(os.path.join(cif_path, cif+".cif"))
    atoms_data[cif] = int(len(atoms))
  except:
    if "FSR" in cif.split("_")[-1]:
    try:
      atoms = read(os.path.join(cif_path, cif.replace("FSR", "")+".cif"))
      atoms_data[cif.replace("FSR", "")] = int(len(atoms))
    except:
      print(cif, "can not find in dataset, skip")
      continue

rank_len_atoms = dict(sorted(atoms_data.items(), key=lambda x: x[1]))

with open(os.path.join(save_path, "n_atoms_all.json"), "w") as f:
  json.dump(rank_len_atoms, f, indent=2)

benchmark_len_atoms = {k: v for k, v in rank_len_atoms.items() if v <= 500}

keys = list(benchmark_len_atoms.keys())
values = list(benchmark_len_atoms.values())
    
idx_ = np.linspace(0, len(keys)-1, 100, dtype=int)
benchmark_data = {keys[i]: values[i] for i in idx_}
    
with open(os.path.join(save_path, "n_atoms_benchmark.json"), "w") as f:
  json.dump(benchmark_data, f, indent=2)
