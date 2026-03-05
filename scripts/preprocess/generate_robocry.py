import json

#dataset = json.load(open("../dataset/n_atoms/n_atoms_benchmark.json"))
dataset = json.load(open("../dataset/n_atoms/n_atoms_all.json"))

from pymatgen.core import Structure
from robocrys import StructureCondenser, StructureDescriber

from tqdm import tqdm
from pathlib import Path

import os

import warnings
warnings.filterwarnings("ignore")

cif_path = "../dataset/cifs/"
robocry_path = "../dataset/robocry/all"

i=0
for cif in tqdm(dataset):
  if i>8865:
    save_dir = os.path.join(robocry_path, cif+".txt")
    if os.path.isfile(save_dir):
      print(cif, "exit.")
    else:
      if int(dataset[cif])<500:
        try:
          structure = Structure.from_file(os.path.join(cif_path, cif+".cif"))
        except:
          print(cif, "not in the database!, skip...")
          continue

        try:
          condenser = StructureCondenser()
          describer = StructureDescriber()

          condensed_structure = condenser.condense_structure(structure)
          description = describer.describe(condensed_structure)

          Path(os.path.join(robocry_path, cif+".txt")).write_text(description, encoding="utf-8") 
        except:
          print(cif, "fail generated")
  i+=1
