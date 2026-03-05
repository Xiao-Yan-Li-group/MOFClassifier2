import pandas as pd
import json
from glob import glob
from pathlib import Path
from tqdm import tqdm

jsons = glob("../../dataset/MOFInputs/Txt/*")
mosaec_data = pd.read_csv("../../dataset/oxidation/MOSAEC_SI_CoRE_2019.csv")
data_all = json.load(open("../../dataset/results/manual_Tom.json"))


i = 0
data_mo = {}
for json_file in tqdm(jsons):
    cif_id = Path(json_file).stem
    with open(json_file, "r") as f:
        data = json.load(f)

    if data_all[cif_id]["has_error"] == 0:
        data_mof = mosaec_data[mosaec_data["CIF"] == cif_id]
        if len(data_mof) > 0:
            mo_all = list(data_mof["ON_network+Outer_Sphere"])
            mo_check = list(set(mo_all))
            if len(mo_check) == 1:
                try:
                    data_mo[cif_id] = int(mo_check[0])
                    i+=1
                except:
                    pass
print(i)            
with open("../../dataset/oxidation/metal_oxidation.json", "w") as f:
    json.dump(data_mo, f, indent=2)