from pyxtal import pyxtal

def structure_to_str(structure):
    px_structure = pyxtal()
    px_structure._from_pymatgen(structure)
    spg_symbol = px_structure.group.number
    
    a, b, c,alpha, beta, gamma = [float(i.replace(' ','')) for i in str(px_structure.lattice).split(',')[:-1]]
    lattice_str = f"|{a:.3f},{b:.3f},{c:.3f},{alpha:.2f},{beta:.2f},{gamma:.2f}|"
    wyckoff_strings = []
    for site in px_structure.atom_sites:
        symbol = site.specie
        letter = site.wp.letter
        multiplicity = site.wp.multiplicity
        position = site.position
        wyckoff_strings.append(f"({symbol}-{multiplicity}{letter}{position})")

    return f"{spg_symbol} {lattice_str} {'->'.join(wyckoff_strings)}"


import os, glob, json
from pathlib import Path
from tqdm import tqdm
from pymatgen.core import Structure

def get_label(all_data, cifid):
    return "True" if all_data[cifid]["has_error"] == 0 else "False"

input_path = "../../dataset/cifs"
jsons = glob.glob("../../dataset/descriptor/*json")
save_path = "./data_prompt"


for json_file in tqdm(jsons[:], desc="generating text from CIF..."):
    try:
        data = {}
        cif_id = Path(json_file).stem
        
        cif_path = os.path.join(input_path, cif_id+".cif")
        structure = Structure.from_file(cif_path)
        prompt = structure_to_str(structure)

        data["cif_id"] = cif_id
        data["struc2str"] = prompt
        with open(os.path.join(save_path, cif_id+".json"), "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(cif_id, e)

