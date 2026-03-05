import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tqdm import tqdm
from xtal2txt.core import TextRep
from pymatgen.core import Structure


dataset = json.load(open("../dataset/n_atoms/n_atoms_all.json"))
mattext_path = "../dataset/MOFText"
cif_path = "../dataset/cifs/"


requested_reps = [
        "cif_p1",
        # "cif_symmetrized",
        "composition",
        "slices",
        "local_env",
        "atom_sequences",
        "atom_sequences_plusplus",
        "crystal_text_llm",
        "zmatrix",
        # "robocrys_rep"
]

for cif in tqdm(dataset):
        save_dir = os.path.join(mattext_path, cif+".json")
        from_file = os.path.join(cif_path, cif+".cif")
        structure = Structure.from_file(from_file, "cif")
        text_rep = TextRep.from_input(structure)
        try:
                mof_text_ = {}
                for rep in requested_reps:
                        requested_text_reps = text_rep.get_requested_text_reps(rep)
                        mof_text_[rep] = requested_text_reps
        except:
                print(cif, "not in the database!, skip...")
                continue
        with open(save_dir, "w") as f:
                json.dump(mof_text_, f, indent=2)