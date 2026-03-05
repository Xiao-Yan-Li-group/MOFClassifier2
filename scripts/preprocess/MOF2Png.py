import json
import os
from tqdm import tqdm
from pymatgen.core import Structure
from pymatviz.structure.figures import structure_2d

cif_path = "../../dataset/cifs/"
output_dir = "../../dataset/MOFInputs/Png/"

cif_files = json.load(open("../../dataset/n_atoms/n_atoms_all.json"))

width = 1800
height = 600
scale = 4

for cif in tqdm(cif_files):
    try:
        structure = Structure.from_file(os.path.join(cif_path, cif+".cif"))
        fig=structure_2d(structure,
                # rotation="0x,0y,0z",
                atomic_radii=None,
                #  elem_colors=None,
                scale=1,
                atom_size=5,
                show_cell_faces=False,
                show_sites=True,
                show_image_sites=True,
                standardize_struct=None,
                cell_boundary_tol=0,
                show_site_vectors=("force", "magmom"),
                vector_kwargs={"force":1, "magmom":1},
                #  hover_text="SiteCoords.cartesian_fractional",
                hover_float_fmt=".4",
                show_bonds=True,
                show_cell=True,
                n_cols=1,
                bond_kwargs={"width": 1},
                subplot_title=False
                )
        
        fig.write_image(os.path.join(output_dir, cif+".png"), width=width, height=height, scale=scale)
    except:
        print(cif, "fail!")