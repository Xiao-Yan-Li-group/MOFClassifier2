import pandas as pd
import json
import os
from tqdm import tqdm

csv_path = "../dataset/label/UNIQUE_errors.csv"
df = pd.read_csv(csv_path)

cif_path = "../dataset/cifs/"
save_path = "../dataset/results/"

error_data = {}

for idx, row in tqdm(df.iterrows(), total=len(df)):
    filename = row['filename']
    has_error = row['has_error']
    
    cif_file = os.path.join(cif_path, filename + ".cif")
    found = False
    actual_filename = filename
    
    if os.path.exists(cif_file):
        found = True
    else:
        parts = filename.split('_')
        if len(parts) > 1:
            parts[-1] = parts[-1].replace('FSR', '')
            new_filename = '_'.join(parts)
            new_cif_file = os.path.join(cif_path, new_filename + ".cif")
            
            if os.path.exists(new_cif_file):
                found = True
                actual_filename = new_filename
    
    if not found:
        print(f"{filename} not found, skip...")
        continue
    
    error_info = {
        "has_error": int(has_error)
    }
    
    if has_error == 1:
        errors = []
        if row['error_h'] == 1:
            errors.append('h')
        if row['error_charge'] == 1:
            errors.append('charge')
        if row['error_disorder'] == 1:
            errors.append('disorder')
        if row['error_other'] == 1:
            errors.append('other')
        
        error_info['error_types'] = errors
    
    error_data[actual_filename] = error_info

output_file = os.path.join(save_path, "manual_Tom.json")
with open(output_file, "w") as f:
    json.dump(error_data, f, indent=2)
