import os
import csv
import json
import pandas as pd

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_text(description_path, input_type):
    descriptor_all = load_json(description_path)
    descriptor = descriptor_all.get(input_type, None)
    if descriptor is None:
        return None
    return descriptor

def get_label(all_data, cifid):
    return 1 if all_data[cifid].get("has_error", 1) == 0 else 2

def main():
    input_types = [
        "cif_p1",
        "composition",
        "slices",
        "local_env",
        "atom_sequences_plusplus",
        "crystal_text_llm",
        "robocry",
    ]
    data_path = "../../dataset/results/manual_Tom.json"
    txt_dir = "../../dataset/MOFInputs/Txt"
    out_dir = "./dataset_csv"
    os.makedirs(out_dir, exist_ok=True)
    all_data = load_json(data_path)
    cifids = list(all_data.keys())
    for input_type in input_types:
        rows = [] 
        for cifid in cifids:
            desc_path = os.path.join(txt_dir, f"{cifid}.json")
            if not os.path.exists(desc_path):
                continue
            try:
                text = generate_text(desc_path, input_type)
            except Exception:
                continue
            if text is None:
                continue
            rows.append(
                {
                    "cifid": cifid,
                    "text": text,
                    "label": get_label(all_data, cifid),
                }
            )
        df = pd.DataFrame(rows, columns=["cifid", "text", "label"])
        out_path = os.path.join(out_dir, f"{input_type}.tsv")
        df.to_csv(
            out_path,
            sep="\t",
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
            lineterminator="\n",
        )
        print(f"[{input_type}] saved {len(df)} rows -> {out_path}")

if __name__ == "__main__":
    main()
