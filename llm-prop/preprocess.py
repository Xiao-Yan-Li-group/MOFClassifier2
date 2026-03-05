import json, glob
from pathlib import Path
import pandas as pd


path = "/Users/sxm13/Desktop/project/NCR-LLM/mofclassifier2"
cif_ids = glob.glob(path + "/dataset/descriptor/*json")
dataset_label = path + "/dataset/results/manual_Tom.json"

def get_label(all_data, cifid):
    return 1 if all_data[cifid]["has_error"] == 0 else 0

def stratified_split(df, label_col="CLscore", train=0.6, val=0.2, test=0.2, seed=42):
    assert abs(train + val + test - 1.0) < 1e-9, "Splits must sum to 1.0"

    parts = []
    for _, g in df.groupby(label_col):
        g = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(g)
        n_train = int(round(n * train))
        n_val   = int(round(n * val))
        n_test  = n - n_train - n_val

        train_g = g.iloc[:n_train]
        val_g   = g.iloc[n_train:n_train + n_val]
        test_g  = g.iloc[n_train + n_val:]

        parts.append((train_g, val_g, test_g))

    train_df = pd.concat([p[0] for p in parts], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df   = pd.concat([p[1] for p in parts], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df  = pd.concat([p[2] for p in parts], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df, test_df

with open(dataset_label, "r") as f:
    all_data = json.load(f)

rows = []
missing = 0

for cif_id_data in cif_ids:
    with open(cif_id_data, "r") as f:
        des = json.load(f)

    cif_id = Path(cif_id_data).stem

    if cif_id not in all_data:
        missing += 1
        continue
    
    description = (des.get("robocry", "") or "").strip()
    if description == "":
        continue

    rows.append([cif_id, des.get("robocry", ""), get_label(all_data, cif_id)])

df = pd.DataFrame(rows, columns=["cif_id", "description", "CLscore"])

train_df, val_df, test_df = stratified_split(df, label_col="CLscore", train=0.6, val=0.2, test=0.2, seed=42)

train_df.to_csv("./data/mofclassifier2/train_robocry.csv", index=False)
val_df.to_csv("./data/mofclassifier2/val_robocry.csv", index=False)
test_df.to_csv("./data/mofclassifier2/test_robocry.csv", index=False)

print("train/val/test:", len(train_df), len(val_df), len(test_df))