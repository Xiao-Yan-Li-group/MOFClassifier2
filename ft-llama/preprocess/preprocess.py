import os, json, random
from tqdm import tqdm

def get_label(all_data, cifid):
    return "True" if all_data[cifid]["has_error"] == 0 else "False"

dataset_label = "../dataset/label/manual_Tom.json"
dataset_des = "../dataset/descriptor"
dataset_path = "../dataset"
data_info_path = "./data/dataset_info.json"
datasets = [
    "cif_p1",
    "composition",
    "slices",
    "local_env",
    "atom_sequences_plusplus",
    "crystal_text_llm",
    "zmatrix",
    "robocry",
    "mof2text",
    "struc2str"
]


data_info = json.load(open(data_info_path))
all_data = json.load(open(dataset_label))

seed = 120
ratio = 0.8

skipped_total = {}

for i, des in enumerate(datasets):
    samples = []
    skipped = {"missing_file": 0, "missing_key": 0, "empty_text": 0, "exception": 0}
    data_info[des] = {}
    data_info[des]["file_name"] = f'mofclass2_{des}_train.json'
    data_info[des+"_test"] = {}
    data_info[des+"_test"]["file_name"] = f'mofclass2_{des}_test.json'
    for fp in tqdm(list(all_data.keys()), desc=f"processing {des}..."):
        try:
            if i < 8:
                path = os.path.join(dataset_des, f"{fp}.json")
                if not os.path.exists(path):
                    skipped["missing_file"] += 1
                    continue
                data_ = json.load(open(path))

                if des not in data_:
                    skipped["missing_key"] += 1
                    continue
                text = data_[des]

            else:
                path = os.path.join(dataset_path, des, f"{fp}.json")
                if not os.path.exists(path):
                    skipped["missing_file"] += 1
                    continue
                data_ = json.load(open(path))

                if "MOF2Text" not in data_:
                    skipped["missing_key"] += 1
                    continue
                text = data_["MOF2Text"]

            if text is None or str(text).strip() == "":
                skipped["empty_text"] += 1
                continue

            samples.append({
                "instruction": f"Is this metal-organic framework structure described by {des} reasonable?",
                "input": text,
                "output": get_label(all_data, fp),
            })

        except Exception:
            skipped["exception"] += 1
            continue

    rng = random.Random(seed + i)
    rng.shuffle(samples)

    n = len(samples)
    n_train = int(n * ratio)
    train_data = samples[:n_train]
    test_data = samples[n_train:]

    train_out = f"./data/mofclass2_{des}_train.json"
    test_out  = f"./data/mofclass2_{des}_test.json"

    with open(train_out, "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(test_out, "w") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"[{des}] kept: {n} | train: {len(train_data)} | test: {len(test_data)} | seed: {seed+i}")
    print(f"[{des}] skipped: {skipped}")

    skipped_total[des] = skipped

print("All done. Skipped summary:")
print(json.dumps(skipped_total, indent=2, ensure_ascii=False))

with open(data_info_path, "w") as f:
    json.dump(data_info, f, indent=2, ensure_ascii=False)