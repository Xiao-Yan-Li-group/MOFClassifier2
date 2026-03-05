import os, json, random
from tqdm import tqdm
import pandas as pd


# def get_mo(data_csv, cifid):
#     data = pd.read_csv(data_csv, low_memory=False)["base_name"] == cifid.split("_")[0]
#     print(cifid.split("_")[0])
#     if len(data) == 1:
#         print(data)
#         print(data["oxidationstate"].values[0])
#         return data["oxidationstate"].values[0]

# data_csv = "../dataset/oxidation/smit_mo.csv"
# dataset_des = "../dataset/descriptor"
# dataset_path = "../dataset"
# data_info_path = "./data/dataset_info.json"
# datasets = [
#     "cif_p1",
#     "composition",
#     "slices",
#     "local_env",
#     "atom_sequences_plusplus",
#     "crystal_text_llm",
#     "zmatrix",
#     "robocry",
#     "mof2text",
#     "struc2str"
# ]

dataset_label = "../dataset/label/manual_Tom.json"
all_data = json.load(open(dataset_label))

data_true = []
# print("good")
for key in all_data:
    if all_data[key]["has_error"] == 0:
        if len(key.split("_")[0]) == 6:
            data_true.append(key.split("_")[0])
            print(key.split("_")[0])

data_false = []
# print("bad")
for key in all_data:
    if all_data[key]["has_error"] == 1:
        if len(key.split("_")[0]) == 6:
            data_false.append(key.split("_")[0])
            print(key.split("_")[0])

# data_info = json.load(open(data_info_path))

# seed = 120
# ratio = 0.8

# skipped_total = {}

# for i, des in enumerate(datasets[:1]):
#     samples = []
#     skipped = {"missing_file": 0, "missing_key": 0, "empty_text": 0, "exception": 0}
#     data_info[des+"_mo"] = {}
#     data_info[des+"_mo"]["file_name"] = f'metal_oxidation_{des}_train.json'
#     data_info[des+"_mo"+"_test"] = {}
#     data_info[des+"_mo"+"_test"]["file_name"] = f'metal_oxidation_{des}_test.json'
#     for fp in tqdm(data_true[:300], desc=f"processing {des}..."):
#         try: 
#             if i < 8:
#                 path = os.path.join(dataset_des, f"{fp}.json")
#                 if not os.path.exists(path):
#                     skipped["missing_file"] += 1
#                     continue
#                 data_ = json.load(open(path))

#                 if des not in data_:
#                     skipped["missing_key"] += 1
#                     continue
#                 text = data_[des]

#             else:
#                 path = os.path.join(dataset_path, des, f"{fp}.json")
#                 if not os.path.exists(path):
#                     skipped["missing_file"] += 1
#                     continue
#                 data_ = json.load(open(path))

#                 if "MOF2Text" not in data_:
#                     skipped["missing_key"] += 1
#                     continue
#                 text = data_["MOF2Text"]

#             if text is None or str(text).strip() == "":
#                 skipped["empty_text"] += 1
#                 continue
            
#             output = get_mo(data_csv, fp)
#             if output is None:
#                 skipped["missing_key"] += 1
#                 continue
#             samples.append({
#                 "instruction": f"What are the oxidation states of the metal centers in the metal-organic framework structure described by {des}?",
#                 "input": text,
#                 "output": output,
#             })

#         except Exception:
#             skipped["exception"] += 1
#             continue

#     rng = random.Random(seed + i)
#     rng.shuffle(samples)

#     n = len(samples)
#     n_train = int(n * ratio)
#     train_data = samples[:n_train]
#     test_data = samples[n_train:]

#     train_out = f"./data/metal_oxidation_{des}_train.json"
#     test_out  = f"./data/metal_oxidation_{des}_test.json"

#     with open(train_out, "w") as f:
#         json.dump(train_data, f, indent=2, ensure_ascii=False)

#     with open(test_out, "w") as f:
#         json.dump(test_data, f, indent=2, ensure_ascii=False)

#     print(f"[{des}] kept: {n} | train: {len(train_data)} | test: {len(test_data)} | seed: {seed+i}")
#     print(f"[{des}] skipped: {skipped}")

#     skipped_total[des] = skipped

# print("All done. Skipped summary:")
# print(json.dumps(skipped_total, indent=2, ensure_ascii=False))

# with open(data_info_path, "w") as f:
#     json.dump(data_info, f, indent=2, ensure_ascii=False)