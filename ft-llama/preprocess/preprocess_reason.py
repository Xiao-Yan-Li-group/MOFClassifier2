import json
import os


def make_instruction():
    return (
        "Explain why a metal-organic framework structure is not reasonable.\nReturn only output of the following format for each reason, and no other information: ### Reason 1. **[Keyword]** [Detailed explanation], ### Reason 2...\n"
    )

dataset_label = "../dataset/label/manual_Tom.json"
all_data = json.load(open(dataset_label, "r", encoding="utf-8"))
out_dir = "./"
os.makedirs(out_dir, exist_ok=True)

instruction = make_instruction()

descriptor_path = "../dataset/descriptor/"

for dataset in ["robocry"]:
    fixed_dataset = []
    open_dataset  = []
    i = 0

    for name, meta in all_data.items():
        if meta.get("has_error", 0) != 1:
            continue
        fp = os.path.join(descriptor_path, name + ".json")
        if not os.path.exists(fp):
            continue
        data_ = json.load(open(fp, "r", encoding="utf-8"))
        des = data_.get(dataset)
        if des is None:
            continue

        fixed_dataset.append({"instruction": instruction, "input": des, "output": name})
        i += 1

    with open(os.path.join(out_dir, f"mofclass2_{dataset}_reason_all.json"), "w", encoding="utf-8") as f:
        json.dump(fixed_dataset, f, ensure_ascii=False, indent=2)
    print(f"{dataset}: {i} entries")


mof2text_path  = "../dataset/mof2text/"
fixed_dataset  = []
open_dataset   = []
i = 0

for name, meta in all_data.items():
    if meta.get("has_error", 0) != 1:
        continue
    fp = os.path.join(mof2text_path, name + ".json")
    if not os.path.exists(fp):
        continue
    data_ = json.load(open(fp, "r", encoding="utf-8"))
    des = data_.get("MOF2Text")
    if des is None:
        continue

    fixed_dataset.append({"instruction": instruction, "input": des, "output": name})
    i += 1

with open(os.path.join(out_dir, "mofclass2_mof2text_reason_all.json"), "w", encoding="utf-8") as f:
    json.dump(fixed_dataset, f, ensure_ascii=False, indent=2)