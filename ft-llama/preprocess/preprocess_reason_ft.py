import os, json, random
from tqdm import tqdm


def get_label(all_data, cifid):
    return all_data[cifid]["has_error"]


def get_reason(all_data, cifid):
    """Returns issues list, e.g. ["h", "charge"] or []"""
    return all_data[cifid]["error_types"]


INSTRUCTION = (
    'You are an expert in metal-organic framework (MOF) chemistry.\n'
    'Given the following MOF structure description, identify which types of issues make it unreasonable.\n\n'
    'Classify the issues from the following categories (one or more may apply):\n'
    '- "h": issues related to hydrogen atoms (missing, misplaced, or incorrect H)\n'
    '- "disorder": structural disorder issues (atomic overlap, ambiguous positions)\n'
    '- "charge": charge balance issues (inconsistent oxidation states or overall charge)\n'
    '- "other": other unreasonable aspects not covered above\n\n'
    'Respond ONLY in the following JSON format, no explanation:\n'
    '{"issues": ["<label1>", "<label2>"]}'
)

dataset_label = "../dataset/label/manual_Tom.json"
dataset_des   = "../dataset/descriptor"
dataset_path  = "../dataset"
data_info_path = "./data/dataset_info.json"

datasets = [
    "robocry",
    "mof2text",
]

seed  = 120
ratio = 0.8

with open(data_info_path, encoding="utf-8") as f:
    data_info = json.load(f)

with open(dataset_label, encoding="utf-8") as f:
    all_data = json.load(f)

skipped_total = {}

for i, des in enumerate(datasets):
    samples  = []
    skipped  = {"missing_file": 0, "missing_key": 0, "empty_text": 0, "exception": 0}

    train_filename = f"mofclass2_{des}_reason_train.json"
    test_filename  = f"mofclass2_{des}_reason_test.json"
    data_info[des]             = {"file_name": train_filename}
    data_info[des + "_test"]   = {"file_name": test_filename}

    for fp in tqdm(list(all_data.keys()), desc=f"processing {des}..."):
        try:
            label = get_label(all_data, fp)

            if des == "robocry":
                path = os.path.join(dataset_des, f"{fp}.json")
                if not os.path.exists(path):
                    skipped["missing_file"] += 1
                    continue
                with open(path, encoding="utf-8") as f_:
                    data_ = json.load(f_)
                if des not in data_:
                    skipped["missing_key"] += 1
                    continue
                text = data_[des]

            else:
                path = os.path.join(dataset_path, des, f"{fp}.json")
                if not os.path.exists(path):
                    skipped["missing_file"] += 1
                    continue
                with open(path, encoding="utf-8") as f_:
                    data_ = json.load(f_)
                if "MOF2Text" not in data_:
                    skipped["missing_key"] += 1
                    continue
                text = data_["MOF2Text"]

            if text is None or str(text).strip() == "":
                skipped["empty_text"] += 1
                continue

            if label != 1:
                continue

            issues = get_reason(all_data, fp)
            output_str = json.dumps({"issues": issues}, ensure_ascii=False)

            samples.append({
                "instruction": INSTRUCTION,
                "input":       text,
                "output":      output_str,
            })

        except Exception as e:
            skipped["exception"] += 1
            print(f"  [WARN] exception on {fp}: {e}")
            continue

    rng = random.Random(seed + i)
    rng.shuffle(samples)

    n       = len(samples)
    n_train = int(n * ratio)
    train_data = samples[:n_train]
    test_data  = samples[n_train:]

    train_out = f"./data/{train_filename}"
    test_out  = f"./data/{test_filename}"

    os.makedirs("./data", exist_ok=True)

    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(test_out, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    n_positive = sum(1 for s in samples if json.loads(s["output"])["issues"])
    n_negative = n - n_positive

    print(f"\n[{des}] total: {n}  |  positive: {n_positive}  |  negative: {n_negative}")
    print(f"[{des}] train: {len(train_data)}  |  test: {len(test_data)}  |  seed: {seed + i}")
    print(f"[{des}] skipped: {skipped}")

    skipped_total[des] = skipped

with open(data_info_path, "w", encoding="utf-8") as f:
    json.dump(data_info, f, indent=2, ensure_ascii=False)

print("\nAll done. Skipped summary:")
print(json.dumps(skipped_total, indent=2, ensure_ascii=False))