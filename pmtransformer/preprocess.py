from moftransformer.utils import prepare_data
import os
import json
import yaml


with open("jobs.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset = json.load(open(config["dataset"]["prop"]))

data_raw_ds = {}
for cif_id in dataset:
    if dataset[cif_id]["has_error"] == 1:
        data_raw_ds[cif_id] = int(0)
    elif dataset[cif_id]["has_error"] == 0:
        data_raw_ds[cif_id] = int(1)

with open(os.path.join(config["dataset"]["root_cifs"], "raw_"+config["dataset"]["downstream"]+".json"), "w") as f:
    json.dump(data_raw_ds, f, indent=2)


prepare_data(
    root_cifs=config["dataset"]["root_cifs"],
    root_dataset=config["dataset"]["root_dataset"],
    train_fraction=config["dataset"]["split"]["train_fraction"],
    test_fraction=config["dataset"]["split"]["test_fraction"],
    downstream=config["dataset"]["downstream"],
    num_workers=config["preprocess"]["num_workers"],
    niggli=config["preprocess"]["niggli"],
    primitive=config["preprocess"]["primitive"],
    graph_method=config["preprocess"]["graph_method"],
)
