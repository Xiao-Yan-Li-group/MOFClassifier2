import torch
torch.autograd.set_detect_anomaly(True)

import moftransformer
import yaml

with open("jobs.yaml", "r") as f:
    config = yaml.safe_load(f)


moftransformer.run(
    root_dataset=config["dataset"]["root_dataset"],
    downstream=config["dataset"]["downstream"],

    log_dir=config["train"]["path"]["log_dir"],
    load_path=config["train"]["path"]["load_path"],

    loss_names=config["train"]["task"]["loss_names"],
    n_classes=config["train"]["task"]["n_classes"],
    visualize = config["train"]["task"]["visualize"],
    test_only=config["train"]["task"]["test_only"],

    batch_size=config["train"]["server"]["batch_size"],
    devices = config["train"]["server"]["devices"],
    per_gpu_batchsize = config["train"]["server"]["per_gpu_batchsize"],
    accelerator = config["train"]["server"]["accelerator"],
    num_nodes = config["train"]["server"]["num_nodes"],
    num_workers = config["train"]["server"]["num_workers"],
    precision = config["train"]["server"]["precision"],

    max_epochs=config["train"]["model"]["max_epochs"],
    # mean=config["train"]["model"]["mean"],
    # std=config["train"]["model"]["std"],
    optim_type = config["train"]["model"]["optim_type"],
    learning_rate = config["train"]["model"]["learning_rate"],
    weight_decay = config["train"]["model"]["weight_decay"],
    seed = config["train"]["model"]["seed"],
    decay_power = config["train"]["model"]["decay_power"],
    max_steps = config["train"]["model"]["max_steps"],
    warmup_steps = config["train"]["model"]["warmup_steps"],
    end_lr = config["train"]["model"]["end_lr"],
    lr_mult = config["train"]["model"]["lr_mult"],
    
    hid_dim = config["train"]["model"]["layer"]["hid_dim"],
    num_heads = config["train"]["model"]["layer"]["num_heads"],
    num_layers = config["train"]["model"]["layer"]["num_layers"],
    mlp_ratio = config["train"]["model"]["layer"]["mlp_ratio"],
    drop_rate = config["train"]["model"]["layer"]["drop_rate"],
    mpp_ratio = config["train"]["model"]["layer"]["mpp_ratio"],
    atom_fea_len = config["train"]["model"]["layer"]["atom_fea_len"],
    nbr_fea_len = config["train"]["model"]["layer"]["nbr_fea_len"],
    max_graph_len = config["train"]["model"]["layer"]["max_graph_len"],
    max_nbr_atoms = config["train"]["model"]["layer"]["max_nbr_atoms"],
    img_size = config["train"]["model"]["layer"]["img_size"],
    patch_size = config["train"]["model"]["layer"]["patch_size"],
    in_chans = config["train"]["model"]["layer"]["in_chans"],
    max_grid_len = config["train"]["model"]["layer"]["max_grid_len"],
    draw_false_grid = config["train"]["model"]["layer"]["draw_false_grid"]
    ) 

