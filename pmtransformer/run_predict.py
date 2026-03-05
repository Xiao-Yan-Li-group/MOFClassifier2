from pathlib import Path
import moftransformer
import yaml


with open("jobs.yaml", "r") as f:
    config = yaml.safe_load(f)


moftransformer.predict(
    
    load_path=Path(config["predict"]["path"]["load_path"]) / f'pretrained_mof_seed{config["predict"]["model"]["seed"]}_from_/version_{config["predict"]["model"]["version"]}/checkpoints/best.ckpt',
    visualize = config["predict"]["task"]["visualize"],
    test_only=config["predict"]["task"]["test_only"],
    batch_size=config["predict"]["server"]["batch_size"],
    devices = config["predict"]["server"]["devices"],
    per_gpu_batchsize = config["predict"]["server"]["per_gpu_batchsize"],
    accelerator = config["predict"]["server"]["accelerator"],
    num_nodes = config["predict"]["server"]["num_nodes"],
    num_workers = config["predict"]["server"]["num_workers"],

    precision = config["train"]["server"]["precision"],
    root_dataset=config["dataset"]["root_dataset"],
    downstream=config["dataset"]["downstream"],
    loss_names=config["train"]["task"]["loss_names"],
    n_classes=config["train"]["task"]["n_classes"],
    max_epochs=config["train"]["model"]["max_epochs"],
    # mean=config["train"]["model"]["mean"],
    # std=config["train"]["model"]["std"],
    seed = config["train"]["model"]["seed"],
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
