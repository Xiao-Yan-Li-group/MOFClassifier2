#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import types

import torch

import numpy as np
import pandas as pd

from time import perf_counter
from typing import Union

from torch_geometric.data import Data

from setc.logger import *
from setc.model import ErrGAT
from setc.graph_gen import generate_graph
from setc.model import ErrGAT

import concurrent.futures
from threading import Thread
from queue import Queue
import multiprocessing as mp

if 'model' not in sys.modules:
    model_module = types.ModuleType('model')
    model_module.ErrGAT = ErrGAT
    sys.modules['model'] = model_module


# collect cli arguments
code_desc = "Predict error types present in given crystal structure file using pretrained GAT models."
parser = argparse.ArgumentParser(description=code_desc)
parser.add_argument("cif", help="path to CIF file OR directory containing CIF files")
parser.add_argument(
    "--node_features",
    nargs="+",
    default=["atomic"],
    choices=["atomic", "localenv"],
    help="select node feature combination.",
)

parser.add_argument(
    "--store_graphs",
    action="store_true",
    help="save graphs as .pt files",
)
parser.add_argument(
    "--log_level",
    default="INFO",
    choices=["INFO", "DEBUG", "WARNING"],
    help="output logging message level",
)

parser.add_argument(
    "--jobs",
    type=int,
    default=os.cpu_count(),
    help="number of parallel worker processes (default: CPU count)",
)

args = parser.parse_args()

# init timer
t_start = perf_counter()

# init logger
logger = logger_setup("classify", "setc.log", args.log_level)

# define device & code paths & GLOBALS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug(f"Device: {device}")

# search for data files
GAT_PATH = os.path.dirname(os.path.realpath(__file__))
logger.debug(f"Code File Path: {GAT_PATH}")

# save graphs
if args.store_graphs:
    dir_sffx = "_".join(sorted(args.node_features))
    SETC_GRAPH = f"tmp_pt_{dir_sffx}"
    os.makedirs(SETC_GRAPH, exist_ok=True)
    logger.debug(f"Temp. Graph File Path: {SETC_GRAPH}")
else:
    SETC_GRAPH = False

# feature combo to be included in output csvs
FEAT_ARGS = {x: True for x in args.node_features}
NODE_FEATS = "-".join(sorted(args.node_features))


STOP = object()

def writer_thread(queue: Queue, csv_file: str):
    header_written = os.path.exists(csv_file)
    while True:
        item = queue.get()
        if item is STOP:
            break
        df = pd.DataFrame([item])
        if header_written and os.path.exists(csv_file):
            df.to_csv(csv_file, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_file, mode="w", header=True, index=False)
            header_written = True

def _multi_worker(cif: str, node_features: list, node_feats_tag: str, store_graphs: bool) -> dict:
    feat_args_local = {x: True for x in node_features}
    local_graph_dir = False
    if store_graphs:
        dir_sffx = "_".join(sorted(node_features))
        local_graph_dir = f"tmp_pt_{dir_sffx}_{os.getpid()}"
        os.makedirs(local_graph_dir, exist_ok=True)

    model = load_model(3).to(device)
    model.eval()

    graph = generate_graph(cif, local_graph_dir, **feat_args_local)
    sclgraph = standard_scale(graph, node_features)
    with torch.inference_mode():
        y = model(sclgraph.to(device)).detach().cpu().numpy().flatten()

    logger.info(f"{cif}|        h: {y[0] > 0.5} ({y[0]:.3f})")
    logger.info(f"{cif}|   charge: {y[1] > 0.5} ({y[1]:.3f})")
    logger.info(f"{cif}| disorder: {y[2] > 0.5} ({y[2]:.3f})")

    return {
        "cif": os.path.basename(cif),
        "h_pred_raw": float(y[0]),
        "h_pred_threshold": float(y[0] > 0.5),
        "charge_pred_raw": float(y[1]),
        "charge_pred_threshold": float(y[1] > 0.5),
        "disorder_pred_raw": float(y[2]),
        "disorder_pred_threshold": float(y[2] > 0.5),
        "node_features": node_feats_tag,
    }


def optim_hypers(param_path: str) -> dict[str, Union[str, int, float]]:
    """
    Read model hyperparameters from optimized file.

        Parameters:
            param_path (str): path to the *.optim file containing
                              optimized hyperparameter

        Returns:
            param_dict (dict): Dictionary of hyperparameters to necessary
                               to load ErrGAT model
    """

    def convert_type(s: str) -> Union[bool, int, float, None]:
        """Convert string object from .optim file to correct data type."""
        try:
            new_s = float(s)
            if new_s % 1 == 0:
                return int(new_s)
            else:
                return new_s
        except Exception as e:
            if "True" in s:
                return True
            elif "False" in s:
                return False
            elif "None" in s:
                return None
            else:
                return s

    # read optimized hyperparameters
    with open(param_path, "r") as ppf:
        param_pairs = [p.split("=") for p in ppf.read().split("\n") if "=" in p]
    ignore = ["batch", "lr"]
    param_dict = {
        pp[0]: convert_type(pp[1]) for pp in param_pairs if pp[0] not in ignore
    }
    # param_dict = {pp[0]: convert_type(pp[1]) for pp in param_pairs if len(pp) == 2}
    logger.debug(f"Loaded Hyperparameter dict ... {str(param_dict)}")
    return param_dict


def load_model(no_targets: int, classifier: str = "multi") -> ErrGAT:
    """
    Load pretrained classification model from torch.state_dict().

        Parameters:
            no_targets (int): output [label] dimension i.e. error types
            classifier (str): model types to load i.e. h, charge, disorder, multi

        Returns:
            model (ErrGAT | torch.nn.Module): classification model
    """

    feats = args.node_features
    feat_len = {"atomic": 8, "localenv": 168}
    model_base = "_".join(sorted(feats)) + f"_{classifier}"
    
    logger.info(f"Loading GAT model ... {model_base}.pt")

    model_saved = os.path.join(GAT_PATH, f"setc/{model_base}.pt")

    # get model parameters
    model_params = {
        "in_chan": sum([feat_len[f] for f in feats]),
        "out_chan": None,
        "activation": "relu",
        "num_targets": no_targets,
    }
    opt_params = optim_hypers(
        os.path.join(GAT_PATH, f"setc/{model_base}.optim")
    )
    model_params.update(opt_params)
    #
 
    model = ErrGAT(**model_params)
    
    try:
        checkpoint = torch.load(model_saved, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    model.eval()
    return model.to(device)


def append_to_csv(row_data: dict[str, Union[str, float]], csv_file: str) -> None:
    """
    Adds the input results to a csv file summarizing error labels.

        Parameters:
            row_data (dict[str, str | float]): dict of predicted labels.
            csv_file (str): path to csv file.

        Returns:
            None
    """

    df = pd.DataFrame([row_data])
    logger.debug(f"Output {str(row_data)} to {csv_file}")
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_file, mode="w", header=True, index=False)
    return 0


def standard_scale(graph: Data, feature_list: list) -> Data:
    """
    Standardize node features of an input graph.

        Parameters:
            graph (torch_geometric.data.Data): structure graph data
            feature_list (list): list containing types of node features
                                       used in the graph representation

        Returns:
            scaled_graph (torch_geometric.data.Data): structure graph
                                        data with scaled node features
    """

    scales = np.load(os.path.join(GAT_PATH, "setc/stnd_scale.npz"))
    #
    means = np.concatenate(
        [scales[f"{feat}_mean"] for feat in feature_list], axis=None
    )
    stds = np.concatenate(
        [scales[f"{feat}_std"] for feat in feature_list], axis=None
    )
    scaled_graph = graph
    scaled_graph.x = (graph.x - means) / stds
    return scaled_graph


def classify_multi(cif_files: list) -> None:
    node_feats_tag = "-".join(sorted(args.node_features))

    q: Queue = Queue(maxsize=2 * (args.jobs or 1))
    wt = Thread(target=writer_thread, args=(q, "../../dataset/5checkers/SETC.csv"), daemon=True)
    wt.start()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    max_workers = max(1, args.jobs or 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_multi_worker, cif, args.node_features, node_feats_tag, args.store_graphs)
            for cif in cif_files
        ]
        for fut in concurrent.futures.as_completed(futures):
            try:
                row = fut.result()
                q.put(row)
            except Exception as e:
                logger.error(f"Failed on a CIF task: {e}", exc_info=True)

    q.put(STOP)
    wt.join()


def main(args):
    # handle single-file/directory of files input options
    if os.path.isfile(args.cif):
        cifs = [args.cif]
    else:
        cifs = glob.glob(f"{args.cif}/*.cif")
    logger.info(f"Number of CIF files found == {len(cifs)}")

    logger.info(
        " ##########    Multi-Label Error Type Classification    ########## "
    )
    classify_multi(cifs)
    
    # clean-up
    t_end = perf_counter()
    logger.info(f"Finished successfully | Elapsed Time: {t_end - t_start}")
    return 0

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main(args)
