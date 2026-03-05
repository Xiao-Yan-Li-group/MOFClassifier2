from moftransformer.visualize import PatchVisualizer
import os, json, yaml, shutil
from pathlib import Path


with open("jobs.yaml", "r") as f:
    config = yaml.safe_load(f)


os.makedirs(config["visualize"]["save_path"], exist_ok=True)
model_path = Path(config["visualize"]["path"]["load_path"]) / f'pretrained_mof_seed{config["visualize"]["model"]["seed"]}_from_/version_{config["visualize"]["model"]["version"]}/checkpoints/best.ckpt'
vis = PatchVisualizer.from_cifname(config["visualize"]["cifname"],
                                   model_path,
                                   config["visualize"]["data_path"],
                                   config["dataset"]["downstream"]
                                   )

fig, ax = vis.draw_graph(
                         view_init=config["visualize"]["graph"]["view_init"],
                         alpha=config["visualize"]["graph"]["alpha"],
                         cmap=config["visualize"]["graph"]["cmap"],
                         minatt=config["visualize"]["graph"]["minatt"],
                         maxatt=config["visualize"]["graph"]["maxatt"],
                         remove_under_minatt=config["visualize"]["graph"]["remove_under_minatt"],
                         grid_scale_factor=config["visualize"]["graph"]["grid_scale_factor"],
                         atomic_scale_factor=config["visualize"]["graph"]["atomic_scale_factor"],
                         att_scale_factor=config["visualize"]["graph"]["att_scale_factor"],
                         return_fig=config["visualize"]["graph"]["return_fig"],
                         show_colorbar=config["visualize"]["graph"]["show_colorbar"]
                         )
fig.savefig(os.path.join(config["visualize"]["save_path"], f"{config["visualize"]["cifname"]}_graph.png"),
                         dpi=config["visualize"]["graph"]["dpi"],
                         bbox_inches=config["visualize"]["graph"]["bbox_inches"],
                         pad_inches=config["visualize"]["graph"]["pad_inches"]
                         )

fig, ax = vis.draw_grid(
                         view_init=config["visualize"]["grid"]["view_init"],
                         alpha=config["visualize"]["grid"]["alpha"],
                         cmap=config["visualize"]["grid"]["cmap"],
                         minatt=config["visualize"]["grid"]["minatt"],
                         maxatt=config["visualize"]["grid"]["maxatt"],
                         remove_under_minatt=config["visualize"]["grid"]["remove_under_minatt"],
                         grid_scale_factor=config["visualize"]["grid"]["grid_scale_factor"],
                         atomic_scale_factor=config["visualize"]["grid"]["atomic_scale_factor"],
                         att_scale_factor=config["visualize"]["grid"]["att_scale_factor"],
                         return_fig=config["visualize"]["grid"]["return_fig"],
                         show_colorbar=config["visualize"]["grid"]["show_colorbar"]
                        )
fig.savefig(os.path.join(config["visualize"]["save_path"], f"{config["visualize"]["cifname"]}_grid.png"),
                         dpi=config["visualize"]["grid"]["dpi"],
                         bbox_inches=config["visualize"]["grid"]["bbox_inches"],
                         pad_inches=config["visualize"]["grid"]["pad_inches"]
                         )
