import os
import sys
import hydra
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

cwd_path = Path(__file__).absolute().parents[0]
root = cwd_path.parents[0]
# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, root.as_posix())  # root


logger = logging.getLogger(__name__)
os.chdir(cwd_path)


def load_demonstrations(datamodule, mode):
    if mode == "training":
        data_loader = datamodule.train_dataloader()
    elif mode == "validation":
        data_loader = datamodule.val_dataloader()

    split_iter = iter(data_loader)
    demos = []
    for i in range(len(split_iter)):
        demo = next(split_iter)
        demos += [demo["robot_obs"].numpy()]

    demos = np.concatenate(demos, axis=0)
    logger.info(f"Dimensions of {mode} demonstrations (NxSxD): {demos.shape}.")

    return demos


@hydra.main(version_base="1.1", config_path="../config", config_name="extract_calvin_demos")
def extract_demos(cfg: DictConfig) -> None:
    """
    This is called to extract demonstrations for a specific skill.
    Args:
        cfg: hydra config
    """
    seed_everything(cfg.seed, workers=True)
    cfg.log_dir = Path(cfg.log_dir).expanduser()
    cfg.demos_dir = Path(cfg.demos_dir).expanduser()
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.demos_dir, exist_ok=True)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage="fit")

    p = Path(Path(cfg.demos_dir).expanduser() / datamodule.skill.name)
    p.mkdir(parents=True, exist_ok=True)
    mode = ["training", "validation"]
    for m in mode:
        demos = load_demonstrations(datamodule, m)
        save_dir = p / m
        np.save(save_dir, demos)


if __name__ == "__main__":
    demos = extract_demos()
