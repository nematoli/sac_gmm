import os
import sys
import wandb
import hydra
import logging
import csv
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import seed_everything
from sac_gmm.utils.utils import print_system_env_info, setup_logger
from sac_gmm.utils.env_maker import make_env

cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[0]
root = sac_gmm_path.parents[0]

# This is to access the locally installed repo clone when using slurm
sys.path.insert(0, sac_gmm_path.as_posix())  # sac_gmm
sys.path.insert(0, os.path.join(root, "calvin_env"))  # root/calvin_env
sys.path.insert(0, root.as_posix())  # root


logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


def play(env, dataset, max_steps, render=False):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    succesful_rollouts, failed_runs = 0, []
    for idx, (xi, d_xi) in enumerate(dataloader):
        log_rank_0(f"Trajectory {idx+1}/{len(dataset)}")

        x0 = xi.squeeze()[0, :].numpy()
        env.reset()
        reached_target = env.calibrate_EE_start_state(desired_pos=x0)
        if reached_target:
            for step in range(max_steps):
                d_x = d_xi.squeeze()[step, :].numpy()
                observation, reward, done, info = env.step(d_x)
                if render:
                    env.render()
                if done:
                    break
            if info["success"]:
                succesful_rollouts += 1
                status = "Success"
            else:
                status = "Fail"
                failed_runs.append(idx)
        else:
            status = "Fail"
            failed_runs.append(idx)

        log_rank_0(f"{idx+1}: {status}!")

    acc = succesful_rollouts / len(dataset.X)

    return acc, failed_runs


@hydra.main(version_base="1.1", config_path="../../config", config_name="play_demos")
def play_demos(cfg: DictConfig) -> None:
    cfg.exp_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]

    seed_everything(cfg.seed, workers=True)
    log_rank_0(f"Playing {cfg.skill.name} demos with the following config:\n{OmegaConf.to_yaml(cfg)}")
    log_rank_0(print_system_env_info())

    # Evaluate by simulating in the CALVIN environment
    env = make_env(cfg.env)
    env.set_skill(cfg.skill)

    # Load dataset
    train_dataset = hydra.utils.instantiate(cfg.datamodule.dataset)
    log_rank_0(f"Skill: {cfg.skill.name}, Train Data: {train_dataset.X.size()}")

    acc, failed = play(env, train_dataset, max_steps=cfg.skill.max_steps, render=cfg.render)
    # Log evaluation output
    log_rank_0(f"{cfg.skill.name} Training Demos Accuracy: {round(acc, 2)}")
    if cfg.remove_failures:
        train_dataset.rm_rw_data(failed)
        log_rank_0(f"Removed {len(failed)} failures from the Training Demos")

    cfg.datamodule.dataset.train = False
    val_dataset = hydra.utils.instantiate(cfg.datamodule.dataset)
    log_rank_0(f"Skill: {cfg.skill.name}, Validation Data: {val_dataset.X.size()}")

    acc, failed = play(
        env,
        val_dataset,
        max_steps=cfg.skill.max_steps,
        render=cfg.render,
    )
    # Log evaluation output
    log_rank_0(f"{cfg.skill.name} Validation Demos Accuracy: {round(acc, 2)}")
    if cfg.remove_failures:
        val_dataset.rm_rw_data(failed)
        log_rank_0(f"Removed {len(failed)} failures from the Validation Demos")


if __name__ == "__main__":
    play_demos()
