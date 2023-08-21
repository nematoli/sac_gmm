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


def evaluate(env, gmm, target, max_steps, render=False, record=False):
    succesful_rollouts, rollout_returns, rollout_lengths = 0, [], []
    # for idx, (xi, d_xi) in enumerate(dataloader):
    val_rollouts = 100
    for idx in range(val_rollouts):
        if (idx % 5 == 0) or (idx == val_rollouts):
            log_rank_0(f"Test Trajectory {idx+1}/{val_rollouts}")
        # x0 = xi.squeeze()[0, :].numpy()
        goal = target
        rollout_return = 0
        observation = env.reset()
        # current_state = observation["position"]
        # # log_rank_0(f'Adjusting EE position to match the initial pose from the dataset')
        # count = 0
        # error_margin = 0.01
        # while np.linalg.norm(x0 - current_state) >= error_margin:
        #     dx = (x0 - current_state) / dt
        #     observation, reward, done, info = env.step(dx)
        #     current_state = observation["position"]
        #     count += 1
        #     if count >= 300:
        #         # x0 = current_state
        #         log_rank_0("CALVIN is struggling to place the EE at the right initial pose")
        #         log_rank_0(f"{np.linalg.norm(x0 - current_state)}")
        #         break
        # # log_rank_0(f'Simulating with gmm')
        # print("count from here")
        # if record:
        #     log_rank_0("Recording Robot Camera Obs")
        #     env.record_frame(size=64)

        for step in range(max_steps):
            d_x = gmm.predict(env.get_pos_from_obs(observation) - goal)
            observation, reward, done, info = env.step(d_x)
            rollout_return += reward
            if record:
                env.record_frame(size=64)
            if render:
                env.render()
            if done:
                break
        status = None
        if info["success"]:
            succesful_rollouts += 1
            status = "Success"
        else:
            status = "Fail"
        log_rank_0(f"{idx+1}: {status}!")
        if record:
            log_rank_0("Saving Robot Camera Obs")
            video_path = env.save_recorded_frames(outdir=env.outdir, fname=idx)
            env.reset_recorded_frames()
            status = None
            gmm.logger.log_table(key="eval", columns=["GMM"], data=[[wandb.Video(video_path, fps=30, format="gif")]])

        rollout_returns.append(rollout_return)
        rollout_lengths.append(step)
    acc = succesful_rollouts / val_rollouts
    gmm.logger.log_table(
        key="stats",
        columns=["skill", "accuracy", "average_return", "average_traj_len"],
        data=[[env.skill.name, acc * 100, np.mean(rollout_returns), np.mean(rollout_lengths)]],
    )

    return acc, np.mean(rollout_returns), np.mean(rollout_lengths)


@hydra.main(version_base="1.1", config_path="../../config", config_name="gmm_eval")
def eval_gmm(cfg: DictConfig) -> None:
    cfg.exp_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]

    seed_everything(cfg.seed, workers=True)
    log_rank_0(f"Evaluating {cfg.skill} skill with the following config:\n{OmegaConf.to_yaml(cfg)}")
    log_rank_0(print_system_env_info())
    log_rank_0(f"Evaluating gmm for {cfg.skill.name} skill with {cfg.skill.state_type} as the input")

    # Load dataset
    # Obtain X_mins and X_maxs from training data to normalize in real-time
    cfg.datamodule.dataset.train = True
    train_dataset = hydra.utils.instantiate(cfg.datamodule.dataset)

    # Create and load models to evaluate
    gmm = hydra.utils.instantiate(cfg.gmm)
    gmm.load_model()

    if "Manifold" in gmm.name:
        gmm.manifold = gmm.make_manifold()

    # Setup logger
    logger_name = f"{cfg.skill.name}_{cfg.skill.state_type}_{gmm.name}_{gmm.n_components}"
    gmm.logger = setup_logger(cfg, name=logger_name)

    # Evaluate by simulating in the CALVIN environment
    env = make_env(cfg.env)
    env.set_skill(cfg.skill)
    env.set_init_pos(train_dataset.start)

    acc, avg_return, avg_len = evaluate(
        env,
        gmm,
        target=train_dataset.goal,
        max_steps=cfg.skill.max_steps,
        render=cfg.render,
        record=cfg.record,
    )

    # Log evaluation output
    log_rank_0(f"{cfg.skill.name} Skill Accuracy: {round(acc, 2)}")


if __name__ == "__main__":
    eval_gmm()
