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
from sac_gmm.envs.skill_env import SkillSpecificEnv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import seed_everything
from sac_gmm.utils.utils import print_system_env_info, setup_logger


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


def make_eval_env(cfg: DictConfig):
    # Create the evaluation env
    new_env_cfg = {**cfg.calvin_env.env}
    new_env_cfg["use_egl"] = False
    new_env_cfg["show_gui"] = False
    new_env_cfg["use_vr"] = False
    new_env_cfg["use_scene_info"] = True
    new_env_cfg["tasks"] = cfg.calvin_env.tasks
    new_env_cfg.pop("_target_", None)
    new_env_cfg.pop("_recursive_", None)
    env = SkillSpecificEnv(**new_env_cfg)
    env.set_state_type(cfg.state_type)
    env.set_outdir(cfg.exp_dir)
    env.set_skill(cfg.skill)

    return env


def evaluate(env, gmm, dataset, max_steps, sampling_dt, render=False, record=False):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    succesful_rollouts, rollout_returns, rollout_lengths = 0, [], []
    for idx, (xi, d_xi) in enumerate(dataloader):
        if (idx % 5 == 0) or (idx == len(dataset)):
            log_rank_0(f"Test Trajectory {idx+1}/{len(dataset)}")
        x0 = xi.squeeze()[0, :].numpy()
        goal = dataset.goal
        rollout_return = 0
        observation = env.reset()
        current_state = observation[gmm.state_type][:3]
        temp = np.append(x0, np.append(dataset.fixed_ori, -1))
        action = env.prepare_action(temp, type="abs")
        # log_rank_0(f'Adjusting EE position to match the initial pose from the dataset')
        count = 0
        error_margin = 0.01
        while np.linalg.norm(current_state - x0) >= error_margin:
            observation, reward, done, info = env.step(action)
            current_state = observation[gmm.state_type][:3]
            count += 1
            if count >= 200:
                # x0 = current_state
                log_rank_0("CALVIN is struggling to place the EE at the right initial pose")
                log_rank_0(f"{np.linalg.norm(current_state - x0)}")
                break
        x = observation[gmm.state_type][:3]
        # log_rank_0(f'Simulating with gmm')
        if record:
            log_rank_0("Recording Robot Camera Obs")
            env.record_frame()
        for step in range(max_steps):
            d_x = gmm.predict(x - goal)
            delta_x = sampling_dt * d_x[:3]
            new_x = x + delta_x
            if gmm.state_type == "pos":
                temp = np.append(new_x, np.append(dataset.fixed_ori, -1))
            elif gmm.state_type == "pos_ori":
                temp = np.append(new_x, np.append(d_x[3:], -1))
            action = env.prepare_action(temp, type="abs")
            observation, reward, done, info = env.step(action)
            x = observation[gmm.state_type][:3]
            rollout_return += reward
            if record:
                env.record_frame()
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
    acc = succesful_rollouts / len(dataset.X)
    # wandb.config.update({"val dataset size": len(dataset.X)})
    gmm.logger.log_table(
        key="stats",
        columns=["skill", "accuracy", "average_return", "average_traj_len"],
        data=[[env.skill_name, acc * 100, np.mean(rollout_returns), np.mean(rollout_lengths)]],
    )

    return acc, np.mean(rollout_returns), np.mean(rollout_lengths)


@hydra.main(version_base="1.1", config_path="../../config", config_name="gmm_eval")
def eval_gmm(cfg: DictConfig) -> None:
    cfg.exp_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]

    seed_everything(cfg.seed, workers=True)
    log_rank_0(f"Evaluating {cfg.skill} skill with the following config:\n{OmegaConf.to_yaml(cfg)}")
    log_rank_0(print_system_env_info())
    log_rank_0(f"Evaluating gmm for {cfg.skill} skill with {cfg.state_type} as the input")

    # Load dataset
    cfg.dataset.skill = cfg.skill
    val_dataset = hydra.utils.instantiate(cfg.dataset)
    log_rank_0(f"Skill: {cfg.skill}, Validation Data: {val_dataset.X.size()}")
    # Obtain X_mins and X_maxs from training data to normalize in real-time
    cfg.dataset.train = True
    train_dataset = hydra.utils.instantiate(cfg.dataset)
    val_dataset.goal = train_dataset.goal

    # Create and load models to evaluate
    gmm = hydra.utils.instantiate(cfg.model.gmm)
    gmm.model_dir = os.path.join(Path(cfg.skills_dir).expanduser(), cfg.state_type, cfg.skill, gmm.name)
    gmm.load_model()
    gmm.state_type = cfg.state_type

    if gmm.name == "ManifoldGMM":
        cfg.dim = val_dataset.X.shape[-1]
        gmm.manifold = gmm.make_manifold(cfg.dim)

    # Setup logger
    logger_name = f"{cfg.skill}_{cfg.state_type}_{gmm.name}_{gmm.n_components}"
    gmm.logger = setup_logger(cfg, name=logger_name)

    # Evaluate by simulating in the CALVIN environment
    env = make_eval_env(cfg)
    acc, avg_return, avg_len = evaluate(
        env,
        gmm,
        val_dataset,
        max_steps=cfg.max_steps,
        render=cfg.render,
        record=cfg.record,
        sampling_dt=cfg.sampling_dt,
    )

    # Log evaluation output
    log_rank_0(f"{cfg.skill} Skill Accuracy: {round(acc, 2)}")


if __name__ == "__main__":
    eval_gmm()
