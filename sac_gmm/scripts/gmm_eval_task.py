import os
import sys
import wandb
import hydra
import logging
import csv
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import seed_everything
from sac_gmm.utils.utils import print_system_env_info, setup_logger
from sac_gmm.utils.env_maker import make_env
from sac_gmm.gmm.utils.rotation_utils import get_relative_quaternion
from sac_gmm.rl.helpers.skill_actor import SkillActor
from collections import Counter

cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[0]
root = sac_gmm_path.parents[0]

# This is to access the locally installed repo clone when using slurm
sys.path.insert(0, sac_gmm_path.as_posix())  # sac_gmm
sys.path.insert(0, os.path.join(root, "calvin_env"))  # root/calvin_env
sys.path.insert(0, root.as_posix())  # root


logger = logging.getLogger(__name__)

import pybullet as p

LETTERS_TO_SKILLS = {
    "A": "open_drawer",
    "B": "turn_on_lightbulb",
    "C": "move_slider_left",
    "D": "turn_on_led",
    "E": "close_drawer",
    "F": "turn_off_lightbulb",
    "G": "move_slider_right",
    "H": "turn_off_led",
}


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


@torch.no_grad()
def evaluate(env, actor, max_steps, render=False, record=False, out_dir=None, num_rollouts=1):
    """Evaluates the actor in the environment"""
    log_rank_0("Evaluation episodes in process")
    succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
    saved_video_path = None
    succesful_skill_ids = []
    # Choose a random episode to record
    rand_idx = np.random.randint(1, num_rollouts + 1)
    for episode in tqdm(range(1, num_rollouts + 1)):
        skill_id = 0
        episode_return, episode_env_steps = 0, 0
        obs = env.reset()
        # Recording setup
        if record and (episode == rand_idx):
            env.reset_recording()
            env.record_frame(size=200)

        while episode_env_steps < max_steps:
            env_action, is_nan = actor.act(obs["robot_obs"], skill_id)
            if is_nan:
                done = True
                log_rank_0("Nan in prediction, aborting episode")
            else:
                obs, reward, done, info = env.step(env_action)
                episode_return += reward
                episode_env_steps += 1

                if reward > 0:
                    succesful_skill_ids.append(skill_id)
                    skill_id = (skill_id + 1) % len(actor.skills)
                    if skill_id != 0:
                        reward = 0
            if record and (episode == rand_idx):
                env.record_frame(size=200)
            if render:
                env.render()
            if done:
                break

            if done:
                skill_id = 0
                break

        if ("success" in info) and info["success"]:
            succesful_episodes += 1
        # Recording setup close
        if record and (episode == rand_idx):
            video_path = env.save_recording(
                outdir=out_dir,
                fname=f"Episode{episode}",
            )
            env.reset_recording()
            saved_video_path = video_path

        episodes_returns.append(episode_return)
        episodes_lengths.append(episode_env_steps)
    accuracy = succesful_episodes / num_rollouts
    return (
        accuracy,
        np.mean(episodes_returns),
        np.mean(episodes_lengths),
        succesful_skill_ids,
        saved_video_path,
    )


@hydra.main(version_base="1.1", config_path="../../config", config_name="gmm_eval_task")
def eval_gmms(cfg: DictConfig) -> None:
    cfg.exp_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]

    task = cfg.task
    task_order = [*task.order]
    task.skills = [LETTERS_TO_SKILLS[skill] for skill in task_order]
    task.max_steps = task.skill_max_steps * len(task.skills)

    # Environment
    env = make_env(cfg.env)
    env.set_task(task.skills)
    env.max_episode_steps = task.max_steps

    # Skill Actor
    actor = SkillActor(task)
    actor.skill_names = task.skills
    actor.make_skills(cfg.gmm)
    actor.load_models()
    for id, skill in enumerate(task.skills):
        cfg.datamodule.dataset.skill.skill = skill
        dataset = hydra.utils.instantiate(cfg.datamodule.dataset)
        actor.skills[id].set_skill_params(dataset)

    # Evaluate by simulating in the CALVIN environment
    acc, avg_return, avg_len, skill_ids, eval_video_path = evaluate(
        env=env,
        actor=actor,
        max_steps=task.max_steps,
        render=cfg.render,
        record=cfg.record,
        out_dir=cfg.exp_dir,
        num_rollouts=cfg.num_rollouts,
    )

    # Log evaluation output
    log_rank_0(f"{actor.skill_names} Task Accuracy: {round(acc, 2)}")
    log_rank_0(f"{actor.skill_names} Task Average Return: {round(avg_return, 2)}")
    log_rank_0(f"{actor.skill_names} Task Average Trajectory Length: {round(avg_len, 2)}")

    if len(skill_ids) > 0:
        skill_id_counts = Counter(skill_ids)
        skill_ids = {f"eval/{actor.skill_names[k]}": v / cfg.num_rollouts for k, v in skill_id_counts.items()}
        # Add 0 values for skills that were not used at all
        unused_skill_ids = set(range(len(actor.skill_names))) - set(skill_id_counts.keys())
        if len(unused_skill_ids) > 0:
            skill_ids.update({f"eval/{actor.skill_names[k]}": 0 for k in list(unused_skill_ids)})
    else:
        skill_ids = {f"eval/{k}": 0 for k in actor.skill_names}
    log_rank_0(f"Skill Distribution: {skill_ids}")


if __name__ == "__main__":
    eval_gmms()
