import os
import sys
import wandb
import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig

cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[0]
root = sac_gmm_path.parents[0]

# This is to access the locally installed repo clone when using slurm
sys.path.insert(0, sac_gmm_path.as_posix())  # sac_gmm
sys.path.insert(0, os.path.join(root, "calvin_env"))  # root/calvin_env
sys.path.insert(0, root.as_posix())  # root

import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from sac_gmm.envs.skill_env import SkillSpecificEnv


class SkillEvaluator(object):
    """Python wrapper that allows you to evaluate learned gmm skills
    in the CALVIN environment.
    """

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.skill = self.cfg.skill
        self.logger = logging.getLogger("SkillEvaluator")
        self.robot_obs = self.cfg.state_type

    def evaluate(self, gmm, dataset, max_steps, sampling_dt, render=False, record=False):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        succesful_rollouts, rollout_returns, rollout_lengths = 0, [], []
        for idx, (xi, d_xi) in enumerate(dataloader):
            if (idx % 5 == 0) or (idx == len(dataset)):
                self.logger.info(f"Test Trajectory {idx+1}/{len(dataset)}")
            x0 = xi.squeeze()[0, :].numpy()
            goal = dataset.goal
            rollout_return = 0
            observation = self.env.reset()
            current_state = observation[self.robot_obs][:3]
            temp = np.append(x0, np.append(dataset.fixed_ori, -1))
            action = self.env.prepare_action(temp, type="abs")
            # self.logger.info(f'Adjusting EE position to match the initial pose from the dataset')
            count = 0
            error_margin = 0.01
            while np.linalg.norm(current_state - x0) >= error_margin:
                observation, reward, done, info = self.env.step(action)
                current_state = observation[self.robot_obs][:3]
                count += 1
                if count >= 200:
                    # x0 = current_state
                    self.logger.info("CALVIN is struggling to place the EE at the right initial pose")
                    self.logger.info(f"{np.linalg.norm(current_state - x0)}")
                    break
            x = observation[self.robot_obs][:3]
            # self.logger.info(f'Simulating with gmm')
            if record:
                self.logger.info("Recording Robot Camera Obs")
                self.env.record_frame()
            for step in range(max_steps):
                d_x = gmm.predict(x - goal)
                delta_x = sampling_dt * d_x[:3]
                new_x = x + delta_x
                if dataset.state_type == "pos":
                    temp = np.append(new_x, np.append(dataset.fixed_ori, -1))
                elif dataset.state_type == "pos_ori":
                    temp = np.append(new_x, np.append(d_x[3:], -1))
                action = self.env.prepare_action(temp, type="abs")
                observation, reward, done, info = self.env.step(action)
                x = observation[self.robot_obs][:3]
                rollout_return += reward
                if record:
                    self.env.record_frame()
                if render:
                    self.env.render()
                if done:
                    break
            status = None
            if info["success"]:
                succesful_rollouts += 1
                status = "Success"
            else:
                status = "Fail"
            self.logger.info(f"{idx+1}: {status}!")
            if record:
                self.logger.info("Saving Robot Camera Obs")
                video_path = self.env.save_recorded_frames()
                self.env.reset_recorded_frames()
                status = None
                if self.cfg.wandb:
                    wandb.log(
                        {
                            f"{self.env.skill_name} {status} {self.env.record_count}": wandb.Video(
                                video_path, fps=30, format="gif"
                            )
                        }
                    )
            rollout_returns.append(rollout_return)
            rollout_lengths.append(step)
        acc = succesful_rollouts / len(dataset.X)
        if self.cfg.wandb:
            wandb.config.update({"val dataset size": len(dataset.X)})
            wandb.log(
                {
                    "skill": self.env.skill_name,
                    "accuracy": acc * 100,
                    "average_return": np.mean(rollout_returns),
                    "average_traj_len": np.mean(rollout_lengths),
                }
            )
        return acc, np.mean(rollout_returns), np.mean(rollout_lengths)

    def run(self):
        skill_accs = {}
        self.env.set_skill(self.skill)

        # Get validation dataset
        self.cfg.dataset.skill = self.skill
        val_dataset = hydra.utils.instantiate(self.cfg.dataset)

        # Create and load models to evaluate
        self.cfg.dim = val_dataset.X.shape[-1]
        gmm = hydra.utils.instantiate(self.cfg.gmm)
        gmm.model_dir = os.path.join(self.cfg.skills_dir, self.cfg.state_type, self.skill, gmm.name)

        # Obtain X_mins and X_maxs from training data to normalize in real-time
        self.cfg.dataset.train = True
        train_dataset = hydra.utils.instantiate(self.cfg.dataset)
        val_dataset.goal = train_dataset.goal

        gmm.skills_dir = gmm.model_dir
        gmm.load_model()
        gmm.state_type = self.cfg.state_type
        if gmm.name == "ManifoldGMM":
            gmm.manifold = gmm.make_manifold(self.cfg.dim)

        self.logger.info(f"Evaluating {self.skill} skill with {self.cfg.state_type} input on CALVIN environment")
        self.logger.info(f"Test/Val Data: {val_dataset.X.size()}")

        if self.cfg.wandb:
            config = {
                "state_type": self.cfg.state_type,
                "sampling_dt": self.cfg.sampling_dt,
                "max steps": self.cfg.max_steps,
            }
            wandb.init(
                project="gmm-evaluation",
                entity="in-ac",
                config=config,
                name=f"{self.cfg.skill}_{gmm.state_type}_{gmm.name}_{gmm.n_components}",
            )

        # Evaluate by simulating in the CALVIN environment
        acc, avg_return, avg_len = self.evaluate(
            gmm,
            val_dataset,
            max_steps=self.cfg.max_steps,
            render=self.cfg.render,
            record=self.cfg.record,
            sampling_dt=self.cfg.sampling_dt,
        )
        skill_accs[self.skill] = [str(acc), str(avg_return), str(avg_len)]
        self.env.count = 0

        if self.cfg.wandb:
            wandb.finish()

        # Log evaluation output
        self.logger.info(f"{self.skill} Skill Accuracy: {round(acc, 2)}")

        # Write accuracies to a file
        with open(os.path.join(self.env.outdir, f"skill_gmm_acc_{self.cfg.state_type}.txt"), "w") as f:
            writer = csv.writer(f)
            for row in skill_accs.items():
                writer.writerow(row)


@hydra.main(version_base="1.1", config_path="../../config", config_name="gmm_eval")
def main(cfg: DictConfig) -> None:
    cfg.log_dir = Path(cfg.log_dir).expanduser()
    cfg.demos_dir = Path(cfg.demos_dir).expanduser()
    cfg.skills_dir = Path(cfg.skills_dir).expanduser()

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
    env.set_outdir(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])

    eval = SkillEvaluator(cfg, env)
    eval.run()


if __name__ == "__main__":
    main()
