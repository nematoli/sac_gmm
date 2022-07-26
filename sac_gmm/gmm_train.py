from distutils.command.config import config
import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm

import numpy as np
import hydra
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

from extract_demos import extract_demos
from utils.env_maker import make_env

from sklearn.mixture import BayesianGaussianMixture
from gmm import GMM
from envs.turn_off_bulb import TurnOffBulbEnv
from gmm.utils.utils import plot_3d_trajectories

# from . import ModifiedGMM

import pdb

logger = logging.getLogger(__name__)


class GMMTrainer(object):
    """Python wrapper that allows you to learn a GMM on a specific
    skill demonstrations extracted from CALVIN dataset.
    """

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.gmm = None
        if cfg.gmm_type == "base":
            self.gmm = GMM
        else:
            raise Exception("GMM type not supported")

    def fit_gmm(self, data_dir):
        train_data = np.load(data_dir / self.cfg.skill / "train.npy")[:1]
        # plot_3d_trajectories(train_data[:, :, 1:4])
        init_poses = train_data[:, 0, 1:7]
        train_data = train_data.reshape(1, -1, train_data.shape[-1]).squeeze(0)
        val_data = np.load(data_dir / self.cfg.skill / "val.npy")
        # Get velocities from pose
        train_data = train_data[:, 1:4]
        dt = 2 / 30
        X_dot = (train_data[2:] - train_data[:-2]) / dt
        X = train_data[1:-1]
        X_train = np.hstack((X, X_dot))

        logger.info(f"Fitting Bayesian GM with {self.cfg.gmm_components} components for good priors")
        bgmm = BayesianGaussianMixture(
            n_components=self.cfg.gmm_components, max_iter=500, random_state=self.cfg.seed
        ).fit(X_train)
        logger.info(f"Fitting {self.cfg.gmm_type} GMM with {self.cfg.gmm_components} components")
        gmm = self.gmm(
            n_components=self.cfg.gmm_components,
            priors=bgmm.weights_,
            means=bgmm.means_,
            covariances=bgmm.covariances_,
            random_state=self.cfg.seed,
        )
        gmm.from_samples(X=X_train)

        if self.cfg.plot_gmr:
            sampled_path = []
            x = init_poses[0, :3]
            sampling_dt = 1 / 30
            for t in range(64):
                sampled_path.append(x)
                cgmm = gmm.condition([0, 1, 2], x)
                x_dot = cgmm.sample_confidence_region(1, alpha=0.7).reshape(-1)
                x = x + sampling_dt * x_dot
            sampled_path = np.array(sampled_path)
            plot_3d_trajectories(X, sampled_path)

        return gmm, init_poses

    def evaluate(self, gmm, env, init_poses, max_steps=500, num_episodes=5, render=False):
        sampling_dt = 1 / 30  # increases sampling frequency
        succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
        for episode in tqdm(range(1, num_episodes + 1), desc="Evaluating GMM model"):
            observation = env.reset()
            current_pose = observation[:6]
            init_pose = init_poses[episode % len(init_poses)]
            init_pos, init_orn = init_pose[:3], init_pose[3:]
            action = np.array([init_pos, init_orn, -1], dtype=object)
            while np.linalg.norm(current_pose - init_pose) > 0.005:
                observation, reward, done, info = env.step(action)
                current_pose = observation[:6]
            current_pos = current_pose[:3]
            episode_return = 0
            for step in range(max_steps):
                cgmm = gmm.condition([0, 1, 2], current_pos)
                delta_pos = sampling_dt * cgmm.sample_confidence_region(1, alpha=0.7)[0]
                action = np.append(delta_pos, np.append(np.zeros(3), -1))
                observation, reward, done, info = env.step(action)
                current_pos = observation[:3]
                episode_return += reward
                if render:
                    env.render()
                if done:
                    break
            if info["success"]:
                succesful_episodes += 1
            episodes_returns.append(episode_return)
            episodes_lengths.append(step)
        accuracy = succesful_episodes / num_episodes

        logger.info(
            f"Evaluation Results - Accuracy: {accuracy}, Avg. Episode Returns: {np.mean(episodes_returns)}, Avg. Episode Lenths: {np.mean(episodes_lengths)}"
        )

    def run(self):
        demos_dir = Path(self.cfg.demos_dir).expanduser()
        # Extract demonstrations
        if self.cfg.use_existing_demos:
            print((demos_dir / self.cfg.skill / "train.npy").is_file())
            if (demos_dir / self.cfg.skill / "train.npy").is_file() & (
                demos_dir / self.cfg.skill / "val.npy"
            ).is_file():
                logger.info(f"Using demonstration available at {demos_dir}/{self.cfg.skill}")
            else:
                raise Exception(f"Missing demonstrations at {demos_dir / self.cfg.skill}")
        else:
            extract_demos(self.cfg)

        # Train a GMM
        fitted_gmm, init_poses = self.fit_gmm(demos_dir)

        # Evaluate in Calvin environment
        self.evaluate(gmm=fitted_gmm, env=self.env, init_poses=init_poses, render=self.cfg.render)


@hydra.main(config_path="../config", config_name="gmm")
def main(cfg: DictConfig) -> None:
    new_env_cfg = {**cfg.calvin_env.env}
    new_env_cfg["tasks"] = cfg.calvin_env.tasks
    new_env_cfg.pop("_target_", None)
    new_env_cfg.pop("_recursive_", None)
    env = TurnOffBulbEnv(**new_env_cfg)

    trainer = GMMTrainer(cfg, env)
    trainer.run()


if __name__ == "__main__":
    main()
