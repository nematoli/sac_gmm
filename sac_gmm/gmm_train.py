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
        if cfg.gmm_type == 'base':
            self.gmm = GMM
        else: 
            raise Exception('GMM type not supported')

    def fit_gmm(self, data):
        logger.info(f'Fitting Bayesian GM with {self.cfg.gmm_components} components for good priors')
        bgmm = BayesianGaussianMixture(n_components=self.cfg.gmm_components, max_iter=500,
                                        random_state=self.cfg.seed).fit(data)
        logger.info(f'Fitting {self.cfg.gmm_type} GMM with {self.cfg.gmm_components} components')
        gmm = self.gmm(n_components=self.cfg.gmm_components, priors=bgmm.weights_, 
                        means=bgmm.means_, covariances=bgmm.covariances_, 
                        random_state=self.cfg.seed)
        return gmm.from_samples(X=data)

    def evaluate(self, gmm, env, init_poses, max_steps=500, num_episodes=5, render=False):
        sampling_dt = 0.2  # increases sampling frequency
        succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
        for episode in tqdm(range(1, num_episodes + 1), desc="Evaluating GMM model"):
            observation = env.reset()
            pose = init_poses[episode%len(init_poses)]
            action = np.array([pose, np.zeros(3), -1], dtype=object)
            observation, reward, done, info = env.step(action)
            observation = observation[:3]
            episode_return = 0
            for step in range(max_steps):
                cgmm = gmm.condition([0, 1, 2], observation)
                pose = pose + sampling_dt * cgmm.sample_confidence_region(1, alpha=0.7)[0]
                action = np.append(pose, np.append(np.zeros(3), -1))
                observation, reward, done, info = env.step(action)
                observation = observation[:3]
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
        return accuracy, np.mean(episodes_returns), np.mean(episodes_lengths)


    def run(self):
        # Extract demonstrations
        if self.cfg.use_existing_demos:
            print(Path().absolute())
            print((Path(self.cfg.demos_dir) / self.cfg.skill / 'train.npy').is_file())
            if (Path(self.cfg.demos_dir) / self.cfg.skill / 'train.npy').is_file() &\
                            (Path(self.cfg.demos_dir) / self.cfg.skill / 'val.npy').is_file():
                logger.info(f'Using demonstration available at {self.cfg.demos_dir}/{self.cfg.skill}')
            else:
                raise Exception(f'Missing demonstrations at {Path(self.cfg.demos_dir) / self.cfg.skill}')
        else:
            extract_demos(self.cfg)

        # Train a GMM
        train_data = np.load(Path(self.cfg.demos_dir) / self.cfg.skill / 'train.npy')
        train_data = train_data.reshape(1, -1, train_data.shape[-1]).squeeze(0)
        val_data = np.load(Path(self.cfg.demos_dir) / self.cfg.skill / 'val.npy')
        # Get velocities from pose
        train_data = train_data[:, 1:4]
        dt = 1.0
        X_dot = (train_data[2:] - train_data[:-2]) / dt
        X = train_data[1:-1]
        X_train = np.hstack((X, X_dot))
        fitted_gmm = self.fit_gmm(X_train)

        # Evaluate in Calvin environment
        train_data = np.load(Path(self.cfg.demos_dir) / self.cfg.skill / 'train.npy')
        init_poses = train_data[:, 0, 1:4]
        acc, ep_returns, ep_lens = self.evaluate(gmm=fitted_gmm, env=self.env, init_poses=init_poses)
        logger.info(f'Evaluation Results - Accuracy: {acc}, Avg. Episode Returns: {ep_returns}, Avg. Episode Lenths: {ep_lens}')

@hydra.main(config_path="../config", config_name="gmm")
def main(cfg: DictConfig) -> None:
    new_env_cfg = {**cfg.calvin_env.env}
    new_env_cfg["tasks"] = cfg.calvin_env.tasks
    new_env_cfg.pop('_target_', None)
    new_env_cfg.pop('_recursive_', None)
    env = TurnOffBulbEnv(**new_env_cfg)

    trainer = GMMTrainer(cfg, env)
    trainer.run()


if __name__ == '__main__':
    main()