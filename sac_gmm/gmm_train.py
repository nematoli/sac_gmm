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

from gmm.gmr.gmm import GMM
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
        logger.info(f'Fitting {self.cfg.gmm_type} GMM with {self.cfg.gmm_components} components')
        gmm = self.gmm(n_components=self.cfg.gmm_components, random_state=self.cfg.seed)
        return gmm.from_samples(X=data)

    def evaluate(self, gmm, env, max_steps=500, num_episodes=10, render=False):
        sampling_dt = 0.2  # increases sampling frequency
        succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
        for episode in tqdm(range(1, num_episodes + 1), desc="Evaluating GMM model"):
            observation = env.reset()
            observation = observation['robot_obs'][:3]
            action = gmm.priors
            episode_return = 0
            for step in range(max_steps):
                cgmm = gmm.condition([0, 1, 2], observation)
                action = action + sampling_dt * cgmm.sample_confidence_region(1, alpha=0.7)[0]
                pdb.set_trace()
                observation, reward, done, info = env.step(action)
                observation = observation['robot_obs'][:3]
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
        train_data = train_data[:, :3]
        dt = 1.0
        X_dot = (train_data[2:] - train_data[:-2]) / dt
        X = train_data[1:-1]
        X_train = np.hstack((X, X_dot))
        fitted_gmm = self.fit_gmm(X_train)

        # Evaluate in Calvin environment
        self.evaluate(fitted_gmm, self.env)

@hydra.main(config_path="../config", config_name="gmm")
def main(cfg: DictConfig) -> None:
    # Run the following once to save the full env config
    # with initialize(config_path="../config/calvin_env/"):
    #     env_cfg = compose(config_name="default.yaml", overrides=["cameras=static_and_gripper"])
    #     env_cfg.env["use_egl"] = False
    #     env_cfg.env["show_gui"] = False
    #     env_cfg.env["use_vr"] = False
    #     env_cfg.env["use_scene_info"] = True
    # pdb.set_trace()
    env = hydra.utils.instantiate(cfg.calvin_env.env)
    #     OmegaConf.save(config=cfg, f='../config/full_default.yaml')
    # initialize(config_path="../config/")
    # cfg = compose(config_name="gmm.yaml")
    trainer = GMMTrainer(cfg, env)
    trainer.run()


if __name__ == '__main__':
    main()