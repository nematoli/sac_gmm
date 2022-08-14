#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
from pathlib import Path

from envs.turn_off_bulb import TurnOffBulbEnv
from gmm.utils.utils import plot_3d_trajectories
from utils.video import VideoRecorder
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer
from utils.misc import set_seed_everywhere
from extract_demos import extract_demos
import gym
import utils

import hydra

import pdb
import logging

logging.basicConfig(level=logging.DEBUG, filename=f"{__name__}.log", filemode="w")


class SACGMMTrainer(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        # pdb.set_trace()
        self.logger = logging.getLogger(__name__)

        self.stats_logger = Logger(
            self.work_dir, save_tb=cfg.log_save_tb, log_frequency=cfg.log_frequency, agent=cfg.name
        )

        set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        new_env_cfg = {**cfg.calvin_env.env}
        new_env_cfg["tasks"] = cfg.calvin_env.tasks
        new_env_cfg.pop("_target_", None)
        new_env_cfg.pop("_recursive_", None)
        self.env = TurnOffBulbEnv(**new_env_cfg)
        # self.env = hydra.utils.instantiate(cfg.calvin_env.env)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape, self.env.action_space.shape, int(cfg.replay_buffer_capacity), self.device
        )

        self.dyn_sys = hydra.utils.instantiate(cfg.dyn_sys.cfg)
        self.fit_gmm()
        self.initial_dyn_sys = self.dyn_sys

        # Set agent observation and action spaces appropriately
        latent_dim = cfg.agent.autoencoder.hidden_dim
        pose_dim = 3
        param_space = self.dyn_sys.get_params_range()
        priors_high = np.ones(param_space["priors"].shape[0])
        means_high = np.ones(param_space["means"].shape[0])
        action_high = np.concatenate((priors_high, means_high), axis=-1)
        action_low = -action_high
        self.env.action_space = gym.spaces.Box(action_low, action_high)
        gmm_params_dim = param_space["priors"].shape[0] + param_space["means"].shape[0]
        cfg.agent.agent.obs_dim = gmm_params_dim + pose_dim + latent_dim
        cfg.agent.agent.action_dim = self.env.action_space.shape[0]
        cfg.agent.agent.action_range = [float(self.env.action_space.low.min()), float(self.env.action_space.high.max())]
        pdb.set_trace()
        self.agent = hydra.utils.instantiate(cfg.agent.agent)

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(f"{self.step}.mp4")
        average_episode_reward /= self.cfg.num_eval_episodes
        self.stats_logger.log("eval/episode_reward", average_episode_reward, self.step)
        self.stats_logger.dump(self.step)

    def run(self):
        self.dyn_sys.copy_model(self.initial_dyn_sys)
        obs = self.env.reset()
        pdb.set_trace()
        gmm_change = self.agent.act(obs, sample=True)
        self.dyn_sys.update_gmm(gmm_change)

        step = 0
        episode_reward = 0
        curr_observation = 0
        while step <= self.cfg.gmm_window_size:
            vel = self.dyn_sys.predict_vel_from_obs(curr_observation)
            curr_observation, reward, done, info = self.env.step(vel)
            episode_reward += reward
            if done:
                break

        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.stats_logger.log("train/duration", time.time() - start_time, self.step)
                    start_time = time.time()
                    self.stats_logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.stats_logger.log("eval/episode", episode, self.step)
                    self.evaluate()

                self.stats_logger.log("train/episode_reward", episode_reward, self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.stats_logger.log("train/episode", episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.stats_logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

    def fit_gmm(self, plot=False):
        demos_dir = Path(self.cfg.demos_dir).expanduser()
        # Extract demonstrations
        if self.cfg.use_existing_demos:
            print((demos_dir / self.cfg.skill / "train.npy").is_file())
            if (demos_dir / self.cfg.skill / "train.npy").is_file() & (
                demos_dir / self.cfg.skill / "val.npy"
            ).is_file():
                self.logger.info(f"Using demonstration available at {demos_dir}/{self.cfg.skill}")
            else:
                raise Exception(f"Missing demonstrations at {demos_dir / self.cfg.skill}")
        else:
            extract_demos(self.cfg)

        train_data = np.load(demos_dir / self.cfg.skill / "train.npy")[:1]
        # plot_3d_trajectories(train_data[:, :, 1:4])
        init_poses = train_data[:, 0, 1:7]
        train_data = train_data.reshape(1, -1, train_data.shape[-1]).squeeze(0)
        val_data = np.load(demos_dir / self.cfg.skill / "val.npy")
        # Get velocities from pose
        train_data = train_data[:, 1:4]
        dt = 2 / 30
        X_dot = (train_data[2:] - train_data[:-2]) / dt
        X = train_data[1:-1]
        X_train = np.hstack((X, X_dot))

        self.logger.info(f"Fitting Bayesian GM with {self.cfg.dyn_sys.gmm_components} components for good priors")
        bgmm = self.dyn_sys.fit_bgmm(
            X=X_train, n_components=self.cfg.dyn_sys.gmm_components, max_iter=500, random_state=self.cfg.seed
        )
        self.logger.info(f"Fitting {self.cfg.dyn_sys.gmm_type} GMM with {self.cfg.dyn_sys.gmm_components} components")
        self.dyn_sys.copy_bgm_model(bgmm)
        self.dyn_sys.from_samples(X=X_train)

        if self.cfg.dyn_sys.plot_gmr:
            sampled_path = []
            x = init_poses[0, :3]
            sampling_dt = 1 / 30
            for t in range(64):
                sampled_path.append(x)
                cgmm = self.dyn_sys.condition([0, 1, 2], x)
                x_dot = cgmm.sample_confidence_region(1, alpha=0.7).reshape(-1)
                x = x + sampling_dt * x_dot
            sampled_path = np.array(sampled_path)
            plot_3d_trajectories(X, sampled_path)

        return init_poses


@hydra.main(config_path="../config", config_name="sac_gmm_train")
def main(cfg):
    trainer = SACGMMTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
