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
from tqdm import tqdm

from envs.skill_env import SkillSpecificEnv
from gmm.utils.utils import plot_3d_trajectories
from utils.video import VideoRecorder
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer
from utils.misc import set_seed_everywhere
from utils.transforms import PreprocessImage
from extract_demos import extract_demos
import gym
import utils

import hydra

import pdb
import logging

logging.basicConfig(level=logging.DEBUG, filename="SAC_GMM_TRAIN.log", filemode="w")


class SACGMMTrainer(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        # pdb.set_trace()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.stats_logger = Logger(
            self.work_dir, save_tb=cfg.log_save_tb, log_frequency=cfg.log_frequency, agent=cfg.name
        )

        set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        new_env_cfg = {**cfg.calvin_env.env}
        new_env_cfg["tasks"] = cfg.calvin_env.tasks
        new_env_cfg.pop("_target_", None)
        new_env_cfg.pop("_recursive_", None)
        self.env = SkillSpecificEnv(**new_env_cfg)
        self.env.set_skill(cfg.skill)
        self.env.max_episode_steps = cfg.gmm_max_episode_steps
        # self.env = hydra.utils.instantiate(cfg.calvin_env.env)

        self.dyn_sys = hydra.utils.instantiate(cfg.dyn_sys.cfg)
        self.logger.info("Fitting GMM on Training Trajectory Set!")
        self.initial_points = self.fit_gmm()
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
        cfg.agent.obs_dim = gmm_params_dim + pose_dim + latent_dim
        cfg.agent.action_dim = self.env.action_space.shape[0]
        cfg.agent.action_range = [float(self.env.action_space.low.min()), float(self.env.action_space.high.max())]
        # Set autoencoder params
        cam_obs_shape = PreprocessImage()(self.env.get_camera_obs()["rgb_obs"]["rgb_gripper"]).squeeze(0).shape
        cfg.agent.autoencoder.hd_input_space = (cam_obs_shape[1], cam_obs_shape[2])
        cfg.agent.autoencoder.in_channels = cam_obs_shape[0]

        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(
            (cfg.agent.obs_dim,),
            cam_obs_shape[1:],
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device,
        )

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        self.logger.info("Evaluating GMM on Validation Trajectory Set!")
        total_reward = 0
        total_steps = 0
        val_data = self.get_skill_data(val=True)
        init_idx = np.arange(0, val_data.shape[0], self.cfg.skill_length - 2)
        init_poses = val_data[init_idx]
        obs = self.env.reset()
        for init_pose in init_poses:
            action = np.array([init_pose[:3], obs[3:6], obs[-1]], dtype=object)
            error_margin = 0.005
            while True:
                curr_obs, _, _, _ = self.env.step(action)
                if np.linalg.norm(curr_obs[:3] - init_pose[:3]) < error_margin:
                    break
            step = 0
            while step < self.cfg.gmm_max_episode_steps:
                action = self.dyn_sys.predict_action(curr_obs, relative=True)
                curr_obs, reward, done, info = self.env.step(action)
                step += 1
                if done:
                    total_reward += 1
                    total_steps += step
                    break
        self.logger.info(
            f"Evaluation Results - Accuracy: {total_reward/val_data.shape[0]}, Avg. Episode Lenths: {total_steps/val_data.shape[0]}"
        )
        self.stats_logger.log("eval/episode_reward", total_reward, self.step)
        self.stats_logger.dump(self.step)

    def prepare_agent_input(self, obs):
        pose = obs[:3]
        image = PreprocessImage()(self.env.get_camera_obs()["rgb_obs"]["rgb_gripper"])
        gmm_params = self.dyn_sys.get_params()
        agent_obs = {"gmm_params": gmm_params, "pose": pose, "image": image}
        return agent_obs

    def prepare_agent_output(self, gmm_change, priors_scale=0.1, means_scale=0.1):
        dict = {}
        param_space = self.dyn_sys.get_params_range()
        priors_size = param_space["priors"].shape[0]
        priors_change, means_change = np.split(gmm_change, [priors_size])
        priors_change *= priors_scale
        means_change *= means_scale

        dict["priors"] = priors_change
        dict["means"] = means_change

        vect = np.concatenate((priors_change, means_change), axis=-1)
        return dict, vect

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                # Log training duration
                if self.step > 0:
                    self.stats_logger.log("train/duration", time.time() - start_time, self.step)
                    start_time = time.time()
                    self.stats_logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                # Evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.stats_logger.log("eval/episode", episode, self.step)
                    self.evaluate()

                self.stats_logger.log("train/episode_reward", episode_reward, self.step)

                obs = self.env.reset()
                # First absolute action step
                rand_idx = int(np.random.choice(np.arange(0, len(self.initial_points)), 1))
                target_obs = self.initial_points[rand_idx]
                action = np.array([target_obs[:3], obs[3:6], obs[-1]], dtype=object)
                # This makes sure the EE is close to initial position
                error_margin = 0.005
                while True:
                    curr_obs, _, _, _ = self.env.step(action)
                    if np.linalg.norm(curr_obs[:3] - target_obs[:3]) < error_margin:
                        break
                obs = curr_obs
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                self.logger.info(f"Episode {episode}, Step {episode_step}: Begins")
                self.stats_logger.log("train/episode", episode, self.step)
            # Change GMM
            self.dyn_sys.copy_from(self.initial_dyn_sys)
            agent_obs = self.prepare_agent_input(obs)
            gmm_change_raw = self.agent.act(agent_obs, sample=True)
            gmm_change_dict, gmm_change_vect = self.prepare_agent_output(
                gmm_change_raw, priors_scale=0.1, means_scale=0.1
            )
            self.dyn_sys.update_gmm(gmm_change_dict)

            # Act with GMM in the world
            ds_reward = 0
            step = 0
            while step < self.cfg.gmm_window_size:
                action = self.dyn_sys.predict_action(curr_obs, relative=True)
                curr_obs, reward, done, info = self.env.step(action)
                ds_reward += reward
                step += 1
                if done:
                    break
            self.step += step
            episode_step += step
            # Store GMM experience in replay buffer
            # Prepare and store agent observation vectors
            obs = self.agent.prepare_obs(self.prepare_agent_input(obs)).cpu().numpy()
            next_obs = self.agent.prepare_obs(self.prepare_agent_input(curr_obs)).cpu().numpy()
            # Get current and next state camera observations (images)
            cam_obs = (
                PreprocessImage()(self.env.get_camera_obs()["rgb_obs"]["rgb_gripper"]).squeeze(0).squeeze(0).numpy()
            )
            next_cam_obs = (
                PreprocessImage()(self.env.get_camera_obs()["rgb_obs"]["rgb_gripper"]).squeeze(0).squeeze(0).numpy()
            )

            # Allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env.max_episode_steps else done
            episode_reward += ds_reward
            self.replay_buffer.add(obs, cam_obs, gmm_change_vect, ds_reward, next_obs, next_cam_obs, done, done_no_max)
            # Run agent training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.stats_logger, self.step)

            obs = next_obs
            episode_step += 1
            self.step += 1

    def fit_gmm(self, plot=False):
        X_train = self.get_skill_data(val=False)
        # plot_3d_trajectories(train_data[:, :, 1:4])
        init_idx = np.arange(0, X_train.shape[0], self.cfg.skill_length - 2)
        init_poses = X_train[init_idx]
        self.logger.info(
            f"GMM Fit: Fitting Bayesian GM with {self.cfg.dyn_sys.gmm_components} components for good priors"
        )
        bgmm = self.dyn_sys.fit_bgmm(
            X=X_train, n_components=self.cfg.dyn_sys.gmm_components, max_iter=500, random_state=self.cfg.seed
        )
        self.logger.info(
            f"GMM Fit: Fitting {self.cfg.dyn_sys.gmm_type} GMM with {self.cfg.dyn_sys.gmm_components} components"
        )
        self.dyn_sys.copy_from_bgm(bgmm)
        self.dyn_sys.from_samples(X=X_train)

        if self.cfg.dyn_sys.plot_gmr:
            sampled_path = []
            x = init_poses[0, :3]
            sampling_dt = 1 / 30
            for t in range(self.cfg.skill_length):
                sampled_path.append(x)
                cgmm = self.dyn_sys.condition([0, 1, 2], x)
                x_dot = cgmm.sample_confidence_region(1, alpha=0.7).reshape(-1)
                x = x + sampling_dt * x_dot
            sampled_path = np.array(sampled_path)
            plot_3d_trajectories(X_train[:, :3], sampled_path)

        return init_poses

    def get_skill_data(self, val=False):
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

        if val:
            dataset = np.load(demos_dir / self.cfg.skill / "val.npy")
        else:
            dataset = np.load(demos_dir / self.cfg.skill / "train.npy")

        final_set = None
        for data in dataset:
            data = data[:, 1:4]
            dt = 2 / 30
            X_dot = (data[2:] - data[:-2]) / dt
            X = data[1:-1]
            if final_set is None:
                final_set = np.hstack((X, X_dot))
            else:
                final_set = np.append(final_set, np.hstack((X, X_dot)), axis=0)
        return final_set


@hydra.main(config_path="../config", config_name="sac_gmm_train")
def main(cfg):
    trainer = SACGMMTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
