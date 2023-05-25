#!/usr/bin/env python3
import numpy as np
import torch
import os
import time
from pathlib import Path

from envs.skill_env import SkillSpecificEnv
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer
from utils.misc import set_seed_everywhere, torch_tensor_cam_obs, preprocess_agent_in, postprocess_agent_out
import copy
import gym

import hydra

import pdb
import logging

logging.basicConfig(level=logging.DEBUG, filename="SAC_GMM_TRAIN.log", filemode="w")


class SACGMMTrainer(object):
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.env.set_skill(self.cfg.skill)
        self.env.set_state_type(self.cfg.state_type)
        self.env.set_outdir(self.cfg.exp_dir)
        self.env.obs_allowed = self.cfg.obs_allowed
        self.robot_obs, self.cam_obs = self.env.obs_allowed

        self.exp_dir = self.cfg.exp_dir

        self.logger = logging.getLogger(self.__class__.__name__)
        self.stats_logger = Logger(
            self.exp_dir, save_tb=cfg.log_save_tb, log_frequency=cfg.log_frequency, agent=cfg.name
        )

        set_seed_everywhere(self.cfg.seed)
        self.device = torch.device(self.cfg.device)

        self.dyn_sys = hydra.utils.instantiate(self.cfg.dyn_sys)
        self.dyn_sys.model_dir = os.path.join(
            self.cfg.skills_dir, self.cfg.state_type, self.cfg.skill, self.dyn_sys.name
        )
        self.dyn_sys.load_model()
        self.dyn_sys.state_type = self.robot_obs
        self.dyn_sys.dataset = hydra.utils.instantiate(self.cfg.dataset)
        if "Manifold" in self.dyn_sys.name:
            self.dyn_sys.manifold = self.dyn_sys.make_manifold(self.dyn_sys.state_size)

        # Sample some initial points to start experiments from there
        self.initial_pos = self.dyn_sys.sample_starts(size=20, scale=0.1)
        self.initial_dyn_sys = copy.deepcopy(self.dyn_sys)

        # Set env obs spaces, SACGMM agent's state and action spaces
        self.env.obs_space = self.env.get_obs_space()
        param_space = self.dyn_sys.get_update_range_parameter_space()
        action_space = self.get_action_space(param_space)

        # SACGMM obs space = robots_obs_dim + cam_obs_latent_dim + gmm_params_dim
        gmm_params_dim = action_space.shape[0]
        robot_obs_dim = self.dyn_sys.state_size
        cam_obs_latent_dim = self.cfg.agent.autoencoder.hidden_dim
        self.cfg.agent.obs_dim = gmm_params_dim + robot_obs_dim + cam_obs_latent_dim
        self.cfg.agent.action_dim = action_space.shape[0]
        self.cfg.agent.action_range = [float(action_space.low.min()), float(action_space.high.max())]

        # Set autoencoder params
        # cam_obs_shape = self.env.obs_space[cam_obs].shape
        self.cfg.agent.autoencoder.hd_input_space = (64, 64)  # networks work with only 64 now
        self.cfg.agent.autoencoder.in_channels = 1  # grayscale image

        self.agent = hydra.utils.instantiate(self.cfg.agent)

        self.replay_buffer = ReplayBuffer(
            (self.agent.obs_dim,),
            (1, 64, 64),
            action_space.shape,
            int(self.cfg.replay_buffer_capacity),
            self.device,
        )

        self.step = 0
        self.robot_obs_dim = robot_obs_dim
        if self.cfg.record:
            self.video_dir = os.path.join(self.exp_dir, "videos")
            os.makedirs(self.video_dir, exist_ok=True)

    def get_action_space(self, param_space):
        priors_high = np.ones(param_space["priors"].shape[0])
        means_high = np.ones(param_space["mu"].shape[0])
        action_high = np.concatenate((priors_high, means_high), axis=-1)
        if self.cfg.adapt_cov:
            sigma_high = np.ones(param_space["sigma"].shape[0])
            action_high = np.concatenate((action_high, sigma_high), axis=-1)
        action_low = -action_high
        return gym.spaces.Box(action_low, action_high)

    def evaluate(self):
        self.logger.info("Evaluating GMM on Validation Trajectory Set!")
        total_reward = 0
        total_steps = 0
        episode_step = 0

        init_pos = self.dyn_sys.sample_starts(size=self.cfg.num_eval_episodes, scale=0.1)
        rand_idx = np.random.randint(0, self.cfg.num_eval_episodes)  # rand episode to be recorded
        for p, i in enumerate(init_pos):
            self.env.reset()
            temp = np.append(p, np.append(self.dyn_sys.dataset.fixed_ori, -1))
            action = self.env.prepare_action(temp, type="abs")
            error_margin = 0.01
            check_count = 0
            while True:
                curr_obs, _, _, _ = self.env.step(action)
                if np.linalg.norm(curr_obs[self.robot_obs][: self.robot_obs_dim] - p) < error_margin:
                    break
                check_count += 1
                if check_count > 15:
                    break
            step = 0
            done = False

            if self.cfg.record and (i == rand_idx):
                self.env.reset_recorded_frames()
                self.env.record_frame(size=64)

            if not done and episode_step < self.cfg.gmm_max_episode_steps:
                while step < self.cfg.gmm_window_size:
                    d_x = self.dyn_sys.predict(
                        curr_obs[self.robot_obs][: self.robot_obs_dim] - self.dyn_sys.dataset.goal
                    )
                    delta_x = self.cfg.sampling_dt * d_x[: self.robot_obs_dim]
                    new_x = curr_obs[self.robot_obs][: self.robot_obs_dim] + delta_x
                    if self.robot_obs == "pos":
                        temp = np.append(new_x, np.append(self.dyn_sys.dataset.fixed_ori, -1))
                    action = self.env.prepare_action(temp, type="abs")
                    curr_obs, reward, done, info = self.env.step(action)
                    total_reward += reward
                    step += 1
                    episode_step += 1

                    if self.cfg.record and (i == rand_idx):
                        self.env.record_frame(size=64)

                    if done:
                        total_reward += 1
                        total_steps += step
                        break
            if self.cfg.record:
                video_path = self.env.save_recorded_frames(outdir=self.video_dir, fname=f"{self.step}_{i}")
                self.env.reset_recorded_frames()
            episode_step = 0
        self.logger.info(
            f"Evaluation Results - Accuracy: {total_reward/self.cfg.num_eval_episodes}, Avg. Episode Lenths: {total_steps/self.cfg.num_eval_episodes}"
        )
        self.stats_logger.log("eval/episode_reward", total_reward, self.step)
        self.stats_logger.dump(self.step)

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
                rand_idx = int(np.random.choice(np.arange(0, len(self.initial_pos)), 1))
                x0 = self.initial_pos[rand_idx]
                temp = np.append(x0, np.append(self.dyn_sys.dataset.fixed_ori, -1))
                action = self.env.prepare_action(temp, type="abs")
                # This makes sure the EE is close to initial position
                error_margin = 0.01
                check_count = 0
                while True:
                    curr_obs, _, _, _ = self.env.step(action)
                    if (
                        np.linalg.norm(curr_obs[self.robot_obs][: self.robot_obs_dim] - x0[: self.robot_obs_dim])
                        < error_margin
                    ):
                        break
                    check_count += 1
                    if check_count > 15:
                        break
                obs = curr_obs
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                self.logger.info(f"Episode {episode}, Step {episode_step}:")
                self.stats_logger.log("train/episode", episode, self.step)

            # Change GMM
            self.dyn_sys.copy_model(self.initial_dyn_sys)
            agent_in = preprocess_agent_in(self, obs)
            gmm_change_raw = self.agent.act(agent_in, sample=True)
            gmm_change_dict = postprocess_agent_out(self, gmm_change_raw, priors_scale=0.1, means_scale=0.1)
            self.dyn_sys.update_model(gmm_change_dict)

            # Act with GMM in the world
            dynsys_reward = 0
            step = 0
            curr_obs = obs
            while step < self.cfg.gmm_window_size:
                d_x = self.dyn_sys.predict(curr_obs[self.robot_obs][: self.robot_obs_dim] - self.dyn_sys.dataset.goal)
                delta_x = self.cfg.sampling_dt * d_x[: self.robot_obs_dim]
                new_x = curr_obs[self.robot_obs][: self.robot_obs_dim] + delta_x
                if self.robot_obs == "pos":
                    temp = np.append(new_x, np.append(self.dyn_sys.dataset.fixed_ori, -1))
                action = self.env.prepare_action(temp, type="abs")
                curr_obs, reward, done, info = self.env.step(action)
                dynsys_reward += reward
                step += 1
                if done:
                    break
            self.step += step
            episode_step += step
            # Store GMM experience in replay buffer
            # Prepare and store agent observation and next observation vectors
            agent_obs = agent_in.cpu().numpy()
            agent_next_obs = preprocess_agent_in(self, curr_obs).cpu().numpy()
            # Get current camera observations (image)
            cam_obs_tensor = torch_tensor_cam_obs(curr_obs[self.cam_obs]).cpu().numpy()

            # Allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env.max_episode_steps else done
            episode_reward += dynsys_reward
            self.replay_buffer.add(
                agent_obs, cam_obs_tensor, gmm_change_raw, dynsys_reward, agent_next_obs, done, done_no_max
            )
            # Run agent training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.stats_logger, self.step)

            obs = curr_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path="../config", config_name="sac_gmm_train")
def main(cfg):
    cfg.log_dir = Path(cfg.log_dir).expanduser()
    cfg.demos_dir = Path(cfg.demos_dir).expanduser()
    cfg.skills_dir = Path(cfg.skills_dir).expanduser()
    cfg.exp_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]).expanduser()

    new_env_cfg = {**cfg.calvin_env.env}
    new_env_cfg["use_egl"] = False
    new_env_cfg["show_gui"] = False
    new_env_cfg["use_vr"] = False
    new_env_cfg["use_scene_info"] = True
    new_env_cfg["tasks"] = cfg.calvin_env.tasks
    new_env_cfg.pop("_target_", None)
    new_env_cfg.pop("_recursive_", None)

    env = SkillSpecificEnv(**new_env_cfg)

    trainer = SACGMMTrainer(cfg, env)
    trainer.run()


if __name__ == "__main__":
    main()
