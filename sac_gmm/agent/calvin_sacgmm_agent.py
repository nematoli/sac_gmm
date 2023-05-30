from omegaconf import OmegaConf
import os
import gym
import torch
import numpy as np
from tqdm import tqdm
import logging
import hydra
import copy
from sac_gmm.envs.skill_env import SkillSpecificEnv
from sac_gmm.utils.misc import transform_to_tensor
from sac_gmm.agent.calvin_sac_agent import CALVINSACAgent


class CALVINSACGMMAgent(CALVINSACAgent):
    def __init__(
        self,
        calvin_env,
        dyn_sys,
        dataset,
        skill,
        state_type,
        adapt_cov,
        mean_shift,
        env_obs_allowed,
        device,
        num_train_steps,
        num_seed_steps,
        eval_frequency,
        num_eval_episodes,
        gmm_window_size,
        max_episode_steps,
        record,
        render,
        wandb,
        exp_dir,
        skills_dir,
        demos_dir,
    ) -> None:
        super().__init__(
            calvin_env,
            dataset,
            skill,
            state_type,
            env_obs_allowed,
            device,
            num_train_steps,
            num_seed_steps,
            eval_frequency,
            num_eval_episodes,
            max_episode_steps,
            record,
            render,
            wandb,
            exp_dir,
            skills_dir,
            demos_dir,
        )
        new_cfg = OmegaConf.create(
            {
                "adapt_cov": adapt_cov,
                "mean_shift": mean_shift,
                "gmm_window_size": gmm_window_size,
            }
        )
        self.cfg = OmegaConf.merge(self.cfg, new_cfg)

        # Dynamical System
        self.dyn_sys = hydra.utils.instantiate(dyn_sys)
        self.dyn_sys.model_dir = os.path.join(
            self.cfg.skills_dir, self.cfg.state_type, self.cfg.skill, self.dyn_sys.name
        )
        self.dyn_sys.load_model()
        self.dyn_sys.state_type = self.robot_obs
        self.dyn_sys.dataset = self.dataset
        if "Manifold" in self.dyn_sys.name:
            self.dyn_sys.manifold = self.dyn_sys.make_manifold(self.dyn_sys.state_size)

        # Initial DS
        self.initial_dyn_sys = copy.deepcopy(self.dyn_sys)

        # dt
        self.dt = self.dataset.dt

        # Logging
        self.cons_logger = logging.getLogger("CALVINSACGMMAgent")

    def get_action_space(self):
        if not hasattr(self, "action_space"):
            if self.cfg.mean_shift:
                mu_high = np.ones(self.dyn_sys.mu.shape[0])
                self.action_space = gym.spaces.Box(-mu_high, mu_high)
            else:
                parameter_space = self.dyn_sys.get_update_range_parameter_space()
                priors_high = np.ones(parameter_space["priors"].shape[0])
                mu_high = np.ones(parameter_space["mu"].shape[0])
                action_high = np.concatenate((priors_high, mu_high), axis=-1)
                if self.cfg.adapt_cov:
                    sigma_high = np.ones(parameter_space["sigma"].shape[0])
                    action_high = np.concatenate((action_high, sigma_high), axis=-1)
                action_low = -action_high
                self.action_space = gym.spaces.Box(action_low, action_high)
        return self.action_space

    def update_gaussians(self, gmm_change):
        if self.cfg.mean_shift:
            mu = np.hstack([gmm_change.reshape((self.dyn_sys.means.shape[0], 1)) * 0.01] * self.dyn_sys.means.shape[1])
            change_dict = {"mu": mu}
            self.dyn_sys.update_model(change_dict)
        else:
            parameter_space = self.dyn_sys.get_update_range_parameter_space()
            size_priors = parameter_space["priors"].shape[0]
            size_mu = parameter_space["mu"].shape[0]
            priors = gmm_change[:size_priors] * parameter_space["priors"].high
            mu = gmm_change[size_priors : size_priors + size_mu] * parameter_space["mu"].high
            change_dict = {"mu": mu, "priors": priors}
            if self.cfg.adapt_cov:
                change_dict["sigma"] = gmm_change[size_priors + size_mu :] * parameter_space["sigma"].high
            self.dyn_sys.update_model(change_dict)

    def play_step(self, actor, strategy="stochastic", replay_buffer=None):
        """Perform a step in the environment and add the transition
        tuple to the replay buffer"""
        # Change dynamical system
        self.dyn_sys.copy_model(self.initial_dyn_sys)
        gmm_change = self.get_action(actor, self.observation, strategy)
        self.update_gaussians(gmm_change)

        # Act with the dynamical system in the environment
        dyn_sys_reward = 0
        x = self.observation[self.robot_obs]
        for _ in range(self.cfg.gmm_window_size):
            dx = self.dyn_sys.predict(x - self.dataset.goal)
            new_x = x + self.dt * dx
            env_action = np.append(new_x, np.append(self.fixed_ori, -1))
            env_action = self.env.prepare_action(env_action, type="abs")
            next_observation, reward, done, info = self.env.step(env_action)
            dyn_sys_reward += reward
            x = next_observation[self.robot_obs]
            self.env_steps += 1
            if done:
                break

        replay_buffer.add(
            self.observation,
            gmm_change,
            dyn_sys_reward,
            next_observation,
            done,
        )
        self.observation = next_observation

        self.steps += 1
        self.episode_steps += 1
        if done or (self.episode_steps >= self.cfg.max_episode_steps):
            self.reset()
            done = True
        return reward, done

    @torch.no_grad()
    def evaluate(self, actor):
        """Evaluates the actor in the environment"""
        self.cons_logger.info("Evaluation episodes in process")
        succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
        saved_video_path = None
        # Choose a random episode to record
        rand_idx = np.random.randint(0, self.cfg.num_eval_episodes)
        for episode in tqdm(range(1, self.cfg.num_eval_episodes + 1)):
            episode_steps = 0
            episode_return = 0
            self.observation = self.env.reset()
            # Start from a known starting point
            self.observation = self.calibrate_EE_start_position()
            # Recording setup
            if self.cfg.record and (episode == rand_idx):
                self.env.reset_recorded_frames()
                self.env.record_frame(size=64)
            x = self.observation[self.robot_obs]
            while episode_steps < self.cfg.max_episode_steps:
                # Change dynamical system
                self.dyn_sys.copy_model(self.initial_dyn_sys)
                gmm_change = self.get_action(actor, self.observation, "deterministic")
                self.update_gaussians(gmm_change)

                # Act with the dynamical system in the environment
                dyn_sys_reward = 0
                for _ in range(self.cfg.gmm_window_size):
                    dx = self.dyn_sys.predict(x - self.dataset.goal)
                    new_x = x + self.dt * dx
                    env_action = np.append(new_x, np.append(self.fixed_ori, -1))
                    env_action = self.env.prepare_action(env_action, type="abs")
                    next_observation, reward, done, info = self.env.step(env_action)
                    dyn_sys_reward += reward
                    x = next_observation[self.robot_obs]
                    if done:
                        break
                episode_steps += 1
                episode_return += dyn_sys_reward
                if self.cfg.record and (episode == rand_idx):
                    self.env.record_frame(size=64)
                if done:
                    self.reset()
                    break
            if ("success" in info) and info["success"]:
                succesful_episodes += 1
            # Recording setup close
            if self.cfg.record and (episode == rand_idx):
                video_path = self.env.save_recorded_frames(
                    outdir=self.video_dir,
                    fname=f"{self.steps}_{self.env_steps}_{episode}",
                )
                self.env.reset_recorded_frames()
                saved_video_path = video_path

            episodes_returns.append(episode_return)
            episodes_lengths.append(episode_steps)
        accuracy = succesful_episodes / self.cfg.num_eval_episodes

        return (
            accuracy,
            np.mean(episodes_returns),
            np.mean(episodes_lengths),
            saved_video_path,
        )
