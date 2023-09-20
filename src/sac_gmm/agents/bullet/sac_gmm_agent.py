import logging
import hydra
from omegaconf import DictConfig
import os
import gym
import torch
import numpy as np
from tqdm import tqdm
import copy
from pytorch_lightning.utilities import rank_zero_only
from sac_gmm.agents.bullet.bullet_agent import BulletAgent

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class SACGMMAgent(BulletAgent):
    def __init__(
        self,
        name: str,
        bullet_env: DictConfig,
        datamodule: DictConfig,
        num_init_steps: int,
        num_eval_episodes: int,
        skill: DictConfig,
        gmm: DictConfig,
        encoder: DictConfig,
        kp_mock: DictConfig,
        priors_change_range: float,
        mu_change_range: float,
        adapt_per_episode: int,
        render: bool,
    ) -> None:
        super(SACGMMAgent, self).__init__(
            name=name,
            env=bullet_env,
            datamodule=datamodule,
            num_init_steps=num_init_steps,
            num_eval_episodes=num_eval_episodes,
            skill=skill,
            gmm=gmm,
            encoder=encoder,
            kp_mock=kp_mock,
            render=render,
        )

        # GMM refine setup
        self.priors_change_range = priors_change_range
        self.mu_change_range = mu_change_range
        self.adapt_per_episode = adapt_per_episode
        self.gmm_window = self.skill.max_steps // self.adapt_per_episode

        self.reset()

    def get_update_range_parameter_space(self):
        param_space = {}
        if self.priors_change_range != 0:
            param_space["priors"] = gym.spaces.Box(
                low=-self.priors_change_range, high=self.priors_change_range, shape=(self.gmm.priors.size,)
            )
        else:
            param_space["priors"] = gym.spaces.Box(low=0, high=0, shape=(0,))
        if self.mu_change_range != 0:
            param_space["mu"] = gym.spaces.Box(
                low=-self.mu_change_range, high=self.mu_change_range, shape=(self.gmm.means.size,)
            )
        else:
            param_space["mu"] = gym.spaces.Box(low=0, high=0, shape=(0,))

        return gym.spaces.Dict(param_space)

    def get_action_space(self):
        parameter_space = self.get_update_range_parameter_space()
        mu_high = np.ones(parameter_space["mu"].shape[0])

        priors_high = np.ones(parameter_space["priors"].shape[0])
        action_high = np.concatenate((priors_high, mu_high), axis=-1)

        action_low = -action_high
        self.action_space = gym.spaces.Box(action_low, action_high)
        return self.action_space

    def update_gaussians(self, gmm_change):
        parameter_space = self.get_update_range_parameter_space()
        size_priors = parameter_space["priors"].shape[0]
        size_mu = parameter_space["mu"].shape[0]

        priors = gmm_change[:size_priors] * parameter_space["priors"].high
        mu = gmm_change[size_priors : size_priors + size_mu] * parameter_space["mu"].high

        change_dict = {"mu": mu, "priors": priors}
        self.gmm.update_model(change_dict)

    @torch.no_grad()
    def detect_target(self, obs, device):
        keypoint_out = self.kp_mock.keypoint(np.zeros(1))
        objectness = keypoint_out[0][self.kp_mock.dim - 1]

        keypoint_out = self.kp_mock.to_world(keypoint_out).squeeze()
        keypoint_pos = keypoint_out[: self.kp_mock.dim - 1]

        return keypoint_pos + self.kp_target_shift

    @torch.no_grad()
    def play_step(self, actor, strategy="stochastic", replay_buffer=None, device="cuda"):
        """Perform a step in the environment and add the transition
        tuple to the replay buffer"""
        # detect the target point
        target_pos = self.detect_target(obs=self.obs, device=device)
        # Change dynamical system
        self.gmm.copy_model(self.initial_gmm)
        gmm_change = self.get_action(actor, self.obs, strategy, device)
        self.update_gaussians(gmm_change)

        # Act with the dynamical system in the environment
        gmm_reward = 0
        curr_obs = self.obs
        for _ in range(self.gmm_window):
            dx = self.gmm.predict(curr_obs["position"] - target_pos)
            action = {"motion": dx, "gripper": 0}
            curr_obs, reward, done, info = self.env.step(action)
            gmm_reward += reward
            self.episode_env_steps += 1
            self.total_env_steps += 1
            if done:
                break

        if self.episode_env_steps >= self.skill.max_steps:
            done = True

        replay_buffer.add(self.obs, gmm_change, gmm_reward, curr_obs, done)
        self.obs = curr_obs

        self.episode_play_steps += 1
        self.total_play_steps += 1

        if done:
            self.reset()
        return gmm_reward, done
