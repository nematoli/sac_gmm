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
from sac_gmm.models.sac.agent.agent import Agent

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class CALVINSACGMMAgent(Agent):
    def __init__(
        self,
        calvin_env: DictConfig,
        datamodule: DictConfig,
        num_init_steps: int,
        num_eval_episodes: int,
        skill: DictConfig,
        gmm: DictConfig,
        adapt_cov: bool,
        mean_shift: bool,
        adapt_per_episode: int,
        exp_dir: str,
        render: bool,
        record: bool,
    ) -> None:
        super(CALVINSACGMMAgent, self).__init__(
            env=calvin_env,
            num_init_steps=num_init_steps,
            num_eval_episodes=num_eval_episodes,
        )

        self.skill = skill

        # Environment
        self.env.set_skill(self.skill)
        self.robot_obs, self.cam_obs = self.env.obs_allowed
        # # TODO: find the transforms for this
        # env.set_obs_transforms(cfg.datamodule.transforms)

        # GMM refine setup
        self.gmm = hydra.utils.instantiate(gmm)
        self.gmm.load_model()
        if "Manifold" in self.gmm.name:
            self.gmm.manifold = self.gmm.make_manifold()
        self.initial_gmm = copy.deepcopy(self.gmm)
        self.dt = self.skill.dt
        self.adapt_cov = adapt_cov
        self.mean_shift = mean_shift
        self.adapt_per_episode = adapt_per_episode
        self.gmm_window = self.skill.max_steps // self.adapt_per_episode

        # record setup
        self.video_dir = os.path.join(exp_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)
        self.env.set_outdir(self.video_dir)
        self.render = render
        self.record = record

        # Dataset (helps with EE start positions)
        self.datamodule = hydra.utils.instantiate(datamodule)
        self.reset()

    def reset(self) -> None:
        """Resets the environment, moves the EE to a good start state and updates the agent state"""
        super().reset()
        self.obs = self.env.sample_start_position(self.datamodule.dataset)

    def get_state_dim(self):
        """Returns the size of the state based on env's observation space"""
        state_dim = 0
        robot_obs, cam_obs = self.env.obs_allowed
        if "pos" in robot_obs:
            state_dim += 3
        if "ori" in robot_obs:
            state_dim += 4
        if "joint" in robot_obs:
            state_dim = 7

        return state_dim

    def get_action_space(self):
        parameter_space = self.get_update_range_parameter_space()
        mu_high = np.ones(parameter_space["mu"].shape[0])
        if self.mean_shift:
            action_high = mu_high
        else:
            priors_high = np.ones(parameter_space["priors"].shape[0])
            action_high = np.concatenate((priors_high, mu_high), axis=-1)
            if self.adapt_cov:
                sigma_high = np.ones(parameter_space["sigma"].shape[0])
                action_high = np.concatenate((action_high, sigma_high), axis=-1)

        action_low = -action_high
        self.action_space = gym.spaces.Box(action_low, action_high)
        return self.action_space

    def play_step(self, actor, strategy="stochastic", replay_buffer=None):
        """Perform a step in the environment and add the transition
        tuple to the replay buffer"""
        # Change dynamical system
        self.gmm.copy_model(self.initial_gmm)
        gmm_change = self.get_action(actor, self.obs, strategy)
        self.update_gaussians(gmm_change)

        # Act with the dynamical system in the environment
        gmm_reward = 0
        x = self.obs[self.robot_obs]
        for _ in range(self.gmm_window):
            dx = self.gmm.predict(x - self.datamodule.dataset.goal)
            new_x = x + self.dt * dx
            env_action = np.append(new_x, np.append(self.datamodule.dataset.fixed_ori, -1))
            env_action = self.env.prepare_action(env_action, type="abs")
            next_obs, reward, done, info = self.env.step(env_action)
            gmm_reward += reward
            x = next_obs[self.robot_obs]
            self.episode_env_steps += 1
            self.total_env_steps += 1
            if done:
                break

        replay_buffer.add(self.obs, gmm_change, gmm_reward, next_obs, done)
        self.obs = next_obs

        self.episode_play_steps += 1
        self.total_play_steps += 1

        if done or (self.episode_env_steps >= self.skill.max_steps):
            self.reset()
            done = True
        return reward, done

    @torch.no_grad()
    def evaluate(self, actor):
        """Evaluates the actor in the environment"""
        log_rank_0("Evaluation episodes in process")
        succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
        saved_video_path = None
        # Choose a random episode to record
        rand_idx = np.random.randint(0, self.num_eval_episodes)
        for episode in tqdm(range(0, self.num_eval_episodes)):
            episode_env_steps = 0
            episode_return = 0
            self.obs = self.env.reset()
            # Start from a known starting point
            self.obs = self.env.sample_start_position(self.datamodule.dataset)
            # Recording setup
            if self.record and (episode == rand_idx):
                self.env.reset_recorded_frames()
                self.env.record_frame(size=64)
            while episode_env_steps < self.skill.max_steps:
                # Change dynamical system
                self.gmm.copy_model(self.initial_gmm)
                gmm_change = self.get_action(actor, self.obs, "deterministic")
                self.update_gaussians(gmm_change)

                # Act with the dynamical system in the environment
                gmm_reward = 0
                x = self.obs[self.robot_obs]
                for _ in range(self.gmm_window):
                    dx = self.gmm.predict(x - self.datamodule.dataset.goal)
                    new_x = x + self.dt * dx
                    env_action = np.append(new_x, np.append(self.datamodule.dataset.fixed_ori, -1))
                    env_action = self.env.prepare_action(env_action, type="abs")
                    next_obs, reward, done, info = self.env.step(env_action)
                    gmm_reward += reward
                    x = next_obs[self.robot_obs]
                    episode_env_steps += 1

                    if self.record and (episode == rand_idx):
                        self.env.record_frame(size=64)
                    if self.render:
                        self.env.render()
                    if done:
                        break

                episode_return += gmm_reward
                self.obs = next_obs
                if done:
                    self.reset()
                    break

            if ("success" in info) and info["success"]:
                succesful_episodes += 1
            # Recording setup close
            if self.record and (episode == rand_idx):
                video_path = self.env.save_recorded_frames(
                    outdir=self.video_dir,
                    fname=f"{self.total_play_steps}_{self.total_env_steps }_{episode}",
                )
                self.env.reset_recorded_frames()
                saved_video_path = video_path

            episodes_returns.append(episode_return)
            episodes_lengths.append(episode_env_steps)
        accuracy = succesful_episodes / self.num_eval_episodes

        return (
            accuracy,
            np.mean(episodes_returns),
            np.mean(episodes_lengths),
            saved_video_path,
        )

    def get_update_range_parameter_space(self):
        """Returns GMM parameters range as a gym.spaces.Dict for the agent to predict

        Returns:
            param_space : gym.spaces.Dict
                Range of GMM parameters parameters
        """
        # TODO: make low and high config variables
        param_space = {}
        param_space["priors"] = gym.spaces.Box(low=-0.1, high=0.1, shape=(self.gmm.priors.size,))
        param_space["mu"] = gym.spaces.Box(low=-0.01, high=0.01, shape=(self.gmm.means.size,))

        dim = self.gmm.means.shape[1] // 2
        num_gaussians = self.gmm.means.shape[0]
        sigma_change_size = int(num_gaussians * dim * (dim + 1) / 2 + dim * dim * num_gaussians)
        param_space["sigma"] = gym.spaces.Box(low=-1e-6, high=1e-6, shape=(sigma_change_size,))
        return gym.spaces.Dict(param_space)

    def update_gaussians(self, gmm_change):
        parameter_space = self.get_update_range_parameter_space()
        size_mu = parameter_space["mu"].shape[0]
        if self.mean_shift:
            # TODO: check low and high here
            mu = np.hstack([gmm_change.reshape((size_mu, 1)) * parameter_space["mu"].high] * self.gmm.means.shape[1])

            change_dict = {"mu": mu}
            self.gmm.update_model(change_dict)
        else:
            size_priors = parameter_space["priors"].shape[0]

            priors = gmm_change[:size_priors] * parameter_space["priors"].high
            mu = gmm_change[size_priors : size_priors + size_mu] * parameter_space["mu"].high

            change_dict = {"mu": mu, "priors": priors}
            if self.adapt_cov:
                change_dict["sigma"] = gmm_change[size_priors + size_mu :] * parameter_space["sigma"].high
            self.gmm.update_model(change_dict)
