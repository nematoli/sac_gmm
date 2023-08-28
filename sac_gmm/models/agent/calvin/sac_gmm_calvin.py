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
from sac_gmm.models.agent.agent import Agent

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
        priors_change_range: float,
        mu_change_range: float,
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

        skill.skill = skill.name  # This is a hack to make things consistent with TaskActorWithNSACGMM
        self.skill = skill

        # Environment
        self.env.set_skill(self.skill)

        # Dataset

        datamodule.dataset.skill.skill = datamodule.dataset.skill.name
        self.datamodule = hydra.utils.instantiate(datamodule)

        # GMM refine setup
        gmm.skill.skill = gmm.skill.name
        self.gmm = hydra.utils.instantiate(gmm)
        self.gmm.load_model()
        if "Manifold" in self.gmm.name:
            self.gmm.manifold = self.gmm.make_manifold()
        self.gmm.set_skill_params(self.datamodule.dataset)
        self.initial_gmm = copy.deepcopy(self.gmm)
        self.priors_change_range = priors_change_range
        self.mu_change_range = mu_change_range
        self.adapt_cov = adapt_cov
        self.mean_shift = mean_shift
        self.adapt_per_episode = adapt_per_episode
        self.gmm_window = self.skill.max_steps // self.adapt_per_episode

        # # record setup
        self.video_dir = os.path.join(exp_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)
        self.render = render
        self.record = record

        self.env.set_init_pos(self.gmm.start)
        self.reset()

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

    @torch.no_grad()
    def play_step(self, actor, strategy="stochastic", replay_buffer=None, device="cuda"):
        """Perform a step in the environment and add the transition
        tuple to the replay buffer"""
        # Change dynamical system
        self.gmm.copy_model(self.initial_gmm)
        gmm_change = self.get_action(actor, self.obs, strategy, device)
        self.update_gaussians(gmm_change)

        # Act with the dynamical system in the environment
        gmm_reward = 0
        curr_obs = self.obs
        for _ in range(self.gmm_window):
            conn = self.env.isConnected()
            if not conn:
                done = False
                break
            dx_pos, dx_ori = self.gmm.predict(curr_obs["robot_obs"])
            # env_action = np.append(dx_pos, np.append(dx_ori, -1))
            curr_obs, reward, done, info = self.env.step(dx_pos)
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

        if done or not conn:
            self.reset()
        return gmm_reward, done

    @torch.no_grad()
    def evaluate(self, actor, device="cuda"):
        """Evaluates the actor in the environment"""
        log_rank_0("Evaluation episodes in process")
        succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
        saved_video_path = None
        # Choose a random episode to record
        rand_idx = np.random.randint(1, self.num_eval_episodes + 1)
        for episode in tqdm(range(1, self.num_eval_episodes + 1)):
            episode_return, episode_env_steps = 0, 0

            self.obs = self.env.reset()
            # Recording setup
            if self.record and (episode == rand_idx):
                self.env.reset_recording()
                self.env.record_frame(size=64)

            while episode_env_steps < self.skill.max_steps:
                # Change dynamical system
                self.gmm.copy_model(self.initial_gmm)
                gmm_change = self.get_action(actor, self.obs, "deterministic", device)
                self.update_gaussians(gmm_change)

                # Act with the dynamical system in the environment
                for _ in range(self.gmm_window):
                    dx_pos, dx_ori = self.gmm.predict(self.obs["robot_obs"])
                    # env_action = np.append(dx_pos, np.append(dx_ori, -1))
                    self.obs, reward, done, info = self.env.step(dx_pos)
                    episode_return += reward
                    episode_env_steps += 1

                    if self.record and (episode == rand_idx):
                        self.env.record_frame(size=64)
                    if self.render:
                        self.env.render()
                    if done:
                        break

                if done:
                    self.reset()
                    break

            if ("success" in info) and info["success"]:
                succesful_episodes += 1
            # Recording setup close
            if self.record and (episode == rand_idx):
                video_path = self.env.save_recording(
                    outdir=self.video_dir,
                    fname=f"{self.total_play_steps}_{self.total_env_steps }_{episode}",
                )
                self.env.reset_recording()
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
        param_space["priors"] = gym.spaces.Box(
            low=-self.priors_change_range, high=self.priors_change_range, shape=(self.gmm.priors.size,)
        )
        param_space["mu"] = gym.spaces.Box(
            low=-self.mu_change_range, high=self.mu_change_range, shape=(self.gmm.means.size,)
        )

        # dim = self.gmm.means.shape[1] // 2
        # num_gaussians = self.gmm.means.shape[0]
        # sigma_change_size = int(num_gaussians * dim * (dim + 1) / 2 + dim * dim * num_gaussians)
        # param_space["sigma"] = gym.spaces.Box(low=-1e-6, high=1e-6, shape=(sigma_change_size,))
        return gym.spaces.Dict(param_space)

    def update_gaussians(self, gmm_change):
        parameter_space = self.get_update_range_parameter_space()
        size_priors = parameter_space["priors"].shape[0]
        size_mu = parameter_space["mu"].shape[0]

        priors = gmm_change[:size_priors] * parameter_space["priors"].high
        mu = gmm_change[size_priors : size_priors + size_mu] * parameter_space["mu"].high

        change_dict = {"mu": mu, "priors": priors}
        # if self.adapt_cov:
        #     change_dict["sigma"] = gmm_change[size_priors + size_mu :] * parameter_space["sigma"].high
        self.gmm.update_model(change_dict)

        # if self.mean_shift:
        #     # TODO: check low and high here
        #     mu = np.hstack([gmm_change.reshape((size_mu, 1)) * parameter_space["mu"].high] * self.gmm.means.shape[1])

        #     change_dict = {"mu": mu}
        #     self.gmm.update_model(change_dict)
        # else:

    def get_state_from_observation(self, encoder, obs, device="cuda"):
        if isinstance(obs, dict):
            # Robot obs
            if "position" in obs:
                fc_input = torch.tensor(obs["position"]).to(device)
            if "orientation" in obs:
                fc_input = torch.cat((fc_input, obs["orientation"].float()), dim=-1).to(device)
            if "robot_obs" in obs:
                if obs["robot_obs"].ndim > 1:
                    fc_input = torch.tensor(obs["robot_obs"][:, :3]).to(device)
                else:
                    fc_input = torch.tensor(obs["robot_obs"][:3]).to(device)
            if "rgb_gripper" in obs:
                x = obs["rgb_gripper"]
                if not torch.is_tensor(x):
                    x = torch.tensor(x).to(device)
                if len(x.shape) < 4:
                    x = x.unsqueeze(0)
                features = encoder(x)
                if features is not None:
                    fc_input = torch.cat((fc_input, features.squeeze()), dim=-1)
                # fc_input = features.squeeze()
            # if "obs" in obs:
            #     fc_input = torch.cat((fc_input, torch.tensor(obs["obs"]).to(device)), dim=-1)
            return fc_input.float()

        return obs.float()
