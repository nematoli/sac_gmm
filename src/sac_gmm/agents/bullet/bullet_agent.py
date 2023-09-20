import logging
import hydra
from omegaconf import DictConfig
import torch
import gym
import numpy as np
from tqdm import tqdm
import copy
from pytorch_lightning.utilities import rank_zero_only
from sac_gmm.utils.env_maker import make_env
from sac_gmm.agents.agent import Agent

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class BulletAgent(Agent):
    def __init__(
        self,
        name: str,
        env: DictConfig,
        datamodule: DictConfig,
        num_init_steps: int,
        num_eval_episodes: int,
        skill: DictConfig,
        gmm: DictConfig,
        encoder: DictConfig,
        kp_mock: DictConfig,
        render: bool,
    ) -> None:
        super(BulletAgent, self).__init__(
            name=name,
            env=env,
            datamodule=datamodule,
            num_init_steps=num_init_steps,
            num_eval_episodes=num_eval_episodes,
            skill=skill,
            gmm=gmm,
            encoder=encoder,
            kp_mock=kp_mock,
            render=render,
        )
        self.agent_name = name

        self.reset()

    def get_state_dim(self):
        """Returns the size of the state based on env's observation space"""
        state_dim = 0
        observation_space = self.env.observation_space
        keys = list(observation_space.keys())
        if "position" in keys:
            state_dim += 3
        if "rgb_gripper" in keys:
            state_dim += self.encoder.feature_size
        return state_dim

    def get_features_from_observation(self, obs, device="cuda"):
        if "rgb_gripper" in obs:
            x = obs["rgb_gripper"]
            if not torch.is_tensor(x):
                x = torch.tensor(x).to(device)
            if len(x.shape) < 4:
                x = x.unsqueeze(0)
            features = self.encoder.to(device)(x)
        else:
            features = None

        return features

    def get_state_from_observation(self, features, obs, device="cuda"):
        if isinstance(obs, dict):
            # Robot obs
            if "position" in obs:
                fc_input = torch.tensor(obs["position"]).to(device)
            if features is not None:
                if "fc_input" in locals():
                    fc_input = torch.cat((fc_input, features.squeeze()), dim=-1)
                else:
                    fc_input = features
            return fc_input.float()
        return obs.float()

    def get_action(self, actor, observation, strategy="stochastic", device="cuda"):
        if actor is None:
            return None
        actor.eval()
        if strategy == "random":
            return self.get_action_space().sample()
        elif strategy == "zeros":
            return np.zeros(self.get_action_space().shape)
        elif strategy == "stochastic":
            deterministic = False
        elif strategy == "deterministic":
            deterministic = True
        else:
            raise Exception("Strategy not implemented")
        features = self.get_features_from_observation(observation, device)
        state = self.get_state_from_observation(features, observation, device)
        action, _ = actor.get_actions(state, deterministic=deterministic, reparameterize=False)
        actor.train()
        return action.detach().cpu().numpy()

    @torch.no_grad()
    def evaluate(self, actor, device="cuda"):
        """Evaluates the actor in the environment"""
        log_rank_0("Evaluation episodes in process")
        succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
        for episode in tqdm(range(1, self.num_eval_episodes + 1)):
            episode_return, episode_env_steps = 0, 0

            self.obs = self.env.reset()
            while episode_env_steps < self.skill.max_steps:
                # detect target
                target_pos = self.detect_target(obs=self.obs, device=device)
                if isinstance(target_pos, torch.Tensor):
                    target_pos = target_pos.cpu().numpy()

                self.gmm.copy_model(self.initial_gmm)
                if self.agent_name in ["SACGMM", "MimicSACGMM", "KISGMM"]:
                    # Change dynamical system
                    gmm_change = self.get_action(actor, self.obs, "deterministic", device)
                    self.update_gaussians(gmm_change)

                # Act with the dynamical system in the environment
                for _ in range(self.gmm_window):
                    dx = self.gmm.predict(self.obs["position"] - target_pos)
                    action = {"motion": dx, "gripper": 0}
                    self.obs, reward, done, info = self.env.step(action)
                    episode_return += reward
                    episode_env_steps += 1

                    if self.render:
                        self.env.render()
                    if done:
                        break
                if done:
                    break

            self.reset()

            if ("success" in info) and info["success"]:
                succesful_episodes += 1

            episodes_returns.append(episode_return)
            episodes_lengths.append(episode_env_steps)
        accuracy = succesful_episodes / self.num_eval_episodes

        return (accuracy, np.mean(episodes_returns), np.mean(episodes_lengths))
