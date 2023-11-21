import logging
from omegaconf import DictConfig
import torch
import numpy as np
from tqdm import tqdm
from pytorch_lightning.utilities import rank_zero_only
from sac_gmm.utils.env_maker import make_env

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class BaseAgent(object):
    def __init__(
        self,
        env: DictConfig,
        num_init_steps: int,
        num_eval_episodes: int,
    ) -> None:
        self.env = make_env(env)

        self.num_init_steps = num_init_steps

        self.num_eval_episodes = num_eval_episodes

        # Trackers
        # number of "play_step"s in a given episode
        self.episode_play_steps = 0
        # number of "env_step"s in a given episode
        self.episode_env_steps = 0

        # Total "play_steps" taken in an experiment
        self.total_play_steps = 0
        # Total environment steps taken in an experiment
        self.total_env_steps = 0

        # State variable
        self.obs = None
        # Agent resets - env and state variable self.obs

    def reset(self, target_skill=None, start_skill=None) -> None:
        """Resets the environment, moves the EE to a good start state and updates the agent state"""
        if target_skill is not None:
            self.obs = self.env.reset(target_skill=target_skill)
        if start_skill is not None:
            self.obs = self.env.reset(start_skill=start_skill)
        if target_skill is None and start_skill is None:
            self.obs = self.env.reset()
        self.episode_play_steps = 0
        self.episode_env_steps = 0

    def get_state_dim(self, feature_size=0):
        """Returns the size of the state based on env's observation space"""
        state_dim = 0
        observation_space = self.env.get_observation_space()
        keys = list(observation_space.keys())
        if "position" in keys:
            state_dim += 3
        if "orientation" in keys:
            state_dim += 3
        if "robot_obs" in keys:
            state_dim += 3
        if "rgb_gripper" in keys:
            # state_dim += feature_size
            state_dim = feature_size
        if "obs" in keys:
            state_dim = 21
        if "state" in keys:
            state_dim = 33
        return state_dim

    def get_action_space(self):
        raise NotImplementedError

    def populate_replay_buffer(self, actor, model, replay_buffer):
        """
        Carries out several steps through the environment to initially fill
        up the replay buffer with experiences from the GMM
        Args:
            steps: number of random steps to populate the buffer with
            strategy: strategy to follow to select actions to fill the replay buffer
        """
        log_rank_0("Populating replay buffer with random warm up steps")
        if model is not None:
            for _ in tqdm(range(self.num_init_steps)):
                self.play_step(actor, model, critic=None, strategy="random", replay_buffer=replay_buffer)
        else:
            # SACGMM
            for _ in tqdm(range(self.num_init_steps)):
                self.play_step(actor, strategy="random", replay_buffer=replay_buffer)
        replay_buffer.save()

    def get_action(self, actor, model, observation, strategy="stochastic", device="cuda"):
        raise NotImplementedError

    def play_step(self, actor, strategy="stochastic", replay_buffer=None):
        """Perform a step in the environment and add the transition
        tuple to the replay buffer"""
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, actor):
        """Evaluates the actor in the environment"""
        raise NotImplementedError

    def sample_start_position(self, error_margin=0.01, max_checks=15):
        """Samples a random starting point and moves the end effector to that point"""
        raise NotImplementedError

    def get_state_from_observation(self, encoder, obs, device="cuda"):
        """get state from observation"""
        raise NotImplementedError
