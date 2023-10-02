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
from sac_gmm.keypoint.key_nets import KeypointMock

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class Agent(object):
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
        self.name = name
        # GMM setup
        self.gmm = hydra.utils.instantiate(gmm)
        self.gmm.load_model()
        self.initial_gmm = copy.deepcopy(self.gmm)

        # Dataset (helps with EE start positions)
        self.datamodule = hydra.utils.instantiate(datamodule)

        self.skill = skill
        self.env = make_env(env, skill, self.datamodule.dataset.start)

        # Encoder
        self.encoder = hydra.utils.instantiate(encoder)

        self.demos_target = self.datamodule.dataset.goal
        kp_mock.env_is_source = self.env.is_source
        self.kp_mock = hydra.utils.instantiate(kp_mock)
        self.gt_keypoint = None
        self.kp_target_shift = None

        self.kp_det = None

        self.num_init_steps = num_init_steps
        self.num_eval_episodes = num_eval_episodes

        self.gmm_window = self.skill.max_steps

        self.render = render

        # Trackers
        self.episode_play_steps = 0
        self.episode_env_steps = 0
        self.total_play_steps = 0
        self.total_env_steps = 0

        # State variable
        self.obs = None

    def reset_mock(self, gt_keypoint):
        self.kp_mock.reset_gt(gt_keypoint)
        self.gt_keypoint = self.kp_mock.init_pos
        self.kp_target_shift = self.demos_target - self.gt_keypoint
        self.kp_mock.reset_position()

    def reset(self) -> None:
        """Resets the environment, moves the EE to a good start state and updates the agent state"""
        self.obs = self.env.reset()
        self.reset_mock(self.env.gt_keypoint)
        self.episode_play_steps = 0
        self.episode_env_steps = 0

    def populate_replay_buffer(self, actor, replay_buffer):
        """
        Carries out several steps through the environment to initially fill
        up the replay buffer with experiences from the GMM
        Args:
            steps: number of random steps to populate the buffer with
            strategy: strategy to follow to select actions to fill the replay buffer
        """
        log_rank_0("Populating replay buffer with random warm up steps")
        for _ in tqdm(range(self.num_init_steps)):
            self.play_step(actor=actor, strategy="random", replay_buffer=replay_buffer)
        replay_buffer.save()

    def get_update_range_parameter_space(self):
        """Returns GMM parameters range as a gym.spaces.Dict for the agent to predict"""
        raise NotImplementedError

    def get_state_dim(self):
        """Returns the size of the state based on env's observation space"""
        raise NotImplementedError

    def get_state_from_observation(self):
        raise NotImplementedError

    def get_action_space(self):
        raise NotImplementedError

    def get_action(self):
        """Interface to get action from the agent ready to be used in the environment"""
        raise NotImplementedError

    def play_step(self, actor, strategy="stochastic", replay_buffer=None):
        """Perform a step in the environment and add the transition
        tuple to the replay buffer"""
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, actor, device="cuda"):
        """Evaluates the actor in the environment"""
        raise NotImplementedError
