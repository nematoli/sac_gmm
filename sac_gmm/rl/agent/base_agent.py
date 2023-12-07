import logging
from omegaconf import DictConfig
import torch
import numpy as np
from tqdm import tqdm
from pytorch_lightning.utilities import rank_zero_only
from sac_gmm.utils.env_maker import make_env
import gym

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

    def populate_replay_buffer(self, actor, model, replay_buffer):
        """
        Carries out several steps through the environment to initially fill
        up the replay buffer with experiences from the GMM
        Args:
            steps: number of random steps to populate the buffer with
            strategy: strategy to follow to select actions to fill the replay buffer
        """
        log_rank_0("Populating replay buffer with random warm up steps")
        for _ in tqdm(range(self.num_init_steps)):
            self.play_step(actor, model, strategy="random", replay_buffer=replay_buffer)
        replay_buffer.save()

    def populate_replay_buffer_with_critic(self, actor, model, critic, replay_buffer):
        """
        Carries out several steps through the environment to initially fill
        up the replay buffer with experiences from the GMM
        Args:
            steps: number of random steps to populate the buffer with
            strategy: strategy to follow to select actions to fill the replay buffer
        """
        log_rank_0("Populating replay buffer with random warm up steps")
        for _ in tqdm(range(self.num_init_steps)):
            self.play_step(actor, model, critic, strategy="random", replay_buffer=replay_buffer)
        replay_buffer.save()

    def get_action(self, actor, model, observation, strategy="stochastic", device="cuda"):
        raise NotImplementedError

    def play_step(self, actor, model, strategy="stochastic", replay_buffer=None):
        """Perform a step in the environment and add the transition
        tuple to the replay buffer"""
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, actor):
        """Evaluates the actor in the environment"""
        raise NotImplementedError

    def get_state_from_observation(self, encoder, obs, device="cuda"):
        """get state from observation"""
        raise NotImplementedError

    def get_action_space(self):
        parameter_space = self.get_update_range_parameter_space()
        mu_high = np.ones(parameter_space["mu"].shape[0])
        action_high = mu_high
        if "priors" in parameter_space.spaces:
            priors_high = np.ones(parameter_space["priors"].shape[0])
            action_high = np.concatenate((priors_high, action_high), axis=-1)
        if self.adapt_cov:
            sigma_high = np.ones(parameter_space["sigma"].shape[0])
            action_high = np.concatenate((action_high, sigma_high), axis=-1)
        if "quat" in parameter_space.spaces:
            quat_high = np.ones(parameter_space["quat"].shape[0])
            action_high = np.concatenate((action_high, quat_high), axis=-1)

        action_low = -action_high
        self.action_space = gym.spaces.Box(action_low, action_high)
        return self.action_space

    def get_update_range_parameter_space(self):
        """Returns GMM parameters range as a gym.spaces.Dict for the agent to predict

        Returns:
            param_space : gym.spaces.Dict
                Range of GMM parameters parameters
        """
        # TODO: make low and high config variables
        param_space = {}
        if self.priors_change_range > 0:
            param_space["priors"] = gym.spaces.Box(
                low=-self.priors_change_range,
                high=self.priors_change_range,
                shape=(self.skill_actor.priors_size,),
            )
        if self.mu_change_range > 0:
            if self.skill_actor.gmm_type in [1, 4]:
                param_space["mu"] = gym.spaces.Box(
                    low=-self.mu_change_range, high=self.mu_change_range, shape=(self.skill_actor.means_size,)
                )
            elif self.skill_actor.gmm_type in [2, 5]:
                param_space["mu"] = gym.spaces.Box(
                    low=-self.mu_change_range,
                    high=self.mu_change_range,
                    shape=(self.skill_actor.means_size // 2,),
                )
            else:
                # Only update position means for now
                total_size = self.skill_actor.means_size
                just_positions_size = total_size - self.skill_actor.priors_size * 4
                param_space["mu"] = gym.spaces.Box(
                    low=-self.mu_change_range,
                    high=self.mu_change_range,
                    shape=(just_positions_size // 2,),
                )
                # Update orientations (quaternion) means also
                if self.quat_change_range > 0:
                    just_orientations_size = self.skill_actor.priors_size * 4
                    param_space["quat"] = gym.spaces.Box(
                        low=-self.quat_change_range,
                        high=self.quat_change_range,
                        shape=(just_orientations_size,),
                    )

        return gym.spaces.Dict(param_space)

    def update_gaussians(self, gmm_change, skill_id=None):
        parameter_space = self.get_update_range_parameter_space()
        change_dict = {}
        if "priors" in parameter_space.spaces:
            size_priors = parameter_space["priors"].shape[0]
            priors = gmm_change[:size_priors] * parameter_space["priors"].high
            change_dict.update({"priors": priors})
        else:
            size_priors = 0

        size_mu = parameter_space["mu"].shape[0]
        mu = gmm_change[size_priors : size_priors + size_mu] * parameter_space["mu"].high
        change_dict.update({"mu": mu})

        if "quat" in parameter_space.spaces:
            quat = gmm_change[size_priors + size_mu :] * parameter_space["quat"].high
            change_dict.update({"quat": quat})
        if skill_id is not None:
            self.skill_actor.update_model(change_dict, skill_id)
        else:
            self.gmm.update_model(change_dict)
