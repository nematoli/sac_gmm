import logging
from omegaconf import DictConfig
import torch
import gym
import numpy as np
from pytorch_lightning.utilities import rank_zero_only
from sac_gmm.agents.calvin.calvin_agent import CALVINAgent
from sac_gmm.utils.projections import xyz_to_XYZ

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class KISGMMAgent(CALVINAgent):
    def __init__(
        self,
        name: str,
        calvin_env: DictConfig,
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
        mimic_sac_gmm: bool,
        render: bool,
    ) -> None:
        super(KISGMMAgent, self).__init__(
            name=name,
            env=calvin_env,
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

        self.mimic_sac_gmm = mimic_sac_gmm

        self.reset()

    def set_keypoint_detector(self, kp_det):
        self.kp_det = kp_det

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

    def cam_from_obs(self, obs):
        viewm = torch.tensor(obs["view_mtx"])
        intrinsics = torch.tensor(obs["intrinsics"])

        if len(viewm.shape) == 1:
            viewm = viewm.unsqueeze(dim=0)
        return viewm, intrinsics

    @torch.no_grad()
    def detect_target(self, obs, device):
        self.kp_det.eval()
        if self.mimic_sac_gmm or self.env.is_source:
            keypoint_out = self.kp_mock.keypoint(np.zeros(1))
            objectness = keypoint_out[0][self.kp_mock.dim - 1]

            keypoint_out = self.kp_mock.to_world(keypoint_out).squeeze()
            keypoint_pos = keypoint_out[: self.kp_mock.dim - 1]

        else:
            cv, intrinsics = self.cam_from_obs(obs)

            features = self.get_features_from_observation(obs, device)
            keypoint_out = self.kp_det.keypoint(features)
            objectness = keypoint_out[0][self.kp_det.dim - 1]

            hw = (84, 84)
            x, y = keypoint_out[0, 0].floor().int(), keypoint_out[0, 1].floor().int()
            z = keypoint_out[0, 2]

            if (x >= 0) and (y >= 0) and (x < hw[0]) and (y < hw[1]):
                xyz = xyz_to_XYZ(z, cv, (x, y), hw, intrinsics)
                keypoint_pos = xyz
            else:
                keypoint_pos = torch.zeros(3).type_as(keypoint_out)

        if not torch.is_tensor(keypoint_pos):
            keypoint_pos = torch.tensor(keypoint_pos)

        keypoint_mean = keypoint_pos
        self.kp_det.train()
        return keypoint_mean + torch.from_numpy(self.kp_target_shift)

    def get_gt_keypoint(self):
        return self.gt_keypoint

    @torch.no_grad()
    def play_step(self, actor, strategy="stochastic", replay_buffer=None, device="cuda"):
        """Perform a step in the environment and add the transition
        tuple to the replay buffer"""
        # detect the target point
        target_pos = self.detect_target(obs=self.obs, device=device).cpu().numpy()
        # Change dynamical system
        self.gmm.copy_model(self.initial_gmm)
        gmm_change = self.get_action(actor, self.obs, strategy, device)
        self.update_gaussians(gmm_change)

        # Act with the dynamical system in the environment
        gmm_reward = 0
        curr_obs = self.obs
        for _ in range(self.gmm_window):
            dx = self.gmm.predict(curr_obs["position"] - target_pos)
            curr_obs, reward, done, info = self.env.step(dx)
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
