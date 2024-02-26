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
from sac_gmm.rl.agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


OBS_KEY = "rgb_static"
# OBS_KEY = "rgb_gripper"
# OBS_KEY = "robot_obs"


class CALVINSACGMMAgent(BaseAgent):
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
        quat_change_range: float,
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

        # GMM
        gmm.skill.skill = gmm.skill.name
        self.gmm = hydra.utils.instantiate(gmm)
        self.gmm.load_model()
        if "Manifold" in self.gmm.name:
            self.gmm.manifold, self.gmm.manifold2 = self.gmm.make_manifold()
        self.gmm.set_skill_params(self.datamodule.dataset)
        self.initial_gmm = copy.deepcopy(self.gmm)

        # Refine parameters
        self.priors_change_range = priors_change_range
        self.mu_change_range = mu_change_range
        self.quat_change_range = quat_change_range
        self.adapt_cov = adapt_cov
        self.mean_shift = mean_shift
        self.gmm_window = 10

        # Record setup
        self.video_dir = os.path.join(exp_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)
        self.render = render
        self.record = record

        self.reset()

        self.nan_counter = 0

    @torch.no_grad()
    def play_step(self, actor, model, strategy="stochastic", replay_buffer=None, device="cuda"):
        """Perform a step in the environment and add the transition
        tuple to the replay buffer"""
        # Change dynamical system
        self.gmm.copy_model(self.initial_gmm)
        gmm_change = self.get_action(actor, model, self.obs, strategy, device)
        self.update_gaussians(gmm_change)

        # Act with the dynamical system in the environment
        gmm_reward = 0
        curr_obs = self.obs
        for _ in range(self.gmm_window):
            conn = self.env.isConnected()
            if not conn:
                done = False
                break
            dx_pos, dx_ori, is_nan = self.gmm.predict(curr_obs["robot_obs"])
            if is_nan:
                self.nan_counter += 1
                done = True
                log_rank_0("Nan in prediction, aborting episode")
            else:
                env_action = np.append(dx_pos, np.append(dx_ori, -1))
                curr_obs, reward, done, info = self.env.step(env_action)
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
    def evaluate(self, actor, model, device="cuda"):
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
                self.env.record_frame(size=200)

            while episode_env_steps < self.skill.max_steps:
                # Change dynamical system
                self.gmm.copy_model(self.initial_gmm)
                gmm_change = self.get_action(actor, model, self.obs, "deterministic", device)
                self.update_gaussians(gmm_change)

                # Act with the dynamical system in the environment
                for _ in range(self.gmm_window):
                    dx_pos, dx_ori, is_nan = self.gmm.predict(self.obs["robot_obs"])
                    if is_nan:
                        done = True
                        log_rank_0("Nan in prediction, aborting episode")
                    else:
                        env_action = np.append(dx_pos, np.append(dx_ori, -1))
                        self.obs, reward, done, info = self.env.step(env_action)
                        episode_return += reward
                        episode_env_steps += 1

                    if self.record and (episode == rand_idx):
                        self.env.record_frame(size=200)
                    if self.render:
                        self.env.render()
                    if done:
                        break

                if done:
                    self.reset()
                    break

            if not is_nan and ("success" in info) and info["success"]:
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

    def get_state_from_observation(self, encoder, obs, device="cuda"):
        if isinstance(obs, dict):
            if "rgb_gripper" in obs or "rgb_static" in obs:
                x = obs["rgb_gripper"] if "rgb_gripper" in obs else obs["rgb_static"]
                if not torch.is_tensor(x):
                    x = torch.tensor(x).to(device)
                if len(x.shape) < 4:
                    x = x.unsqueeze(0)
                with torch.no_grad():
                    features = encoder(x)
            fc_input = features.squeeze()
            return fc_input.float()

    def get_action(self, actor, model, observation, strategy="stochastic", device="cuda"):
        """Interface to get action from SAC Actor,
        ready to be used in the environment"""
        actor.eval()
        model.eval()
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
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation[OBS_KEY]).to(device)
            state = model.encoder({"obs": obs_tensor.float()}).squeeze(0)
        action, _ = actor.get_actions(state, deterministic=deterministic, reparameterize=False)
        actor.train()
        model.train()
        return action.detach().cpu().numpy()

    def get_update_range_parameter_space(self):
        """Returns GMM parameters range as a gym.spaces.Dict for the agent to predict

        Returns:
            param_space : gym.spaces.Dict
                Range of GMM parameters parameters
        """
        # TODO: make low and high config variables
        param_space = {}
        priors_size, means_size, _ = self.gmm.get_params_size()
        if self.priors_change_range > 0:
            param_space["priors"] = gym.spaces.Box(
                low=-self.priors_change_range,
                high=self.priors_change_range,
                shape=(priors_size,),
            )
        if self.mu_change_range > 0:
            if self.gmm.gmm_type in [1, 4]:
                param_space["mu"] = gym.spaces.Box(
                    low=-self.mu_change_range, high=self.mu_change_range, shape=(means_size,)
                )
            elif self.gmm.gmm_type in [2, 5]:
                param_space["mu"] = gym.spaces.Box(
                    low=-self.mu_change_range,
                    high=self.mu_change_range,
                    shape=(means_size // 2,),
                )
            else:
                # Only update position means for now
                total_size = means_size
                just_positions_size = total_size - priors_size * 4
                param_space["mu"] = gym.spaces.Box(
                    low=-self.mu_change_range,
                    high=self.mu_change_range,
                    shape=(just_positions_size // 2,),
                )
                # Update orientations (quaternion) means also
                if self.quat_change_range > 0:
                    just_orientations_size = priors_size * 4
                    param_space["quat"] = gym.spaces.Box(
                        low=-self.quat_change_range,
                        high=self.quat_change_range,
                        shape=(just_orientations_size,),
                    )

        return gym.spaces.Dict(param_space)

    def update_gaussians(self, gmm_change):
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
        self.gmm.update_model(change_dict)
