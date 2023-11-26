import logging
import hydra
from omegaconf import DictConfig
import os
from enum import Enum
import gym
import torch
import numpy as np
from tqdm import tqdm
import copy
from pytorch_lightning.utilities import rank_zero_only
from sac_gmm.rl.agent.base_agent import BaseAgent
from sac_gmm.rl.helpers.skill_actor import SkillActor

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


OBS_KEY = "rgb_gripper"
# OBS_KEY = "robot_obs"


class CALVIN_SACNGMMAgent(BaseAgent):
    def __init__(
        self,
        calvin_env: DictConfig,
        datamodule: DictConfig,
        num_init_steps: int,
        num_eval_episodes: int,
        task: DictConfig,
        gmm: DictConfig,
        priors_change_range: float,
        mu_change_range: float,
        adapt_cov: bool,
        mean_shift: bool,
        adapt_per_skill: int,
        exp_dir: str,
        root_dir: str,
        render: bool,
        record: bool,
        device: str,
        sparse_reward: bool,
    ) -> None:
        super().__init__(
            env=calvin_env,
            num_init_steps=num_init_steps,
            num_eval_episodes=num_eval_episodes,
        )

        self.task = task
        self.task.max_steps = self.task.skill_max_steps

        # Environment
        self.env.set_task(self.task.skills)
        self.env.max_episode_steps = self.task.max_steps
        self.env.sparse_reward = sparse_reward

        # Refine parameters
        self.priors_change_range = priors_change_range
        self.mu_change_range = mu_change_range
        self.adapt_cov = adapt_cov
        self.mean_shift = mean_shift
        self.adapt_per_skill = adapt_per_skill
        self.gmm_window = 16  # self.task.max_steps // (self.adapt_per_skill * len(self.task.skills))

        # One SkillActor per set of skills
        self.skill_actor = SkillActor(self.task)
        # The order of skills inside actor should always be the same as the order of skills in the SKILLS enum
        self.skill_actor.skill_names = self.task.skills
        # GMM
        self.skill_actor.make_skills(gmm)
        # Load GMM weights of each skill
        self.skill_actor.load_models()
        # Use Dataset to set skill parameters - goal, fixed_ori, pos_dt, ori_dt
        self.skill_actor.set_skill_params(datamodule.dataset)
        if "Manifold" in self.skill_actor.name:
            self.skill_actor.make_manifolds()
        self.initial_gmms = copy.deepcopy(self.skill_actor.skills)
        self.skill_params_stacked = torch.from_numpy(self.skill_actor.get_all_skill_params(self.initial_gmms))

        # Store skill info - starts, goals, fixed_ori, pos_dt, ori_dt
        # self.env.store_skill_info(self.skill_actor.skills)

        # Record setup
        self.video_dir = os.path.join(exp_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)
        self.render = render
        self.record = record

        self.reset()
        self.skill_id = self.task.skills.index(self.env.target_skill)

        self.nan_counter = 0
        self.one_hot_skill_vector = False

    @torch.no_grad()
    def play_step(self, refine_actor, model, strategy="stochastic", replay_buffer=None, device="cuda"):
        """Perform a step in the environment and add the transition
        tuple to the replay buffer"""
        # Change dynamical system
        self.skill_actor.copy_model(self.initial_gmms, self.skill_id)
        gmm_change = self.get_action(refine_actor, model, self.skill_id, self.obs, strategy, device)
        self.update_gaussians(gmm_change, self.skill_id)

        # Act with the dynamical system in the environment
        gmm_reward = 0
        curr_obs = self.obs
        for _ in range(self.gmm_window):
            conn = self.env.isConnected()
            if not conn:
                done = False
                break
            env_action, is_nan = self.skill_actor.act(curr_obs["robot_obs"], self.skill_id)
            if is_nan:
                self.nan_counter += 1
                done = True
                log_rank_0("Nan in prediction, aborting episode")
            else:
                curr_obs, reward, done, info = self.env.step(env_action)
                gmm_reward += reward
                self.episode_env_steps += 1
                self.total_env_steps += 1
                if reward > 0:
                    # self.skill_id = (self.skill_id + 1) % len(self.task.skills)
                    done = True

            if done:
                break

        if self.episode_env_steps >= self.task.skill_max_steps:
            done = True

        # if done and (self.episode_env_steps < self.task.max_steps):
        #     # If the episode is done, the self.skill_counter rotates back to 0
        #     # but this is not the true next skill, so we need to set it to the actual last skill of the task
        #     next_skill_id = len(self.task.skills) - 1
        # else:
        next_skill_id = self.skill_id

        replay_buffer.add(self.obs, self.skill_id, gmm_change, gmm_reward, curr_obs, next_skill_id, done)
        self.obs = curr_obs

        self.episode_play_steps += 1
        self.total_play_steps += 1

        if done or not conn:
            self.reset()
            self.skill_id = self.task.skills.index(self.env.target_skill)
        return gmm_reward, done

    @torch.no_grad()
    def evaluate(self, refine_actor, model, device="cuda"):
        """Evaluates the actor in the environment"""
        log_rank_0("Evaluation episodes in process")
        succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
        saved_video_paths = {}
        succesful_skill_ids = []
        # Choose a random episode to record
        for skill in self.task.skills:
            rand_idx = np.random.randint(1, self.num_eval_episodes + 1)
            for episode in tqdm(range(1, self.num_eval_episodes + 1)):
                episode_return, episode_env_steps = 0, 0
                self.obs = self.env.reset(target_skill=skill)
                skill_id = self.task.skills.index(skill)
                # log_rank_0(f"Skill: {skill} - Obs: {self.obs['robot_obs']}")
                # Recording setup
                if self.record and (episode == rand_idx):
                    self.env.reset_recording()
                    self.env.record_frame(size=200)

                while episode_env_steps < self.task.max_steps:
                    # Change dynamical system
                    self.skill_actor.copy_model(self.initial_gmms, skill_id)
                    gmm_change = self.get_action(refine_actor, model, skill_id, self.obs, "deterministic", device)
                    self.update_gaussians(gmm_change, skill_id)

                    # Act with the dynamical system in the environment
                    # x = self.obs["position"]

                    for _ in range(self.gmm_window):
                        env_action, is_nan = self.skill_actor.act(self.obs["robot_obs"], skill_id)
                        if is_nan:
                            done = True
                            log_rank_0("Nan in prediction, aborting episode")
                        else:
                            self.obs, reward, done, info = self.env.step(env_action)
                            episode_return += reward
                            episode_env_steps += 1

                            if reward > 0:
                                succesful_skill_ids.append(skill_id)
                        if self.record and (episode == rand_idx):
                            self.env.record_frame(size=200)
                        if self.render:
                            self.env.render()
                        if done:
                            break

                    if done:
                        self.reset(target_skill=skill)
                        skill_id = self.task.skills.index(skill)
                        break

                if ("success" in info) and info["success"]:
                    succesful_episodes += 1
                # Recording setup close
                if self.record and (episode == rand_idx):
                    video_path = self.env.save_recording(
                        outdir=self.video_dir,
                        fname=f"PlaySteps{self.total_play_steps}_EnvSteps{self.total_env_steps }_{skill}",
                    )
                    self.env.reset_recording()
                    saved_video_paths[skill] = video_path

            episodes_returns.append(episode_return)
            episodes_lengths.append(episode_env_steps)
        accuracy = succesful_episodes / (self.num_eval_episodes * len(self.task.skills))
        return (
            accuracy,
            np.mean(episodes_returns),
            np.mean(episodes_lengths),
            succesful_skill_ids,
            saved_video_paths,
        )

    def get_action_space(self):
        parameter_space = self.get_update_range_parameter_space()
        mu_high = np.ones(parameter_space["mu"].shape[0])
        priors_high = np.ones(parameter_space["priors"].shape[0])
        action_high = np.concatenate((priors_high, mu_high), axis=-1)
        if self.adapt_cov:
            sigma_high = np.ones(parameter_space["sigma"].shape[0])
            action_high = np.concatenate((action_high, sigma_high), axis=-1)

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
        param_space["priors"] = gym.spaces.Box(
            low=-self.priors_change_range,
            high=self.priors_change_range,
            shape=(self.skill_actor.priors_size,),
        )

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

        # dim = self.gmm.means.shape[1] // 2
        # num_gaussians = self.gmm.means.shape[0]
        # sigma_change_size = int(num_gaussians * dim * (dim + 1) / 2 + dim * dim * num_gaussians)
        # param_space["sigma"] = gym.spaces.Box(low=-1e-6, high=1e-6, shape=(sigma_change_size,))
        return gym.spaces.Dict(param_space)

    def update_gaussians(self, gmm_change, skill_id):
        parameter_space = self.get_update_range_parameter_space()
        size_priors = parameter_space["priors"].shape[0]
        size_mu = parameter_space["mu"].shape[0]

        priors = gmm_change[:size_priors] * parameter_space["priors"].high
        mu = gmm_change[size_priors : size_priors + size_mu] * parameter_space["mu"].high

        change_dict = {"mu": mu, "priors": priors}
        self.skill_actor.update_model(change_dict, skill_id)

    def get_skill_vector(self, skill_id, device="cuda"):
        if type(skill_id) is int:  # When skill_id is of shape (Batch x 1)
            if self.one_hot_skill_vector:
                skill_vector = torch.eye(len(self.task.skills))[skill_id]
            else:
                skill_vector = self.skill_params_stacked[skill_id].squeeze(0)
        else:
            if self.one_hot_skill_vector:
                skill_vector = torch.eye(len(self.task.skills))[skill_id[:, 0].cpu().int()]
            else:
                skill_vector = self.skill_params_stacked[skill_id[:, 0].cpu().int()]
        return skill_vector.to(device)

    def get_state_from_observation(self, encoder, obs, skill_id, device="cuda"):
        skill_vector = self.get_skill_vector(skill_id, device=device)
        if isinstance(obs, dict):
            # Robot obs
            if "state" in obs:
                name = "state"
            elif "robot_obs" in obs:
                name = "robot_obs"
            # RGB obs
            if "rgb_gripper" in obs:
                x = obs["rgb_gripper"]
                if not torch.is_tensor(x):
                    x = torch.tensor(x).to(device)
                if len(x.shape) < 4:
                    x = x.unsqueeze(0)
                with torch.no_grad():
                    features = encoder(x)
                    # features = encoder({"obs": x.float()})
                # If position are part of the input state
                # if features is not None:
                #     fc_input = torch.cat((fc_input, features.squeeze()), dim=-1).to(device)
                # Else
                fc_input = features.squeeze()

            fc_input = torch.cat((fc_input, skill_vector.squeeze()), dim=-1).to(device)
            return fc_input.float()

        return skill_vector

    def get_action(self, actor, model, skill_id, observation, strategy="stochastic", device="cuda"):
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
            skill_vector = self.get_skill_vector(skill_id, device)
            obs_tensor = torch.from_numpy(observation[OBS_KEY]).to(device)
            enc_ob = model.encoder({"obs": obs_tensor.float()}).squeeze(0)
            actor_input = torch.cat((enc_ob, skill_vector), dim=-1).to(device).float()
            action, _ = actor.get_actions(actor_input, deterministic=deterministic, reparameterize=False)
        actor.train()
        model.train()
        return action.detach().cpu().numpy()
