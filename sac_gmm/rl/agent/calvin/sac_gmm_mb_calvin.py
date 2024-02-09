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
from sac_gmm.gmm.batch_gmm import BatchGMM
from sac_gmm.utils.utils import LinearDecay

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


OBS_KEY = "rgb_static"
# OBS_KEY = "rgb_gripper"
# OBS_KEY = "robot_obs"


class CALVINSACGMM_MB_Agent(BaseAgent):
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
        root_dir: str,
        cem_cfg: DictConfig,
    ) -> None:
        super(CALVINSACGMM_MB_Agent, self).__init__(
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
        self.gmm_window = None

        # Record setup
        self.video_dir = os.path.join(exp_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)
        self.render = render
        self.record = record

        self.reset()
        self.root_dir = root_dir
        self.cem_cfg = cem_cfg
        self._std_decay = LinearDecay(cem_cfg.max_std, cem_cfg.min_std, cem_cfg.std_step)
        self._horizon_decay = LinearDecay(1, 1, cem_cfg.horizon_step)

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
    def play_step_with_critic(self, actor, model, critic, strategy="stochastic", replay_buffer=None, device="cuda"):
        """Perform a step in the environment and add the transition
        tuple to the replay buffer"""
        # Change dynamical system
        self.gmm.copy_model(self.initial_gmm)
        gmm_change = self.get_action_with_critic(actor, model, critic, self.obs, strategy, device)
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

    @torch.no_grad()
    def estimate_value(self, refine_actor, model, critic, state, ac, dyn_ac, horizon):
        """Imagine a trajectory for `horizon` steps, and estimate the value."""
        value, discount = 0, 1
        for t in range(horizon):
            state = model.imagine_step(state, dyn_ac[t])
            reward = model.imagine_reward(state, ac[t])
            value += discount * reward
            discount *= self.cem_cfg.cem_discount
        next_rv, _ = refine_actor.get_actions(state)
        value += discount * torch.min(*critic(state, next_rv))
        return value

    @torch.no_grad()
    def plan(self, refine_actor, model, critic, rgb_ob, robot_ob, prev_mean=None, is_train=True, device="cuda"):
        cfg = self.cem_cfg
        step = self.total_env_steps
        horizon = int(self._horizon_decay(step))
        clamp_max = 0.999
        state = model.encoder({"obs": rgb_ob.float()}).squeeze(0)
        # Sample policy trajectories.
        robot_obs = robot_ob.repeat(cfg.num_policy_traj, 1)
        z = state.repeat(cfg.num_policy_traj, 1)
        policy_ac = []
        policy_rv = []
        for t in range(horizon):
            rv, _ = refine_actor.get_actions(z)
            ac = batch_actions(self.initial_gmm, robot_obs[:, :3].cpu(), rv, self.gmm_window).to(device)
            policy_ac.append(ac)
            policy_rv.append(rv)
            z = model.imagine_step(z, policy_ac[t])

        policy_ac = torch.stack(policy_ac, dim=0)
        policy_rv = torch.stack(policy_rv, dim=0)

        # CEM optimization.
        z = state.repeat(cfg.num_policy_traj + cfg.num_sample_traj, 1)
        robot_obs = robot_ob.repeat(cfg.num_policy_traj + cfg.num_sample_traj, 1)
        mean = torch.zeros(horizon, policy_rv.shape[-1], device=device)
        std = 2 * torch.ones(horizon, policy_rv.shape[-1], device=device)
        if prev_mean is not None and horizon > 1 and prev_mean.shape[0] == horizon:
            mean[:-1] = prev_mean[1:]

        for _ in range(cfg.cem_iter):
            sample_rv = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                horizon, cfg.num_sample_traj, self.action_space.shape[0], device=device
            )
            sample_rv = torch.clamp(sample_rv, -clamp_max, clamp_max)
            rv = torch.cat([sample_rv, policy_rv], dim=1)

            sample_ac = (
                batch_actions(
                    self.initial_gmm,
                    robot_obs[: cfg.num_sample_traj, :3].cpu(),
                    sample_rv.squeeze(0).cpu(),
                    self.gmm_window,
                )
                .to(device)
                .unsqueeze(0)
            )

            ac = torch.cat([sample_ac, policy_ac], dim=1)

            imagine_return = self.estimate_value(refine_actor, model, critic, z, rv, ac, horizon).squeeze(-1)
            _, idxs = imagine_return.sort(dim=0)
            idxs = idxs[-cfg.num_elites :]
            elite_value = imagine_return[idxs]
            elite_action = rv[:, idxs]

            # Weighted aggregation of elite plans.
            score = torch.exp(cfg.cem_temperature * (elite_value - elite_value.max()))
            score = (score / score.sum()).view(1, -1, 1)
            new_mean = (score * elite_action).sum(dim=1)
            new_std = torch.sqrt(torch.sum(score * (elite_action - new_mean.unsqueeze(1)) ** 2, dim=1))

            mean = cfg.cem_momentum * mean + (1 - cfg.cem_momentum) * new_mean
            std = torch.clamp(new_std, self._std_decay(step), 2)

        # Sample action for MPC.
        score = score.squeeze().cpu().numpy()
        rv = elite_action[0, np.random.choice(np.arange(cfg.num_elites), p=score)]
        if is_train:
            rv += std[0] * torch.randn_like(std[0])
        return torch.clamp(rv, -clamp_max, clamp_max), mean

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
        # elif strategy == "cem":
        #     critic.eval()
        #     rgb_obs = torch.from_numpy(observation[OBS_KEY]).to(device)
        #     action, _ = self.plan(actor, model, critic, rgb_obs, observation["robot_obs"], device=device)
        #     actor.train()
        #     model.train()
        #     critic.train()
        #     return action.detach().cpu().numpy()
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

    def get_action_with_critic(self, actor, model, critic, observation, strategy="stochastic", device="cuda"):
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
        elif strategy == "cem":
            critic.eval()
            rgb_ob = torch.from_numpy(observation[OBS_KEY]).to(device)
            robot_ob = torch.from_numpy(observation["robot_obs"]).to(device)
            action, _ = self.plan(actor, model, critic, rgb_ob, robot_ob, device=device)
            actor.train()
            model.train()
            critic.train()
            return action.detach().cpu().numpy()
        else:
            raise Exception("Strategy not implemented")
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation[OBS_KEY]).to(device)
            state = model.encoder({"obs": obs_tensor.float()}).squeeze(0)
        action, _ = actor.get_actions(state, deterministic=deterministic, reparameterize=False)
        actor.train()
        model.train()
        return action.detach().cpu().numpy()


def batch_actions(gmm, batch_x, batch_rv, skill_horizon):
    """
    Batch acts skill_horizon times
    """
    batch_size = batch_x.shape[0]
    out = torch.zeros((batch_size, skill_horizon, batch_x.shape[1]))

    batch_priors = np.repeat(np.expand_dims(gmm.priors, 0), batch_size, axis=0)
    batch_means = np.repeat(np.expand_dims(gmm.means, 0), batch_size, axis=0)
    batch_covariances = np.repeat(np.expand_dims(gmm.covariances, 0), batch_size, axis=0)
    # Batch refine (only means)
    batch_means += batch_rv.cpu().numpy().reshape(batch_means.shape) * 0.05

    # Batch predict
    for i in range(skill_horizon):
        batch_dx = batch_predict(
            gmm.n_components,
            batch_priors,
            batch_means,
            batch_covariances,
            batch_x,
            gmm.random_state,
        )
        out[:, i, :] = batch_dx
        batch_x += batch_dx * gmm.pos_dt

    return out.reshape((batch_size, -1))


def batch_predict(n_components, batch_priors, batch_means, batch_covariances, batch_x, random_state):
    """
    Batch Predict function for BayesianGMM

    Along the batch dimension, you have different means, covariances, and priors and input.
    The function outputs the predicted delta x for each batch.
    """
    batch_condition = BatchGMM(
        n_components=n_components,
        priors=batch_priors,
        means=batch_means,
        covariances=batch_covariances,
        random_state=random_state,
    ).condition([0, 1, 2], batch_x)
    return batch_condition.one_sample_confidence_region(alpha=0.7)
