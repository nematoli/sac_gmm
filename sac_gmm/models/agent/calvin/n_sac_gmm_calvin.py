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
from sac_gmm.models.agent.agent import Agent
from sac_gmm.models.skill_actor_nsacgmm import SkillActorWithNSACGMM

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


# This helps load each skill's model checkpoints in an set order
# irrespective of the skill order given in the config file
class SKILLS(Enum):
    open_drawer = 0
    turn_on_lightbulb = 1
    move_slider_left = 2
    turn_on_led = 3


class CALVIN_NSACGMMAgent(Agent):
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
        sac: DictConfig,
        model_ckpts: list,
        device: str,
    ) -> None:
        super().__init__(
            env=calvin_env,
            num_init_steps=num_init_steps,
            num_eval_episodes=num_eval_episodes,
        )

        self.task = task

        # Environment
        self.env.set_task(self.task.skills)
        self.env.max_episode_steps = self.task.max_steps

        # Refine parameters
        self.priors_change_range = priors_change_range
        self.mu_change_range = mu_change_range
        self.adapt_cov = adapt_cov
        self.mean_shift = mean_shift
        self.adapt_per_skill = adapt_per_skill
        self.gmm_window = self.task.max_steps // (self.adapt_per_skill * len(self.task.skills))

        # One SkillActorWithNSACGMM per set of skills
        self.actor = SkillActorWithNSACGMM(self.task)
        # The order of skills inside actor should always be the same as the order of skills in the SKILLS enum
        self.actor.skill_names = [e.name for e in SKILLS]
        # GMM
        self.actor.make_skills(gmm)
        # Load GMM weights of each skill
        self.actor.load_models()
        # Use Dataset to set skill parameters - goal, fixed_ori, pos_dt, ori_dt
        self.actor.set_skill_params(datamodule.dataset)
        if "Manifold" in self.actor.name:
            self.actor.make_manifolds()
        self.initial_gmms = copy.deepcopy(self.actor.skills)
        # Refine Actors to refine each skill
        self.sacgmms = self.actor.make_sacgmms(sac, model_ckpts, root_dir, device, self.get_action_space())

        # Set initial pos and orn
        self.env.store_skill_info(self.actor.skills)
        # # record setup
        self.video_dir = os.path.join(exp_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

        self.render = render
        self.record = record

        self.reset()
        self.skill_count = 0

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
    def evaluate(self, device="cuda"):
        """Evaluates the actor in the environment"""
        log_rank_0("Evaluation episodes in process")
        succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
        saved_video_path = None
        # Choose a random episode to record
        rand_idx = np.random.randint(1, self.num_eval_episodes + 1)
        self.env.eval_mode = True
        for episode in tqdm(range(1, self.num_eval_episodes + 1)):
            episode_return, episode_env_steps = 0, 0
            self.obs = self.env.reset()
            skill_count = 0
            # Recording setup
            if self.record:  # and (episode == rand_idx):
                self.env.reset_recording()
                self.env.record_frame(size=100)

            while episode_env_steps < self.task.max_steps:
                # Change dynamical system
                skill_id = SKILLS[self.task.skills[skill_count]].value
                self.actor.copy_model(self.initial_gmms, skill_id)
                gmm_change = self.get_action(self.sacgmms[skill_id], self.obs, "deterministic", device)
                self.update_gaussians(gmm_change, skill_id)

                # Act with the dynamical system in the environment
                # x = self.obs["position"]

                for _ in range(self.gmm_window):
                    env_action = self.actor.act(self.obs["robot_obs"], skill_id)
                    self.obs, reward, done, info = self.env.step(env_action)
                    episode_return += reward
                    episode_env_steps += 1

                    if self.record:  # and (episode == rand_idx):
                        self.env.record_frame(size=100)
                    if self.render:
                        self.env.render()
                    if reward > 0:
                        skill_count = (skill_count + 1) % len(self.task.skills)
                    if done:
                        break

                if done:
                    self.reset()
                    skill_count = 0
                    break

            if ("success" in info) and info["success"]:
                succesful_episodes += 1
            # Recording setup close
            if self.record:  # and (episode == rand_idx):
                video_path = self.env.save_recording(
                    outdir=self.video_dir,
                    fname=f"{self.total_play_steps}_{self.total_env_steps }_{episode}",
                )
                self.env.reset_recording()
                saved_video_path = video_path

            episodes_returns.append(episode_return)
            episodes_lengths.append(episode_env_steps)
        accuracy = succesful_episodes / self.num_eval_episodes
        self.env.eval_mode = False
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
            low=-self.priors_change_range, high=self.priors_change_range, shape=(self.actor.skills[0].priors.size,)
        )
        param_space["mu"] = gym.spaces.Box(
            low=-self.mu_change_range, high=self.mu_change_range, shape=(self.actor.skills[0].means.size,)
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
        # if self.adapt_cov:
        #     change_dict["sigma"] = gmm_change[size_priors + size_mu :] * parameter_space["sigma"].high
        self.actor.update_model(change_dict, skill_id)

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


@hydra.main(config_path="../../../../config", config_name="n_sac_gmm_run")
def main(cfg: DictConfig) -> None:
    cfg.agent.exp_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    agent = hydra.utils.instantiate(cfg.agent)
    acc, ep_ret, ep_len, saved_video_path = agent.evaluate()
    log_rank_0(f"Accuracy: {acc}, Average Return: {ep_ret}, Average Trajectory Length: {ep_len}")
    log_rank_0(f"Saved video at {saved_video_path}")


if __name__ == "__main__":
    main()
