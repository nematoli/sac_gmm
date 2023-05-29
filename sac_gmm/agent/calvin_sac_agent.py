from omegaconf import OmegaConf
import os
import gym
import torch
import numpy as np
from tqdm import tqdm
import logging
import hydra
import copy
from sac_gmm.envs.skill_env import SkillSpecificEnv
from sac_gmm.utils.misc import transform_to_tensor, get_state_from_observation


class CALVINSACAgent(object):
    def __init__(
        self,
        calvin_env,
        dataset,
        skill,
        state_type,
        env_obs_allowed,
        device,
        num_train_steps,
        num_seed_steps,
        eval_frequency,
        num_eval_episodes,
        gmm_window_size,
        max_episode_steps,
        record,
        render,
        wandb,
        exp_dir,
        skills_dir,
        demos_dir,
    ) -> None:
        self.cfg = OmegaConf.create(
            {
                "skill": skill,
                "state_type": state_type,
                "env_obs_allowed": env_obs_allowed,
                "num_train_steps": num_train_steps,
                "num_seed_steps": num_seed_steps,
                "eval_frequency": eval_frequency,
                "num_eval_episodes": num_eval_episodes,
                "gmm_window_size": gmm_window_size,
                "max_episode_steps": max_episode_steps,
                "record": record,
                "render": render,
                "wandb": wandb,
                "exp_dir": exp_dir,
                "skills_dir": skills_dir,
                "demos_dir": demos_dir,
            }
        )
        self.device = device

        # Environment
        self.calvin_env_cfg = calvin_env
        self.env = self.make_env()
        self.action_space = self.get_action_space()
        self.robot_obs, self.cam_obs = self.env.obs_allowed

        # Dataset (helps with EE start positions)
        dataset.state_type = "pos"
        self.dataset = hydra.utils.instantiate(dataset)
        self.fixed_ori = self.dataset.fixed_ori

        # Trackers
        # State variable
        self.observation = None
        # At any point in time, my agent can only perform self.cfg.max_episode_steps number of
        # "play_step"s in a given episode, this tracks that
        self.episode_steps = 0
        # This tracks total "play_steps" taken in an experiment
        self.steps = 0
        # This tracks total environment steps taken in an experiment
        self.env_steps = 0
        # Agent resets - env and state variable self.observation
        self.reset()

        # Logging
        self.cons_logger = logging.getLogger("CALVINSeqblendRLAgent")
        if self.cfg.record:
            self.video_dir = os.path.join(self.cfg.exp_dir, "videos")
            os.makedirs(self.video_dir, exist_ok=True)

    def reset(self) -> None:
        """Resets the environment, moves the EE to a good start state and updates the agent state"""
        self.observation = self.env.reset()
        self.observation = self.calibrate_EE_start_position()
        self.episode_steps = 0

    def make_env(self):
        new_env_cfg = {**self.calvin_env_cfg.env}
        new_env_cfg["use_egl"] = False
        new_env_cfg["show_gui"] = False
        new_env_cfg["use_vr"] = False
        new_env_cfg["use_scene_info"] = True
        new_env_cfg["tasks"] = self.calvin_env_cfg.tasks
        new_env_cfg.pop("_target_", None)
        new_env_cfg.pop("_recursive_", None)

        env = SkillSpecificEnv(**new_env_cfg)
        env.set_skill(self.cfg.skill)
        env.set_state_type(self.cfg.state_type)
        env.set_state_type(self.cfg.state_type)

        self.video_dir = os.path.join(self.cfg.exp_dir, "videos")
        env.set_outdir(self.video_dir)

        env.set_obs_allowed(self.cfg.env_obs_allowed)
        env.observation_space = env.get_obs_space()

        return env

    def get_state_dim(self, compact_rep_size):
        """Returns the size of the state based on env's observation space"""
        state_dim = 0
        robot_obs, cam_obs = self.env.obs_allowed
        if "pos" in robot_obs:
            state_dim += 3
        if "ori" in robot_obs:
            state_dim += 4
        if "joint" in robot_obs:
            state_dim = 7

        if cam_obs is not None:
            state_dim += compact_rep_size

        return state_dim

    def get_action_space(self):
        return self.env.action_space

    def play_step(self, actor, strategy="stochastic", replay_buffer=None):
        """Perform a step in the environment and add the transition
        tuple to the replay buffer"""
        action = self.get_action(actor, self.observation, strategy)
        action_with_gripper = np.append(action, -1)
        action_with_gripper = self.env.prepare_action(action_with_gripper, type="abs")
        next_observation, reward, done, info = self.env.step(action_with_gripper)

        replay_buffer.add(
            self.observation[self.robot_obs],
            action,
            reward,
            next_observation[self.robot_obs],
            done,
        )
        self.observation = next_observation

        self.steps += 1
        self.episode_steps += 1
        if done or (self.episode_steps >= self.cfg.max_episode_steps):
            self.reset()
            done = True
        return reward, done

    def populate_replay_buffer(self, actor, replay_buffer):
        """
        Carries out several steps through the environment to initially fill
        up the replay buffer with experiences from the GMM
        Args:
            steps: number of random steps to populate the buffer with
            strategy: strategy to follow to select actions to fill the replay buffer
        """
        self.cons_logger.info("Populating replay buffer with random warm up steps")
        for _ in tqdm(range(self.cfg.num_seed_steps)):
            self.play_step(actor=actor, strategy="random", replay_buffer=replay_buffer)

        replay_buffer.save()

    def get_action(self, actor, observation, strategy="stochastic"):
        """Interface to get action from SAC Actor,
        ready to be used in the environment"""
        if strategy == "random":
            return self.action_space.sample()
        elif strategy == "zeros":
            return np.zeros(self.action_space.shape)
        elif strategy == "stochastic":
            deterministic = False
        elif strategy == "deterministic":
            deterministic = True
        else:
            raise Exception("Strategy not implemented")
        observation = transform_to_tensor(observation, device=self.device)
        action, _ = actor.get_actions(observation, deterministic=deterministic, reparameterize=False)

        return action.detach().cpu().numpy()

    @torch.no_grad()
    def evaluate(self, actor):
        """Evaluates the actor in the environment"""
        self.cons_logger.info("Evaluation episodes in process")
        succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
        saved_video_path = None
        # Choose a random episode to record
        rand_idx = np.random.randint(0, self.cfg.num_eval_episodes)
        for episode in tqdm(range(1, self.cfg.num_eval_episodes + 1)):
            episode_steps = 0
            episode_return = 0
            self.observation = self.env.reset()
            # Start from a known starting point
            self.observation = self.calibrate_EE_start_position()
            # Recording setup
            if self.cfg.record and (episode == rand_idx):
                self.env.reset_recorded_frames()
                self.env.record_frame(size=64)
            obs = self.observation
            while episode_steps < self.cfg.max_episode_steps:
                action = self.get_action(actor, obs, "deterministic")
                action = self.env.prepare_action(action, type="joint")
                obs, reward, done, info = self.env.step(action)
                episode_steps += 1
                episode_return += reward
                if self.cfg.record and (episode == rand_idx):
                    self.env.record_frame(size=64)
                if done:
                    self.reset()
                    break
            if ("success" in info) and info["success"]:
                succesful_episodes += 1
            # Recording setup close
            if self.cfg.record and (episode == rand_idx):
                video_path = self.env.save_recorded_frames(
                    outdir=self.video_dir,
                    fname=f"{self.steps}_{self.env_steps}_{episode}",
                )
                self.env.reset_recorded_frames()
                saved_video_path = video_path

            episodes_returns.append(episode_return)
            episodes_lengths.append(episode_steps)
        accuracy = succesful_episodes / self.cfg.num_eval_episodes

        return (
            accuracy,
            np.mean(episodes_returns),
            np.mean(episodes_lengths),
            saved_video_path,
        )

    def calibrate_EE_start_position(self, error_margin=0.01, max_checks=15):
        """Samples a random starting point and moves the end effector to that point"""
        desired_start_pos = self.dataset.sample_start(size=1, sigma=0.05)
        count = 0
        start_state = np.append(desired_start_pos, np.append(self.fixed_ori, -1))
        # Change env params temporarily
        temp = self.env.state_type
        temp_obs_allowed = self.env.obs_allowed
        self.env.set_state_type("pos")
        self.env.set_obs_allowed(["pos", None])
        obs = self.env.reset()
        action = self.env.prepare_action(start_state, type="abs")
        robot_obs = obs["pos"]
        while np.linalg.norm(robot_obs - desired_start_pos) > error_margin:
            obs, _, _, _ = self.env.step(action)
            count += 1
            if count >= max_checks:
                # self.cons_logger.info(
                #     f"CALVIN is struggling to place the EE at the right initial pose. \
                #         Difference: {np.linalg.norm(obs - desired_start)}"
                # )
                break
            robot_obs = obs["pos"]
        # Change them back to original
        self.env.set_state_type(temp)
        self.env.set_obs_allowed(temp_obs_allowed)
        obs = self.env.get_obs()

        return obs
