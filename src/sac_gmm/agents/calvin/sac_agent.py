import logging
import hydra
from omegaconf import DictConfig
import os
import gym
import torch
import numpy as np
from tqdm import tqdm
from pytorch_lightning.utilities import rank_zero_only
from sac_gmm.agents.calvin.calvin_agent import CALVINAgent

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class SACAgent(CALVINAgent):
    def __init__(
        self,
        calvin_env: DictConfig,
        datamodule: DictConfig,
        num_init_steps: int,
        num_eval_episodes: int,
        skill: DictConfig,
        exp_dir: str,
        render: bool,
        record: bool,
    ) -> None:
        super(SACAgent, self).__init__(
            name="SAC",
            env=calvin_env,
            num_init_steps=num_init_steps,
            num_eval_episodes=num_eval_episodes,
        )

        self.skill = skill

        # Environment
        self.env.set_skill(self.skill)
        self.robot_obs, self.cam_obs = self.env.obs_allowed
        # # TODO: find the transforms for this
        # env.set_obs_transforms(cfg.datamodule.transforms)

        # record setup
        self.video_dir = os.path.join(exp_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)
        self.env.set_outdir(self.video_dir)
        self.render = render
        self.record = record

        # Dataset (helps with EE start positions)
        self.datamodule = hydra.utils.instantiate(datamodule)
        self.reset()

    def reset(self) -> None:
        """Resets the environment, moves the EE to a good start state and updates the agent state"""
        super().reset()
        self.obs = self.env.sample_start_position(self.datamodule.dataset)

    def get_state_dim(self):
        """Returns the size of the state based on env's observation space"""
        state_dim = 0
        robot_obs, cam_obs = self.env.obs_allowed
        if "pos" in robot_obs:
            state_dim += 3
        if "ori" in robot_obs:
            state_dim += 4
        if "joint" in robot_obs:
            state_dim = 7

        return state_dim

    def get_action_space(self):
        pos = np.ones(3)
        if self.skill.state_type == "pos":
            action_high = pos
        elif self.skill.state_type == "pos_ori":
            ori = np.ones(4)
            action_high = np.concatenate((pos, ori), axis=-1)
        action_low = -action_high
        self.action_space = gym.spaces.Box(action_low, action_high)

        return self.action_space

    @torch.no_grad()
    def play_step(self, actor, strategy="stochastic", replay_buffer=None):
        """Perform a step in the environment and add the transition
        tuple to the replay buffer"""
        action = self.get_action(actor, self.obs, strategy)
        action_with_fixori = np.concatenate((action, np.zeros(3)))
        action_with_gripper = np.append(action_with_fixori, -1)
        action_with_gripper = self.env.prepare_action(action_with_gripper, type="rel")
        next_obs, reward, done, info = self.env.step(action_with_gripper)

        replay_buffer.add(self.obs, action, reward, next_obs, done)
        self.obs = next_obs

        self.episode_env_steps += 1
        self.total_env_steps += 1

        self.episode_play_steps += 1
        self.total_play_steps += 1

        if self.episode_env_steps >= self.skill.max_steps:
            done = True

        if done:
            self.reset()
        return reward, done

    @torch.no_grad()
    def evaluate(self, actor):
        """Evaluates the actor in the environment"""
        log_rank_0("Evaluation episodes in process")
        succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
        saved_video_path = None
        # Choose a random episode to record
        rand_idx = np.random.randint(1, self.num_eval_episodes + 1)
        for episode in tqdm(range(1, self.num_eval_episodes + 1)):
            episode_env_steps = 0
            episode_return = 0
            self.obs = self.env.reset()
            # Start from a known starting point
            self.obs = self.env.sample_start_position(self.datamodule.dataset)
            # Recording setup
            if self.record and (episode == rand_idx):
                self.env.reset_recorded_frames()
                self.env.record_frame(size=64)
            obs = self.obs
            while episode_env_steps < self.skill.max_steps:
                action = self.get_action(actor, obs, "deterministic")
                action_with_fixori = np.concatenate((action, np.zeros(3)))
                action_with_gripper = np.append(action_with_fixori, -1)
                action = self.env.prepare_action(action_with_gripper, type="rel")
                obs, reward, done, info = self.env.step(action)
                episode_env_steps += 1
                episode_return += reward

                if self.record and (episode == rand_idx):
                    self.env.record_frame(size=64)
                if self.render:
                    self.env.render()
                if done:
                    self.reset()
                    break

            if ("success" in info) and info["success"]:
                succesful_episodes += 1
            # Recording setup close
            if self.record and (episode == rand_idx):
                video_path = self.env.save_recorded_frames(
                    outdir=self.video_dir,
                    fname=f"{self.total_play_steps}_{self.total_env_steps }_{episode}",
                )
                self.env.reset_recorded_frames()
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
