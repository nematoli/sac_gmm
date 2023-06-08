import os
import sys
from pathlib import Path

cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[0]
root = sac_gmm_path.parents[0]

# This is to access the locally installed repo clone when using slurm
sys.path.insert(0, sac_gmm_path.as_posix())  # sac_gmm
sys.path.insert(0, os.path.join(root, "calvin_env"))  # root/calvin_env
sys.path.insert(0, root.as_posix())  # Root

import gym
from typing import Dict

# from sac_gmm.utils.misc import resize_cam_obs
from sac_gmm.datasets.utils.load_utils import get_transforms
from calvin_env.envs.play_table_env import PlayTableSimEnv
import hydra
import imageio
import cv2
import numpy as np


class SkillSpecificEnv(PlayTableSimEnv):
    def __init__(self, tasks: dict = {}, **kwargs):
        super(SkillSpecificEnv, self).__init__(**kwargs)
        # For this example we will modify the observation to
        # only retrieve the end effector pose
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,))
        self.obs_allowed = ["pos", None]  # robot obs, cam obs (order is important)
        self.observation_space = None
        # We can use the task utility to know if the task was executed correctly
        self.tasks = hydra.utils.instantiate(tasks)

        self.skill = None

        self.frames = []
        self.outdir = None
        self.record_count = 1

        self.transform_rgb, self.transform_dpt = None, None

    def set_skill(self, skill):
        """Set skill"""
        self.skill = skill

    def set_obs_allowed(self, obs_allowed):
        """Set what observations env should return"""
        assert len(obs_allowed) == 2, "Input must be a list of size 2"
        self.obs_allowed = obs_allowed

    def set_outdir(self, outdir):
        """Set output directory where recordings can/will be saved"""
        self.outdir = outdir

    def set_obs_transforms(self, transforms: Dict):
        """Set the transformation on observations for the RL agent"""
        self.transform_rgb = get_transforms(transforms.rgb)
        self.transform_dpt = get_transforms(transforms.depth)
        return

    def reset(self):
        obs = super().reset()
        self.start_info = self.get_info()
        self.reset_recorded_frames()
        return obs

    def get_obs(self):
        """Overwrite robot obs to only retrieve end effector position"""
        obs = {}
        robot_obs_allowed, cam_obs_allowed = self.obs_allowed
        robot_obs, robot_info = self.robot.get_observation()

        if robot_obs_allowed == "pos":
            obs[robot_obs_allowed] = robot_obs[:3]
        elif robot_obs_allowed == "pos_ori":
            obs[robot_obs_allowed] = robot_obs[:7]
        elif robot_obs_allowed == "joint":
            obs[robot_obs_allowed] = robot_obs[8:15]
        else:
            raise ValueError("Invalid value (robot obs) inside obs_allowed list")

        if cam_obs_allowed is not None:
            rgb_obs, depth_obs = self.get_camera_obs(transform=True)
            if "rgb" in cam_obs_allowed:
                obs[cam_obs_allowed] = rgb_obs[cam_obs_allowed]
            elif "depth" in cam_obs_allowed:
                obs[cam_obs_allowed] = depth_obs[cam_obs_allowed]
            else:
                raise ValueError("Invalid value (camera obs) inside obs_allowed list")
        return obs

    def get_camera_obs(self, transform=False):
        """Collect camera, robot and scene observations."""
        assert self.cameras is not None
        rgb_obs = {}
        depth_obs = {}
        for cam in self.cameras:
            rgb, depth = cam.render()
            if transform:
                rgb_obs[f"rgb_{cam.name}"] = self.transform_rgb(rgb)
                # TODO: make sure transformed depth obs is what you want
                depth_obs[f"depth_{cam.name}"] = self.transform_dpt(depth)
            else:
                rgb_obs[f"rgb_{cam.name}"] = rgb
                depth_obs[f"depth_{cam.name}"] = depth
        return rgb_obs, depth_obs

    def record_frame(self, obs_type="rgb", cam_type="static", size=208):
        """Record RGB obsservation"""
        rgb_obs, depth_obs = self.get_camera_obs()
        if obs_type == "rgb":
            frame = rgb_obs[f"{obs_type}_{cam_type}"]
        else:
            frame = depth_obs[f"{obs_type}_{cam_type}"]
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        self.frames.append(frame)

    def reset_recorded_frames(self):
        """Reset recorded frames"""
        self.frames = []

    def save_recorded_frames(self, outdir, fname):
        """Save recorded frames as a video"""
        fname = f"{fname}.mp4"
        fpath = os.path.join(outdir, fname)
        imageio.mimsave(fpath, self.frames, fps=30)
        return fpath

    def _success(self):
        """Returns a boolean indicating if the task was performed correctly"""
        current_info = self.get_info()
        task_filter = [self.skill.name]
        task_info = self.tasks.get_task_info_for_set(self.start_info, current_info, task_filter)
        return self.skill.name in task_info

    def _reward(self):
        """Returns the reward function that will be used
        for the RL algorithm"""
        reward = int(self._success()) * 10
        r_info = {"reward": reward}
        return reward, r_info

    def _termination(self):
        """Indicates if the robot has reached a terminal state"""
        success = self._success()
        done = success
        d_info = {"success": success}
        return done, d_info

    def get_valid_columns(self):
        if "joint" in self.skill.state_type:
            start, end = 8, 15
        elif "pos_ori" in self.skill.state_type:
            start, end = 1, 7
        elif "pos" in self.skill.state_type:
            start, end = 1, 4
        elif "ori" in self.skill.state_type:
            start, end = 4, 7
        elif "grip" in self.skill.state_type:
            start, end = 7, 8
        return start - 1, end - 1

    def prepare_action(self, input, type):
        action = []
        if self.skill.state_type == "joint":
            action = {"type": f"joint_{type}", "action": None}
        elif "pos" in self.skill.state_type:
            action = {"type": f"cartesian_{type}", "action": None}

        action["action"] = input
        return action

    def step(self, action):
        """Performing a relative action in the environment
        input:
            action: 7 tuple containing
                    Position x, y, z.
                    Angle in rad x, y, z.
                    Gripper action
                    each value in range (-1, 1)
        output:
            observation, reward, done info
        """
        # Transform gripper action to discrete space
        env_action = action.copy()
        env_action["action"][-1] = (int(action["action"][-1] >= 0) * 2) - 1
        self.robot.apply_action(env_action)
        for i in range(self.action_repeat):
            self.p.stepSimulation(physicsClientId=self.cid)
        obs = self.get_obs()
        info = self.get_info()
        reward, r_info = self._reward()
        done, d_info = self._termination()
        info.update(r_info)
        info.update(d_info)
        return obs, reward, done, info

    def get_obs_space(self):
        obs_space = {}
        robot_obs, cam_obs = self.obs_allowed

        # Robot EE position
        if robot_obs == "pos":
            obs_space[robot_obs] = gym.spaces.Box(low=-np.ones(3), high=np.ones(3))
        elif robot_obs == "pos_ori":
            obs_space[robot_obs] = gym.spaces.Box(low=-np.ones(7), high=np.ones(7))
        elif robot_obs == "joint":
            obs_space[robot_obs] = gym.spaces.Box(low=-np.ones(7), high=np.ones(7))

        # Camera
        if cam_obs is not None:
            rgb, depth = self.get_camera_obs(transform=True)
            high = None
            if "rgb" in cam_obs:
                high = np.ones((rgb[cam_obs].shape[0], rgb[cam_obs].shape[1], rgb[cam_obs].shape[2]))
            else:
                high = np.ones((depth[cam_obs].shape[0], depth[cam_obs].shape[1]))
            obs_space[cam_obs] = gym.spaces.Box(low=-high, high=high)
        return gym.spaces.Dict(obs_space)

    def sample_start_position(self, dataset, error_margin=0.01, max_checks=15):
        """Samples a random starting point and moves the end effector to that point"""
        desired_start_pos = dataset.sample_start(size=1, sigma=0.05)
        count = 0
        start_state = np.append(desired_start_pos, np.append(dataset.fixed_ori, -1))
        # Change env params temporarily
        temp = self.skill.state_type
        temp_obs_allowed = self.obs_allowed
        self.set_state_type("pos")
        self.set_obs_allowed(["pos", None])
        obs = self.reset()
        action = self.prepare_action(start_state, type="abs")
        robot_obs = obs["pos"]
        while np.linalg.norm(robot_obs - desired_start_pos) > error_margin:
            obs, _, _, _ = self.step(action)
            count += 1
            if count >= max_checks:
                # self.cons_logger.info(
                #     f"CALVIN is struggling to place the EE at the right initial pose. \
                #         Difference: {np.linalg.norm(obs - desired_start)}"
                # )
                break
            robot_obs = obs["pos"]
        # Change them back to original
        self.set_state_type(temp)
        self.set_obs_allowed(temp_obs_allowed)
        obs = self.get_obs()

        return obs
