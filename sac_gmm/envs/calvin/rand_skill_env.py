import copy
from collections import defaultdict
import os
import gc
import pybullet as p
import pybullet_utils.bullet_client as bc
import pkgutil
import sys
import hydra
import numpy as np
import gym
import cv2
import imageio
from calvin_env.envs.play_table_env import PlayTableSimEnv

import logging

logger = logging.getLogger(__name__)

GYM_POSITION_INDICES = np.array([0, 1, 2])
GYM_ORIENTATION_INDICES = np.array([3, 4, 5])


class CalvinRandSkillEnv(PlayTableSimEnv):
    def __init__(self, cfg):
        pt_cfg = {**cfg.calvin_env.env}
        pt_cfg.pop("_target_", None)
        pt_cfg.pop("_recursive_", None)
        self.robot_cfg = pt_cfg["robot_cfg"]
        self.scene_cfg = pt_cfg["scene_cfg"]
        self.cameras_c = pt_cfg["cameras"]
        super(CalvinRandSkillEnv, self).__init__(**pt_cfg)

        self.gripper_width = 64

        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.tasks = hydra.utils.instantiate(cfg.calvin_env.tasks)
        self.target_tasks = list(self.tasks.tasks.keys())
        self.target_skill = np.random.choice(self.target_tasks)
        self._t = 0
        self.max_episode_steps = 0
        self.reward_scale = 10

        self.init_base_pos, self.init_base_orn = self.p.getBasePositionAndOrientation(self.robot.robot_uid)
        self.ee_noise = np.array([0.4, 0.3, 0.1])  # Units: meters
        self.init_pos = None
        self.eval_mode = False

        self.skill_starts = {}
        self.skill_goals = {}
        self.skill_oris = {}
        self.centroid = np.array([0.036, -0.13, 0.509])

    @staticmethod
    def get_action_space():
        """End effector position and gripper width relative displacement"""
        return gym.spaces.Box(low=-1, high=1, shape=(7,))

    def get_observation_space(self):
        """Return only position and gripper_image by default"""
        observation_space = {}
        observation_space["robot_obs"] = gym.spaces.Box(low=-1, high=1, shape=(21,))
        observation_space["rgb_gripper"] = gym.spaces.Box(
            low=-1, high=1, shape=(self.gripper_width, self.gripper_width, 3)
        )
        return gym.spaces.Dict(observation_space)

    def get_obs(self):
        obs = super().get_obs()

        nobs = {}
        nobs["robot_obs"] = np.concatenate([obs["robot_obs"], obs["scene_obs"]])[:21]
        nobs["rgb_gripper"] = cv2.resize(
            obs["rgb_obs"]["rgb_gripper"], (self.gripper_width, self.gripper_width), interpolation=cv2.INTER_AREA
        )
        return nobs

    def set_task(self, task):
        self.target_tasks = task
        self.target_skill = np.random.choice(self.target_tasks)

    def reset(self, robot_obs=None, scene_obs=None, target_skill=None):
        if not self.isConnected():
            h, w = self.rhw
            self.initialize_bullet(self.bts, w, h)
            self.load()
        self.scene.reset(scene_obs)
        if robot_obs is None:
            npos, norn = self.sample_base_shift()
            self.p.resetBasePositionAndOrientation(self.robot.robot_uid, npos, norn)
        self.robot.reset(robot_obs)
        self.p.stepSimulation(physicsClientId=self.cid)

        if target_skill is None:
            self.target_skill = np.random.choice(self.target_tasks)
        else:
            self.target_skill = target_skill

        self.calibrate_scene(self.target_skill)
        self.calibrate_EE_start_state(self.get_state_obs()["robot_obs"])
        self.start_info = self.get_info()
        self._t = 0
        self.completed_tasks = []
        self.solved_subtasks = defaultdict(lambda: 0)
        return self.get_obs()

    def calibrate_scene_for_close_drawer(self):
        """Calibrate the scene for the close_drawer skill"""
        self.scene.doors[1].reset(0.2)
        self.scene.doors[1].initial_state = 0.2

    def calibrate_scene_for_turn_off_lightbulb(self):
        """Calibrate the scene for the turn_off_lightbulb skill"""
        self.scene.lights[0].reset(1)
        self.scene.switches[0].reset(0.08)

    def calibrate_scene_for_move_slider_right(self):
        """Calibrate the scene for the move_slider_right skill"""
        self.scene.doors[0].reset(0.2)
        self.scene.doors[0].initial_state = 0.2

    def calibrate_scene_for_turn_off_led(self):
        """Calibrate the scene for the turn_off_led skill"""
        self.scene.lights[1].reset(1)
        self.scene.buttons[0].reset(0)

    def reset_close_drawer_scene(self):
        """Reset the scene for the close_drawer skill"""
        self.scene.doors[1].reset(0)
        self.scene.doors[1].initial_state = 0

    def reset_turn_off_lightbulb_scene(self):
        """Reset the scene for the turn_off_lightbulb skill"""
        self.scene.lights[0].reset(0)
        self.scene.switches[0].reset(0)

    def reset_move_slider_right_scene(self):
        """Reset the scene for the move_slider_right skill"""
        self.scene.doors[0].reset(0)
        self.scene.doors[0].initial_state = 0

    def reset_turn_off_led_scene(self):
        """Reset the scene for the turn_off_led skill"""
        self.scene.lights[1].reset(0)
        self.scene.buttons[0].reset(0)

    def calibrate_scene(self, skill):
        """
        Change scene based on the skill to be performed.

        Logic: Set scene for one scene but reset others
        """
        if skill == "close_drawer":
            self.calibrate_scene_for_close_drawer()
            self.reset_turn_off_lightbulb_scene()
            self.reset_move_slider_right_scene()
            self.reset_turn_off_led_scene()
        elif skill == "turn_off_lightbulb":
            self.calibrate_scene_for_turn_off_lightbulb()
            self.reset_close_drawer_scene()
            self.reset_move_slider_right_scene()
            self.reset_turn_off_led_scene()
        elif skill == "move_slider_right":
            self.calibrate_scene_for_move_slider_right()
            self.reset_close_drawer_scene()
            self.reset_turn_off_lightbulb_scene()
            self.reset_turn_off_led_scene()
        elif skill == "turn_off_led":
            self.calibrate_scene_for_turn_off_led()
            self.reset_close_drawer_scene()
            self.reset_turn_off_lightbulb_scene()
            self.reset_move_slider_right_scene()
        else:
            # reset all
            self.reset_close_drawer_scene()
            self.reset_turn_off_lightbulb_scene()
            self.reset_move_slider_right_scene()
            self.reset_turn_off_led_scene()

    def reset_to_state(self, state):
        return super().reset(robot_obs=state[:15], scene_obs=state[15:])

    def _success(self):
        """Returns a boolean indicating if the task was performed correctly"""
        current_info = self.get_info()
        task_filter = [self.target_skill]
        task_info = self.tasks.get_task_info_for_set(self.start_info, current_info, task_filter)
        return self.target_skill in task_info

    def _reward(self):
        """Returns the reward function that will be used
        for the RL algorithm"""
        reward = int(self._success()) * self.reward_scale
        r_info = {"reward": reward}
        return reward, r_info

    def _termination(self, reward):
        """Indicates if the robot has reached a terminal state"""
        success = reward > 0
        done = success or self._t >= self.max_episode_steps
        d_info = {"success": success}
        return done, d_info

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
        if len(action) == 3:
            coords = action
            orientation = np.zeros(3)
            gripper_action = [-1]
            env_action = np.concatenate([coords, orientation, gripper_action], axis=0)
        else:
            env_action = action.copy()

        self.robot.apply_action(env_action)
        for _ in range(self.action_repeat):
            self.p.stepSimulation(physicsClientId=self.cid)
        self.scene.step()
        obs = self.get_obs()
        info = self.get_info()
        reward, r_info = self._reward()
        self._t += 1
        done, d_info = self._termination(reward)
        info.update(r_info)
        info.update(d_info)
        return obs, reward, done, info

    def get_episode_info(self):
        completed_tasks = self.completed_tasks if len(self.completed_tasks) > 0 else [None]
        info = dict(solved_subtask=completed_tasks, tasks_to_complete=self.tasks_to_complete)
        info.update(self.solved_subtasks)
        return info

    def sample_base_shift(self):
        # TODO: add noise here

        bp, born = self.init_base_pos, self.init_base_orn
        return bp, born

    def set_init_pos(self, init_pos=None):
        """Sets the initial position of the end effector based on the task."""
        if init_pos is not None:
            first_skill = self.tasks_to_complete[0]
            reference = {
                "open_drawer": np.array([0.082, -0.175, 0.532]),
                "turn_on_lightbulb": np.array([0.059, -0.133, 0.493]),
                "move_slider_left": np.array([0.021, -0.116, 0.542]),
                "turn_on_led": np.array([0.053, -0.092, 0.505]),
            }
            if first_skill in reference.keys():
                self.init_pos = reference[first_skill]
            else:
                ValueError(f"Skill {first_skill} is not recognized.")
        else:
            self.init_pos = init_pos

    def store_skill_info(self, skills):
        """Stores each skill's info - start, goal, fixed_ori."""
        for skill in skills:
            self.skill_starts[skill.skill] = np.round(skill.start, 3)
            self.skill_goals[skill.skill] = np.round(skill.goal, 3)
            self.skill_oris[skill.skill] = skill.fixed_ori
        self.centroid = np.concatenate([list(self.skill_goals.values()) + list(self.skill_starts.values())])
        self.centroid = np.mean(self.centroid, axis=0)

    def get_init_pos(self, strategy="starts"):
        """Gets the initial position of the end effector based on the chosen skill.
        When strategy is "starts", the initial position is the skill's start.
        When strategy is "goals", the initial position is a random goal of the other skills
        or the skills's own start.
        """
        if strategy == "starts":
            return self.skill_starts[self.target_skill]
        elif strategy == "goals":
            goals = [v for k, v in self.skill_goals.items() if k != self.target_skill]
            # Add chosen skill's start to the list of other skills' goals
            goals.append(self.skill_starts[self.target_skill])
            return goals[np.random.choice(range(len(goals)))]

    def get_init_orn(self):
        """Gets the initial orientation of the end effector based on the chosen skill."""
        return np.array([3.14, -0.3, 1.5])  # Default

    def sample_ee_pose(self):
        """Samples a random end effector pose within a small range around the initial pose."""
        # if self.init_pos is None:
        #     self.init_gripper_pos = self.robot.target_pos
        # else:
        #     self.init_gripper_pos = self.init_pos
        # self.init_gripper_pos = self.get_init_pos()
        # self.init_gripper_orn = self.robot.target_orn
        self.init_gripper_orn = self.get_init_orn()
        offset = [0, 0, 0]
        np.random.seed(np.random.randint(0, 1000))
        offset[0] = np.random.uniform(-self.ee_noise[0], self.ee_noise[0], 1)[0]
        offset[1] = np.random.uniform(-self.ee_noise[1] / 1.5, self.ee_noise[1] / 2, 1)[0]
        offset[2] = np.random.uniform(-self.ee_noise[2] / 2, self.ee_noise[2], 1)[0]
        gripper_pos = self.centroid + offset
        gripper_orn = self.init_gripper_orn
        return gripper_pos, gripper_orn

    def custom_step(self, action):
        """Called only by calibrate_EE_start_state to perform a absolute steps in the environment."""
        # Transform gripper action to discrete space
        env_action = action.copy()
        self.robot.apply_action(env_action)
        for _ in range(self.action_repeat):
            self.p.stepSimulation(physicsClientId=self.cid)
        self.scene.step()
        obs = self.get_state_obs()["robot_obs"]  # Cheaper than get_obs()
        return obs

    def calibrate_EE_start_state(self, obs, error_margin=0.01, max_checks=15):
        """Samples a random but good starting point and moves the end effector to that point."""
        ee_pos, ee_orn = self.sample_ee_pose()
        count = 0
        action = np.array([ee_pos, ee_orn, -1], dtype=object)
        while np.linalg.norm(obs[:3] - ee_pos) > error_margin:
            obs = self.custom_step(action)
            if count >= max_checks:
                print("CALVIN is struggling to place the EE at the right initial pose.")
                print("Current EE pos: ", obs[:3])
                print("Desired EE pos: ", ee_pos)
                # Sample and try again
                ee_pos, ee_orn = self.sample_ee_pose()
                action = np.array([ee_pos, ee_orn, -1], dtype=object)
                count = 0
            count += 1
        self.robot.update_target_pose()

    def record_frame(self, obs_type="rgb", cam_type="static", size=200):
        """Record RGB obsservations"""
        rgb_obs, depth_obs = self.get_camera_obs()
        if obs_type == "rgb":
            frame = rgb_obs[f"{obs_type}_{cam_type}"]
        else:
            frame = depth_obs[f"{obs_type}_{cam_type}"]
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        self.frames.append(frame)

    def save_recording(self, outdir, fname):
        """Save recorded frames as a video"""
        if len(self.frames) == 0:
            # This shouldn't happen but if it does, the function
            # call exits gracefully
            return None
        fname = f"{fname}.gif"
        kargs = {"duration": 33}
        fpath = os.path.join(outdir, fname)
        imageio.mimsave(fpath, np.array(self.frames), "GIF", **kargs)
        return fpath

    def reset_recording(self):
        """Reset recorded frames"""
        self.frames = []

    def load(self):
        logger.info("Resetting simulation")
        self.p.resetSimulation(physicsClientId=self.cid)
        logger.info("Setting gravity")
        self.p.setGravity(0, 0, -9.8, physicsClientId=self.cid)

        self.robot = hydra.utils.instantiate(self.robot_cfg, cid=self.cid)
        self.scene = hydra.utils.instantiate(self.scene_cfg, p=self.p, cid=self.cid, np_random=self.np_random)

        self.robot.load()
        self.scene.load()

        # init cameras after scene is loaded to have robot id available
        self.cameras = [
            hydra.utils.instantiate(
                self.cameras_c[name], cid=self.cid, robot_id=self.robot.robot_uid, objects=self.scene.get_objects()
            )
            for name in self.cameras_c
        ]

    def initialize_bullet(self, bullet_time_step, render_width, render_height):
        self.bts = bullet_time_step
        self.rhw = (render_height, render_width)
        if not self.isConnected():
            self.ownsPhysicsClient = True
            if self.use_vr:
                self.p = bc.BulletClient(connection_mode=p.SHARED_MEMORY)
                cid = self.p._client
                if cid < 0:
                    logger.error("Failed to connect to SHARED_MEMORY bullet server.\n" " Is it running?")
                    sys.exit(1)
                self.p.setRealTimeSimulation(enableRealTimeSimulation=1, physicsClientId=cid)
            elif self.show_gui:
                self.p = bc.BulletClient(connection_mode=p.GUI)
                cid = self.p._client
                if cid < 0:
                    logger.error("Failed to connect to GUI.")
            elif self.use_egl:
                options = f"--width={self.rhw[1]} --height={self.rhw[0]}"
                self.p = bc.BulletClient(p.DIRECT, options=options)
                self.p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
                cid = self.p._client
                self.p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=cid)
                self.p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=cid)
                self.p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=cid)

                egl = pkgutil.get_loader("eglRenderer")
                logger.info("Loading EGL plugin (may segfault on misconfigured systems)...")
                if egl:
                    plugin = self.p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                else:
                    plugin = self.p.loadPlugin("eglRendererPlugin")
                if plugin < 0:
                    logger.error("\nPlugin Failed to load!\n")
                    sys.exit()
                # set environment variable for tacto renderer
                os.environ["PYOPENGL_PLATFORM"] = "egl"
                logger.info("Successfully loaded egl plugin")
            else:
                self.p = bc.BulletClient(p.DIRECT)
                cid = self.p._client
                if cid < 0:
                    logger.error("Failed to start DIRECT bullet mode.")
            logger.info(f"Connected to server with id: {cid}")

            self.cid = cid
            self.p.resetSimulation(physicsClientId=self.cid)
            self.p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, physicsClientId=self.cid)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0, physicsClientId=self.cid)
            logger.info(f"Connected to server with id: {self.cid}")
            self.p.setTimeStep(1.0 / self.bts, physicsClientId=self.cid)
            return cid

    def close(self):
        if self.ownsPhysicsClient:
            print("disconnecting id %d from server" % self.cid)
            if self.cid >= 0 and self.p is not None:
                print("CID: " + str(self.cid))
                try:
                    self.p.disconnect(physicsClientId=self.cid)
                except:
                    pass

        else:
            print("does not own physics client id")

    def isConnected(self):
        connected = False
        if self.p is not None:
            connected = self.p.getConnectionInfo()["isConnected"] == 1
            if not connected:
                try:
                    self.p.disconnect(self.cid)
                    del p
                except:
                    pass
                gc.collect()
        return connected
