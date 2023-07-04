import hydra

from calvin_env.envs.play_table_env import PlayTableSimEnv
import pybullet as p
import pybullet_utils.bullet_client as bc

import logging
import gc
import sys
import os
import pkgutil
import gym
from gym import spaces
import numpy as np

# A logger for this file
log = logging.getLogger(__name__)

# here are the indices of data in the observation vector
# here [0,1,2] - gripper position; [3,4,5] - gripper orientation, [14] - gripper action, [6] - gripper width
GYM_POSITION_INDICES = np.array([0, 1, 2])
GYM_ORIENTATION_INDICES = np.array([3, 4, 5])
GYM_6D_INDICES = np.concatenate([GYM_POSITION_INDICES, GYM_ORIENTATION_INDICES])


class CalvinEnv(PlayTableSimEnv):
    def __init__(self, cfg):
        pt_cfg = {**cfg.calvin_env.env}
        pt_cfg.pop("_target_", None)
        pt_cfg.pop("_recursive_", None)
        self.robot_cfg = pt_cfg["robot_cfg"]
        self.scene_cfg = pt_cfg["scene_cfg"]
        self.cameras_c = pt_cfg["cameras"]
        super(CalvinEnv, self).__init__(**pt_cfg)

        self.dt = cfg.dt
        self.init_gripper_pos = self.robot.target_pos
        self.init_gripper_orn = self.robot.target_orn
        self.init_base_pos, self.init_base_orn = self.p.getBasePositionAndOrientation(self.robot.robot_uid)

        self.ee_position_deviation = cfg.ee_position_deviation
        self.ee_position_shift = cfg.ee_position_shift
        self.name = cfg.name
        self.task_name = cfg.task_name
        self.max_episode_steps = cfg.max_episode_steps
        # Set interaction parameters
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()
        # We can use the task utility to know if the task was executed correctly
        self.tasks = hydra.utils.instantiate(cfg.calvin_env.tasks)

    def load(self):
        log.info("Resetting simulation")
        self.p.resetSimulation(physicsClientId=self.cid)
        log.info("Setting gravity")
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
                    log.error("Failed to connect to SHARED_MEMORY bullet server.\n" " Is it running?")
                    sys.exit(1)
                self.p.setRealTimeSimulation(enableRealTimeSimulation=1, physicsClientId=cid)
            elif self.show_gui:
                self.p = bc.BulletClient(connection_mode=p.GUI)
                cid = self.p._client
                if cid < 0:
                    log.error("Failed to connect to GUI.")
            elif self.use_egl:
                options = f"--width={self.rhw[1]} --height={self.rhw[0]}"
                self.p = bc.BulletClient(p.DIRECT, options=options)
                self.p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
                cid = self.p._client
                self.p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=cid)
                self.p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=cid)
                self.p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=cid)

                egl = pkgutil.get_loader("eglRenderer")
                log.info("Loading EGL plugin (may segfault on misconfigured systems)...")
                if egl:
                    plugin = self.p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                else:
                    plugin = self.p.loadPlugin("eglRendererPlugin")
                if plugin < 0:
                    log.error("\nPlugin Failed to load!\n")
                    sys.exit()
                # set environment variable for tacto renderer
                os.environ["PYOPENGL_PLATFORM"] = "egl"
                log.info("Successfully loaded egl plugin")
            else:
                self.p = bc.BulletClient(p.DIRECT)
                cid = self.p._client
                if cid < 0:
                    log.error("Failed to start DIRECT bullet mode.")
            log.info(f"Connected to server with id: {cid}")

            self.cid = cid
            self.p.resetSimulation(physicsClientId=self.cid)
            self.p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, physicsClientId=self.cid)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0, physicsClientId=self.cid)
            log.info(f"Connected to server with id: {self.cid}")
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

    def set_ee_position_shift(self, shift):
        self.ee_position_shift = shift

    @staticmethod
    def get_observation_space():
        """Return only position and gripper_width by default"""
        observation_space = {}

        observation_space["position"] = spaces.Box(low=np.array([0.3, -0.85, 0]), high=np.array([0.85, 0.85, 0.8]))
        # observation_space["orientation"] = spaces.Box(low=-1, high=1, shape=[3])
        # observation_space["gripper_width"] = spaces.Box(low=-1, high=1, shape=[1])
        # observation_space["gripper_action"] = spaces.Box(low=0, high=1, shape=[1], dtype=np.int)
        observation_space["depth_gripper"] = spaces.Box(low=-1, high=1, shape=(1, 84, 84))
        observation_space["rgb_gripper"] = spaces.Box(low=-1, high=1, shape=(3, 84, 84))
        observation_space["rgbd_gripper"] = spaces.Box(low=-1, high=1, shape=(4, 84, 84))
        return gym.spaces.Dict(observation_space)

    @staticmethod
    def get_action_space():
        """End effector position and gripper width relative displacement"""
        return spaces.Box(np.array([-1] * 7), np.array([1] * 7))

    @staticmethod
    def clamp_action(action):
        """Assure every action component is scaled between -1, 1"""
        max_action = np.max(np.abs(action))
        if max_action > 1:
            action /= max_action
        return action

    def sample_base_shift(self):
        shift = np.array(self.ee_position_shift)
        pos_shift = shift + np.random.normal(0.0, self.ee_position_deviation, 3)
        align_in_target_front = self.task_object_position() - self.init_gripper_pos

        target_ee_pos = self.init_gripper_pos + shift + align_in_target_front

        color = [0, 1, 1]
        self.p.addUserDebugLine(self.task_object_position(), target_ee_pos, color, 5)

        sample_ee_pos = self.init_gripper_pos + pos_shift + align_in_target_front
        color = [1, 1, 0]
        self.p.addUserDebugLine(sample_ee_pos, target_ee_pos, color, 5)

        bp, born = self.init_base_pos, self.init_base_orn
        return bp + pos_shift + align_in_target_front, born

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

    def reset(self, robot_obs=None, scene_obs=None):
        if not self.isConnected():
            h, w = self.rhw
            self.initialize_bullet(self.bts, w, h)
            self.load()
        self.scene.reset(scene_obs)
        if robot_obs is None:
            npos, norn = self.sample_base_shift()
            self.p.resetBasePositionAndOrientation(self.robot.robot_uid, npos, norn)
            self.robot.reset()
        else:
            self.robot.reset(robot_obs)
        self.p.stepSimulation(physicsClientId=self.cid)
        self.start_info = self.get_info()
        return self.get_obs()

    def object_by_name(self, name):
        doors = [o for o in self.scene.doors if o.name == name]
        buttons = [o for o in self.scene.buttons if o.effect == name]
        if len(doors) != 0:
            return doors[0]
        elif len(buttons) != 0:
            return buttons[0]

    def object_position(self, obj):
        base = self.scene.fixed_objects[0]
        ln = self.p.getJointInfo(base.uid, obj.joint_index, physicsClientId=self.cid).link_name.decode()
        return self.p.getLinkState(base.uid, base.get_info()["links"][ln], physicsClientId=self.cid)[
            "link_world_position"
        ]

    def task_object_position(self):
        object = self.object_by_name(self.tasks.tasks[self.task_name].args[0])
        return self.object_position(object)

    def _success(self):
        """Returns a boolean indicating if the task was performed correctly"""
        current_info = self.get_info()
        task_filter = [self.task_name]
        task_info = self.tasks.get_task_info_for_set(self.start_info, current_info, task_filter)
        return self.task_name in task_info

    def _reward(self):
        """Returns the reward function that will be used
        for the RL algorithm"""
        mult = 10  # TODO implement dense reward
        reward = int(self._success()) * mult
        r_info = {"reward": reward}
        return reward, r_info

    def _termination(self):
        """Indicates if the robot has reached a terminal state"""
        success = self._success()
        done = success
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
        coords = action[:3] * self.dt
        orientation = np.zeros(3) * self.dt
        # TODO discuss what to do with gripper width
        gripper_action = [-1]
        env_action = np.concatenate([coords, orientation, gripper_action], axis=0)

        self.robot.apply_action(env_action)
        for i in range(self.action_repeat):
            self.p.stepSimulation(physicsClientId=self.cid)

        self.scene.step()
        obs = self.get_obs()
        info = self.get_info()
        reward, r_info = self._reward()
        done, d_info = self._termination()
        info.update(r_info)
        info.update(d_info)
        return obs, reward, done, info

    def get_obs(self):
        obs = super(CalvinEnv, self).get_obs()
        nobs = {}
        robs = np.array(obs["robot_obs"])
        nobs["position"] = robs[GYM_POSITION_INDICES]
        cam = self.cam_by_name("gripper")
        nobs["gripper_view_mtx"] = np.array(cam.view_matrix)
        nobs["gripper_projection_mtx"] = np.array(cam.projection_matrix)
        # nobs['orientation'] = robs[GYM_ORIENTATION_INDICES] - self.target[3:]
        # nobs['gripper_action'] = np.array(coords[6])
        # nobs['gripper_width'] = np.array(coords[6] - self.target[6])
        gr_rgb = obs["rgb_obs"]["rgb_gripper"]
        nobs["rgb_gripper"] = np.moveaxis(gr_rgb, 2, 0)
        nobs["depth_gripper"] = np.expand_dims(obs["depth_obs"]["depth_gripper"], axis=0)
        nobs["rgbd_gripper"] = np.concatenate([nobs["rgb_gripper"], nobs["depth_gripper"]], 0)
        return nobs

    def cam_by_name(self, cam_name):
        return [c for c in self.cameras if c.name == cam_name][0]

    @staticmethod
    def value_to_range(value, lower_bound, upper_bound):
        return value * (upper_bound - lower_bound) + lower_bound

    @staticmethod
    def normalize(value, lower_bound, upper_bound):
        return (value - lower_bound) / (upper_bound - lower_bound)


def make_env_calvin_env_v0(cfg):
    env = gym.make("sac_gmm:calvin-env-v0", cfg=cfg)
    env._max_episode_steps = cfg.max_episode_steps
    return env
