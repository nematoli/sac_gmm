import glob
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import pdb
import pybullet as p
from pathlib import Path
from omegaconf import DictConfig


class CALVINDynSysDataset2(Dataset):
    def __init__(
        self,
        skill: DictConfig,
        train: bool,
        goal_centered: bool,
        demos_dir: str,
    ):
        self.skill = skill
        self.train = train
        self.goal_centered = goal_centered
        self.demos_dir = Path(demos_dir).expanduser()
        self.state_type = self.skill.state_type
        self.pos_dt = self.skill.pos_dt
        self.ori_dt = self.skill.ori_dt
        self.normalized = self.skill.normalized
        self.norm_range = [-1, 1]
        self.X_mins = None
        self.X_maxs = None
        self.fixed_ori = None
        self.start = None
        self.goal = None
        if self.train:
            fname = "training"
        else:
            fname = "validation"
        assert self.demos_dir.is_dir(), "Demos directory does not exist!"
        self.data_file = glob.glob(str(self.demos_dir / self.skill.skill / f"{fname}.npy"))[0]

        start_idx, end_idx = self.get_valid_columns(self.state_type)
        data = np.load(self.data_file)[:, :, start_idx:end_idx]

        # Get the euler angles best for the skill
        if self.skill.skill in ["open_drawer", "close_drawer", "turn_on_led"]:
            self.fixed_ori = np.array([3.14, 0.0, 1.5])
        elif self.skill.skill in ["turn_on_lightbulb", "move_slider_left"]:
            self.fixed_ori = np.array([3.14, -0.5, 1.5])

        self.X_pos = data[:, :, :3]
        self.X_ori = data[:, :, 3:]

        oris = np.apply_along_axis(p.getQuaternionFromEuler, -1, self.X_ori)
        # Make all quaternions positive
        for traj in range(oris.shape[0]):
            for t_step in range(oris.shape[1]):
                if oris[traj, t_step, 0] < 0:
                    oris[traj, t_step, :] *= -1
        self.X_ori = np.copy(oris)
        # if self.goal_centered:
        #     # Make X goal centered i.e., subtract each trajectory with its goal
        #     self.X_pos = self.X_pos - np.expand_dims(self.X_pos[:, -1, :], axis=1)

        # Input: Pos, Output: Next Pos and Ori
        self.X = np.copy(self.X_pos[:, :-1, :])
        dX_pos = np.copy(self.X_pos[:, 1:, :])
        dX_ori = np.copy(self.X_ori[:, 1:, :])
        self.dX = np.concatenate([dX_pos, dX_ori], axis=-1)

        self.start = np.mean(self.X_pos[:, 0, :], axis=0)
        self.goal = np.mean(self.X_pos[:, -1, :], axis=0)

        self.X = torch.from_numpy(self.X).type(torch.FloatTensor)
        self.dX = torch.from_numpy(self.dX).type(torch.FloatTensor)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.dX[idx]

    def get_valid_columns(self, state_type):
        if "joint" in state_type:
            start, end = 7, 14
        elif "pos_ori" in state_type:
            start, end = 0, 6
        elif "pos" in state_type:
            start, end = 0, 3
        elif "ori" in state_type:
            start, end = 3, 6
        elif "grip" in state_type:
            start, end = 6, 7
        return start, end
