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
    """
    Use this to train GMM: Position -> Next Position
    """

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

        # Get position only
        self.X_pos = np.load(self.data_file)[:, :, :3]

        # Record the euler angles best for the skill
        if self.skill.skill in ["turn_on_lightbulb", "move_slider_left"]:
            self.fixed_ori = np.array([3.14, -0.5, 1.5])
        else:
            self.fixed_ori = np.array([3.14, 0.0, 1.5])

        # Store average start and goal positions
        self.start = np.mean(self.X_pos[:, 0, :], axis=0)
        self.goal = np.mean(self.X_pos[:, -1, :], axis=0)

        # Make X goal centered i.e., subtract each trajectory with its goal
        if self.goal_centered:
            self.X_pos[:, :, :] = self.X_pos[:, :, :] - np.expand_dims(self.X_pos[:, -1, :], axis=1)

        self.dX_pos = self.dX_pos = np.copy(self.X_pos[:, 1:, :])
        self.X_pos = np.copy(self.X_pos[:, :-1, :])

        self.X_pos = torch.from_numpy(self.X_pos).type(torch.FloatTensor)
        self.dX_pos = torch.from_numpy(self.dX_pos).type(torch.FloatTensor)

    def __len__(self):
        return len(self.X_pos)

    def __getitem__(self, idx):
        return self.X_pos[idx], self.dX_pos[idx]
