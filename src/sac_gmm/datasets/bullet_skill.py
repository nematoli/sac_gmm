import glob
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import pybullet as p
from pathlib import Path
from omegaconf import DictConfig
import copy


class BulletSkillDataset(Dataset):
    def __init__(
        self,
        skill: DictConfig,
        goal_centered: bool,
        demos_dir: str,
    ):
        self.skill = skill
        self.goal_centered = goal_centered
        self.demos_dir = Path(demos_dir).expanduser()
        self.dt = self.skill.dt
        self.start = None
        self.goal = None

        assert self.demos_dir.is_dir(), "Demos directory does not exist!"
        skill_demo_dir = self.demos_dir / self.skill.name
        x_list, dx_list, start_points, goal_points = [], [], [], []
        for filename in os.listdir(skill_demo_dir):
            if filename.endswith(".npz"):
                file_path = os.path.join(skill_demo_dir, filename)
                data = np.load(file_path)
                X = data["position"]
                dX = np.zeros_like(X)
                start_points.append(copy.deepcopy(X[0]))
                goal_points.append(copy.deepcopy(X[-1]))

                if self.goal_centered:
                    # Make X goal centered i.e., subtract each trajectory with its goal
                    X[:, :] = X[:, :] - X[-1, :]
                dX[:-1, :] = (X[1:, :] - X[:-1, :]) / self.dt
                dX[-1, :] = 0
                x_list.append(X)
                dx_list.append(dX)

        self.X = np.vstack(x_list)
        self.dX = np.vstack(dx_list)

        self.start = np.mean(np.stack(start_points, axis=0), axis=0)
        self.goal = np.mean(np.stack(goal_points, axis=0), axis=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.dX[idx]
