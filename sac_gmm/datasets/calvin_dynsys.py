import glob
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import pdb
import pybullet as p
from pathlib import Path
from omegaconf import DictConfig


class CALVINDynSysDataset(Dataset):
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
        self.X = np.load(self.data_file)[:, :, start_idx:end_idx]

        # Get the last orientation from the trajectory (this is bad for orientation dependant tasks)
        # s_idx, e_idx = self.get_valid_columns("ori")
        # temp_ori = np.load(self.data_file)[:, :, s_idx:e_idx]
        # self.fixed_ori = temp_ori[0, -1, :]

        # Get the euler angles best for the skill
        if self.skill.skill in ["open_drawer", "close_drawer", "turn_on_led"]:
            self.fixed_ori = np.array([3.14, 0.0, 1.5])
        elif self.skill.skill in ["turn_on_lightbulb", "move_slider_left"]:
            self.fixed_ori = np.array([3.14, -0.5, 1.5])

        if self.state_type == "ori" and self.is_quaternion:
            self.X = np.apply_along_axis(p.getQuaternionFromEuler, -1, self.X)
        elif self.state_type == "pos_ori" and self.is_quaternion:
            oris = np.apply_along_axis(p.getQuaternionFromEuler, -1, self.X[:, :, 3:])
            self.X = np.concatenate([self.X[:, :, :3], oris], axis=-1)

        self.start = np.mean(self.X[:, 0, :3], axis=0)
        self.goal = np.mean(self.X[:, -1, :3], axis=0)
        if self.goal_centered:
            # Make X goal centered i.e., subtract each trajectory with its goal
            self.X[:, :, :3] = self.X[:, :, :3] - np.expand_dims(self.X[:, -1, :3], axis=1)

        if self.normalized:
            self.set_mins_and_maxs(self.X)
            self.X = self.normalize(self.X)

        self.dX = np.zeros_like(self.X)
        self.dX[:, :-1, :3] = (self.X[:, 1:, :3] - self.X[:, :-1, :3]) / self.pos_dt
        self.dX[:, -1, :3] = np.zeros(self.dX.shape[-1])

        if self.state_type == "pos_ori":
            self.Ori = self.X[:, :, 3:]
            self.Ori = torch.from_numpy(self.Ori).type(torch.FloatTensor)
            self.X = self.X[:, :, :3]

        self.X = torch.from_numpy(self.X).type(torch.FloatTensor)
        self.dX = torch.from_numpy(self.dX).type(torch.FloatTensor)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.dX[idx]

    def set_mins_and_maxs(self, data=None):
        self.X_mins = np.min(data.reshape(-1, data.shape[-1]), axis=0)
        self.X_maxs = np.max(data.reshape(-1, data.shape[-1]), axis=0)

    def normalize(self, x):
        """See this link for clarity: https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1"""
        assert self.X_mins is not None, "Cannot normalize with X_mins as None"
        assert self.X_maxs is not None, "Cannot normalize with X_maxs as None"
        return (self.norm_range[-1] - self.norm_range[0]) * (x - self.X_mins) / (
            self.X_maxs - self.X_mins
        ) + self.norm_range[0]

    def undo_normalize(self, x):
        """See this link for clarity: https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1"""
        assert self.X_mins is not None, "Cannot undo normalization with X_mins as None"
        assert self.X_maxs is not None, "Cannot undo normalization with X_maxs as None"
        return (x - self.norm_range[0]) * (self.X_maxs - self.X_mins) / (
            self.norm_range[-1] - self.norm_range[0]
        ) + self.X_mins

    def sample_start(self, size=1, sigma=0.15):
        start = self.start
        sampled = sample_gaussian_norm_ball(start, sigma, size)
        if size == 1:
            return sampled[0]
        else:
            return sampled

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

    def plot_random(self):
        sampled_path = []
        rand_idx = np.random.randint(0, len(self.X))
        true_x = self.X[rand_idx, :, :].numpy()
        x = true_x[0]
        for t in range(len(true_x)):
            sampled_path.append(x)
            delta_x = self.dt * self.dX[rand_idx, t, :].numpy()
            x = x + delta_x
        sampled_path = np.array(sampled_path)
        plot_3d_trajectories(true_x, sampled_path)

    def rm_rw_data(self, list_indicis):
        self.X = np.load(self.data_file)
        new_X = np.delete(self.X, list_indicis, axis=0)
        np.save(self.data_file, new_X)


def plot_3d_trajectories(demos, repro=None, goal=None, figsize=(4, 4)):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    if goal is not None:
        ax.scatter(goal[0], goal[1], goal[2], marker="*", s=15, label="Goal", c="red")
    if repro is None:
        for i in range(demos.shape[0]):
            x_val = demos[i, :, 0]
            y_val = demos[i, :, 1]
            z_val = demos[i, :, 2]
            ax.scatter(x_val, y_val, z_val, s=10)
    else:
        ax.scatter(demos[:, 0], demos[:, 1], demos[:, 2], alpha=0.5, s=1, label="Demonstration")
        ax.scatter(repro[:, 0], repro[:, 1], repro[:, 2], alpha=0.5, s=1, label="Reproduction")
    plt.legend()
    plt.tight_layout()
    plt.show()


def sample_gaussian_norm_ball(reference_point, sigma, num_samples):
    samples = []
    for _ in range(num_samples):
        # Step 1: Sample from standard Gaussian distribution
        offset = np.random.randn(3)

        # Step 2: Normalize the offset
        normalized_offset = offset / np.linalg.norm(offset)

        # Step 3: Scale the normalized offset
        scaled_offset = normalized_offset * np.random.normal(0, sigma)

        # Step 4: Translate the offset
        sampled_point = reference_point + scaled_offset

        samples.append(sampled_point)

    return samples
