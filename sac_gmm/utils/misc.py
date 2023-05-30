import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
from collections import deque
import random
import math
from utils.transforms import PreprocessImage, ResizeImage, GrayscaleImage


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def calc_out_size(w, h, kernel_size, padding=0, stride=1):
    width = (w - kernel_size + 2 * padding) // stride + 1
    height = (h - kernel_size + 2 * padding) // stride + 1
    return width, height


def transform_to_tensor(x, dtype=torch.float, grad=True, device="cuda"):
    hd_input_keys = [
        "rgb_gripper",
        "depth_gripper",
        "rgb_static",
        "depth_static",
    ]
    if isinstance(x, dict):
        tensor = {}
        for k, v in x.items():
            if k in hd_input_keys:
                tensor[k] = v.clone().detach().requires_grad_(grad).to(dtype).to(device)
            else:
                tensor[k] = torch.tensor(v, dtype=dtype, device=device, requires_grad=grad)
    else:
        tensor = torch.tensor(x, dtype=dtype, device=device, requires_grad=grad)  # B, S_D
    return tensor


def preprocess_cam_obs(cam_obs):
    """Converts RGB or Depth images to 1x64x64 (grayscale) images"""
    cam_obs = PreprocessImage()(cam_obs)
    return cam_obs


def resize_cam_obs(cam_obs):
    """Resizes camera obs"""
    cam_obs = ResizeImage()(cam_obs)
    return cam_obs


def grayscale_cam_obs(cam_obs):
    """Grayscales camera obs"""
    cam_obs = GrayscaleImage()(cam_obs)
    return cam_obs


def get_state_from_observation(hd_input_encoder, obs, detach_encoder):
    if isinstance(obs, dict):
        # Robot obs
        if "pos" in obs:
            fc_input = obs["pos"].float()
        elif "pos_ori" in obs:
            fc_input = obs["pos_ori"].float()
        elif "joint" in obs:
            fc_input = obs["joint"].float()

        # Added for Residual action
        if "gmm_action" in obs:
            fc_input = torch.cat((fc_input, obs["gmm_action"]), dim=-1)

        # Camera obs (High dimensional inputs)
        hd_input_keys = [
            "rgb_gripper",
            "depth_gripper",
            "rgb_static",
            "depth_static",
        ]
        for key in hd_input_keys:
            if key in obs:
                if type(obs[key]) is list:
                    obs[key] = torch.stack(obs[key], dim=1)
                compact_repr = hd_input_encoder(grayscale_cam_obs(obs[key].float()), detach_encoder)
                fc_input = torch.cat((fc_input, compact_repr), dim=-1)

        return fc_input.float()

    return obs.float()


def preprocess_agent_in(trainer, obs):
    """Prepares a dictionary of observations suitable for SAC training.

    Args:
        trainer (SACGMMTrainer): SACGMMTrainer Object
        obs (gym.spaces.Dict): Observation dict given by the env

    Returns:
        agent_in (torch.tensor): gmm_params, robot obs and cam obs latents flattened as a torch.tensor
    """
    robot_obs = torch.from_numpy(obs[trainer.robot_obs]).to(trainer.device)
    cam_obs = preprocess_cam_obs(obs[trainer.cam_obs]).to(trainer.device)
    cam_obs_rep = trainer.agent.ae.get_image_rep(cam_obs.to(trainer.device))
    gmm_params = trainer.dyn_sys.model_params(trainer.cfg.adapt_cov)
    gmm_params = torch.from_numpy(gmm_params).to(trainer.device)

    agent_in = torch.concat((gmm_params, robot_obs, cam_obs_rep), axis=-1).to(trainer.device)
    return agent_in


def postprocess_agent_out(trainer, gmm_change, priors_scale=0.1, means_scale=0.1):
    """Processes SAC agent's output into a dictionary with labels

    Args:
        gmm_change (np.array): SAC actor's output
        priors_scale (float): Scale factor to scale priors in agent's output
        means_scale (float): Scale factor to scale means in agent's output

    Returns:
        dict (dict): Dictionary format of the processed output
    """
    dict = {}
    param_space = trainer.dyn_sys.get_update_range_parameter_space()
    priors_size = param_space["priors"].shape[0]
    means_size = param_space["mu"].shape[0]
    priors = gmm_change[:priors_size] * param_space["priors"].high
    mu = gmm_change[priors_size : priors_size + means_size] * param_space["mu"].high
    dict = {"mu": mu, "priors": priors}
    if trainer.cfg.adapt_cov:
        dict["sigma"] = gmm_change[priors_size + means_size :] * param_space["sigma"].high

    return dict
