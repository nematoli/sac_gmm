import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
from collections import deque
import random
import math
from utils.transforms import PreprocessImage


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


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


def torch_tensor_cam_obs(cam_obs):
    cam_obs = PreprocessImage()(cam_obs).squeeze(1)
    return cam_obs


def preprocess_agent_in(trainer, obs):
    """Prepares a dictionary of observations suitable for SAC training.

    Args:
        trainer (SACGMMTrainer): SACGMMTrainer Object
        obs (gym.spaces.Dict): Observation dict given by the env

    Returns:
        agent_in (torch.tensor): gmm_params, robot obs and cam obs latents flattened as a torch.tensor
    """
    robot_obs = torch.from_numpy(obs[trainer.robot_obs]).to(trainer.device)
    cam_obs = torch_tensor_cam_obs(obs[trainer.cam_obs]).to(trainer.device)
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
