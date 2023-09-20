import torch
import torch.nn as nn
import torch.nn.functional as F
from sac_gmm.utils.distributions import TanhNormal
from torch.distributions.normal import Normal
from collections import OrderedDict


LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
MEAN_MIN = -9.0
MEAN_MAX = 9.0


class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        # Action parameters
        self.action_space = None
        action_dim = action_dim

        # TODO: replace ELU with SiLU
        # TODO: Batchnorm?
        self.fc_layers = nn.Sequential(
            OrderedDict(
                [
                    ("fc_1", nn.Linear(input_dim, hidden_dim)),
                    ("bn_1", nn.BatchNorm1d(hidden_dim)),
                    ("elu_1", nn.ELU()),
                    ("fc_2", nn.Linear(hidden_dim, hidden_dim)),
                    ("bn_2", nn.BatchNorm1d(hidden_dim)),
                    ("elu_2", nn.ELU()),
                ]
            )
        )
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        single_dim = len(state.shape) == 1
        if single_dim:
            state = state.unsqueeze(0)

        state = self.fc_layers(state)
        mean = self.fc_mean(state)
        log_std = self.fc_log_std(state)
        # log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Avoid -inf when std -> 0
        if single_dim:
            mean = mean.squeeze()
            log_std = log_std.squeeze()

        return mean, log_std

    def get_actions(self, obs, deterministic=False, reparameterize=False, epsilon=1e-6):
        mean, log_std = self.forward(obs)
        if deterministic:
            actions = torch.tanh(mean)
            log_pi = torch.zeros_like(actions)
        else:
            actions, log_pi = self.sample_actions(mean, log_std, reparameterize, epsilon=epsilon)
        return self.scale_actions(actions), log_pi

    def sample_actions(self, means, log_stds, reparameterize, epsilon=1e-6):
        stds = log_stds.exp()
        normal = Normal(means, stds)
        if reparameterize:
            z = normal.rsample()
        else:
            z = normal.sample()
        actions = torch.tanh(z)
        log_pi = normal.log_prob(z) - torch.log(1 - actions.square() + epsilon)
        log_pi = log_pi.sum(-1, keepdim=True)
        return actions, log_pi

    def set_action_space(self, action_space):
        self.action_space = action_space

    def scale_actions(self, action):
        action_high = torch.tensor(self.action_space.high, dtype=torch.float, device=action.device)
        action_low = torch.tensor(self.action_space.low, dtype=torch.float, device=action.device)
        slope = (action_high - action_low) / 2
        action = action_low + slope * (action + 1)
        return action
