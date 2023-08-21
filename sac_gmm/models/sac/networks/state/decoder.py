from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from .utils import MLP, get_activation
from .distributions import Normal, TanhNormal, MixedDistribution, OneHot
from .distributions import Bernoulli, Symlog, SymlogDiscrete

from . import rmap
from .pytorch import symlog


class DenseDecoder(nn.Module):
    """MLP decoder returns normal distribution of prediction."""

    def __init__(self, input_dim, output_dim, hidden_dims, activation, loss):
        super().__init__()

        if loss == "normal":
            self.fc = MLP(input_dim, output_dim * 2, hidden_dims, activation, norm=True)
        elif loss == "mse":
            self.fc = MLP(input_dim, output_dim, hidden_dims, activation, norm=True)
        elif loss == "symlog_mse":
            self.fc = MLP(input_dim, output_dim, hidden_dims, activation, norm=True)
        elif loss == "symlog_discrete":
            self.fc = MLP(input_dim, output_dim * 255, hidden_dims, activation, norm=True, small_weight=True)
        elif loss == "binary":
            self.fc = MLP(input_dim, output_dim, hidden_dims, activation, norm=True)
        else:
            raise ValueError(f"Loss type is not available: {loss}")
        self._loss = loss

    def forward(self, feat):
        out = self.fc(feat)
        if self._loss == "normal":
            mean, std = out.chunk(2, dim=-1)
            std = (std.tanh() + 1) * 0.7 + 0.1  # [0.1, 1.5]
            return Normal(mean, std, event_dim=1)
        elif self._loss == "mse":
            return Normal(out, 1, event_dim=1)
        elif self._loss == "symlog_mse":
            return Symlog(out, event_dim=1)
        elif self._loss == "symlog_discrete":
            shape = list(out.shape[:-1]) + [out.shape[-1] // 255, 255]
            out = out.reshape(shape)
            return SymlogDiscrete(out, event_dim=1)
        elif self._loss == "binary":
            return Bernoulli(logits=out, event_dim=1)


class Decoder(nn.Module):
    def __init__(self, cfg, state_dim, ob_space):
        super().__init__()
        self._ob_space = ob_space
        self.decoders = nn.ModuleDict()
        self.decoders["obs"] = DenseDecoder(
            state_dim,
            self._ob_space.shape[0],
            cfg.hidden_dims,
            cfg.dense_act,
            cfg.state_loss,
        )

    def forward(self, feat):
        return MixedDistribution(OrderedDict([(k, self.decoders[k](feat)) for k in ["obs"]]))
