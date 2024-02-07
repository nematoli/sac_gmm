import numpy as np
import torch
import torch.nn as nn
import gym.spaces
from collections import OrderedDict

from .utils.utils import MLP, get_activation, weight_init
from .utils.distributions import TanhNormal, Normal, MixedDistribution
from .dreamer_model import Encoder as DreamerEncoder
from .dreamer_model import Decoder as DreamerDecoder


class TDMPCCustomModel(nn.Module):
    """Task-Oriented Latent Dynamics model
    with DreamerV3 Observation Encoder and Decoder."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def make_model_dynamics(self, cfg, input_dim, action_dim):
        self.cfg = None
        self.dynamics = MLP(
            input_dim + action_dim,
            input_dim,
            [cfg.num_units] * cfg.num_layers,
            cfg.dense_act,
        )
        self.reward = Critic(
            input_dim + action_dim,
            [cfg.num_units] * cfg.num_layers,
            1,
            cfg.dense_act,
        )

    def make_enc_dec(self, cfg, ob_space, state_dim):
        self.encoder = DreamerEncoder(cfg.encoder, ob_space, state_dim)
        self.decoder = DreamerDecoder(cfg.decoder, state_dim, ob_space)

    def imagine_step(self, state, ac):
        out = torch.cat([state, ac], dim=-1)
        return self.dynamics(out)

    def load_state_dict(self, ckpt):
        try:
            super().load_state_dict(ckpt, strict=False)
        except:
            from collections import OrderedDict

            ckpt = OrderedDict([(k, v) for k, v in ckpt.items() if not k.startswith("critic")])
            super().load_state_dict(ckpt, strict=False)


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dims, ensemble, activation):
        super().__init__()
        self._ensemble = ensemble
        h = hidden_dims
        assert len(h) > 0
        self.fcs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, h[0]),
                    nn.LayerNorm(h[0]),
                    nn.Tanh(),
                    MLP(h[0], 1, h[1:], activation, small_weight=True),
                )
                for _ in range(ensemble)
            ]
        )

    def forward(self, state, ac=None):
        if ac is not None:
            state = torch.cat([state, ac], dim=-1)
        q = [fc(state).squeeze(-1) for fc in self.fcs]
        return q[0] if self._ensemble == 1 else q
