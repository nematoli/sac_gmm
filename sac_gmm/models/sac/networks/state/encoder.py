"""The following code is directly taken from:
    https://github.com/youngwoon/robot-learning/blob/dreamerv3/rolf/networks/tdmpc_model.py
"""
#####################################################################
"""
Code reference:
  https://github.com/MishaLaskin/rad/blob/master/encoder.py
"""

import torch
import torch.nn as nn
import gym

from .utils import MLP, get_activation, weight_init


class Encoder(nn.Module):
    def __init__(self, cfg, ob_space, state_dim):
        super().__init__()
        self._ob_space = ob_space
        self.encoders = nn.ModuleDict()
        self.encoders["obs"] = DenseEncoder(
            self._ob_space.shape[0],
            cfg.embed_dim,
            cfg.hidden_dims,
            cfg.dense_act,
        )
        self.fc = MLP(cfg.embed_dim, state_dim, [], cfg.dense_act)
        self.act = get_activation(cfg.dense_act)
        self.output_dim = state_dim

    def forward(self, ob):
        embeddings = [self.act(self.encoders[k](v)) for k, v in ob.items()]
        return self.fc(torch.cat(embeddings, -1))


class DenseEncoder(nn.Module):
    def __init__(self, shape, embed_dim, hidden_dims, activation):
        super().__init__()
        self.fc = MLP(shape, embed_dim, hidden_dims, activation)
        self.output_dim = embed_dim

    def forward(self, ob):
        return self.fc(ob)
