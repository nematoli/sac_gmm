import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super(QNetwork, self).__init__()
        # TODO: change ELU with SiLU
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
                    ("fc_3", nn.Linear(hidden_dim, 1)),
                ]
            )
        )

    def forward(self, state, action):
        fc_input = torch.cat((state, action), dim=-1)
        return self.fc_layers(fc_input)


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.Q1 = QNetwork(input_dim=input_dim, hidden_dim=hidden_dim)
        self.Q2 = QNetwork(input_dim=input_dim, hidden_dim=hidden_dim)

    def forward(self, state, action):
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return q1, q2
