import numpy as np
import torch
import torch.nn as nn
import gym.spaces
from collections import OrderedDict

from .utils.utils import MLP, get_activation, weight_init
from .utils.distributions import TanhNormal, Normal, MixedDistribution


class TDMPCModel(nn.Module):
    """Task-Oriented Latent Dynamics model."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def make_enc_dec(self, cfg, ob_space, state_dim):
        self.encoder = Encoder(cfg.encoder, ob_space, state_dim)
        self.decoder = Decoder(cfg.decoder, state_dim, ob_space)

    def imagine_step(self, state, ac):
        out = torch.cat([state, ac], dim=-1)
        return self.dynamics(out), self.reward(out).squeeze(-1)

    def load_state_dict(self, ckpt):
        try:
            super().load_state_dict(ckpt, strict=False)
        except:
            from collections import OrderedDict

            ckpt = OrderedDict([(k, v) for k, v in ckpt.items() if not k.startswith("critic")])
            super().load_state_dict(ckpt, strict=False)


class Encoder(nn.Module):
    def __init__(self, cfg, ob_space, state_dim):
        super().__init__()
        self._ob_space = ob_space
        self.encoders = nn.ModuleDict()
        enc_dim = 0
        for k, v in ob_space.spaces.items():
            if len(v.shape) == 3:
                self.encoders[k] = ConvEncoder(
                    cfg.image_shape,
                    cfg.kernel_size,
                    cfg.stride,
                    cfg.conv_dim,
                    cfg.cnn_act,
                )
            elif len(v.shape) == 1:
                self.encoders[k] = DenseEncoder(
                    gym.spaces.flatdim(v),
                    cfg.embed_dim,
                    cfg.hidden_dims,
                    cfg.dense_act,
                )
            else:
                raise ValueError("Observations should be either vectors or RGB images")
            enc_dim += self.encoders[k].output_dim
        self.fc = MLP(enc_dim, state_dim, [], cfg.dense_act)
        self.act = get_activation(cfg.dense_act)
        self.output_dim = state_dim

    def forward(self, ob):
        embeddings = [self.act(self.encoders[k](v)) for k, v in ob.items()]
        return self.fc(torch.cat(embeddings, -1))
        # return torch.cat(embeddings, -1)


class DenseEncoder(nn.Module):
    def __init__(self, shape, embed_dim, hidden_dims, activation):
        super().__init__()
        self.fc = MLP(shape, embed_dim, hidden_dims, activation)
        self.output_dim = embed_dim

    def forward(self, ob):
        return self.fc(ob)


class ConvEncoder(nn.Module):
    def __init__(self, shape, kernel_size, stride, conv_dim, activation):
        super().__init__()
        convs = []
        activation = get_activation(activation)
        h, w, d_prev = shape
        for k, s, d in zip(kernel_size, stride, conv_dim):
            convs.append(nn.Conv2d(d_prev, d, k, s))
            convs.append(activation)
            d_prev = d
            h = int(np.floor((h - k) / s + 1))
            w = int(np.floor((w - k) / s + 1))

        self.convs = nn.Sequential(*convs)
        self.output_dim = h * w * d_prev
        self.apply(weight_init)

    def forward(self, ob):
        shape = list(ob.shape[:-3]) + [-1]
        x = ob.reshape([-1] + list(ob.shape[-3:]))
        # x = x.permute(0, 3, 1, 2)
        x = self.convs(x)
        return x.reshape(shape)


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


class Decoder(nn.Module):
    def __init__(self, cfg, state_dim, ob_space):
        super().__init__()
        self._ob_space = ob_space
        self.decoders = nn.ModuleDict()
        for k, v in ob_space.spaces.items():
            if len(v.shape) == 3:
                self.decoders[k] = ConvDecoder(
                    state_dim,
                    cfg.image_shape,
                    cfg.kernel_size,
                    cfg.stride,
                    cfg.conv_dim,
                    cfg.cnn_act,
                )
            elif len(v.shape) == 1:
                self.decoders[k] = DenseDecoder(
                    state_dim,
                    gym.spaces.flatdim(v),
                    cfg.hidden_dims,
                    cfg.dense_act,
                )
            else:
                raise ValueError("Observations should be either vectors or RGB images")

    def forward(self, feat):
        return MixedDistribution(OrderedDict([(k, self.decoders[k](feat)) for k in self._ob_space.spaces]))


class ConvDecoder(nn.Module):
    """CNN decoder returns normal distribution of prediction with std 1."""

    def __init__(self, input_dim, shape, kernel_size, stride, conv_dim, activation):
        super().__init__()
        self._shape = list(shape)
        self._conv_dim = conv_dim

        self.fc = MLP(input_dim, conv_dim[0], [], None)

        d_prev = conv_dim[0]
        conv_dim = conv_dim + [shape[-1]]
        activation = get_activation(activation)
        deconvs = []
        for k, s, d in zip(kernel_size, stride, conv_dim[1:]):
            deconvs.append(nn.ConvTranspose2d(d_prev, d, k, s))
            deconvs.append(activation)
            d_prev = d
        deconvs.pop()
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, feat):
        shape = list(feat.shape[:-1]) + self._shape
        x = self.fc(feat)
        x = x.reshape([-1, self._conv_dim[0], 1, 1])
        x = self.deconvs(x)
        # x = x.permute(0, 2, 3, 1)
        # x = x.reshape(shape)
        return Normal(x, 1, event_dim=1)


class DenseDecoder(nn.Module):
    """MLP decoder returns normal distribution of prediction."""

    def __init__(self, input_dim, output_dim, hidden_dims, activation):
        super().__init__()
        self.fc = MLP(input_dim, output_dim * 2, hidden_dims, activation)

    def forward(self, feat):
        mean, std = self.fc(feat).chunk(2, dim=-1)
        std = (std.tanh() + 1) * 0.7 + 0.1  # [0.1, 1.5]
        return Normal(mean, std, event_dim=1)
