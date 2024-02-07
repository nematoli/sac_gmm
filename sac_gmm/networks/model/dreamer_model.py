import numpy as np
import torch
import torch.nn as nn
import gym.spaces
from collections import OrderedDict

from .utils.utils import MLP, get_activation, weight_init
from .utils.distributions import TanhNormal, Normal, MixedDistribution
from .utils.pytorch import symlog
from .utils.utils import rmap
from .utils.distributions import Bernoulli, Symlog, SymlogDiscrete


class DreamerModel(nn.Module):
    """DreamerV3's Observation Encoder and Decoder."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.state_dim = 0

    def make_enc_dec(self, cfg, ob_space, state_dim, dtype=torch.float64):
        self.encoder = Encoder(cfg.encoder, ob_space, state_dim)
        self.decoder = Decoder(cfg.decoder, state_dim, ob_space)

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
        self.output_dim = 0
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
                    cfg.hidden_dims,
                    cfg.dense_act,
                    cfg.symlog,
                )
            else:
                raise ValueError("Observations should be either vectors or RGB images")
            self.output_dim += self.encoders[k].output_dim
        self.fc = MLP(self.output_dim, state_dim, [], cfg.dense_act)
        self.act = get_activation(cfg.dense_act)
        self.output_dim = state_dim

    def forward(self, ob):
        embeddings = [self.act(self.encoders[k](v)) for k, v in ob.items()]
        return self.fc(torch.cat(embeddings, -1))


class DenseEncoder(nn.Module):
    def __init__(self, shape, hidden_dims, activation, symlog):
        super().__init__()
        self.output_dim = hidden_dims[-1]
        self.fc = MLP(shape, hidden_dims[-1], hidden_dims[:-1], activation, norm=True)
        self._symlog = symlog

    def forward(self, ob):
        if self._symlog:
            ob = rmap(symlog, ob)
        return self.fc(ob)


class ConvEncoder(nn.Module):
    def __init__(self, shape, kernel_size, stride, conv_dim, activation):
        super().__init__()
        convs = []
        activation = get_activation(activation)
        h, w, d_prev = shape
        for k, s, d in zip(kernel_size, stride, conv_dim[1:]):
            h = int(np.floor((h - k + 2) / s + 1))
            w = int(np.floor((w - k + 2) / s + 1))
            convs.append(nn.Conv2d(d_prev, d, k, s, padding=1, bias=False))
            convs.append(nn.LayerNorm([d, h, w]))
            # convs.append(nn.GroupNorm(d, d))  # Use GroupNorm to match official code
            convs.append(activation)
            d_prev = d

        self.convs = nn.Sequential(*convs)
        self.output_dim = h * w * d_prev

    def forward(self, ob):
        shape = list(ob.shape[:-3]) + [-1]
        x = ob.reshape([-1] + list(ob.shape[-3:]))
        x = x.permute(0, 3, 1, 2)
        x = self.convs(x)
        return x.reshape(shape)


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
                    cfg.seed_shape,
                    cfg.kernel_size,
                    cfg.stride,
                    cfg.conv_dim,
                    cfg.cnn_act,
                    cfg.image_loss,
                )
            elif len(v.shape) == 1:
                self.decoders[k] = DenseDecoder(
                    state_dim,
                    gym.spaces.flatdim(v),
                    cfg.hidden_dims,
                    cfg.dense_act,
                    cfg.state_loss,
                )
            else:
                raise ValueError("Observations should be either vectors or RGB images")

    def forward(self, feat):
        return MixedDistribution(OrderedDict([(k, self.decoders[k](feat)) for k in self._ob_space.spaces]))


class ConvDecoder(nn.Module):
    """CNN decoder returns normal distribution of prediction with std 1."""

    def __init__(self, input_dim, out_shape, seed_shape, kernel_size, stride, conv_dim, activation, loss):
        super().__init__()
        self._shape = list(out_shape)
        self._seed_shape = list(seed_shape)
        self._seed_shape = self._seed_shape[2:] + self._seed_shape[:2]
        self._conv_dim = conv_dim
        self._loss = loss
        activation = get_activation(activation)

        self.fc = MLP(input_dim, np.prod(seed_shape), [], activation, norm=True)

        d_prev, h, w = self._seed_shape
        deconvs = []
        for k, s, d in zip(kernel_size, stride, conv_dim[1:]):
            h = (h - 1) * s + k - 2
            w = (w - 1) * s + k - 2
            deconvs.append(nn.ConvTranspose2d(d_prev, d, k, s, padding=1, bias=False))
            deconvs.append(nn.LayerNorm([d, h, w]))
            # deconvs.append(nn.GroupNorm(d, d))  # Use GroupNorm to match official code
            deconvs.append(activation)
            d_prev = d
        deconvs = deconvs[:-3]
        deconvs.append(nn.ConvTranspose2d(conv_dim[-2], conv_dim[-1], k, s, padding=1))
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, feat):
        shape = list(feat.shape[:-1]) + self._shape
        x = self.fc(feat)
        x = x.reshape([-1, *self._seed_shape])
        x = self.deconvs(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(shape)
        return Normal(x, 1, event_dim=3)


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
