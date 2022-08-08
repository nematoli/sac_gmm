import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Tuple
from collections import OrderedDict

import numpy as np
import utils


class AutoEncoder(object):
    def __init__(self, hd_input_space, in_channels, hidden_dim, late_fusion, latent_lambda):
        self.encoder = Encoder(in_channels, hd_input_space, hidden_dim, late_fusion)
        self.decoder = Decoder(hidden_dim, hd_input_space, in_channels, late_fusion)
        self.latent_lambda = latent_lambda

    def forward(self, x):
        x = self.encoder.forward(x)
        x = self.decoder.forward(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, obs_space, out_dim, late_fusion):
        if isinstance(obs_space, Tuple):
            h, w = obs_space[0].shape
        else:
            h, w = obs_space.shape
        h, w = utils.misc.calc_out_size(h, w, 8, stride=4)
        h, w = utils.misc.calc_out_size(h, w, 4, stride=2)
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("enc_cnn_1", nn.Conv2d(in_channels, 16, 8, stride=4)),
                    ("enc_cnn_elu_1", nn.ELU()),
                    ("enc_cnn_2", nn.Conv2d(16, 32, 4, stride=2)),
                    (
                        "spatial_softmax",
                        SpatialSoftmax(h, w),
                    ),
                ]
            )
        )

    def forward(self, x, detach_encoder=False):
        if x.ndim == 3:
            x = x.unsqueeze(0)

        if self.late_fusion:
            for i in range(self.in_channels):
                output = []
                for i in range(self.in_channels):
                    aux = self.net(x[:, i].unsqueeze(1)).squeeze()
                    output.append(aux)
                output = torch.cat(output, dim=-1)
        else:
            output = self.net(x).squeeze()

        if detach_encoder:
            output = output.detach()
        output = self.fc(output)
        return output


class Decoder(nn.Module):
    def __init__(self, input_dim, obs_space, out_channels, late_fusion):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.late_fusion = late_fusion
        if isinstance(obs_space, Tuple):
            self.i_h, self.i_w = obs_space[0].shape
        else:
            self.i_h, self.i_w = obs_space.shape
        h, w = self.i_h, self.i_w
        h, w = utils.misc.calc_out_size(h, w, 8, stride=4)
        self.h, self.w = utils.misc.calc_out_size(h, w, 4, stride=2)

        if self.late_fusion:
            self.fc = nn.Linear(input_dim, 64 * out_channels)
            out_channels = 1
        else:
            self.fc = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, self.h * self.w * 32)

        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("dec_cnn_trans_1", nn.ConvTranspose2d(32, 16, 4, stride=2)),
                    ("dec_cnn_trans_elu_1", nn.ELU()),
                    (
                        "dec_cnn_trans_2",
                        nn.ConvTranspose2d(16, out_channels, 8, stride=4),
                    ),
                ]
            )
        )

    def forward(self, x):
        if len(list(x.size())) <= 1:  # if latent space have just one sample
            batch_size = 1
        else:
            batch_size = x.shape[0]
        x = F.elu(self.fc(x))

        if self.late_fusion:
            output = []
            for i in range(self.out_channels):
                aux = F.elu(self.fc2(x[:, 64 * i : 64 * (i + 1)]))
                aux = aux.view(batch_size, 32, self.h, self.w)
                aux = self.net(aux)
                output.append(aux)
            output = torch.cat(output, dim=1)
        else:
            output = F.elu(self.fc2(x))
            output = output.view(batch_size, 32, self.h, self.w)
            output = self.net(output)

        output = F.interpolate(output, size=(self.i_h, self.i_w))
        return output


class SpatialSoftmax(nn.Module):
    # reference: https://arxiv.org/pdf/1509.06113.pdf
    def __init__(self, height, width):
        super(SpatialSoftmax, self).__init__()
        x_map = np.empty([height, width], np.float32)
        y_map = np.empty([height, width], np.float32)

        for i in range(height):
            for j in range(width):
                x_map[i, j] = (i - height / 2.0) / height
                y_map[i, j] = (j - width / 2.0) / width

        self.x_map = torch.from_numpy(np.array(x_map.reshape((-1)), np.float32))  # W*H
        self.y_map = torch.from_numpy(np.array(x_map.reshape((-1)), np.float32))  # W*H

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])  # batch, C, W*H
        x = F.softmax(x, dim=2)  # batch, C, W*H
        fp_x = torch.matmul(x, self.x_map)  # batch, C
        fp_y = torch.matmul(x, self.y_map)  # batch, C
        x = torch.cat((fp_x, fp_y), 1)
        return x  # batch, C*2
