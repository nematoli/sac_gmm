import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import numpy as np

from pl_bolts.models import AE
from torchvision.models import resnet18, ResNet18_Weights  # , resnet50, ResNet50_Weights
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


def calc_out_size(w, h, kernel_size, padding=0, stride=1):
    width = (w - kernel_size + 2 * padding) // stride + 1
    height = (h - kernel_size + 2 * padding) // stride + 1
    return width, height


class PTEncoder(nn.Module):
    def __init__(self, model_name):
        super(PTEncoder, self).__init__()
        self.no_encoder = False
        if model_name == "cifar10-resnet18":
            self.wrap = AE(input_height=32).from_pretrained("cifar10-resnet18").encoder
            self.mean = np.array([x / 255.0 for x in [125.3, 123.0, 113.9]])
            self.std = np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
            self.feature_size = 512
            self.wrap.eval()
        elif model_name == "imagenet-resnet18":
            self.wrap = torch.nn.Sequential(*(list(resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-2]))
            preprocess = ResNet18_Weights.DEFAULT.transforms()
            self.mean = np.array(preprocess.mean)
            self.std = np.array(preprocess.std)
            self.feature_size = 512
            self.wrap.eval()
        else:
            self.no_encoder = True
            self.feature_size = 21

    def preprocess(self, x, mean, std):
        norm = T.Compose(
            [
                T.Resize(32, interpolation=InterpolationMode.BILINEAR),
                T.Normalize(mean=mean, std=std),
            ]
        )
        o = norm(x / 255.0)
        return o

    def forward(self, x):
        with torch.no_grad():
            if self.no_encoder:
                return None
            o = self.wrap(self.preprocess(x, self.mean, self.std))
        return o


class AutoEncoder(nn.Module):
    def __init__(self, img_res, in_channels, hidden_dim, late_fusion, latent_lambda, device):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(in_channels, img_res, hidden_dim, late_fusion, device).to(device)
        self.decoder = Decoder(hidden_dim, img_res, in_channels, late_fusion, device).to(device)
        self.latent_lambda = latent_lambda

    def forward(self, x):
        x = self.encoder.forward(x)
        x = self.decoder.forward(x)
        return x

    def get_image_rep(self, x):
        self.encoder.eval()
        return self.encoder.forward(x, detach_encoder=True).detach()


class Encoder(nn.Module):
    def __init__(self, in_channels, obs_space, out_dim, late_fusion, device):
        super(Encoder, self).__init__()
        self.late_fusion = late_fusion
        self.device = device
        self.in_channels = in_channels
        h, w = obs_space, obs_space
        h, w = calc_out_size(h, w, 8, stride=4)
        h, w = calc_out_size(h, w, 4, stride=2)
        if late_fusion:
            num_channels = 1
        else:
            num_channels = in_channels
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("enc_cnn_1", nn.Conv2d(num_channels, 16, 8, stride=4)),
                    ("enc_cnn_elu_1", nn.ELU()),
                    ("enc_cnn_2", nn.Conv2d(16, 32, 4, stride=2)),
                    (
                        "spatial_softmax",
                        SpatialSoftmax(h, w, self.device).to(self.device),
                    ),
                ]
            )
        ).to(self.device)

        if self.late_fusion:
            self.fc = nn.Linear(64 * in_channels, out_dim).to(self.device)
        else:
            self.fc = nn.Linear(64, out_dim).to(self.device)

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
    def __init__(self, input_dim, obs_space, out_channels, late_fusion, device):
        super(Decoder, self).__init__()
        self.device = device
        self.out_channels = out_channels
        self.late_fusion = late_fusion
        self.i_h, self.i_w = obs_space, obs_space
        h, w = self.i_h, self.i_w
        h, w = calc_out_size(h, w, 8, stride=4)
        self.h, self.w = calc_out_size(h, w, 4, stride=2)

        if self.late_fusion:
            self.fc = nn.Linear(input_dim, 64 * out_channels).to(self.device)
            out_channels = 1
        else:
            self.fc = nn.Linear(input_dim, 64).to(self.device)
        self.fc2 = nn.Linear(64, self.h * self.w * 32).to(self.device)

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
        ).to(self.device)

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
    def __init__(self, height, width, device):
        super(SpatialSoftmax, self).__init__()
        x_map = np.empty([height, width], np.float32)
        y_map = np.empty([height, width], np.float32)

        for i in range(height):
            for j in range(width):
                x_map[i, j] = (i - height / 2.0) / height
                y_map[i, j] = (j - width / 2.0) / width

        self.x_map = torch.from_numpy(np.array(x_map.reshape((-1)), np.float32)).to(device)  # W*H
        self.y_map = torch.from_numpy(np.array(x_map.reshape((-1)), np.float32)).to(device)  # W*H

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])  # batch, C, W*H
        x = F.softmax(x, dim=2)  # batch, C, W*H
        fp_x = torch.matmul(x, self.x_map)  # batch, C
        fp_y = torch.matmul(x, self.y_map)  # batch, C
        x = torch.cat((fp_x, fp_y), 1)
        return x  # batch, C*2
