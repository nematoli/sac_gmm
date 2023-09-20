import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from sac_gmm.utils.projections import cam_to_world


class DecoderNet(nn.Module):
    def __init__(self, feature_size):
        super(DecoderNet, self).__init__()

        self.feature_size = feature_size
        self.i_h, self.i_w = 84, 84

        self.fc_layers = nn.Sequential(
            OrderedDict(
                [
                    ("fc_1", nn.Linear(self.feature_size, self.feature_size)),
                    ("bn_1", nn.BatchNorm1d(self.feature_size)),
                    ("elu_1", nn.ELU()),
                    ("fc_2", nn.Linear(self.feature_size, self.feature_size * 4 * 4)),
                    ("bn_2", nn.BatchNorm1d(self.feature_size * 4 * 4)),
                    ("elu_2", nn.ELU()),
                ]
            )
        )

        self.conv_1 = self.conv_layer(in_ch=self.feature_size, out_ch=256, out_ks=1, factor=2)
        self.conv_2 = self.conv_layer(in_ch=256, out_ch=256, out_ks=3, factor=2)
        self.conv_3 = self.conv_layer(in_ch=256, out_ch=64, out_ks=3, factor=2)
        self.conv_4 = self.conv_layer(in_ch=64, out_ch=1, out_ks=5, factor=None)

    def conv_layer(self, in_ch, out_ch, out_ks, factor=None):
        layers = [
            ("conv2d_1", nn.ConvTranspose2d(in_ch, in_ch, kernel_size=1, stride=1)),
            ("conv_bn_1", nn.BatchNorm2d(in_ch)),
            ("dec_relu_1", nn.ReLU()),
            ("conv2d_2", nn.ConvTranspose2d(in_ch, out_ch, kernel_size=out_ks, stride=1)),
            ("conv_bn_2", nn.BatchNorm2d(out_ch)),
            ("dec_relu_2", nn.ReLU()),
        ]

        if factor is not None and type(factor) is int:
            layers.append(("ups", nn.Upsample(scale_factor=factor)))

        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.fc_layers(x)

        x = x.view(x.size(0), self.feature_size, 4, 4)

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)

        output = F.interpolate(x, mode="bilinear", size=(self.i_h, self.i_w))
        return output


class KeypointDetector(nn.Module):
    def __init__(self, feature_size):
        super(KeypointDetector, self).__init__()

        self.feature_size = feature_size
        self.dim = 4

        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("fc_1", nn.Linear(self.feature_size, int(self.feature_size / 4))),
                    ("bn_1", nn.BatchNorm1d(int(self.feature_size / 4))),
                    ("elu_1", nn.ELU()),
                    ("fc_2", nn.Linear(int(self.feature_size / 4), int(self.feature_size / 16))),
                    ("bn_2", nn.BatchNorm1d(int(self.feature_size / 16))),
                    ("elu_2", nn.ELU()),
                    ("fc_3", nn.Linear(int(self.feature_size / 16), 1)),
                    ("sig_1", nn.Sigmoid()),
                ]
            )
        )

        self.depth_decoder = DecoderNet(feature_size=self.feature_size)
        self.heatmap_decoder = DecoderNet(feature_size=self.feature_size)
        self.depth_last = nn.Sequential(
            OrderedDict(
                [
                    ("ups", nn.Upsample(scale_factor=1)),
                    ("conv2d", nn.Conv2d(1, 1, 1, stride=1)),
                    ("bn_1", nn.BatchNorm2d(1)),
                ]
            )
        )
        self.sm = nn.Softmax(2)

    def is_mock(self):
        return False

    def keypoint(self, x):
        heatmap, depth_map, objectness = self.forward(x)
        xy = self.pixel_from_heatmap(heatmap)
        z = self.depth_from_maps(depth_map, heatmap)
        xyzo = torch.cat([xy, z, objectness], dim=1)
        return xyzo

    def forward(self, x):
        features = x.type_as(next(self.parameters()))

        objectness = self.fc(features)

        dd = self.depth_decoder(features)
        depth_map = self.depth_last(dd)

        hd = self.heatmap_decoder(features)
        heatmap = self.sm(hd.view(*hd.size()[:2], -1)).view_as(hd)
        return heatmap, depth_map, objectness

    @staticmethod
    def pixel_from_heatmap(map):
        if len(map.shape) == 4:
            map = map.squeeze(1)
        o = torch.zeros_like(map).type_as(map)
        x = torch.arange((map.shape[1])).type_as(map)
        y = torch.arange((map.shape[2])).type_as(map)
        x_o = (o.permute((0, 2, 1)) + x).permute((0, 2, 1))
        y_o = o + y
        xx = (x_o * map).sum(dim=(1, 2)).unsqueeze(1)
        yy = (y_o * map).sum(dim=(1, 2)).unsqueeze(1)
        xy = torch.cat([yy, xx], dim=1)
        return xy

    @staticmethod
    def depth_from_maps(depths_map, heatmap):
        if len(depths_map.shape) == 4:
            depths_map = depths_map.squeeze(1)
        if len(heatmap.shape) == 4:
            heatmap = heatmap.squeeze(1)
        map = depths_map * heatmap
        re = map.sum(dim=(1, 2)).unsqueeze(1)
        return re

    def to_world(self, x, cam_viewm):
        if len(x.shape) == 1:
            x = x[: self.dim - 1]
        else:
            x = x[:, : self.dim - 1]
        cam_viewm = cam_viewm.float().type_as(next(self.parameters()))
        w_pos = cam_to_world(x, cam_viewm).squeeze()
        return w_pos


class KeypointMock:
    def __init__(self, kp_noise_high, kp_noise_low):
        self.dim = 4
        self.init_pos = None
        self.pos = None
        self.kp_noise_high = np.array(kp_noise_high)
        self.kp_noise_low = np.array(kp_noise_low)

    def reset_gt(self, gt_keypoint=None):
        if gt_keypoint is None:
            return
        self.init_pos = np.array(gt_keypoint)
        self.pos = self.init_pos

    def eval(self):
        return

    def train(self):
        return

    def is_mock(self):
        return True

    def keypoint(self, x):
        return self.forward(x)

    def reset_position(self):
        self.pos = self.sample_keypoint()
        return

    def forward(self, x):
        return np.concatenate((self.pos, np.ones(1)), axis=0).reshape(1, -1).repeat(x.shape[0], 1)

    def to_world(self, x, cam_viewm=None):
        return x

    def sample_keypoint(self):
        dist = self.kp_noise_high - self.kp_noise_low
        shift = (np.random.uniform(0.0, 1.0, [self.init_pos.shape[0]]) * 2.0 - 1.0) * dist
        shift += self.kp_noise_low * np.sign(shift)
        # shift = np.random.normal(0.0, self.target_deviation, [self.init_pos.shape[0]])
        return np.copy(self.init_pos) + shift
