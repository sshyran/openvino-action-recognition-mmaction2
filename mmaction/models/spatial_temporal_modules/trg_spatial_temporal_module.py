import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init

from ...core.ops.nonlinearities import HSwish
from ..registry import SPATIAL_TEMPORAL_MODULES


class TRGLayer(nn.Module):
    """Based on TRG network: https://arxiv.org/pdf/1908.09995.pdf
    """

    def __init__(self, num_channels, num_heads, embed_size, spatial_size, temporal_size):
        super(TRGLayer, self).__init__()

        self.out_channels = num_channels
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.spatial_size = np.prod(spatial_size) if not isinstance(spatial_size, int) else spatial_size ** 2
        self.temporal_size = temporal_size
        self.factor = 1.0 / np.sqrt(self.spatial_size * self.embed_size)

        self.theta = self._project(num_channels, embed_size * num_heads)
        self.phi = self._project(num_channels, embed_size * num_heads)
        self.g = nn.Sequential(
            self._project(num_channels, num_channels * num_heads, with_bn=True),
            HSwish())

        self.adjacent_softmax = nn.Softmax(dim=-1)
        self.gcn_non_linearity = HSwish()

        self.heads_weights = nn.Parameter(torch.Tensor(num_heads, num_heads))
        self.heads_weights.data.normal_()
        self.heads_softmax = nn.Softmax(dim=-1)

        self.out_non_linearity = HSwish()

    @staticmethod
    def _project(in_channels, out_channels, with_bn=False):
        layers = [nn.Conv3d(in_channels, in_channels, bias=False, groups=in_channels,
                            kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1),
                  nn.BatchNorm3d(in_channels),
                  HSwish(),
                  nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=not with_bn)]

        if with_bn:
            layers.append(nn.BatchNorm3d(out_channels, eps=1e-05, momentum=0.9, affine=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        theta = self.theta(x) \
            .view(-1, self.num_heads, self.embed_size, self.temporal_size, self.spatial_size) \
            .permute(0, 1, 3, 2, 4) \
            .reshape(-1, self.num_heads, self.temporal_size, self.embed_size * self.spatial_size)
        phi = self.phi(x) \
            .view(-1, self.num_heads, self.embed_size, self.temporal_size, self.spatial_size) \
            .permute(0, 1, 2, 4, 3) \
            .reshape(-1, self.num_heads, self.embed_size * self.spatial_size, self.temporal_size)
        g = self.g(x) \
            .view(-1, self.num_heads, self.out_channels, self.temporal_size, self.spatial_size) \
            .permute(0, 1, 3, 2, 4) \
            .reshape(-1, self.num_heads, self.temporal_size, self.out_channels * self.spatial_size)

        adjacent_matrix = self.adjacent_softmax(self.factor * torch.matmul(theta, phi))
        gcn = self.gcn_non_linearity(torch.matmul(adjacent_matrix, g))

        heads_features = gcn.mean(dim=-1).view(-1, self.num_heads, self.temporal_size).permute(0, 2, 1)
        heads_attention = self.heads_softmax(torch.matmul(heads_features, self.heads_weights))

        z = torch.matmul(gcn.permute(0, 2, 3, 1), heads_attention.view(-1, self.temporal_size, self.num_heads, 1)) \
            .view(-1, self.temporal_size, self.out_channels, self.spatial_size) \
            .permute(0, 2, 1, 3) \
            .reshape_as(x)
        out = self.out_non_linearity(x + z)

        return out


@SPATIAL_TEMPORAL_MODULES.register_module()
class TRGSpatialTemporalModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=4,
                 num_heads=8, embed_size=32, spatial_size=7, temporal_size=1):
        super(TRGSpatialTemporalModule, self).__init__()

        layers = []
        if out_channels != in_channels:
            layers.extend([nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                           nn.BatchNorm3d(out_channels, eps=1e-05, momentum=0.9, affine=True)])

        layers.extend([TRGLayer(out_channels, num_heads, embed_size, spatial_size, temporal_size)
                       for _ in range(num_layers)])

        self.features = nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1.0, 0.0)
            elif isinstance(m, nn.Parameter):
                m.data.normal_()

    def forward(self, x, return_extra_data=False):
        if return_extra_data:
            return self.features(x), dict()
        else:
            return self.features(x)
