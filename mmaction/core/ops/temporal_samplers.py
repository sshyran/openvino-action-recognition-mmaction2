import torch
import torch.nn as nn
import numpy as np

from ...core.ops import HSwish, conv_1x1x1_bn


class SimilarityGuidedSampling(nn.Module):
    def __init__(self, in_planes, num_bins, internal_factor=0.5, embd_size=8, norm='none'):
        super(SimilarityGuidedSampling, self).__init__()

        hidden_dim = int(internal_factor * in_planes)
        layers = [
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            *conv_1x1x1_bn(in_planes, hidden_dim, norm=norm),
            HSwish(),
            *conv_1x1x1_bn(hidden_dim, embd_size, norm=norm),
        ]
        self.encoder = nn.Sequential(*layers)

        self.num_bins = num_bins
        self.embd_size = embd_size

        centers = np.array([2.0 * i + 1 for i in range(self.num_bins)], dtype=np.float32)
        self.register_buffer('centers', torch.from_numpy(centers))

    def forward(self, x):
        embds = self.encoder(x).squeeze(4).squeeze(3)
        norms = torch.sum(embds ** 2, dim=1)

        with torch.no_grad():
            range_size = torch.max(norms, dim=1)[0] - torch.min(norms, dim=1)[0]
            gamma = 0.5 * range_size / float(self.num_bins)
            centers = gamma.view(-1, 1, 1) * self.centers.view(1, 1, -1)

        diff = norms.unsqueeze(2) - centers
        coeff = torch.clamp_min(1.0 - torch.abs(diff) / gamma.view(-1, 1, 1), 0.0)

        scaled_features = x.unsqueeze(3) * coeff.unsqueeze(1).unsqueeze(4).unsqueeze(5)
        out_features = torch.sum(scaled_features, dim=2)

        return out_features
