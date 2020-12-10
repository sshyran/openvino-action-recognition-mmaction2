import torch
import torch.nn as nn
import numpy as np

from ...core.ops import Conv3d, HSwish, conv_1x1x1_bn


class SimilarityGuidedSampling(nn.Module):
    def __init__(self, in_planes, num_bins, internal_factor=2.0, embd_size=8, norm='none'):
        super(SimilarityGuidedSampling, self).__init__()

        hidden_dim = int(internal_factor * in_planes)
        layers = [
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            *conv_1x1x1_bn(in_planes, hidden_dim, norm=norm),
            HSwish(),
            Conv3d(hidden_dim, embd_size, kernel_size=1, normalization=norm)
            # *conv_1x1x1_bn(hidden_dim, embd_size, norm=norm),
        ]
        self.encoder = nn.Sequential(*layers)

        self.num_bins = num_bins
        assert self.num_bins >= 2
        self.interval_scale = 1.0 / float(4 * (self.num_bins - 2) + 2)

        centers = np.array([4 * i - 1 for i in range(self.num_bins)], dtype=np.float32)
        self.register_buffer('centers', torch.from_numpy(centers))

    def forward(self, x):
        embds = self.encoder(x).squeeze(4).squeeze(3)
        norms = torch.sum(embds ** 2, dim=1)

        with torch.no_grad():
            min_norm = torch.min(norms, dim=1)[0]
            max_norm = torch.max(norms, dim=1)[0]

            gamma = self.interval_scale * (max_norm - min_norm)
            centers = min_norm.view(-1, 1, 1) + gamma.view(-1, 1, 1) * self.centers.view(1, 1, -1)

        diff = norms.unsqueeze(2) - centers
        unscaled_coeff = torch.clamp_min(1.0 - 0.5 * torch.abs(diff) / gamma.view(-1, 1, 1), 0.0)

        with torch.no_grad():
            sum_coeff = torch.sum(unscaled_coeff, dim=1, keepdim=True)
            scales = torch.where(sum_coeff > 0.0, 1.0 / sum_coeff, torch.ones_like(sum_coeff))

        scaled_coeff = scales * unscaled_coeff
        scaled_features = x.unsqueeze(3) * scaled_coeff.unsqueeze(1).unsqueeze(4).unsqueeze(5)
        out_features = torch.sum(scaled_features, dim=2)

        return out_features
