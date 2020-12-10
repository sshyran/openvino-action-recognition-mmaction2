import torch
import torch.nn as nn

from ...core.ops import HSwish, conv_1x1x1_bn, normalize


class SimilarityGuidedSampling(nn.Module):
    def __init__(self, in_planes, num_bins, internal_factor=2.0, embd_size=32, scale=5.0, norm='none'):
        super(SimilarityGuidedSampling, self).__init__()

        self.num_bins = num_bins
        assert self.num_bins >= 2
        self.scale = scale
        assert self.scale > 0.0

        hidden_dim = int(internal_factor * in_planes)
        layers = [
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            *conv_1x1x1_bn(in_planes, hidden_dim, norm=norm),
            HSwish(),
            *conv_1x1x1_bn(hidden_dim, embd_size, norm=norm),
        ]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        embds = self.encoder(x).squeeze(4).squeeze(3)
        norm_embd = normalize(embds, dim=1)

        with torch.no_grad():
            neighbour_similarities = torch.sum(norm_embd[:, :, 1:] * norm_embd[:, :, :-1], dim=1)

            smallest_similarities = torch.topk(neighbour_similarities, self.num_bins - 1, largest=False)[0]
            threshold = smallest_similarities[:, -1]
            edges = (neighbour_similarities > threshold.view(-1, 1)).int()

            init_interval = torch.zeros(edges.size(0), 1, dtype=edges.dtype, device=edges.device)
            interval_ends = torch.cat((init_interval, 1 - edges), dim=1)
            groups = torch.cumsum(interval_ends, dim=1)

            group_ids = torch.arange(self.num_bins, dtype=groups.dtype, device=groups.device)
            group_mask = (groups.unsqueeze(2).repeat(1, 1, self.num_bins) == group_ids.view(1, 1, -1)).float()

            group_sizes = torch.sum(group_mask, dim=1, keepdim=True)
            centers_sum = torch.sum(norm_embd.unsqueeze(3) * group_mask.unsqueeze(1), dim=2, keepdim=True)
            norm_centers = normalize(centers_sum / group_sizes.unsqueeze(1), dim=1)

        similarities = torch.sum(norm_embd.unsqueeze(3) * norm_centers, dim=1)
        weights = torch.softmax(self.scale * similarities, dim=2)

        with torch.no_grad():
            sum_weights = torch.sum(weights, dim=1, keepdim=True)
            scales = torch.where(sum_weights > 0.0, 1.0 / sum_weights, torch.ones_like(sum_weights))

        norm_weights = scales * weights
        scaled_features = x.unsqueeze(3) * norm_weights.unsqueeze(1).unsqueeze(4).unsqueeze(5)
        out_features = torch.sum(scaled_features, dim=2)

        return out_features
