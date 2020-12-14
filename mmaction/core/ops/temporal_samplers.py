import torch
import torch.nn as nn
import torch.nn.functional as F

from .math import normalize
from .nonlinearities import HSwish
from .conv import conv_1x1x1_bn
from .pooling import AdaptivePool3D


class SimilarityGuidedSampling(nn.Module):
    def __init__(self, in_planes, num_bins, internal_factor=2.0, embd_size=32,
                 scale=10.0, pos_score=0.5, pool_method='avg+max', norm='none'):
        super(SimilarityGuidedSampling, self).__init__()

        self.num_bins = num_bins
        assert self.num_bins >= 2
        self.scale = float(scale)
        assert self.scale > 0.0
        self.pos_score = float(pos_score)
        assert self.pos_score >= 0.0

        hidden_dim = int(internal_factor * in_planes)
        layers = [
            AdaptivePool3D((None, 1, 1), method=pool_method),
            *conv_1x1x1_bn(in_planes, hidden_dim, norm=norm),
            HSwish(),
            *conv_1x1x1_bn(hidden_dim, embd_size, norm=norm),
        ]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x, return_extra_data=False):
        embds = self.encoder(x).squeeze(4).squeeze(3)
        norm_embd = normalize(embds, dim=1)

        with torch.no_grad():
            neighbour_similarities = torch.sum(norm_embd[:, :, 1:] * norm_embd[:, :, :-1], dim=1)
            break_idx = torch.topk(neighbour_similarities, self.num_bins - 1, dim=1, largest=False)[1]
            breaks = torch.zeros_like(neighbour_similarities, dtype=torch.int32).scatter(1, break_idx, 1)

            init_interval = torch.zeros(breaks.size(0), 1, dtype=breaks.dtype, device=breaks.device)
            interval_ends = torch.cat((init_interval, breaks), dim=1)
            groups = torch.cumsum(interval_ends, dim=1)

            group_ids = torch.arange(self.num_bins, dtype=groups.dtype, device=groups.device)
            group_mask = (groups.unsqueeze(2).repeat(1, 1, self.num_bins) == group_ids.view(1, 1, -1)).float()
            group_sizes = torch.sum(group_mask, dim=1, keepdim=True)

            sum_weights = torch.sum(group_mask, dim=1, keepdim=True)
            scales = torch.where(sum_weights > 0.0, 1.0 / sum_weights, torch.ones_like(sum_weights))
            norm_weights = scales * group_mask

            # pos_scores = self.pos_score * group_mask

        centers_sum = torch.sum(norm_embd.unsqueeze(3) * group_mask.unsqueeze(1), dim=2, keepdim=True)
        norm_centers = normalize(centers_sum / group_sizes.unsqueeze(1), dim=1)

        similarities = torch.sum(norm_embd.unsqueeze(3) * norm_centers, dim=1)
        # weights = torch.softmax(self.scale * (similarities + pos_scores), dim=2)
        #
        # with torch.no_grad():
        #     sum_weights = torch.sum(weights, dim=1, keepdim=True)
        #     scales = torch.where(sum_weights > 0.0, 1.0 / sum_weights, torch.ones_like(sum_weights))
        #
        # norm_weights = scales * weights
        weighted_features = x.unsqueeze(3) * norm_weights.unsqueeze(1).unsqueeze(4).unsqueeze(5)
        out_features = torch.sum(weighted_features, dim=2)

        if return_extra_data:
            return out_features, dict(similarities=similarities, groups=groups)
        else:
            return out_features

    def loss(self, similarities, groups, **kwargs):
        scores = self.scale * similarities.view(-1, self.num_bins)
        labels = groups.view(-1)

        loss = F.cross_entropy(scores, labels)

        return loss
