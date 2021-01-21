import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .math import normalize
from .nonlinearities import HSwish
from .conv import conv_1x1x1_bn, conv_1xkxk_bn


class SimilarityGuidedSampling(nn.Module):
    def __init__(self, in_planes, num_bins, internal_factor=2.0,
                 ce_scale=10.0, center_threshold=0.5, norm='none'):
        super(SimilarityGuidedSampling, self).__init__()

        self.num_bins = num_bins
        assert self.num_bins >= 2
        self.ce_scale = float(ce_scale)
        assert self.ce_scale > 0.0
        self.center_threshold = float(center_threshold)
        assert self.center_threshold >= 0.0

        hidden_dim = int(internal_factor * in_planes)
        layers = [
            *conv_1x1x1_bn(in_planes, hidden_dim, norm=norm),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            *conv_1xkxk_bn(hidden_dim, hidden_dim, k=3, groups=hidden_dim, norm=norm),
            nn.ReLU(),
            *conv_1x1x1_bn(hidden_dim, 1, norm=norm)
            # *conv_1x1x1_bn(in_planes, hidden_dim, norm=norm),
            # HSwish(),
            # *conv_1x1x1_bn(hidden_dim, 1, norm=norm)
        ]
        self.encoder = nn.Sequential(*layers)

        valid_mask = np.triu(np.ones((self.num_bins, self.num_bins), dtype=np.float32), k=1)
        self.register_buffer('valid_mask', torch.from_numpy(valid_mask))

    def forward(self, x, return_extra_data=False):
        embds = self.encoder(x)
        _, _, enc_t, enc_h, enc_w = embds.size()
        norm_embd = normalize(embds.view(-1, enc_t, enc_h * enc_w).transpose(1, 2), dim=1)

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

            # import matplotlib.pyplot as plt
            # maps = norm_embd.transpose(1, 2).view(-1, enc_t, enc_h, enc_w)[0].cpu().numpy()
            # _, axs = plt.subplots(4, 4)
            # for ii in range(4):
            #     for jj in range(4):
            #         axs[ii, jj].imshow(maps[ii * 4 + jj])
            # plt.show()

        centers_sum = torch.sum(norm_embd.unsqueeze(3) * group_mask.unsqueeze(1), dim=2, keepdim=True)
        norm_centers = normalize(centers_sum / group_sizes.unsqueeze(1), dim=1)

        similarities = torch.sum(norm_embd.unsqueeze(3) * norm_centers, dim=1).clamp(-1.0, 1.0)

        with torch.no_grad():
            if self.training:
                scores = (1.0 + similarities) * group_mask
                flat_scores = scores.transpose(1, 2).reshape(-1, scores.size(1))

                temporal_pos = torch.multinomial(flat_scores, num_samples=1).view(-1, self.num_bins)
            else:
                temporal_pos = torch.argmax(similarities * group_mask, dim=1)

            _, orig_c, _, orig_h, orig_w = x.size()
            ind = temporal_pos.unsqueeze(1).unsqueeze(3).unsqueeze(4).repeat(1, orig_c, 1, orig_h, orig_w)

        out_features = torch.gather(x, 2, ind)

        if return_extra_data:
            return out_features, dict(similarities=similarities, groups=groups)
        else:
            return out_features

    def loss(self, similarities, groups, **kwargs):
        ce_loss = self._ce_loss(similarities, groups, self.ce_scale, self.num_bins)
        return ce_loss

    @staticmethod
    def _ce_loss(similarities, groups, scale, num_classes):
        scores = scale * similarities.view(-1, num_classes)
        labels = groups.view(-1)
        return F.cross_entropy(scores, labels)
