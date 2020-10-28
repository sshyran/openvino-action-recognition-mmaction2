import torch

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class LocalPushLoss(BaseWeightedLoss):
    def __init__(self, margin=0.1, smart_margin=True, **kwargs):
        super(LocalPushLoss, self).__init__(**kwargs)

        self.margin = margin
        assert self.margin >= 0.0

        self.smart_margin = smart_margin

    def _forward(self, normalized_embeddings, cos_theta, target):
        similarity = normalized_embeddings.matmul(normalized_embeddings.permute(1, 0))

        with torch.no_grad():
            pairs_mask = target.view(-1, 1) != target.view(1, -1)

            if self.smart_margin:
                center_similarity = cos_theta[torch.arange(cos_theta.size(0), device=target.device), target]
                threshold = center_similarity.clamp(min=self.margin).view(-1, 1) - self.margin
            else:
                threshold = self.margin
            similarity_mask = similarity > threshold

            mask = pairs_mask & similarity_mask

        filtered_similarity = torch.where(mask, similarity - threshold, torch.zeros_like(similarity))
        losses, _ = filtered_similarity.max(dim=-1)

        return losses.mean()
