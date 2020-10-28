import torch

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class ClipMixingLoss(BaseWeightedLoss):
    MODES = 'embd', 'logits'

    def __init__(self, mode='', default_scale=10.0, **kwargs):
        super(ClipMixingLoss, self).__init__(**kwargs)

        assert mode in self.MODES
        self.mode = mode
        self.default_scale = default_scale

    def _forward(self, logits, norm_embd, num_clips, scale=None):
        assert num_clips > 1

        if self.mode == 'embd':
            norm_embd = norm_embd.view(norm_embd.size(0) // num_clips, num_clips, -1)

            similarity = torch.matmul(norm_embd, norm_embd.permute(0, 2, 1))
            losses = 1.0 - similarity

            ind_range = torch.arange(num_clips, dtype=torch.int64, device=norm_embd.device)
            mask = ind_range.view(-1, 1) < ind_range.view(1, -1)

            valid_losses = losses[mask.view(-1, num_clips, num_clips).repeat(norm_embd.size(0), 1, 1)]
        else:
            scale = scale if scale is not None else self.default_scale
            logits = scale * logits.view(logits.size(0) // num_clips, num_clips, -1)

            with torch.no_grad():
                probs = torch.softmax(logits, dim=2)
                trg_probs = probs.mean(dim=1, keepdim=True)

            log_probs = torch.log_softmax(logits, dim=2)
            valid_losses = (trg_probs * log_probs).sum(dim=2).neg()

        return valid_losses.mean()
