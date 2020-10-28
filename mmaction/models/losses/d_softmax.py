import torch

from ..registry import LOSSES
from .metric_learning_base import BaseMetricLearningLoss


@LOSSES.register_module()
class DSoftmaxLoss(BaseMetricLearningLoss):
    """Computes the D-Softmax loss: https://arxiv.org/pdf/1908.01281.pdf
    """

    def __init__(self, end_point=0.9, **kwargs):
        super(DSoftmaxLoss, self).__init__(**kwargs)

        self.d = end_point
        assert self.d > 0.0

    def _calculate(self, cos_theta, target, scale):
        num_classes = cos_theta.size(1)

        intra_values = cos_theta[torch.arange(cos_theta.size(0), device=target.device), target]
        intra_losses = torch.log(1.0 + torch.exp(scale * (self.d - intra_values)))

        inter_mask = torch.arange(num_classes, device=target.device).view(1, -1) != target.detach().view(-1, 1)
        inter_values = cos_theta.masked_select(inter_mask).view(-1, num_classes - 1)
        inter_losses = torch.log(1.0 + torch.sum(torch.exp(scale * inter_values), dim=-1))

        out_losses = intra_losses + inter_losses

        return out_losses
