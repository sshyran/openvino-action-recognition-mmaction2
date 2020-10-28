import torch
import torch.nn.functional as F

from ..registry import LOSSES
from .metric_learning_base import BaseMetricLearningLoss


@LOSSES.register_module()
class ArcLoss(BaseMetricLearningLoss):
    """Computes the Arc loss: https://arxiv.org/pdf/1904.13148.pdf
    """

    def __init__(self, **kwargs):
        super(ArcLoss, self).__init__(**kwargs)

    def _calculate(self, cos_theta, target, scale):
        theta = torch.acos(cos_theta)
        out_losses = F.cross_entropy(scale * theta, target.detach().view(-1), reduction='none')

        return out_losses
