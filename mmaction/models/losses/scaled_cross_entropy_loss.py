import torch.nn.functional as F

from ..registry import LOSSES
from .metric_learning_base import BaseMetricLearningLoss


@LOSSES.register_module()
class ScaledCrossEntropyLoss(BaseMetricLearningLoss):
    def __init__(self, **kwargs):
        super(ScaledCrossEntropyLoss, self).__init__(**kwargs)

    def _calculate(self, cos_theta, target, scale):
        out_losses = F.cross_entropy(scale * cos_theta, target.detach().view(-1), reduction='none')

        return out_losses
