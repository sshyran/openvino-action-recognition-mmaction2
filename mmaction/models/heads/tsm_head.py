import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from .base import BaseHead


@HEADS.register_module()
class TSMHead(BaseHead):
    """Class head for TSM.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        is_shift (bool): Indicating whether the feature is shifted.
            Default: True.
        temporal_pool (bool): Indicating whether feature is temporal pooled.
            Default: False.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 spatial_type='avg',
                 consensus=dict(type='AvgConsensus', dim=1),
                 init_std=0.001,
                 is_shift=True,
                 temporal_pool=False,
                 **kwargs):
        super().__init__(consensus=consensus, **kwargs)

        self.init_std = init_std
        self.is_shift = is_shift
        self.temporal_pool = temporal_pool

        self.avg_pool = None
        if spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x, num_segments, **kwargs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int): Number of segments into which a video
                is divided.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N * num_segs, in_channels, 7, 7]
        x = self.avg_pool(x)
        # [N * num_segs, in_channels, 1, 1]
        x = torch.flatten(x, 1)
        # [N * num_segs, in_channels]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N * num_segs, num_classes]
        cls_score = self.fc_cls(x)

        if self.is_shift and self.temporal_pool:
            # [2 * N, num_segs // 2, num_classes]
            cls_score = cls_score.view((-1, num_segments // 2) +
                                       cls_score.size()[1:])
        else:
            # [N, num_segs, num_classes]
            cls_score = cls_score.view((-1, num_segments) +
                                       cls_score.size()[1:])
        # [N, 1, num_classes]
        cls_score = self.consensus(cls_score)
        # [N, num_classes]
        return cls_score.squeeze(1)
