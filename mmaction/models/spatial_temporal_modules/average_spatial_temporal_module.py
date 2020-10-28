import torch.nn as nn
from ..registry import SPATIAL_TEMPORAL_MODULES


@SPATIAL_TEMPORAL_MODULES.register_module()
class AverageSpatialTemporalModule(nn.Module):
    def __init__(self, spatial_size=7, temporal_size=1):
        super(AverageSpatialTemporalModule, self).__init__()

        self.spatial_size = spatial_size if not isinstance(spatial_size, int) else (spatial_size, spatial_size)
        self.temporal_size = temporal_size
        self.pool_size = (self.temporal_size, ) + self.spatial_size

        self.op = nn.AvgPool3d(self.pool_size, stride=1, padding=0)

    def init_weights(self):
        pass

    def forward(self, x, return_extra_data=False):
        if return_extra_data:
            return self.op(x), dict()
        else:
            return self.op(x)
