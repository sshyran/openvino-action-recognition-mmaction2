import torch.nn as nn
import torch.nn.functional as F


class AdaptivePool3D(nn.Module):
    methods = 'avg', 'max', 'avg+max'

    def __init__(self, output_size, method='avg'):
        super(AdaptivePool3D, self).__init__()

        self.output_size = output_size
        self.method = method
        assert self.method in self.methods

    def forward(self, x):
        if self.method == 'avg':
            return F.adaptive_avg_pool3d(x, self.output_size)
        elif self.method == 'avg':
            return F.adaptive_max_pool3d(x, self.output_size)
        else:
            avg_pooled = F.adaptive_avg_pool3d(x, self.output_size)
            max_pooled = F.adaptive_max_pool3d(x, self.output_size)
            return avg_pooled + max_pooled
