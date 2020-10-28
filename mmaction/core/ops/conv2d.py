import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, normalization='none', eps=1e-12):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups, bias)

        self.eps = eps
        self.normalization = normalization
        assert self.normalization in ['none', 'ws', 'l2']

    def forward(self, x):
        weight = self.weight

        if self.normalization == 'ws':
            weight_mean = weight.mean(dim=(1, 2, 3), keepdim=True)
            weight_std = weight.std(dim=(1, 2, 3), keepdim=True)
            norm_weight = (weight - weight_mean) / weight_std.clamp_min(self.eps)
        elif self.normalization == 'l2':
            norm_weight = F.normalize(weight.view(weight.size(0), -1), dim=1).view_as(weight)
        else:
            norm_weight = weight

        return F.conv2d(x, norm_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
