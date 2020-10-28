import torch.nn as nn
import torch.nn.functional as F


class Conv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 center_weight=None, normalization='none', eps=1e-12):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups, bias)

        self.eps = eps
        self.center_weight = center_weight
        self.normalization = normalization
        assert self.normalization in ['none', 'ws', 'l2']

        self.use_center_conv = self.center_weight is not None and self.center_weight > 0.0
        if self.use_center_conv:
            assert kernel_size[0] % 2 == 1

    def forward(self, x):
        weight = self.weight

        if self.normalization == 'ws':
            weight_mean = weight.mean(dim=(1, 2, 3, 4), keepdim=True)
            weight_std = weight.std(dim=(1, 2, 3, 4), keepdim=True)
            norm_weight = (weight - weight_mean) / weight_std.clamp_min(self.eps)
        elif self.normalization == 'l2':
            norm_weight = F.normalize(weight.view(weight.size(0), -1), dim=1).view_as(weight)
        else:
            norm_weight = weight

        y = F.conv3d(x, norm_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        if self.use_center_conv:
            temp_anchor = norm_weight.shape[2] // 2
            center_weight = norm_weight[:, :, :temp_anchor].sum(dim=(2, 3, 4), keepdim=True) + \
                            norm_weight[:, :, (temp_anchor + 1):].sum(dim=(2, 3, 4), keepdim=True)

            center_y = F.conv3d(x, center_weight, groups=self.groups)
            y = y - self.center_weight * center_y

        return y
