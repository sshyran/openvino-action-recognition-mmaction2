import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init

from ..registry import SPATIAL_TEMPORAL_MODULES


@SPATIAL_TEMPORAL_MODULES.register_module()
class NonLocalModule(nn.Module):
    def __init__(self, in_channels=1024, nonlocal_type="gaussian", dim=3,
                 embed=True, embed_dim=None, embed_factor=2, spatial_sub_sample=True, use_bn=True):
        super(NonLocalModule, self).__init__()

        assert nonlocal_type in ['gaussian', 'dot', 'concat']
        assert dim == 2 or dim == 3
        assert embed_factor >= 1

        self.nonlocal_type = nonlocal_type
        self.embed = embed
        self.embed_dim = embed_dim if embed_dim is not None else in_channels // embed_factor
        self.sub_sample = spatial_sub_sample
        self.use_bn = use_bn

        if self.embed:
            self.theta = self._conv_1x1(in_channels, self.embed_dim, dim)
            self.phi = self._conv_1x1(in_channels, self.embed_dim, dim)
            self.g = self._conv_1x1(in_channels, self.embed_dim, dim)

        if self.nonlocal_type == 'gaussian':
            self.softmax = nn.Softmax(dim=2)
        elif self.nonlocal_type == 'concat':
            self.concat_proj = nn.Sequential(self._conv_1x1(self.embed_dim * 2, 1, dim),
                                             nn.ReLU())

        if spatial_sub_sample:
            if dim == 2:
                self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
            elif dim == 3:
                self.max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2))

            self.g = nn.Sequential(self.max_pool, self.g)
            self.phi = nn.Sequential(self.max_pool, self.phi)

        w_modules = [self._conv_1x1(self.embed_dim, in_channels, dim)]

        if use_bn:
            if dim == 2:
                w_modules.append(nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.9, affine=True))
            elif dim == 3:
                w_modules.append(nn.BatchNorm3d(in_channels, eps=1e-05, momentum=0.9, affine=True))

        self.W = nn.Sequential(*w_modules)

    @staticmethod
    def _conv_1x1(in_channels, out_channels, dim):
        if dim == 2:
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        elif dim == 3:
            return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
               kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
               constant_init(m, 1.0, 0.0)

    def forward(self, x, return_extra_data=False):
        if self.embed:
            theta = self.theta(x)
            phi = self.phi(x)
            g = self.g(x)
        else:
            theta = x
            phi = x
            g = x

        if self.nonlocal_type in ['gaussian', 'dot']:
            # reshape [BxC'xTxHxW] to [BxC'x(T)HW]
            theta = theta.view(theta.shape[:2] + (-1,))
            phi = phi.view(phi.shape[:2] + (-1,))
            g = g.view(g.shape[:2] + (-1,))
            theta_phi = torch.matmul(theta.transpose(1, 2), phi)
            if self.nonlocal_type == 'gaussian':
                p = self.softmax(theta_phi)
            elif self.nonlocal_type == 'dot':
                N = theta_phi.size(-1)
                p = theta_phi / N
        elif self.non_local_type == 'concat':
            # reshape [BxC'xTxHxW] to [BxC'x(T)HWx1]
            theta = theta.view(theta.shape[:2] + (-1,1))
            # reshape [BxC'xTxHxW] to [BxC'x1x(T)HW]
            phi = phi.view(theta.shape[:2] + (1,-1))
            theta_x = theta.repeat(1, 1, 1, phi.size(3))
            phi_x = phi.repeat(1, 1, theta.size(2), 1)
            theta_phi = torch.cat([theta_x, phi_x], dim=1)
            theta_phi = self.concat_proj(theta_phi)
            theta_phi = theta_phi.squeeze()
            N = theta_phi.size(-1)
            p = theta_phi / N
        else:
            raise NotImplementedError

        # BxC'xddd , Bxdxddd => BxC'xd
        y = torch.matmul(g, p.transpose(1, 2))
        y = y.view(y.shape[:2] + x.shape[2:])
        z = self.W(y) + x

        if return_extra_data:
            return z, dict()
        else:
            return z
