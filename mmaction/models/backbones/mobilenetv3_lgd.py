import torch.nn as nn

from ...core.ops import HSwish, conv_1x1x1_bn
from ..registry import BACKBONES
from .mobilenetv3_s3d import MobileNetV3_S3D


@BACKBONES.register_module()
class MobileNetV3_LGD(MobileNetV3_S3D):
    def __init__(self, mix_paths, **kwargs):
        super(MobileNetV3_LGD, self).__init__(**kwargs)

        assert len(mix_paths) == len(self.cfg)
        mix_idx = [idx for idx in range(len(self.cfg)) if mix_paths[idx] > 0]
        assert len(mix_idx) > 0
        assert mix_idx[0] > 0
        assert mix_idx[-1] < len(self.cfg)

        self.glob_to_local_idx = mix_idx
        self.local_to_glob_idx = [mix_idx[0] - 1] + mix_idx[:-1]
        self.glob_idx = mix_idx[:-1]

        self.glob_channels_num = [self.channels_num[idx] for idx in self.local_to_glob_idx]

        self.upsample_modules = nn.ModuleDict({
            f'upsample_{idx}': self._build_upsample_module(
                glob_channels,
                self.channels_num[idx],
                norm=self.weight_norm
            )
            for idx, glob_channels in zip(self.glob_to_local_idx, self.glob_channels_num)
        })
        self.pooling_modules = nn.ModuleDict({
            f'pooling_{idx}': self._build_pooling_module(
                self.channels_num[idx],
                self.channels_num[idx],
                norm=self.weight_norm
            )
            for idx in self.local_to_glob_idx
        })
        self.glob_modules = nn.ModuleDict({
            f'glob_{idx}': self._build_glob_module(
                glob_channels,
                self.channels_num[idx],
                norm=self.weight_norm
            )
            for idx, glob_channels in zip(self.glob_idx, self.glob_channels_num[:-1])
        })

    @staticmethod
    def _build_upsample_module(in_planes, out_planes, factor=3, norm='none'):
        hidden_dim = int(factor * in_planes)
        layers = [
            *conv_1x1x1_bn(in_planes, hidden_dim, norm=norm),
            HSwish(),
            *conv_1x1x1_bn(hidden_dim, out_planes, norm=norm),
        ]
        return nn.Sequential(*layers)

    @staticmethod
    def _build_pooling_module(in_planes, out_planes, factor=3, norm='none'):
        hidden_dim = int(factor * in_planes)
        layers = [
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            *conv_1x1x1_bn(in_planes, hidden_dim, norm=norm),
            HSwish(),
            *conv_1x1x1_bn(hidden_dim, out_planes, norm=norm),
        ]
        return nn.Sequential(*layers)

    @staticmethod
    def _build_glob_module(in_planes, out_planes, factor=3, norm='none'):
        hidden_dim = int(factor * in_planes)
        layers = [
            *conv_1x1x1_bn(in_planes, hidden_dim, norm=norm),
            HSwish(),
            *conv_1x1x1_bn(hidden_dim, out_planes, norm=norm),
        ]
        return nn.Sequential(*layers)

    def forward(self, x, return_extra_data=False, enable_extra_modules=True):
        y = self._norm_input(x)

        local_y = y
        glob_y = None

        local_outs = []
        feature_data, att_data = dict(), dict()
        for module_idx in range(len(self.features)):
            local_y = self._infer_module(
                local_y, module_idx, return_extra_data, enable_extra_modules, feature_data, att_data
            )

            if module_idx in self.glob_to_local_idx:
                assert glob_y is not None

                upsample_module = self.upsample_modules[f'upsample_{module_idx}']
                local_y = upsample_module(glob_y) + local_y

            if module_idx in self.local_to_glob_idx:
                pooling_module = self.pooling_modules[f'pooling_{module_idx}']
                pooled_local_y = pooling_module(local_y)

                if glob_y is not None:
                    glob_module = self.glob_modules[f'glob_{module_idx}']
                    glob_y = glob_module(glob_y) + pooled_local_y
                else:
                    glob_y = pooled_local_y

            if module_idx in self.out_ids:
                local_outs.append(local_y)

        local_outs = self._out_conv(local_outs, return_extra_data, enable_extra_modules, att_data)

        if return_extra_data:
            return local_outs, dict(feature_data=feature_data, att_data=att_data)
        else:
            return local_outs
