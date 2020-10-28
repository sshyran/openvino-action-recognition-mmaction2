import logging

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint, _load_checkpoint
from mmcv.cnn import constant_init, kaiming_init

from ...core.ops import (Dropout, Conv3d, HSigmoid, HSwish, conv_kxkxk_bn, conv_1xkxk_bn,
                         conv_kx1x1_bn, conv_1x1x1_bn, gumbel_sigmoid)
from ..registry import BACKBONES
from ..losses import TotalVarianceLoss
from .mobilenetv3 import make_divisible, MobileNetV3


class SELayer_3D(nn.Module):
    def __init__(self, in_channels, reduction=4, reduce_temporal=False, norm='none'):
        super(SELayer_3D, self).__init__()

        self.reduce_temporal = reduce_temporal

        self.avg_pool = nn.AdaptiveAvgPool3d((1 if reduce_temporal else None, 1, 1))
        self.fc = nn.Sequential(
            Conv3d(in_channels, in_channels // reduction, kernel_size=1, padding=0, normalization=norm),
            nn.ReLU(inplace=True),
            Conv3d(in_channels // reduction, in_channels, kernel_size=1, padding=0, normalization=norm),
            HSigmoid()
        )

    def forward(self, x):
        if torch.onnx.is_in_onnx_export() and not self.reduce_temporal:
            glob_context = F.avg_pool3d(x, (1, int(x.shape[3]), int(x.shape[4])), stride=1, padding=0)
        else:
            glob_context = self.avg_pool(x)

        mask = self.fc(glob_context)
        return mask * x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1.0, 0.0)


class ResidualAttention(nn.Module):
    def __init__(self, in_channels, kernels=3, scale=1.0, add_temporal=True, gumbel=True,
                 norm='none', residual=True, reg_weight=1.0, adaptive_threshold=True,
                 threshold=0.5, neg_fraction=0.7, tv_loss=True, gt_regression=False, enable_loss=True):
        super(ResidualAttention, self).__init__()

        self.gumbel = gumbel
        self.residual = residual
        self.scale = scale
        assert self.scale > 0.0
        self.reg_weight = reg_weight
        assert self.reg_weight > 0.0
        self.adaptive_threshold = adaptive_threshold
        self.threshold = threshold
        assert 0.0 < self.threshold < 1.0
        self.neg_fraction = neg_fraction
        assert 0.0 < self.neg_fraction < 1.0
        self.enable_loss = enable_loss
        self.gt_regression = gt_regression

        if enable_loss and tv_loss:
            temp_kernel = 3 if add_temporal else 1
            self.tv_loss = TotalVarianceLoss(
                spatial_kernels=3, temporal_kernels=temp_kernel, num_channels=1,
                hard_values=True, limits=(0.0, 1.0), threshold=0.5
            )
        else:
            self.tv_loss = None

        self.spatial_logits = nn.Sequential(
            *conv_1xkxk_bn(in_channels, in_channels, kernels, 1, groups=in_channels, norm=norm),
            HSwish(),
            *conv_1x1x1_bn(in_channels, 1, norm=norm)
        )

        if add_temporal:
            self.temporal_logits = nn.Sequential(
                nn.Sequential(),
                *conv_kx1x1_bn(in_channels, in_channels, kernels, 1, groups=in_channels, norm=norm),
                HSwish(),
                *conv_1x1x1_bn(in_channels, 1, norm=norm)
            )
        else:
            self.temporal_logits = None

    def forward(self, x, return_extra_data=False):
        spatial_logits = self.spatial_logits(x)

        if self.temporal_logits is not None:
            pooled_x = F.avg_pool3d(x, (1, int(x.shape[3]), int(x.shape[4])), stride=1, padding=0)
            temporal_logits = self.temporal_logits(pooled_x)
            logits = spatial_logits + temporal_logits
        else:
            logits = spatial_logits

        if self.gumbel and self.training:
            soft_mask = gumbel_sigmoid(self.scale * logits)
        else:
            soft_mask = torch.sigmoid(self.scale * logits)

        if self.residual:
            out = soft_mask * x + x
        else:
            out = soft_mask * x

        if return_extra_data:
            return out, dict(attention_logits=logits, attention_scores=soft_mask)
        else:
            return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1.0, 0.0)

    def loss(self, attention_logits, attention_scores, attention_mask=None):
        if not self.enable_loss:
            return None

        spatial_size = attention_scores.size(3) * attention_scores.size(4)

        if attention_mask is not None and self.gt_regression:
            with torch.no_grad():
                trg_values = F.interpolate(attention_mask, size=attention_logits.size()[2:], mode='nearest')

            losses = F.binary_cross_entropy_with_logits(attention_logits, trg_values, reduction='none')

            pos_mask = trg_values > 0
            pos_losses = losses[pos_mask]
            neg_losses = losses[~pos_mask]

            num_trg_losses = 0
            out_loss = 0.0
            if pos_losses.numel() > 0:
                out_loss += 1.0 / float(pos_losses.numel()) * pos_losses.sum()
                num_trg_losses += 1
            if neg_losses.numel() > 0:
                out_loss += 1.0 / float(neg_losses.numel()) * neg_losses.sum()
                num_trg_losses += 1
            out_loss = out_loss / float(num_trg_losses)
        elif self.tv_loss is not None:
            out_loss = self.tv_loss(attention_scores)
        else:
            conf = attention_scores.view(-1, spatial_size)

            if self.adaptive_threshold:
                num_values = int(self.neg_fraction * spatial_size)
                threshold, _ = conf.kthvalue(num_values, dim=-1, keepdim=True)
            else:
                threshold = self.threshold

            losses = torch.where(conf < threshold, conf, 1.0 - conf)
            out_loss = losses.mean()

        return self.reg_weight * out_loss


class DynamicAttention(nn.Module):
    def __init__(self, in_channels, kernels=3, scale=1.0, norm='none', residual=True, **kwargs):
        super(DynamicAttention, self).__init__()

        self.scale = scale
        self.residual = residual

        self.key = nn.Sequential(
            *conv_1xkxk_bn(in_channels, in_channels, kernels, 1, groups=in_channels, norm=norm),
            HSwish(),
            *conv_1x1x1_bn(in_channels, in_channels, norm=norm),
            nn.AdaptiveMaxPool3d((1, 1, 1))
        )
        self.value = nn.Sequential(
            *conv_1xkxk_bn(in_channels, in_channels, kernels, 1, groups=in_channels, norm=norm),
            HSwish(),
            *conv_1x1x1_bn(in_channels, in_channels, norm=norm)
        )

    def forward(self, x):
        key = self.key(x)
        values = self.value(x)

        responses = (key * values).sum(dim=1, keepdim=True)
        soft_mask = torch.sigmoid(self.scale * responses)

        if self.residual:
            out = soft_mask * x + x
        else:
            out = soft_mask * x

        return out


class InvertedResidual_S3D(nn.Module):
    def __init__(self, in_planes, hidden_dim, out_planes, spatial_kernels, temporal_kernels,
                 spatial_stride, temporal_stride, use_se, use_hs,
                 temporal_avg_pool=False, dropout_cfg=None, dw_temporal=True,
                 norm='none', scale=None, center_weight=None):
        super(InvertedResidual_S3D, self).__init__()
        assert spatial_stride in [1, 2]

        self.identity = spatial_stride == 1 and temporal_stride == 1 and in_planes == out_planes
        self.scale = scale

        if in_planes == hidden_dim:
            conv_layers = [
                # dw
                *conv_1xkxk_bn(hidden_dim, hidden_dim, spatial_kernels, spatial_stride,
                               groups=hidden_dim, norm=norm),
                HSwish() if use_hs else nn.ReLU(inplace=True),

                # Squeeze-and-Excite
                SELayer_3D(hidden_dim, norm=norm) if use_se else nn.Sequential()
            ]
        else:
            # pw
            conv_layers = [
                *conv_1x1x1_bn(in_planes, hidden_dim, norm=norm),
                HSwish() if use_hs else nn.ReLU(inplace=True)
            ]

            # dw
            if dw_temporal:
                conv_layers.extend(conv_kxkxk_bn(hidden_dim, hidden_dim, spatial_kernels, temporal_kernels,
                                                 spatial_stride, temporal_stride, groups=hidden_dim, norm=norm))
            else:
                conv_layers.extend(conv_1xkxk_bn(hidden_dim, hidden_dim, spatial_kernels, spatial_stride,
                                                 groups=hidden_dim, norm=norm))

            # Squeeze-and-Excite
            conv_layers.extend([
                SELayer_3D(hidden_dim, norm=norm) if use_se else nn.Sequential(),
                HSwish() if use_hs else nn.ReLU(inplace=True)
            ])

        # pw-linear
        if dw_temporal:
            conv_layers.extend(conv_1x1x1_bn(hidden_dim, out_planes, norm=norm))
        else:
            if temporal_avg_pool and temporal_stride > 1:
                conv_layers.extend([*conv_kx1x1_bn(hidden_dim, out_planes, temporal_kernels, 1,
                                                   norm=norm, center_weight=center_weight),
                                    nn.AvgPool3d(kernel_size=(temporal_stride, 1, 1),
                                                 stride=(temporal_stride, 1, 1),
                                                 padding=((temporal_stride - 1) // 2, 0, 0))
                                    ])
            else:
                conv_layers.extend(conv_kx1x1_bn(hidden_dim, out_planes, temporal_kernels, temporal_stride,
                                                 norm=norm, center_weight=center_weight))

        self.conv = nn.Sequential(*conv_layers)

        if dropout_cfg is not None and self.identity:
            self.dropout = Dropout(**dropout_cfg)
        else:
            self.dropout = None

    def forward(self, x):
        y = self.conv(x)

        if self.dropout is not None:
            y = self.dropout(y, x)

        if self.identity and self.scale is not None and self.scale != 1.0:
            y *= self.scale

        return x + y if self.identity else y


@BACKBONES.register_module()
class MobileNetV3_S3D(nn.Module):
    def __init__(self,
                 mode,
                 num_input_layers=3,
                 pretrained=None,
                 pretrained2d=True,
                 width_mult=1.0,
                 pool1_stride_t=2,
                 temporal_kernels=3,
                 temporal_strides=1,
                 bn_eval=False,
                 bn_frozen=False,
                 use_temporal_avg_pool=False,
                 use_dw_temporal=0,
                 use_st_att=0,
                 attention_cfg=None,
                 input_bn=False,
                 out_conv=True,
                 out_attention=False,
                 out_ids=None,
                 dropout_cfg=None,
                 weight_norm='none',
                 center_conv_weight=None):
        super(MobileNetV3_S3D, self).__init__()

        # setting of inverted residual blocks
        self.mode = mode
        assert self.mode in ['large', 'small']
        self.cfg = MobileNetV3.arch_settings[self.mode]
        self.width_mult = width_mult
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.temporal_kernels = temporal_kernels \
            if not isinstance(temporal_kernels, int) else (temporal_kernels,) * len(self.cfg)
        self.temporal_strides = temporal_strides \
            if not isinstance(temporal_strides, int) else (temporal_strides,) * len(self.cfg)
        self.use_dw_temporal = use_dw_temporal \
            if not isinstance(use_dw_temporal, int) else (use_dw_temporal,) * len(self.cfg)
        self.use_st_att = use_st_att \
            if not isinstance(use_st_att, int) else (use_st_att,) * len(self.cfg)

        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen

        if input_bn:
            self.input_bn = nn.BatchNorm3d(num_input_layers)
        else:
            self.input_bn = None

        # building first layer
        input_channel = make_divisible(16 * width_mult, 8)

        first_stage_layers = [*conv_1xkxk_bn(num_input_layers, input_channel,
                                             k=3, spatial_stride=2, norm=weight_norm)]
        if pool1_stride_t > 1:
            first_stage_layers.append(nn.AvgPool3d(kernel_size=(pool1_stride_t, 1, 1),
                                                   stride=(pool1_stride_t, 1, 1),
                                                   padding=((pool1_stride_t - 1) // 2, 0, 0)))
        first_stage_layers.append(HSwish())
        layers = [nn.Sequential(*first_stage_layers)]
        num_layers_before = len(layers)

        attention_block = ResidualAttention
        # attention_block = DynamicAttention
        attention_cfg = dict() if attention_cfg is None else attention_cfg
        residual_block = InvertedResidual_S3D

        # building inverted residual blocks
        self.attentions = nn.ModuleDict()
        self.roi_aligns = nn.ModuleDict()
        skip_strides = False
        for layer_id, layer_params in enumerate(self.cfg):
            k, exp_size, c, use_se, use_hs, s = layer_params
            output_channel = make_divisible(c * width_mult, 8)
            spatial_stride = 1 if skip_strides else s
            temporal_stride = 1 if skip_strides else self.temporal_strides[layer_id]

            use_dw_temporal = self.use_dw_temporal[layer_id] > 0

            layers.append(residual_block(
                input_channel, exp_size, output_channel, k, self.temporal_kernels[layer_id],
                spatial_stride, temporal_stride, use_se, use_hs,
                use_temporal_avg_pool, dropout_cfg,
                use_dw_temporal, weight_norm, center_weight=center_conv_weight)
            )
            input_channel = output_channel

            if self.use_st_att[layer_id] > 0:
                att_name = 'st_att_{}'.format(layer_id + num_layers_before)
                self.attentions[att_name] = attention_block(input_channel, **attention_cfg)

        self.features = nn.ModuleList(layers)

        # building last several layers
        if out_conv:
            out_channels = make_divisible(exp_size * width_mult, 8)
            self.conv = nn.Sequential(
                nn.Sequential(*conv_1x1x1_bn(input_channel, out_channels, norm=weight_norm),
                              HSwish()),
                SELayer_3D(out_channels, norm=weight_norm) if mode == 'small' else nn.Sequential()
            )

            if out_attention:
                self.attentions['out_st_att'] = attention_block(out_channels, **attention_cfg)
        else:
            self.conv = None

        self.out_ids = out_ids
        if self.out_ids is None:
            self.out_ids = [len(self.features) - 1]

    def forward(self, x, return_extra_data=False, enable_extra_modules=True):
        if self.input_bn is not None:
            x = self.input_bn(x)

        y = x
        outs = []
        feature_data, att_data = dict(), dict()
        for module_idx, module in enumerate(self.features):
            if return_extra_data and hasattr(module, 'loss'):
                y, feature_extra_data = module(y, return_extra_data=True)
                feature_data[module_idx] = feature_extra_data
            else:
                y = module(y)

            attention_module_name = 'st_att_{}'.format(module_idx)
            if attention_module_name in self.attentions:
                attention_module = self.attentions[attention_module_name]

                if return_extra_data and hasattr(attention_module, 'loss'):
                    y, att_extra_data = attention_module(y, return_extra_data=True)
                    att_data[attention_module_name] = att_extra_data
                elif enable_extra_modules:
                    y = attention_module(y)
                else:
                    y += y  # simulate residual block

            if module_idx in self.out_ids:
                outs.append(y)

        if self.conv is not None:
            assert len(outs) == 1

            y = self.conv(outs[0])

            if 'out_st_att' in self.attentions:
                out_attention = self.attentions['out_st_att']
                if return_extra_data and hasattr(out_attention, 'loss'):
                    y, out_att_extra_data = out_attention(y, return_extra_data=True)
                    att_data['out_st_att'] = out_att_extra_data
                elif enable_extra_modules:
                    y = out_attention(y)
                else:
                    y += y  # simulate residual block

            outs = [y]

        if return_extra_data:
            return outs, dict(feature_data=feature_data, att_data=att_data)
        else:
            return outs

    def loss(self, feature_data, att_data, **kwargs):
        losses = dict()

        reg_losses = []
        for module_idx, module_data in feature_data.items():
            module = self.features[module_idx]

            loss_value = module.loss(**module_data, **kwargs)
            if loss_value is not None:
                reg_losses.append(loss_value)
        if len(reg_losses) > 0:
            losses['loss/freg'] = torch.mean(torch.stack(reg_losses))

        att_losses = []
        for module_name, module_data in att_data.items():
            module = self.attentions[module_name]

            loss_value = module.loss(**module_data, **kwargs)
            if loss_value is not None:
                att_losses.append(loss_value)
        if len(att_losses) > 0:
            losses['loss/att'] = torch.mean(torch.stack(att_losses))

        return losses

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.data.fill_(1)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.data.normal_(0, 0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.zero_()

    def _inflate_weights(self, logger):
        def _is_compatible(shape_a, shape_b):
            return shape_a[0] == shape_b[0]\
                   and shape_a[1] == shape_b[1]\
                   and shape_a[3] == shape_b[3]\
                   and shape_a[4] == shape_b[4]

        state_dict_2d = _load_checkpoint(self.pretrained)
        if 'state_dict' in state_dict_2d:
            state_dict_2d = state_dict_2d['state_dict']

        for name, module in self.named_modules():
            if isinstance(module, nn.Conv3d) and name in state_dict_2d:
                old_weight = state_dict_2d[name + '.weight'].data
                assert len(old_weight.shape) in [2, 4]

                if len(old_weight.shape) == 2:
                    old_weight = old_weight.unsqueeze(2).unsqueeze(3)
                old_weight = old_weight.unsqueeze(2)

                if not _is_compatible(old_weight.shape, module.weight.data.shape):
                    logging.warning('{}. Conv3D not loaded from weights'.format(name))
                    continue

                new_weight = old_weight.expand_as(module.weight) / module.weight.data.shape[2]
                module.weight.data.copy_(new_weight)
                logging.info("{}.weight loaded from weights file into {}".format(name, new_weight.shape))

                if hasattr(module, 'bias') and module.bias is not None:
                    new_bias = state_dict_2d[name + '.bias'].data
                    module.bias.data.copy_(new_bias)
                    logging.info("{}.bias loaded from weights file into {}".format(name, new_bias.shape))
            elif isinstance(module, nn.BatchNorm3d) and name in state_dict_2d:
                for attr_name in ['weight', 'bias', 'running_mean', 'running_var']:
                    old_attr = state_dict_2d[name + '.' + attr_name].data

                    logging.info("{}.{} loaded from weights file into {}"
                                 .format(name, attr_name, old_attr.shape))
                    new_attr = getattr(module, attr_name)
                    new_attr.data.copy_(old_attr)
            else:
                if isinstance(module, nn.Conv3d):
                    logging.warning('{}. Conv3D not loaded from weights'.format(name))
                elif isinstance(module, nn.BatchNorm3d):
                    logging.warning('{}. BN3D not loaded from weights'.format(name))

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            if self.pretrained2d:
                self._inflate_weights(logger)
            else:
                load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            self._init_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def train(self, train_mode=True):
        super(MobileNetV3_S3D, self).train(train_mode)

        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()

                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False

        return self
