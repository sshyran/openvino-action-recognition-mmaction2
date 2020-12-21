import os.path as osp
from argparse import ArgumentParser

import mmcv
import numpy as np
import torch.nn as nn

from mmaction.models import build_recognizer
from mmaction.core import load_checkpoint
from mmaction.utils import get_root_logger, ExtendedDictAction

MODEL_SOURCES = 'modelzoo://', 'torchvision://', 'open-mmlab://', 'http://', 'https://'


def is_valid(model_path):
    if model_path is None:
        return False

    return osp.exists(model_path) or model_path.startswith(MODEL_SOURCES)


def update_config(cfg, args):
    if is_valid(args.load_from):
        cfg.load_from = args.load_from

    if is_valid(args.load2d_from):
        cfg.model.backbone.pretrained = args.load2d_from
        cfg.model.backbone.pretrained2d = True

    cfg.data.videos_per_gpu = 1

    return cfg


def collect_conv_layers(model, eps=1e-5):
    conv_layers = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            weight = m.weight.detach().cpu().numpy()
            bias = m.bias.detach().cpu().numpy() if m.bias is not None else 0.0
            shape = weight.shape

            assert len(shape) == 5
            if shape[2] == shape[3] == shape[4] == 1:
                kernel_type = '1x1x1'
            elif shape[3] == shape[4] == 1:
                kernel_type = 'kx1x1'
            elif shape[2] == 1:
                kernel_type = '1xkxk'
            elif shape[1] == 1:
                kernel_type = 'dw'
            else:
                kernel_type = 'kxkxk'

            conv_layers.append(dict(
                name=name,
                type=kernel_type,
                weight=weight,
                bias=bias,
                updated=False,
            ))
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            assert len(conv_layers) > 0

            last_conv = conv_layers[-1]
            assert not last_conv['updated']

            alpha = m.weight.detach().cpu().numpy()
            beta = m.bias.detach().cpu().numpy()
            running_mean = m.running_mean.detach().cpu().numpy()
            running_var = m.running_var.detach().cpu().numpy()

            scales = alpha / np.sqrt(running_var + eps)
            scale_shape = [-1] + [1] * len(last_conv['weight'].shape)
            scales = scales.reshape(scale_shape)

            last_conv['weight'] = scales * last_conv['weight']
            last_conv['bias'] = scales * (last_conv['bias'] - running_mean) + beta
            last_conv['updated'] = True

    return conv_layers


def show_stat(conv_layers, max_scale=5.0, max_similarity=0.5, sim_percentile=95):
    invalid_weight_scales = []
    invalid_bias_scales = []
    invalid_sim = []
    for conv in conv_layers:
        name = conv['name']
        weights = conv['weight']
        bias = conv['bias']
        kernel_type = conv['type']
        if conv['updated']:
            kernel_type += ', fused'

        num_filters = weights.shape[0]
        filters = weights.reshape([num_filters, -1])

        norms = np.sqrt(np.sum(np.square(filters), axis=-1))
        min_norm, max_norm = np.min(norms), np.max(norms)
        median_norm = np.median(norms)
        scale = max_norm / min_norm

        if num_filters <= filters.shape[1]:
            norm_filters = filters / norms.reshape([-1, 1])
            similarities = np.matmul(norm_filters, np.transpose(norm_filters))

            similarities = np.abs(similarities[np.triu_indices(similarities.shape[0], k=1)])

            num_invalid = np.sum(similarities > max_similarity)
            num_total = len(similarities)
            if num_invalid > 0:
                sim = np.percentile(similarities, sim_percentile)
                invalid_sim.append((name, kernel_type, sim, num_invalid, num_total, num_filters))

        scales = max_norm / norms
        num_invalid = np.sum(scales > max_scale)
        if num_invalid > 0 or median_norm < 0.1:
            invalid_weight_scales.append((name, kernel_type, min_norm, median_norm, scale, num_invalid, num_filters))

        bias_scores = np.abs(bias)
        bias_score = np.percentile(bias_scores, 95)
        if bias_score > 1.0:
            invalid_bias_scales.append((name, kernel_type, bias_score))

    if len(invalid_weight_scales) > 0:
        print('\nFound {} layers with invalid weight norm fraction (max/cur > {}):'
              .format(len(invalid_weight_scales), max_scale))
        for name, kernel_type, min_norm, median_norm, scale, num_invalid, num_filters in invalid_weight_scales:
            print('   - {} ({}): {:.3f} (min={:.3f} median={:.3f} invalid: {} / {})'
                  .format(name, kernel_type, scale, min_norm, median_norm, num_invalid, num_filters))
    else:
        print('\nThere are no layers with invalid weight norm.')

    if len(invalid_bias_scales) > 0:
        print('\nFound {} layers with invalid bias max value (max> {}):'
              .format(len(invalid_bias_scales), 1.0))
        for name, kernel_type, scale in invalid_bias_scales:
            print('   - {} ({}): {:.3f}'
                  .format(name, kernel_type, scale))
    else:
        print('\nThere are no layers with invalid bias.')

    if len(invalid_sim) > 0:
        print('\nFound {} layers with invalid similarity (value > {}):'
              .format(len(invalid_sim), max_similarity))
        for name, kernel_type, sim, num_invalid, num_total, num_filters in invalid_sim:
            print('   - {} ({}): {:.3f} (invalid: {} / {} size={})'
                  .format(name, kernel_type, sim, num_invalid, num_total, num_filters))
    else:
        print('\nThere are no layers with invalid similarity.')


def main():
    parser = ArgumentParser()
    parser.add_argument('config',
                        help='Config file path')
    parser.add_argument('--load_from',
                        help='the checkpoint file to init weights from')
    parser.add_argument('--load2d_from',
                        help='the checkpoint file to init 2D weights from')
    parser.add_argument('--update_config', nargs='+', action=ExtendedDictAction,
                        help='arguments in dict')
    args = parser.parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.update_config is not None:
        cfg.merge_from_dict(args.update_config)
    cfg = update_config(cfg, args)

    net = build_recognizer(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    net.eval()

    if cfg.load_from:
        logger = get_root_logger(log_level=cfg.log_level)
        load_checkpoint(net,
                        cfg.load_from,
                        strict=False,
                        logger=logger,
                        show_converted=True,
                        force_matching=True)

    conv_layers = collect_conv_layers(net)
    show_stat(conv_layers)


if __name__ == '__main__':
    main()
