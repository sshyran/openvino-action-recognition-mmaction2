import argparse
import json

import numpy as np
import torch
from mmcv import Config

from mmaction.models import build_recognizer
from mmaction.utils import ExtendedDictAction
from mmaction.core.ops import Conv2d, Conv3d

try:
    from ptflops import get_model_complexity_info
except ImportError:
    raise ImportError('Please install ptflops: `pip install ptflops`')


def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))
    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if conv_module.bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops
    conv_module.__flops__ += int(overall_flops)


def main():
    parser = argparse.ArgumentParser(description='Measures FLOPs and number of Params')
    parser.add_argument('config', help='model config file path')
    parser.add_argument('--shape', type=int, nargs='+', default=[340, 256], help='model input size')
    parser.add_argument('--out')
    parser.add_argument('--per_layer_stat', action='store_true')
    parser.add_argument('--update_config', nargs='+', action=ExtendedDictAction, help='arguments in dict')
    args = parser.parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2 or len(args.shape) == 3:
        input_shape = (
            1,
            3,
        ) + tuple(args.shape)
    elif len(args.shape) == 4:
        # n, c, h, w = args.shape
        input_shape = tuple(args.shape)
    elif len(args.shape) == 5:
        # n, c, t, h, w = args.shape
        input_shape = tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    if args.update_config is not None:
        cfg.merge_from_dict(args.update_config)
    model = build_recognizer(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_inference'):
        model.forward = model.forward_inference
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    with torch.no_grad():
        macs, params = get_model_complexity_info(
            model,
            input_shape[1:],
            as_strings=True,
            print_per_layer_stat=args.per_layer_stat,
            custom_modules_hooks={
                Conv2d: conv_flops_counter_hook,
                Conv3d: conv_flops_counter_hook,
            }
        )

    split_line = '=' * 30
    print(f'{split_line}\n'
          f'Input shape: {input_shape}\n'
          f'Macs: {macs}\n'
          f'Params: {params}\n'
          f'{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')

    if args.out:
        out = list()
        out.append({'key': 'size', 'display_name': 'Size', 'value': float(params.split(' ')[0]), 'unit': 'Mp'})
        out.append({'key': 'complexity', 'display_name': 'Complexity', 'value': 2 * float(macs.split(' ')[0]),
                    'unit': 'GMac'})
        with open(args.out, 'w') as write_file:
            json.dump(out, write_file, indent=4)


if __name__ == '__main__':
    main()
