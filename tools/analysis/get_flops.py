import argparse
import json

import torch
from mmcv import Config

from mmaction.models import build_recognizer
from mmaction.utils import ExtendedDictAction

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def main():
    parser = argparse.ArgumentParser(description='Measures FLOPs and number of Params')
    parser.add_argument('config', help='model config file path')
    parser.add_argument('--shape', type=int, nargs='+', default=[340, 256], help='model input size')
    parser.add_argument('--out')
    parser.add_argument('--update_config', nargs='+', action=ExtendedDictAction, help='arguments in dict')
    args = parser.parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
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
        flops, params = get_model_complexity_info(
            model,
            input_shape[1:],
            as_strings=True,
            print_per_layer_stat=True
        )

    split_line = '=' * 30
    print(f'{split_line}\n'
          f'Input shape: {input_shape}\n'
          f'Flops: {flops}\n'
          f'Params: {params}\n'
          f'{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')

    if args.out:
        out = list()
        out.append({'key': 'size', 'display_name': 'Size', 'value': float(params.split(' ')[0]), 'unit': 'Mp'})
        out.append({'key': 'complexity', 'display_name': 'Complexity', 'value': 2 * float(flops.split(' ')[0]),
                    'unit': 'GFLOPs'})
        with open(args.out, 'w') as write_file:
            json.dump(out, write_file, indent=4)


if __name__ == '__main__':
    main()
