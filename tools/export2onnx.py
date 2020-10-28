import argparse
from os import makedirs
from os.path import exists, dirname

import torch
import onnx
import mmcv

from mmaction.models import build_recognizer
from mmaction.core import load_checkpoint


def convert_to_onnx(net, input_size, output_file_path, check):
    dummy_input = torch.randn((1, *input_size))
    input_names = ['input']
    output_names = ['output']

    dynamic_axes = {'input': {0: 'batch_size', 1: 'channels', 2: 'length', 3: 'height', 4: 'width'},
                    'output': {0: 'batch_size', 1: 'scores'}}

    torch.onnx.export(net, dummy_input, output_file_path, verbose=True,
                      input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX)

    net_from_onnx = onnx.load(output_file_path)
    if check:
        try:
            onnx.checker.check_model(net_from_onnx)
            print('ONNX check passed.')
        except onnx.onnx_cpp2py_export.checker.ValidationError as ex:
            print('ONNX check failed: {}.'.format(ex))

    return onnx.helper.printable_graph(net_from_onnx.graph)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('output_name', help='Output file')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.videos_per_gpu = 1

    net = build_recognizer(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    net.eval()
    load_checkpoint(net, args.checkpoint, force_matching=True)

    input_time_size = cfg.input_clip_length
    input_image_size = (tuple(cfg.input_img_size)
                        if isinstance(cfg.input_img_size, (list, tuple))
                        else (cfg.input_img_size, cfg.input_img_size))
    input_size = (3, input_time_size) + input_image_size

    output_path = args.output_name
    if not output_path.endswith('.onnx'):
        output_path = '{}.onnx'.format(output_path)

    base_output_dir = dirname(output_path)
    if not exists(base_output_dir):
        makedirs(base_output_dir)

    if hasattr(net, 'forward_inference'):
        net.forward = net.forward_inference

    convert_to_onnx(net, input_size, args.output_name, check=args.check)


if __name__ == '__main__':
    main()
