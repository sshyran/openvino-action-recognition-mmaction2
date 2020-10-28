from os import makedirs
from os.path import exists, join
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy

import torch
import numpy as np
from mmcv import Config, ProgressBar
from mmcv.parallel import scatter, collate, MMDataParallel

from mmaction.core import load_checkpoint
from mmaction.core.utils import propagate_root_dir
from mmaction.datasets import build_dataset
from mmaction.datasets.pipelines import Compose
from mmaction.models import build_model


def update_config(cfg, args, trg_name):
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if cfg.test_cfg is None:
        cfg.test_cfg = dict(average_clips=args.average_clips)

    cfg.data.train.source = trg_name,
    cfg.data.val.source = trg_name,
    cfg.data.test.source = trg_name,

    cfg.data.train.pipeline = cfg.val_pipeline
    cfg.data.val.pipeline = cfg.val_pipeline
    cfg.data.test.pipeline = cfg.val_pipeline

    cfg.data.train.ann_file = 'train.txt'
    cfg.data.val.ann_file = 'val.txt'
    cfg.data.test.ann_file = 'test.txt'

    return cfg


def prepare_tasks(dataset, window_size):
    out_tasks = defaultdict(list)
    for idx in range(len(dataset)):
        ann = dataset.video_infos[idx]

        rel_path = ann['rel_frame_dir']
        video_start = ann['video_start']
        video_end = ann['video_end']

        half_window_size = window_size // 2
        for start in range(video_start - half_window_size, video_end - half_window_size):
            indices = np.array([start + i for i in range(window_size) if video_start <= start + i < video_end])

            num_before = len([True for i in range(window_size) if start + i < video_start])
            num_after = len([True for i in range(window_size) if start + i >= video_end])
            indices = np.concatenate((np.full(num_before, -2, dtype=np.int32),
                                      indices,
                                      np.full(num_after, -2, dtype=np.int32)))

            out_tasks[idx].append((indices, rel_path, start))

    return out_tasks


def process_tasks(tasks, dataset, model, out_dir, batch_size, input_clip_length, pipeline):
    flat_tasks = [(idx, v[0], v[1], v[2]) for idx, sub_tasks in tasks.items() for v in sub_tasks]

    progress_bar = ProgressBar(len(flat_tasks))

    batch = []
    for task in flat_tasks:
        batch.append(task)
        if len(batch) == batch_size:
            process_batch(batch, dataset, model, out_dir, input_clip_length, pipeline)

            for _ in range(batch_size):
                progress_bar.update()
            batch = []

    if len(batch) > 0:
        process_batch(batch, dataset, model, out_dir, input_clip_length, pipeline)

        for _ in range(len(batch)):
            progress_bar.update()


def process_batch(tasks, dataset, model, out_dir, input_clip_length, pipeline):
    data = []
    for idx, indices, _, _ in tasks:
        record = deepcopy(dataset.video_infos[idx])
        record['modality'] = dataset.modality
        record['frame_inds'] = indices + dataset.start_index
        record['num_clips'] = 1
        record['clip_len'] = input_clip_length

        record_data = pipeline(record)
        data.append(record_data)

    data_gpu = scatter(collate(data, samples_per_gpu=len(tasks)),
                       [torch.cuda.current_device()])[0]

    with torch.no_grad():
        net_output = model(return_loss=False, **data_gpu)
        if isinstance(net_output, (list, tuple)):
            assert len(net_output) == 1
            net_output = net_output[0]

    for i, task in enumerate(tasks):
        rel_path = task[2]
        out_path = join(out_dir, rel_path + '.txt')

        with open(out_path, 'a') as output_stream:
            line = ';'.join(['{:.8f}'.format(v) for v in net_output[i]]) + '\n'

            start_pos = task[3]
            output_stream.write('{};'.format(start_pos) + line)


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Test config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint file')
    parser.add_argument('--data_dir', type=str, required=True, help='The dir with dataset')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--gpus', default=1, type=int, help='GPU number used for annotating')
    parser.add_argument('--proc_per_gpu', default=2, type=int, help='Number of processes per GPU')
    parser.add_argument('--mode', choices=['train', 'val', 'test'], default='train')
    args = parser.parse_args()

    assert exists(args.config)
    assert exists(args.checkpoint)
    assert exists(args.data_dir)

    cfg = Config.fromfile(args.config)
    cfg = update_config(cfg, args, trg_name=args.dataset)
    cfg = propagate_root_dir(cfg, args.data_dir)

    dataset = build_dataset(cfg.data, args.mode, dict(test_mode=True))
    data_pipeline = Compose(dataset.pipeline.transforms[1:])
    print('{} dataset:\n'.format(args.mode) + str(dataset))

    tasks = prepare_tasks(dataset, cfg.input_clip_length)
    print('Prepared tasks: {}'.format(sum([len(v) for v in tasks.values()])))

    if not exists(args.out_dir):
        makedirs(args.out_dir)

    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, strict=False)

    batch_size = 4 * cfg.data.videos_per_gpu
    if args.gpus == 1:
        model = MMDataParallel(model, device_ids=[0])
        model.eval()

        process_tasks(tasks, dataset, model, args.out_dir, batch_size, cfg.input_clip_length, data_pipeline)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
