import json
from os import listdir
from os.path import exists, join, isfile
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm
from mmcv import Config
from mmcv.parallel import scatter, collate, MMDataParallel

from mmaction.core import load_checkpoint
from mmaction.core.utils import propagate_root_dir
from mmaction.datasets import build_dataset
from mmaction.datasets.pipelines import Compose
from mmaction.models import build_model

NO_MOTION_LABEL = 12
NEGATIVE_LABEL = 13
IGNORE_LABELS = {NO_MOTION_LABEL, NEGATIVE_LABEL}
STATIC_LABELS = {0, 1, 2, 3, 4, 5, 6, 7}
DYNAMIC_LABELS = {8, 9, 10, 11}


class RawFramesSegmentedRecord:
    def __init__(self, row):
        self._data = row
        assert len(self._data) == 7

    def __repr__(self):
        return ' '.join(self._data)

    @property
    def raw(self):
        return deepcopy(self._data)

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])

    @label.setter
    def label(self, value):
        self._data[1] = str(value)

    @property
    def clip_start(self):
        return int(self._data[2])

    @clip_start.setter
    def clip_start(self, value):
        self._data[2] = str(value)

    @property
    def clip_end(self):
        return int(self._data[3])

    @clip_end.setter
    def clip_end(self, value):
        self._data[3] = str(value)

    @property
    def video_start(self):
        return int(self._data[4])

    @video_start.setter
    def video_start(self, value):
        self._data[4] = str(value)

    @property
    def video_end(self):
        return int(self._data[5])

    @video_end.setter
    def video_end(self, value):
        self._data[5] = str(value)


def load_annotation(ann_full_path):
    return [RawFramesSegmentedRecord(x.strip().split(' ')) for x in open(ann_full_path)]


def parse_predictions_file(file_path):
    predictions = dict()
    with open(file_path) as input_stream:
        for line in input_stream:
            line_parts = line.strip().split(';')
            if len(line_parts) != 15:
                continue

            start_pos = int(line_parts[0])
            scores = [float(v) for v in line_parts[1:]]

            predictions[start_pos] = scores

    starts = list(sorted(predictions.keys()))
    scores = np.array([predictions[s] for s in starts], dtype=np.float32)
    assert len(starts) >= 3

    return starts, scores


def parse_movements_file(file_path):
    movements = dict()
    with open(file_path) as input_stream:
        for line in input_stream:
            line_parts = line.strip().split(';')
            if len(line_parts) != 2:
                continue

            frame_id = int(line_parts[0])
            movement_detected = bool(int(line_parts[1]))

            movements[frame_id] = movement_detected

    frame_ids = list(sorted(movements.keys()))
    assert frame_ids[0] == 0
    assert len(frame_ids) == frame_ids[-1] + 1

    detected_motions = np.array([movements[frame_id] for frame_id in frame_ids], dtype=np.uint8)

    return detected_motions


def parse_kpts_file(file_path):
    with open(file_path) as input_stream:
        hand_kpts = json.load(input_stream)

    if len(hand_kpts) == 0:
        return None

    hand_presented = dict()
    for kpt_idx, kpt_track in hand_kpts.items():
        for frame_id in kpt_track.keys():
            hand_presented[int(frame_id) - 1] = True

    return hand_presented


def load_distributed_data(root_dir, proc_fun, extension):
    file_suffix = '.{}'.format(extension)
    files = [f for f in listdir(root_dir) if isfile(join(root_dir, f)) and f.endswith(file_suffix)]

    out_data = dict()
    for file_name in tqdm(files, desc='Loading data'):
        full_path = join(root_dir, file_name)
        rel_path = file_name.replace(file_suffix, '')

        file_data = proc_fun(full_path)
        if file_data is not None:
            out_data[rel_path] = file_data

    return out_data


def flat_predictions(start_positions, all_scores, trg_label, window_size, num_frames):
    pred_labels = np.argmax(all_scores, axis=1)
    pred_scores = np.max(all_scores, axis=1)
    matched_mask = pred_labels == trg_label

    out_segm = np.zeros([num_frames], dtype=np.uint8)
    out_scores = np.full([num_frames], -1.0, dtype=np.float32)
    for i, start_pos in enumerate(start_positions):
        glob_end = min(start_pos + window_size, num_frames)
        glob_start = max(0, start_pos, glob_end - window_size // 2)

        if 0 <= glob_start < glob_end:
            out_segm[glob_start:glob_end] = matched_mask[i]
            out_scores[glob_start:glob_end] = pred_scores[i] if matched_mask[i] else -1.0

    return out_segm, out_scores


def find_hands(sparse_mask, num_frames):
    if sparse_mask is None:
        return None

    out_seg = np.zeros([num_frames], dtype=np.uint8)
    for frame_id in sparse_mask.keys():
        if 0 <= frame_id < num_frames:
            out_seg[frame_id] = True

    return out_seg


def split_subsequences(values, trg_label=None, min_size=None):
    changes = (np.argwhere(values[1:] != values[:-1]).reshape([-1]) + 1).tolist()
    changes = [0] + changes + [len(values)]

    segments = [[changes[i], changes[i + 1], values[changes[i]]] for i in range(len(changes) - 1)]

    if trg_label is not None:
        segments = [s for s in segments if s[2] == trg_label]

    if min_size is not None:
        segments = [s for s in segments if s[1] - s[0] >= min_size]

    return segments


def get_longest_segment(segments):
    return max(segments, key=lambda tup: tup[1]-tup[0])


def merge_closest(segments, max_distance=1):
    out_data = []

    last_segment = None
    for segment in segments:
        if last_segment is None:
            last_segment = deepcopy(segment)
        else:
            last_end = last_segment[1]
            cur_start = segment[0]
            if cur_start - last_end <= max_distance:
                last_segment[1] = segment[1]
            else:
                out_data.append(last_segment)
                last_segment = deepcopy(segment)

    if last_segment is not None:
        out_data.append(last_segment)

    return out_data


def get_ignore_candidates(records, ignore_labels):
    out_data = []
    for record in records:
        if record.label in ignore_labels:
            out_data.append((record, []))

    return out_data


def get_regular_candidates(records, all_predictions, all_motions, all_hand_kpts,
                           window_size, dynamic,
                           target_labels, negative_label, no_motion_label,
                           min_score=0.99, min_length=5, max_distance=1):
    out_data = []
    ignores = defaultdict(list)
    for record in tqdm(records, desc='Processing gestures'):
        if record.label not in target_labels:
            continue

        if record.path not in all_predictions or record.path not in all_motions:
            continue

        pred_starts, scores = all_predictions[record.path]
        det_motion = all_motions[record.path]
        person_hand_kpts = all_hand_kpts[record.path] if record.path in all_hand_kpts else None

        trg_label_mask, trg_label_scores = flat_predictions(
            pred_starts, scores, record.label, window_size, len(det_motion))
        trg_hand_mask = find_hands(person_hand_kpts, len(det_motion))

        if dynamic:
            motion_segments = split_subsequences(det_motion, trg_label=1, min_size=min_length)
            if len(motion_segments) == 0:
                record.label = no_motion_label
                out_data.append((record, []))

                ignores['no_motion'].append(record)
                continue

            if trg_hand_mask is not None:
                interest_segments = []
                for motion_start, motion_end, _ in motion_segments:
                    segment_mask = trg_hand_mask[motion_start:motion_end]

                    hand_presented_segments = split_subsequences(segment_mask, trg_label=1, min_size=min_length)
                    hand_presented_segments = merge_closest(hand_presented_segments, max_distance)
                    if len(hand_presented_segments) == 0:
                        continue

                    hand_start, hand_end, _ = get_longest_segment(hand_presented_segments)
                    interest_segments.append((hand_start + motion_start, hand_end + motion_start, 1))

                if len(interest_segments) == 0:
                    record.label = negative_label
                    out_data.append((record, []))

                    ignores['no_hands'].append(record)
                    continue
            else:
                interest_segments = motion_segments
        elif trg_hand_mask is not None:
            interest_segments = split_subsequences(trg_hand_mask, trg_label=1, min_size=min_length)
            if len(interest_segments) == 0:
                record.label = negative_label
                out_data.append((record, []))

                ignores['no_hands'].append(record)
                continue
        else:
            interest_segments = [[0, len(det_motion), 1]]

        candidates = []
        for segment_start, segment_end, _ in interest_segments:
            glob_shift = segment_start
            segment_mask = trg_label_mask[segment_start:segment_end]
            segment_scores = trg_label_scores[segment_start:segment_end]

            gesture_segments = split_subsequences(segment_mask, trg_label=1)
            if len(gesture_segments) == 0:
                continue

            gesture_start, gesture_end, _ = get_longest_segment(gesture_segments)

            glob_shift += gesture_start
            gesture_mask = segment_scores[gesture_start:gesture_end] > min_score
            movement_segments = split_subsequences(gesture_mask.astype(np.uint8), trg_label=1)
            if len(movement_segments) == 0:
                continue

            movement_segments = merge_closest(movement_segments, max_distance)
            movement_start, movement_end, _ = get_longest_segment(movement_segments)

            clip_start = glob_shift + movement_start
            clip_end = glob_shift + movement_end
            if clip_end - clip_start >= min_length:
                candidates.append((clip_start, clip_end))

        if len(candidates) == 0:
            record.label = negative_label
            out_data.append((record, []))

            ignores['no_candidates'].append(record)
            continue

        out_data.append((record, candidates))

    return out_data, ignores


def find_best_match(candidates, model, dataset, negative_label, input_clip_length, pipeline):
    idx_map = dict()
    for idx in range(len(dataset)):
        ann = dataset.get_ann_info(idx)
        idx_map[ann['rel_path']] = idx

    out_records, empty_records = [], []
    for record, segments in tqdm(candidates, desc='Fixing annotation'):
        if len(segments) == 0:
            out_records.append(record)
            continue

        data = []
        for segment in segments:
            indices = generate_indices(segment[0], segment[1], input_clip_length)

            idx = idx_map[record.path]
            record = deepcopy(dataset.video_infos[idx])
            record['modality'] = dataset.modality
            record['frame_inds'] = indices + dataset.start_index
            record['num_clips'] = 1
            record['clip_len'] = input_clip_length

            record_data = pipeline(record)
            data.append(record_data)

        data_gpu = scatter(collate(data, samples_per_gpu=len(segments)),
                           [torch.cuda.current_device()])[0]

        with torch.no_grad():
            net_output = model(return_loss=False, **data_gpu)
            if isinstance(net_output, (list, tuple)):
                assert len(net_output) == 1
                net_output = net_output[0]

        pred_labels = np.argmax(net_output, axis=1)
        pred_scores = np.max(net_output, axis=1)

        valid_segments = []
        for segment, pred_label, pred_score in zip(segments, pred_labels, pred_scores):
            if pred_label == record.label:
                valid_segments.append((segment, pred_score))

        if len(valid_segments) == 0:
            record.label = negative_label
            out_records.append(record)

            empty_records.append(record)
            continue

        # find best record
        best_match_record = max(valid_segments, key=lambda tup: tup[1])

        # add positive
        record.clip_start = best_match_record[0][0]
        record.clip_end = best_match_record[0][1]
        out_records.append(record)

        # add negative before clip
        if record.clip_start > record.video_start:
            record_before = RawFramesSegmentedRecord(record.raw)
            record_before.clip_start = record.video_start
            record_before.clip_end = record.clip_start
            record_before.video_end = record.clip_start
            record_before.label = negative_label
            out_records.append(record_before)

        # add negative after clip
        if record.video_end > record.clip_end:
            record_after = RawFramesSegmentedRecord(record.raw)
            record_after.video_start = record.clip_end
            record_after.clip_start = record.clip_end
            record_after.clip_end = record.video_end
            record_after.label = negative_label
            out_records.append(record_after)

    out_stat = dict()
    if len(empty_records) > 0:
        out_stat['invalid_matches'] = empty_records

    return out_records, out_stat


def generate_indices(start, end, out_length, invalid_idx=-2):
    num_frames = end - start
    if num_frames < out_length:
        indices = np.arange(start, end)

        num_rest = out_length - len(indices)
        if num_rest > 0:
            num_before = num_rest // 2
            num_after = num_rest - num_before
            indices = np.concatenate((np.full(num_before, invalid_idx, dtype=np.int32),
                                      indices,
                                      np.full(num_after, invalid_idx, dtype=np.int32)))
    else:
        shift_start = start
        shift_end = end - out_length + 1
        start_pos = (shift_start + shift_end) // 2

        indices = np.array([start_pos + i for i in range(out_length)])

    return indices


def dump_records(records, out_file_path):
    with open(out_file_path, 'w') as output_stream:
        for record in records:
            output_stream.write(str(record) + '\n')


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


def update_stat(old_state, new_state):
    for key, value in new_state.items():
        if key not in old_state:
            old_state[key] = value
        else:
            old_state[key].extend(value)

    return old_state


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--checkpoint', '-w', type=str, required=True)
    parser.add_argument('--dataset_name', '-n', type=str, required=True)
    parser.add_argument('--data_dir', '-d', type=str, required=True)
    parser.add_argument('--predictions', '-p', type=str, required=True)
    parser.add_argument('--movements', '-m', type=str, required=True)
    parser.add_argument('--keypoints', '-k', type=str, required=True)
    parser.add_argument('--out_annotation', '-o', type=str, required=True)
    args = parser.parse_args()

    assert exists(args.config)
    assert exists(args.weights)
    assert exists(args.data_dir)
    assert exists(args.predictions)
    assert exists(args.movements)
    assert exists(args.keypoints)
    assert args.dataset_name is not None and args.dataset_name != ''
    assert args.out_annotation is not None and args.out_annotation != ''

    cfg = Config.fromfile(args.config)
    cfg = update_config(cfg, args, trg_name=args.dataset_name)
    cfg = propagate_root_dir(cfg, args.data_dir)

    dataset = build_dataset(cfg.data, 'train', dict(test_mode=True))
    data_pipeline = Compose(dataset.pipeline.transforms[1:])
    print('{} dataset:\n'.format(args.mode) + str(dataset))

    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, strict=False)
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    annotation_path = join(args.data_dir, cfg.data.train.sources[0], cfg.data.train.ann_file)
    records = load_annotation(annotation_path)
    predictions = load_distributed_data(args.predictions, parse_predictions_file, 'txt')
    movements = load_distributed_data(args.movements, parse_movements_file, 'txt')
    hand_kpts = load_distributed_data(args.keypoints, parse_kpts_file, 'json')
    print('Loaded records: {}'.format(len(records)))

    invalid_stat = dict()
    all_candidates = []

    ignore_candidates = get_ignore_candidates(records, IGNORE_LABELS)
    all_candidates += ignore_candidates

    static_candidates, static_invalids = get_regular_candidates(
        records, predictions, movements, hand_kpts,
        cfg.data.output.length, False,
        STATIC_LABELS, NEGATIVE_LABEL, NO_MOTION_LABEL,
        min_score=0.9, min_length=4, max_distance=1)
    all_candidates += static_candidates
    invalid_stat = update_stat(invalid_stat, static_invalids)
    print('Static candidates: {}'.format(len(static_candidates)))

    if len(invalid_stat) > 0:
        print('Ignored records after static analysis:')
        for ignore_label, ignore_values in invalid_stat.items():
            print('   - {}: {}'.format(ignore_label.replace('_', ' '), len(ignore_values)))

    dynamic_candidates, dynamic_invalids = get_regular_candidates(
        records, predictions, movements, hand_kpts,
        cfg.data.output.length, True,
        DYNAMIC_LABELS, NEGATIVE_LABEL, NO_MOTION_LABEL,
        min_score=0.9, min_length=4, max_distance=1)
    all_candidates += dynamic_candidates
    invalid_stat = update_stat(invalid_stat, dynamic_invalids)
    print('Dynamic candidates: {}'.format(len(dynamic_candidates)))

    if len(invalid_stat) > 0:
        print('Ignored records after dynamic analysis:')
        for ignore_label, ignore_values in invalid_stat.items():
            print('   - {}: {}'.format(ignore_label.replace('_', ' '), len(ignore_values)))

    fixed_records, fix_stat = find_best_match(all_candidates, model, dataset, NEGATIVE_LABEL)
    invalid_stat = update_stat(invalid_stat, fix_stat)
    print('Final records: {}'.format(len(fixed_records)))

    if len(invalid_stat) > 0:
        print('Final ignored records:')
        for ignore_label, ignore_values in invalid_stat.items():
            print('   - {}: {}'.format(ignore_label.replace('_', ' '), len(ignore_values)))
            for ignored_record in ignore_values:
                print('      - {}'.format(ignored_record.path))

    dump_records(fixed_records, args.out_annotation)
    print('Fixed annotation has been stored at: {}'.format(args.out_annotation))


if __name__ == '__main__':
    main()
