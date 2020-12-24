import json
from argparse import ArgumentParser
from collections import defaultdict
from os import listdir
from os.path import exists, join, isfile

import requests
import tensorflow as tf
from tqdm import tqdm
from joblib import delayed, Parallel


def parse_file_names(root_dir):
    return [
        join(root_dir, f)
        for f in listdir(root_dir)
        if isfile(join(root_dir, f)) and f.endswith('.tfrecord')
    ]


def load_dataset(record_files):
    return tf.data.TFRecordDataset(record_files)


def parse_values(value, trg_type=None):
    out_values = []
    for line in str(value).split('\n'):
        line = line.strip()
        if line.startswith('value: '):
            out_values.append(line.replace('value: ', ''))

    if trg_type is not None:
        out_values = [trg_type(v) for v in out_values]

    return out_values


def parse_records(dataset):
    out_records = defaultdict(list)
    all_labels = []
    for raw_record in tqdm(dataset, desc='Parsing records', leave=False):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        video_id = None
        segments = dict(
            start=None,
            end=None,
            label=None,
        )

        for key, value in example.features.feature.items():
            if key == 'id':
                field_values = parse_values(value, str)
                assert len(field_values) == 1

                video_id = field_values[0].replace('"', '')
            elif key == 'segment_start_times':
                segments['start'] = parse_values(value, float)
            elif key == 'segment_end_times':
                segments['end'] = parse_values(value, float)
            elif key == 'segment_labels':
                segments['label'] = parse_values(value, int)

        assert video_id is not None

        num_segments = None
        for record_value in segments.values():
            assert record_value is not None

            if num_segments is None:
                num_segments = len(record_value)
            else:
                assert len(record_value) == num_segments

        records = []
        for start, end, label in zip(segments['start'], segments['end'], segments['label']):
            assert start < end
            assert label >= 0

            records.append((start, end, label))
            all_labels.append(label)

        out_records[video_id].extend(records)

    return out_records, set(all_labels)


def process_video_id(src_video_id, num_attempts=5):
    assert isinstance(src_video_id, str)
    assert len(src_video_id) == 4
    assert num_attempts > 0

    prefix = src_video_id[:2]
    url = f'http://data.yt8m.org/2/j/i/{prefix}/{src_video_id}.js'

    for _ in range(num_attempts):
        response = requests.get(url)
        if response.status_code in [200, 403]:
            break

    if response.status_code != 200:
        print(f'[WARNING] Unable to load {url}: {response.status_code}')
        return src_video_id, None

    answer = response.text
    assert answer.startswith('i(')
    assert answer.endswith(');')

    fields = answer[2:-2].replace('"', '').split(',')
    assert len(fields) == 2

    trg_video_id = fields[-1]

    return src_video_id, trg_video_id


def load_ids_map(file_path):
    if not exists(file_path):
        return dict()

    with open(file_path) as input_stream:
        ids_map = json.load(input_stream)

    return ids_map


def dump_ids_map(ids_map, file_path):
    if len(ids_map) == 0:
        return

    with open(file_path, 'w') as output_stream:
        json.dump(ids_map, output_stream)


def prepare_tasks(in_map, src_ids):
    return list(set(src_ids) - set(in_map.keys()))


def request_ids_map(in_map, tasks, num_jobs=1):
    if num_jobs == 1:
        tuple_list = []
        for src_id in tqdm(tasks, desc='Loading video IDs', leave=False):
            tuple_list.append(process_video_id(src_id))
    else:
        tuple_list = Parallel(n_jobs=num_jobs, verbose=10)(
            delayed(process_video_id)(src_id)
            for src_id in tasks
        )

    ext_map = {src_id: trg_id for src_id, trg_id in tuple_list if trg_id is not None}
    in_map.update(ext_map)

    return in_map


def dump_records(records, ids_map, out_file):
    with open(out_file, 'w') as output_stream:
        for src_video_id, segments in records.items():
            if src_video_id not in ids_map:
                continue

            trg_video_id = ids_map[src_video_id]
            for start, end, label in segments:
                output_stream.write(f'{label},{trg_video_id},{start},{end}\n')


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str, required=True)
    parser.add_argument('--output_ids_map', '-m', type=str, required=True)
    parser.add_argument('--output_annot', '-o', type=str, required=True)
    parser.add_argument('--num_jobs', '-n', type=int, required=False, default=24)
    args = parser.parse_args()

    assert exists(args.input_dir)

    record_files = parse_file_names(args.input_dir)
    print(f'Found {len(record_files)} TFRecord files.')

    dataset = load_dataset(record_files)
    records, labels = parse_records(dataset)

    num_videos = len(records)
    num_records = sum([len(l) for l in records.values()])
    print(f'Collected {num_videos} videos and {num_records} records ({len(labels)} labels).')

    video_ids_map = load_ids_map(args.output_ids_map)
    print(f'Loaded {len(video_ids_map)} IDs map.')

    src_video_ids = records.keys()
    tasks = prepare_tasks(video_ids_map, src_video_ids)
    print(f'Prepared {len(tasks)} tasks.')

    video_ids_map = request_ids_map(video_ids_map, tasks, num_jobs=args.num_jobs)
    print(f'Available {len(video_ids_map)} / {len(src_video_ids)} IDs.')

    dump_ids_map(video_ids_map, args.output_ids_map)
    print(f'Dumped IDs map to: {args.output_ids_map}')

    dump_records(records, video_ids_map, args.output_annot)
    print(f'Dumped records to: {args.output_annot}')


if __name__ == '__main__':
    main()
