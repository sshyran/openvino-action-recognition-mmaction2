import json
import random
from argparse import ArgumentParser
from collections import defaultdict
from os import makedirs, listdir
from os.path import exists, join, isfile, basename


def ensure_dir_exists(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)


def get_valid_sources(all_sources):
    return [s for s in all_sources if exists(s)]


def print_data_sources_stat(data_sources):
    print('Specified {} valid data sources:'.format(len(data_sources)))
    for data_source in data_sources:
        print('   - {}'.format(data_source))


def parse_records(data_sources):
    num_records = defaultdict(int)
    out_records = dict()
    for data_source in data_sources:
        data_type = basename(data_source).split('.')[0]

        with open(data_source) as input_stream:
            for line_id, line in enumerate(input_stream):
                if line_id == 0:
                    continue

                line_elements = line.strip().split(',')
                if len(line_elements) != 4:
                    continue

                label, video_name, start, end = line_elements

                segment_id = num_records[video_name]
                segment_name = f'{video_name}_segment{segment_id}'

                num_records[video_name] += 1
                out_records[segment_name] = dict(
                    label=int(label),
                    data_type=data_type
                )

    return out_records


def validate_videos(records, videos_dir, extension):
    downloaded_videos = set(
        f.replace(f'.{extension}', '')
        for f in listdir(videos_dir)
        if isfile(join(videos_dir, f)) and f.endswith(extension)
    )
    all_videos = set(video_name for video_name in records.keys())

    valid_videos = downloaded_videos & all_videos
    out_records = {video_name: records[video_name] for video_name in valid_videos}

    return out_records


def split_train_val_subsets(records, test_ratio=0.1):
    assert 0.0 < test_ratio < 1.0

    by_labels = defaultdict(list)
    for video_name, content in records.items():
        by_labels[content['label']].append(video_name)

    clustered_segments = dict()
    for label, segments in by_labels.items():
        videos = defaultdict(list)
        for segment in segments:
            video, _ = segment.split('_segment')
            videos[video].append(segment)

        clustered_segments[label] = videos

    out_records = dict()
    for label, videos in clustered_segments.items():
        num_records = len(by_labels[label])
        assert num_records > 1

        video_names = list(videos.keys())
        num_videos = len(video_names)
        assert num_videos > 1

        num_test_samples = min(num_records - 1, max(1, int(num_records * test_ratio)))
        num_test_videos = min(num_videos - 1, max(1, int(num_videos * test_ratio)))

        num_selected_test_samples = 0
        test_videos = []
        for test_video_name in random.sample(video_names, num_test_videos):
            test_videos.append(test_video_name)
            segments = videos[test_video_name]

            for segment in segments:
                out_records[segment] = dict(label=label, data_type='val')

            num_selected_test_samples += len(segments)
            if num_selected_test_samples >= num_test_samples:
                break

        train_videos = list(set(video_names) - set(test_videos))
        for train_video_name in train_videos:
            segments = videos[train_video_name]

            for segment in segments:
                out_records[segment] = dict(label=label, data_type='train')

    return out_records


def build_classmap(records):
    labels = set(record['label'] for record in records.values())
    return {class_name: i for i, class_name in enumerate(sorted(labels))}


def convert_annot(records, classmap, extension):
    out_records = dict()
    for video_name, content in records.items():
        label_id = classmap[content['label']]
        out_records[f'{video_name}.{extension}'] = label_id, content['data_type']

    return out_records


def group_by_type(annotation):
    out_data = defaultdict(list)
    for video_name, (label_id, data_type) in annotation.items():
        out_data[data_type].append((video_name, label_id))

    return out_data


def write_classmap(classmap, out_path):
    with open(out_path, 'w') as output_stream:
        json.dump(classmap, output_stream)


def write_annot(records, out_path):
    with open(out_path, 'w') as output_stream:
        for video_name, label_id in records:
            output_stream.write(f'{video_name} {label_id}\n')


def main():
    parser = ArgumentParser()
    parser.add_argument('--sources', '-s', nargs='+', type=str, required=True)
    parser.add_argument('--videos_dir', '-v', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument('--extension', '-e', type=str, required=False, default='avi')
    parser.add_argument('--test_ratio', '-r', type=float, required=False, default=0.1)
    args = parser.parse_args()

    ensure_dir_exists(args.output_dir)

    data_sources = get_valid_sources(args.sources)
    print_data_sources_stat(data_sources)
    assert len(data_sources) > 0

    records = parse_records(data_sources)
    print(f'Found {len(records)} records.')

    classmap = build_classmap(records)
    print(f'Found {len(classmap)} unique classes.')

    out_classmap_path = join(args.output_dir, 'classmap.json')
    write_classmap(classmap, out_classmap_path)
    print(f'Dumped classmap to: {out_classmap_path}')

    records = validate_videos(records, args.videos_dir, args.extension)
    print(f'Validated {len(records)} videos.')

    records = split_train_val_subsets(records, args.test_ratio)

    annot = convert_annot(records, classmap, args.extension)
    split_annot = group_by_type(annot)

    for data_type, records in split_annot.items():
        out_annot_path = join(args.output_dir, f'{data_type}.txt')
        write_annot(records, out_annot_path)
        print(f'Dumped annot to: {out_annot_path}')


if __name__ == '__main__':
    main()
