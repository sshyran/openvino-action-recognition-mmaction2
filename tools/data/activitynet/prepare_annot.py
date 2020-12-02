import json
from argparse import ArgumentParser
from collections import defaultdict
from os import makedirs, listdir
from os.path import exists, join, isfile


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
    out_records = dict()
    for data_source in data_sources:
        with open(data_source) as input_stream:
            data = json.load(input_stream)

        database = data['database']
        for record in database.values():
            if record['subset'] not in ['training', 'validation']:
                continue

            url = record['url']
            video_name = url.split('?v=')[-1]
            data_type = 'train' if record['subset'] == 'training' else 'test'

            segments = record['annotations']
            for segment_id, segment_annot in enumerate(segments):
                segment_name = f'{video_name}_segment{segment_id}'
                segment_label = segment_annot['label']

                out_records[segment_name] = {
                    'data_type': data_type,
                    'label': segment_label,
                }

    return out_records


def validate_videos(records, out_videos_dir, extension):
    downloaded_videos = set(
        f.replace(f'.{extension}', '')
        for f in listdir(out_videos_dir)
        if isfile(join(out_videos_dir, f)) and f.endswith(extension)
    )
    all_videos = set(video_name for video_name in records.keys())

    valid_videos = downloaded_videos & all_videos
    out_records = {video_name: records[video_name] for video_name in valid_videos}

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


def split_by_type(annotation):
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
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument('--extension', '-e', type=str, required=False, default='avi')
    args = parser.parse_args()

    ensure_dir_exists(args.output_dir)

    data_sources = get_valid_sources(args.sources)
    print_data_sources_stat(data_sources)
    assert len(data_sources) > 0

    records = parse_records(data_sources)
    print(f'Found {len(records)} records.')

    classmap = build_classmap(records)
    print(f'Found {len(classmap)} unique classes.')

    out_classmap_path = join(args.output_dir, '..', 'classmap.json')
    write_classmap(classmap, out_classmap_path)
    print(f'Dumped classmap to: {out_classmap_path}')

    records = validate_videos(records, args.output_dir, args.extension)
    print(f'Validated {len(records)} videos.')

    annot = convert_annot(records, classmap, args.extension)
    split_annot = split_by_type(annot)

    for data_type, records in split_annot.items():
        out_annot_path = join(args.output_dir, '..', f'{data_type}.txt')
        write_annot(records, out_annot_path)
        print(f'Dumped annot to: {out_annot_path}')


if __name__ == '__main__':
    main()
