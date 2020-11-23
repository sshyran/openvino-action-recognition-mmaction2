from collections import defaultdict
from os import makedirs
from os.path import exists, join, basename
from argparse import ArgumentParser

from tqdm import tqdm


def ensure_dir_exists(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)


def get_valid_sources(all_sources):
    return [s for s in all_sources if exists(s)]


def load_class_names(file_path):
    out_data = dict()
    with open(file_path) as input_stream:
        for line in input_stream:
            chunks = line.strip().split(' ')
            assert len(chunks) == 2

            class_id, class_name = chunks
            class_id = int(class_id) - 1

            out_data[class_name] = class_id

    return out_data


def load_annotation(data_sources, class_names_map):
    data = dict()
    for data_source in data_sources:
        file_name = basename(data_source)

        if file_name.startswith('train'):
            data_type = 'train'
        elif file_name.startswith('test'):
            data_type = 'test'
        else:
            raise ValueError('Unknown data type: {}'.format(file_name))

        with open(data_source) as input_stream:
            for line in input_stream:
                chunks = line.strip().split(' ')

                if len(chunks) in [1, 2]:
                    video_name = chunks[0]
                else:
                    raise ValueError('Incorrect data line: {}'.format(line.strip()))

                label_id = class_names_map[video_name.split('/')[0]]
                data[video_name] = label_id, data_type

    return data


def validate_videos(videos_dir, annotation):
    out_data = dict()
    for video_name in tqdm(annotation, desc='Validating videos', leave=False):
        video_path = join(videos_dir, video_name)
        if not exists(video_path):
            print('[WARNING] Video path does not exist: {}'.format(video_path))
            continue

        out_data[video_name] = annotation[video_name]

    return out_data


def split_by_type(annotation):
    out_data = defaultdict(list)
    for video_name, (label_id, data_type) in annotation.items():
        out_data[data_type].append((video_name, label_id))

    return out_data


def write_annot(records, out_path):
    with open(out_path, 'w') as output_stream:
        for video_name, label_id in records:
            output_stream.write(f'{video_name} {label_id}\n')


def main():
    parser = ArgumentParser()
    parser.add_argument('--videos_dir', '-v', type=str, required=True)
    parser.add_argument('--sources', '-s', nargs='+', type=str, required=True)
    parser.add_argument('--classes', '-c', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    args = parser.parse_args()

    assert exists(args.videos_dir)
    ensure_dir_exists(args.output_dir)

    data_sources = get_valid_sources(args.sources)
    assert len(data_sources) > 0

    class_names_map = load_class_names(args.classes)
    annotation = load_annotation(data_sources, class_names_map)
    if len(annotation) > 0:
        print('Found {} files.'.format(len(annotation)))
    else:
        print('No files has been found!')
        return

    annotation = validate_videos(args.videos_dir, annotation)
    split_annot = split_by_type(annotation)

    for data_type, records in split_annot.items():
        out_annot_path = join(args.output_dir, f'{data_type}.txt')
        write_annot(records, out_annot_path)
        print(f'Dumped to: {out_annot_path}')


if __name__ == '__main__':
    main()
