import json
from argparse import ArgumentParser
from collections import defaultdict
from os import makedirs, listdir
from os.path import exists, join, isfile, isdir

from tqdm import tqdm


HMDB_NUM_CLASSES = 51


def ensure_dir_exists(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)


def build_classmap(videos_dir):
    class_names = [d for d in listdir(videos_dir) if isdir(join(videos_dir, d))]
    return {class_name: i for i, class_name in enumerate(sorted(class_names))}


def load_annot_sources(annot_dir, split_id):
    assert split_id in [1, 2, 3]
    split_name = f'_test_split{split_id}.txt'

    annot_files = [f for f in listdir(annot_dir) if isfile(join(annot_dir, f))]
    trg_annot_files = [(join(annot_dir, f), f[:-len(split_name)])
                       for f in annot_files if f.endswith(split_name)]

    return trg_annot_files


def load_annotation(annot_sources, classmap):
    data = dict()
    for data_source, class_name in annot_sources:
        label_id = classmap[class_name]

        with open(data_source) as input_stream:
            for line in input_stream:
                chunks = line.strip().split(' ')
                assert len(chunks) == 2

                video_name, data_split = chunks
                if data_split == '0':
                    continue

                video_rel_path = join(class_name, video_name)
                data_type = 'train' if data_split == '1' else 'test'

                data[video_rel_path] = label_id, data_type

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


def write_classmap(classmap, out_path):
    with open(out_path, 'w') as output_stream:
        json.dump(classmap, output_stream)


def write_annot(records, out_path):
    with open(out_path, 'w') as output_stream:
        for video_name, label_id in records:
            output_stream.write(f'{video_name} {label_id}\n')


def main():
    parser = ArgumentParser()
    parser.add_argument('--videos_dir', '-v', type=str, required=True)
    parser.add_argument('--annot_dir', '-a', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    args = parser.parse_args()

    assert exists(args.videos_dir)
    assert exists(args.annot_dir)
    ensure_dir_exists(args.output_dir)

    classmap = build_classmap(args.videos_dir)
    assert len(classmap) == HMDB_NUM_CLASSES

    out_classmap_path = join(args.output_dir, 'classmap.json')
    write_classmap(classmap, out_classmap_path)
    print(f'Dumped classmap to: {out_classmap_path}')

    annot_sources = load_annot_sources(args.annot_dir, split_id=1)
    assert len(annot_sources) == len(classmap)

    annot = load_annotation(annot_sources, classmap)
    assert len(annot) > 0

    annot = validate_videos(args.videos_dir, annot)
    split_annot = split_by_type(annot)

    for data_type, records in split_annot.items():
        out_annot_path = join(args.output_dir, f'{data_type}.txt')
        write_annot(records, out_annot_path)
        print(f'Dumped annot to: {out_annot_path}')


if __name__ == '__main__':
    main()
