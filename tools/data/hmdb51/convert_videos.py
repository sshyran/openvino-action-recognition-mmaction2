import subprocess
from argparse import ArgumentParser
from os import makedirs, listdir
from os.path import exists, join, isfile, isdir

from tqdm import tqdm


def ensure_dir_exists(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)


def prepare_tasks(input_dir, output_dir):
    class_names = [d for d in listdir(input_dir) if isdir(join(input_dir, d))]
    assert len(class_names) > 0

    tasks = []
    for class_name in class_names:
        in_videos_dir = join(input_dir, class_name)
        out_videos_dir = join(output_dir, class_name)

        ensure_dir_exists(out_videos_dir)

        files = [f for f in listdir(in_videos_dir) if isfile(join(in_videos_dir, f))]
        tasks.extend([(join(in_videos_dir, f), join(out_videos_dir, f)) for f in files])

    return tasks


def convert_videos(tasks):
    for in_video, out_video in tqdm(tasks, desc='Converting videos', leave=False):
        subprocess.run(
            f'ffmpeg'
            f' -i {in_video}'
            f' -c:v libxvid'
            f' -q:v 10'
            f' -an'
            f' -threads 1'
            f' -loglevel panic'
            f' {out_video}'.split(' '),
            check=True
        )


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    args = parser.parse_args()

    assert exists(args.input_dir)
    ensure_dir_exists(args.output_dir)

    tasks = prepare_tasks(args.input_dir, args.output_dir)
    convert_videos(tasks)


if __name__ == '__main__':
    main()
