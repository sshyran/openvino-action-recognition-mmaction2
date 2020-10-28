from os import makedirs, listdir, walk
from os.path import exists, join, isfile, abspath
from shutil import rmtree
from argparse import ArgumentParser

import cv2
from tqdm import tqdm


VIDEO_EXTENSIONS = 'avi', 'mp4', 'mov', 'webm'


def create_dirs(dir_path, override=False):
    if override:
        if exists(dir_path):
            rmtree(dir_path)
        makedirs(dir_path)
    elif not exists(dir_path):
        makedirs(dir_path)


def parse_relative_paths(data_dir, extensions):
    data_dir = abspath(data_dir)
    skip_size = len(data_dir) + 1

    relative_paths = []
    for root, sub_dirs, files in tqdm(walk(data_dir)):
        if len(sub_dirs) == 0 and len(files) > 0:
            valid_files = [f for f in files if f.split('.')[-1].lower() in extensions]
            if len(valid_files) > 0:
                relative_paths.append(root[skip_size:])

    return relative_paths


def prepare_tasks(relative_paths, input_dir, output_dir, extensions):
    out_tasks = []
    for relative_path in tqdm(relative_paths):
        input_videos_dir = join(input_dir, relative_path)
        assert exists(input_videos_dir)

        input_video_files = [f for f in listdir(input_videos_dir)
                             if isfile(join(input_videos_dir, f)) and f.split('.')[-1].lower() in extensions]
        if len(input_video_files) == 0:
            continue

        for input_video_file in input_video_files:
            input_video_path = join(input_videos_dir, input_video_file)

            video_name = input_video_file.split('.')[0].lower()
            output_video_dir = join(output_dir, relative_path, video_name)

            if exists(output_video_dir):
                existed_files = [f for f in listdir(output_video_dir) if isfile(join(output_video_dir, f))]
                existed_frame_ids = [int(f.split('.')[0]) for f in existed_files]
                existed_num_frames = len(existed_frame_ids)
                if min(existed_frame_ids) != 1 or max(existed_frame_ids) != existed_num_frames:
                    rmtree(output_video_dir)
                else:
                    continue

            out_tasks.append((input_video_path, output_video_dir, join(relative_path, video_name)))

    return out_tasks


def estimate_num_frames(tasks):
    total_num_frames = 0
    for video_path, _, _ in tqdm(tasks, desc='Counting num frames'):
        video_capture = cv2.VideoCapture(video_path)

        total_num_frames += int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        video_capture.release()

    return total_num_frames


def dump_frames(video_path, out_dir, image_name_template, max_image_size, pbar):
    video_capture = cv2.VideoCapture(video_path)
    video_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)

    max_side = max(video_width, video_height)
    if max_side > max_image_size:
        scale = float(max_image_size) / float(max_side)
        trg_height, trg_width = int(video_height * scale), int(video_width * scale)
    else:
        trg_height, trg_width = int(video_height), int(video_width)

    success = True
    read_frame_id = 0
    while success:
        success, frame = video_capture.read()
        if success:
            read_frame_id += 1

            resized_image = cv2.resize(frame, (trg_width, trg_height))

            out_image_path = join(out_dir, image_name_template.format(read_frame_id))
            cv2.imwrite(out_image_path, resized_image)

        pbar.update(1)

    video_capture.release()

    return read_frame_id, video_fps


def dump_records(records, out_path):
    with open(out_path, 'w') as output_stream:
        for rel_path, num_frames, fps in records:
            if num_frames == 0:
                continue

            converted_record = rel_path, -1, 0, num_frames, 0, num_frames, fps
            output_stream.write(' '.join([str(r) for r in converted_record]) + '\n')


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument('--out_extension', '-ie', type=str, required=False, default='jpg')
    parser.add_argument('--max_image_size', '-ms', type=int, required=False, default=720)
    parser.add_argument('--clear_dumped', '-c', action='store_true', required=False)
    args = parser.parse_args()

    assert exists(args.input_dir)

    override = args.clear_dumped
    create_dirs(args.output_dir, override=override)

    image_name_template = '{:05}' + '.{}'.format(args.out_extension)

    print('\nPreparing tasks ...')
    relative_paths = parse_relative_paths(args.input_dir, VIDEO_EXTENSIONS)
    tasks = prepare_tasks(relative_paths, args.input_dir, args.output_dir, VIDEO_EXTENSIONS)
    total_num_frames = estimate_num_frames(tasks)
    print('Finished. Found {} videos ({} frames).'.format(len(tasks), total_num_frames))

    print('\nDumping frames ...')
    records = []
    with tqdm(total=total_num_frames) as pbar:
        for input_video_path, output_video_dir, relative_path in tasks:
            create_dirs(output_video_dir)
            video_num_frames, video_fps = dump_frames(
                input_video_path, output_video_dir, image_name_template, args.max_image_size, pbar)

            records.append((relative_path, video_num_frames, video_fps))
    print('Finished.')

    out_annot_path = abspath('{}/../annot.txt'.format(args.output_dir))
    dump_records(records, out_annot_path)
    print('\nAnnotated has been stored at: {}'.format(out_annot_path))


if __name__ == '__main__':
    main()
