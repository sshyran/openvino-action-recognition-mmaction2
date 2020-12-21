import subprocess
from os import makedirs, listdir, remove, popen
from os.path import exists, join, isfile, basename, getsize
from argparse import ArgumentParser

from joblib import delayed, Parallel

try:
    from decord import VideoReader
except ImportError:
    raise ImportError('Please run "pip install decord" to install Decord first.')


class VideoConverter:
    def __init__(self, num_jobs, scale):
        self.num_jobs = num_jobs
        assert self.num_jobs > 0

        self.scale = scale

    @staticmethod
    def _log(message_tuple):
        output_filename, status, msg = message_tuple
        str_template = '   - {}: {}' if status else '   - {}: Error: {}'
        print(str_template.format(output_filename, msg))

    def _process_task(self, input_filename, output_filename):
        output_filename, ok, message = self._convert_video(
            input_filename, output_filename, self.scale
        )
        if not ok:
            return input_filename, False, message

        output_filename, ok, message = self._check_video(
            output_filename
        )
        if not ok:
            return input_filename, False, message

        message_tuple = input_filename, True, 'Converted, Checked'
        self._log(message_tuple)

        return message_tuple

    @staticmethod
    def _convert_video(input_filename, output_filename, scale):
        result = popen(
            f'ffprobe -hide_banner -loglevel error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 {input_filename}'  # noqa:E501
        )
        w, h = [int(d) for d in result.readline().rstrip().split(',')]

        if w > h:
            command = ['ffmpeg',
                       '-hide_banner',
                       '-threads', '1',
                       '-loglevel', '"panic"',
                       '-i', '"{}"'.format(input_filename),
                       '-vf', 'scale=-2:{}'.format(scale),
                       '-c:v', 'libxvid',
                       '-q:v', '10',
                       '-an',
                       '"{}"'.format(output_filename),
                       '-y']
        else:
            command = ['ffmpeg',
                       '-hide_banner',
                       '-threads', '1',
                       '-loglevel', '"panic"',
                       '-i', '"{}"'.format(input_filename),
                       '-vf', 'scale={}:-2'.format(scale),
                       '-c:v', 'libxvid',
                       '-q:v', '10',
                       '-an',
                       '"{}"'.format(output_filename),
                       '-y']
        command = ' '.join(command)

        try:
            _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            return "", False, err.output

        if not exists(output_filename):
            return "", False, "Not converted"
        elif getsize(output_filename) == 0:
            remove(output_filename)
            return output_filename, False, "Empty file"
        else:
            return output_filename, True, "Converted"

    @staticmethod
    def _check_video(video_filename):
        ok = False
        try:
            container = VideoReader(video_filename, num_threads=1)
            if len(container) > 0:
                ok = True

            del container
        except:
            pass

        if ok:
            return video_filename, True, "Checked"
        else:
            remove(video_filename)
            return video_filename, False, "Invalid video file"

    def __call__(self, tasks):
        if len(tasks) == 0:
            return []

        if self.num_jobs == 1:
            status_lst = []
            for in_video_path, out_video_path in tasks:
                status_lst.append(self._process_task(in_video_path, out_video_path))
        else:
            status_lst = Parallel(n_jobs=self.num_jobs)(
                delayed(self._process_task)(in_video_path, out_video_path)
                for in_video_path, out_video_path in tasks
            )

        return status_lst


def ensure_dir_exists(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)


def collect_videos(root_dir):
    return [join(root_dir, f) for f in listdir(root_dir) if isfile(join(root_dir, f))]


def prepare_tasks(in_video_paths, out_videos_dir, extension):
    out_tasks = []
    for in_video_path in in_video_paths:
        video_name = basename(in_video_path).split('.')[0]
        out_video_path = join(out_videos_dir, f'{video_name}.{extension}')
        if exists(out_video_path):
            continue

        out_tasks.append((in_video_path, out_video_path))

    return out_tasks


def print_status(status_lst):
    if len(status_lst) == 0:
        return

    print('Status:')
    for status in status_lst:
        str_template = '   - {}: {}' if status[1] else '   - {}: Error: {}'
        print(str_template.format(status[0], status[2]))


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument('--extension', '-e', type=str, required=False, default='avi')
    parser.add_argument('--scale', '-s', type=int, required=False, default=256)
    parser.add_argument('--num_jobs', '-n', type=int, required=False, default=24)
    args = parser.parse_args()

    assert exists(args.input_dir)
    ensure_dir_exists(args.output_dir)

    input_videos = collect_videos(args.input_dir)
    print('Found {} videos.'.format(len(input_videos)))

    tasks = prepare_tasks(input_videos, args.output_dir, args.extension)
    print('Prepared {} tasks for converting.'.format(len(tasks)))

    converter = VideoConverter(args.num_jobs, args.scale)
    status_lst = converter(tasks)
    print_status(status_lst)


if __name__ == '__main__':
    main()
