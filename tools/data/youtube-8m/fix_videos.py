import subprocess
from os import makedirs, listdir, remove
from os.path import exists, join, isfile, basename, getsize
from argparse import ArgumentParser
from collections import namedtuple
from shutil import rmtree, copyfile

try:
    from decord import VideoReader
except ImportError:
    raise ImportError('Please run "pip install decord" to install Decord first.')

Segment = namedtuple('Segment', 'output_filename, start_time, end_time')


class VideoFixer:
    def __init__(self, num_jobs, tmp_dir, max_num_attempts=5, verbose=10):
        self.num_jobs = num_jobs
        assert self.num_jobs > 0
        self.verbose = verbose
        assert self.verbose >= 0
        self.tmp_dir = tmp_dir
        assert self.tmp_dir is not None and self.tmp_dir != ''
        self.max_num_attempts = max_num_attempts
        assert self.max_num_attempts > 0

    def __call__(self, video_paths):
        if len(video_paths) == 0:
            return

        if not exists(self.tmp_dir):
            makedirs(self.tmp_dir)

        if self.num_jobs == 1:
            for video_path in video_paths:
                self._process_video(video_path)
        else:
            from joblib import delayed, Parallel

            Parallel(n_jobs=self.num_jobs, verbose=self.verbose)(
                delayed(self._process_video)(video_path)
                for video_path in video_paths
            )

        rmtree(self.tmp_dir)

    def _process_video(self, video_path):
        valid, message = self._check_file(video_path)
        if not valid:
            self._log(video_path, False, message)
            return

        valid = self._check_video(video_path)
        if valid:
            return

        for _ in range(self.max_num_attempts):
            tmp_video_path = join(self.tmp_dir, basename(video_path))
            valid, message = self._fix_video(video_path, tmp_video_path)
            if not valid:
                continue

            valid = self._check_video(tmp_video_path)
            if not valid:
                remove(tmp_video_path)
                continue

            remove(video_path)
            copyfile(tmp_video_path, video_path)
            remove(tmp_video_path)

            break

        if not valid:
            self._log(video_path, False, 'Invalid')

    @staticmethod
    def _check_file(video_filename):
        if not exists(video_filename):
            return False, "Not exists"
        elif getsize(video_filename) == 0:
            remove(video_filename)
            return False, "Empty file - removed"
        else:
            return True, "Converted"

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

        return ok

    @staticmethod
    def _fix_video(input_filename, output_filename):
        command = ['ffmpeg',
                   '-hide_banner',
                   '-threads', '1',
                   '-loglevel', '"panic"',
                   '-i', '"{}"'.format(input_filename),
                   '-pix_fmt', ' yuv420p',
                   '-c:v', 'libxvid',
                   '-an',
                   '"{}"'.format(output_filename),
                   '-y']
        command = ' '.join(command)

        try:
            _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            return False, err.output

        if not exists(output_filename):
            return False, "Not converted"
        elif getsize(output_filename) == 0:
            remove(output_filename)
            return False, "Empty file - removed"
        else:
            return True, "Converted"

    @staticmethod
    def _log(*args):
        output_filename, status, msg = args
        str_template = '   - {}: {}' if status else '   - {}: Error: {}'
        print(str_template.format(output_filename, msg))


def ensure_dir_exists(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)


def collect_video_paths(root_dir, extension):
    return [join(root_dir, f) for f in listdir(root_dir) if isfile(join(root_dir, f)) and f.endswith(extension)]


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument('--extension', '-e', type=str, required=False, default='avi')
    parser.add_argument('--num_jobs', '-n', type=int, required=False, default=24)
    parser.add_argument('--tmp_dir', '-t', type=str, required=False, default='/tmp/video_fixer')
    args = parser.parse_args()

    ensure_dir_exists(args.output_dir)

    video_paths = collect_video_paths(args.input_dir, args.extension)
    print(f'Collected {len(video_paths)} videos.')

    fixer = VideoFixer(args.num_jobs, args.tmp_dir)
    fixer(video_paths)


if __name__ == '__main__':
    main()
