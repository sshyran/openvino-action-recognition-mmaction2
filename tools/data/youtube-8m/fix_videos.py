import subprocess
import uuid
import glob
from os import makedirs, listdir, remove, popen
from os.path import exists, join, isfile, basename, getsize
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from shutil import rmtree

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

    def __call__(self, videos):
        if len(videos) == 0:
            return

        if not exists(self.tmp_dir):
            makedirs(self.tmp_dir)

        if self.num_jobs == 1:
            for video in videos:
                self._process_video(video)
        else:
            from joblib import delayed, Parallel

            Parallel(n_jobs=self.num_jobs, verbose=self.verbose)(
                delayed(self._process_video)(video)
                for video in videos
            )

        rmtree(self.tmp_dir)

    def _process_video(self, video_path):
        valid, message = self._check_file(video_path)
        if not valid:
            self._log(video_path, False, message)
            return

        for _ in range(self.max_num_attempts):
            valid = self._check_video(video_path)
            if valid:
                break



        if not valid:
            self._log(video_path, False, 'Invalid format')

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
    def _convert_video(input_filename, start_time, end_time, output_filename, scale):
        result = popen(
            f'ffprobe -hide_banner '
            f'-loglevel error '
            f'-select_streams v:0 '
            f'-show_entries stream=width,height '
            f'-of csv=p=0 {input_filename}'
        )
        w, h = [int(d) for d in result.readline().rstrip().split(',')]

        if w > h:
            command = ['ffmpeg',
                       '-hide_banner',
                       '-threads', '1',
                       '-loglevel', '"panic"',
                       '-i', '"{}"'.format(input_filename),
                       '-pix_fmt', ' yuv420p',
                       '-ss', str(start_time),
                       '-t', str(end_time - start_time),
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
                       '-pix_fmt', ' yuv420p',
                       '-ss', str(start_time),
                       '-t', str(end_time - start_time),
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
    def _log(*args):
        output_filename, status, msg = args
        str_template = '   - {}: {}' if status else '   - {}: Error: {}'
        print(str_template.format(output_filename, msg))


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
        with open(data_source) as input_stream:
            for line_id, line in enumerate(input_stream):
                if line_id == 0:
                    continue

                line_elements = line.strip().split(',')
                if len(line_elements) != 4:
                    continue

                _, video_name, start, end = line_elements

                url = f'https://www.youtube.com/watch?v={video_name}'
                segment_start = float(start)
                segment_end = float(end)

                segment_id = num_records[video_name]
                segment_name = f'{video_name}_segment{segment_id}'

                num_records[video_name] += 1
                out_records[segment_name] = {
                    'url': url,
                    'start': segment_start,
                    'end': segment_end,
                }

    return out_records


def prepare_tasks(records, out_videos_dir, extension):
    downloaded_videos = [join(out_videos_dir, f)
                         for f in listdir(out_videos_dir)
                         if isfile(join(out_videos_dir, f)) and f.endswith(extension)]
    all_videos = [join(out_videos_dir, f'{video_name}.{extension}')
                  for video_name in records.keys()]

    candidate_videos = list(set(all_videos) - set(downloaded_videos))

    out_tasks = []
    out_urls = []
    for video_path in candidate_videos:
        video_name = basename(video_path).replace('.{}'.format(extension), '')

        record = records[video_name]
        out_tasks.append((record['url'], video_path, record['start'], record['end']))
        out_urls.append(record['url'])

    return out_tasks, set(out_urls)


def main():
    parser = ArgumentParser()
    parser.add_argument('--sources', '-i', nargs='+', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument('--extension', '-e', type=str, required=False, default='avi')
    parser.add_argument('--scale', '-s', type=int, required=False, default=256)
    parser.add_argument('--num_jobs', '-n', type=int, required=False, default=24)
    parser.add_argument('--tmp_dir', '-t', type=str, required=False, default='/tmp/video_downloader')
    args = parser.parse_args()

    ensure_dir_exists(args.output_dir)

    data_sources = get_valid_sources(args.sources)
    print_data_sources_stat(data_sources)
    assert len(data_sources) > 0

    all_records = parse_records(data_sources)
    print('Found {} records.'.format(len(all_records)))

    tasks, urls = prepare_tasks(all_records, args.output_dir, args.extension)
    print('Prepared {} tasks for downloading ({} unique videos).'.format(len(tasks), len(urls)))

    downloader = VideoFixer(args.num_jobs, args.tmp_dir, args.scale)
    downloader(tasks)


if __name__ == '__main__':
    main()
