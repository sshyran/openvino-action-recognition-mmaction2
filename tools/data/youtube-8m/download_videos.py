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


class VideoDownloader:
    def __init__(self, num_jobs, tmp_dir, scale, max_num_attempts=5, verbose=10):
        self.num_jobs = num_jobs
        assert self.num_jobs > 0
        self.verbose = verbose
        assert self.verbose >= 0

        self.tmp_dir = tmp_dir
        self.scale = scale

        self.max_num_attempts = max_num_attempts
        assert self.max_num_attempts > 0

    def __call__(self, tasks):
        if len(tasks) == 0:
            return []

        if not exists(self.tmp_dir):
            makedirs(self.tmp_dir)

        grouped_tasks = self._group_tasks(tasks)

        if self.num_jobs == 1:
            for task in grouped_tasks:
                self._process_task(*task)
        else:
            from joblib import delayed, Parallel

            Parallel(n_jobs=self.num_jobs, verbose=self.verbose)(
                delayed(self._process_task)(*task)
                for task in grouped_tasks
            )

        rmtree(self.tmp_dir)

    @staticmethod
    def _group_tasks(tasks):
        grouped_tasks = defaultdict(list)
        for url, output_filename, start_time, end_time in tasks:
            grouped_tasks[url].append(Segment(
                output_filename, start_time, end_time
            ))

        out_tasks = [(url, segments) for url, segments in grouped_tasks.items()]

        return out_tasks

    def _process_task(self, url, segments):
        assert len(segments) > 0

        tmp_video_filename, ok, message = self._download_video(
            self.tmp_dir, url, self.max_num_attempts
        )
        if not ok:
            self._log(url, False, message)
            return

        for segment in segments:
            output_filename, ok, message = self._convert_video(
                tmp_video_filename, segment.start_time, segment.end_time, segment.output_filename, self.scale
            )
            if not ok:
                self._log(output_filename, False, message)
                continue

            output_filename, ok, message = self._check_video(
                output_filename
            )
            if not ok:
                self._log(output_filename, False, message)

        remove(tmp_video_filename)

    @staticmethod
    def _download_video(tmp_dir, url, max_num_attempts):
        tmp_filename = join(tmp_dir, '%s.%%(ext)s' % uuid.uuid4())
        command = ['youtube-dl',
                   '--quiet', '--no-warnings',
                   '-f', 'mp4',
                   '-o', '"%s"' % tmp_filename,
                   '"%s"' % url]
        command = ' '.join(command)

        attempts = 0
        while True:
            try:
                _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err:
                attempts += 1
                if attempts == max_num_attempts:
                    return "", False, err.output
            else:
                break

        downloaded_filename = glob.glob('%s*' % tmp_filename.split('.')[0])[0]
        if getsize(downloaded_filename) > 0:
            return downloaded_filename, True, "Downloaded"
        else:
            remove(downloaded_filename)
            return downloaded_filename, False, "Empty file"

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

    downloader = VideoDownloader(args.num_jobs, args.tmp_dir, args.scale)
    downloader(tasks)


if __name__ == '__main__':
    main()
