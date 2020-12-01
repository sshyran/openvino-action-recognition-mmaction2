import json
import subprocess
import uuid
import glob
from os import makedirs, remove, listdir
from os.path import exists, join, isfile, basename
from shutil import rmtree
from argparse import ArgumentParser

from joblib import delayed, Parallel


class VideoDownloader:
    def __init__(self, num_jobs, tmp_dir, max_num_attempts=5):
        self.num_jobs = num_jobs
        assert self.num_jobs > 0

        self.tmp_dir = tmp_dir

        self.max_num_attempts = max_num_attempts
        assert self.max_num_attempts > 0

    @staticmethod
    def _log(message_tuple):
        output_filename, status, msg = message_tuple
        str_template = '   - {}: {}' if status else '   - {}: Error: {}'
        print(str_template.format(output_filename, msg))

    def _download_video(self, url, output_filename, start_time, end_time):
        status = False
        tmp_filename = join(self.tmp_dir, '%s.%%(ext)s' % uuid.uuid4())
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
                if attempts == self.max_num_attempts:
                    return url, status, err.output
            else:
                break

        tmp_filename = glob.glob('%s*' % tmp_filename.split('.')[0])[0]
        command = ['ffmpeg',
                   '-i', '"%s"' % tmp_filename,
                   '-ss', str(start_time),
                   '-t', str(end_time - start_time),
                   '-c:v', 'copy', '-an',
                   '-threads', '1',
                   '-loglevel', 'panic',
                   '"%s"' % output_filename]
        command = ' '.join(command)
        try:
            _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            return url, status, err.output

        status = exists(output_filename)
        remove(tmp_filename)

        message_tuple = url, status, 'Downloaded'
        self._log(message_tuple)

        return message_tuple

    def __call__(self, tasks):
        if len(tasks) == 0:
            return []

        if not exists(self.tmp_dir):
            makedirs(self.tmp_dir)

        if self.num_jobs == 1:
            status_lst = []
            for url, out_video_path, segment_start, segment_end in tasks:
                status_lst.append(self._download_video(url, out_video_path, segment_start, segment_end))
        else:
            status_lst = Parallel(n_jobs=self.num_jobs)(
                delayed(self._download_video)(url, out_video_path, segment_start, segment_end)
                for url, out_video_path, segment_start, segment_end in tasks
            )

        rmtree(self.tmp_dir)

        return status_lst


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
            url = record['url']
            video_name = url.split('?v=')[-1]

            segments = record['annotations']
            for segment_id, segment_annot in enumerate(segments):
                segment_name = f'{video_name}_segment{segment_id}'
                # segment_label = segment_annot['label']
                segment_borders = segment_annot['segment']

                segment_start = int(segment_borders[0])
                segment_end = int(segment_borders[1])

                out_records[segment_name] = {
                    'url': url,
                    # 'label': segment_label,
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
    for video_path in candidate_videos:
        video_name = basename(video_path).replace('.{}'.format(extension), '')

        record = records[video_name]
        out_tasks.append((record['url'], video_path, record['start'], record['end']))

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
    parser.add_argument('--sources', '-s', nargs='+', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument('--extension', '-e', type=str, required=False, default='mp4')
    parser.add_argument('--num_jobs', '-n', type=int, required=False, default=24)
    parser.add_argument('--tmp_dir', '-t', type=str, required=False, default='/tmp/kinetics700')
    args = parser.parse_args()

    ensure_dir_exists(args.output_dir)

    data_sources = get_valid_sources(args.sources)
    print_data_sources_stat(data_sources)
    assert len(data_sources) > 0

    all_records = parse_records(data_sources)
    print('Found {} records.'.format(len(all_records)))

    tasks = prepare_tasks(all_records, args.output_dir, args.extension)
    print('Prepared {} tasks for downloading.'.format(len(tasks)))

    downloader = VideoDownloader(args.num_jobs, args.tmp_dir)
    status_lst = downloader(tasks)
    print_status(status_lst)


if __name__ == '__main__':
    main()
