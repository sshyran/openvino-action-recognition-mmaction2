from os import makedirs
from os.path import exists
from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm


def ensure_dir_exists(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)


def get_valid_sources(all_sources):
    return [s for s in all_sources if exists(s)]


def parse_hvu_records(data_sources):
    assert len(data_sources) > 0

    out_records = defaultdict(list)
    for data_source in data_sources:
        with open(data_source) as input_stream:
            for line_id, line in enumerate(input_stream):
                if line_id == 0:
                    continue

                line_elements = line.strip().split(',')
                if len(line_elements) != 4:
                    continue

                tags, video_name, start, end = line_elements

                tags = tags.split('|')
                url = f'https://www.youtube.com/watch?v={video_name}'
                segment_start = float(start)
                segment_end = float(end)

                out_records[video_name].append({
                    'url': url,
                    'start': segment_start,
                    'end': segment_end,
                    'tags': tags
                })

    return out_records


def parse_kinetics_records(data_sources):
    assert len(data_sources) > 0

    out_records = defaultdict(list)
    for data_source in data_sources:
        with open(data_source) as input_stream:
            for line_id, line in enumerate(input_stream):
                if line_id == 0:
                    continue

                line_elements = line.strip().split(',')
                if len(line_elements) != 5:
                    continue

                label, video_name, start, end, _ = line_elements

                url = f'https://www.youtube.com/watch?v={video_name}'
                segment_start = float(start)
                segment_end = float(end)

                out_records[video_name].append({
                    'url': url,
                    'start': segment_start,
                    'end': segment_end,
                    'tags': [label],
                })

    return out_records


def merge_records(src_records, candidate_records):
    def _is_same_segment(a, b):
        intersect_start = max(a['start'], b['start'])
        intersect_end = min(a['end'], b['end'])
        return intersect_end > intersect_start

    out_records = src_records
    for video_name, segments in tqdm(candidate_records.items(), leave=False):
        if video_name not in out_records.keys():
            out_records[video_name] = segments
        else:
            cur_segments = out_records[video_name]
            for candidate_segment in segments:
                matches = [
                    True for cur_segment in cur_segments
                    if _is_same_segment(cur_segment, candidate_segment)
                ]

                if len(matches) == 0:
                    out_records[video_name].append(candidate_segment)

    return out_records


def main():
    parser = ArgumentParser()
    parser.add_argument('--hvu_sources', '-hi', nargs='+', type=str, required=True)
    parser.add_argument('--kinetics_sources', '-ci', nargs='+', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    args = parser.parse_args()

    ensure_dir_exists(args.output_dir)

    hvu_data_sources = get_valid_sources(args.hvu_sources)
    hvu_records = parse_hvu_records(hvu_data_sources)
    print('Found {} HVU records.'.format(sum(len(l) for l in hvu_records.values())))

    kinetics_data_sources = get_valid_sources(args.kinetics_sources)
    kinetics_records = parse_kinetics_records(kinetics_data_sources)
    print('Found {} Kinetics records.'.format(sum(len(l) for l in kinetics_records.values())))

    merged_records = merge_records(hvu_records, kinetics_records)
    print('Merged {} records.'.format(sum(len(l) for l in merged_records.values())))


if __name__ == '__main__':
    main()
