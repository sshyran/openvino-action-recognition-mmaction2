from os import makedirs
from os.path import exists, join
from argparse import ArgumentParser

import cv2
import numpy as np
from tqdm import tqdm
from joblib import delayed, Parallel


class RawFramesSegmentedRecord:
    def __init__(self, row):
        self._data = row
        assert len(self._data) == 7

    @property
    def path(self):
        return self._data[0]

    @property
    def start(self):
        return int(self._data[4]) + 1

    @property
    def end(self):
        return int(self._data[5]) + 1


def load_annotation(ann_full_path):
    return [RawFramesSegmentedRecord(x.strip().split(' ')) for x in open(ann_full_path)]


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def load_images(record, data_dir):
    out_data = dict()

    in_images_dir = join(data_dir, record.path)
    for image_id in range(record.start, record.end):
        image = cv2.imread(join(in_images_dir, '{:05d}.jpg'.format(image_id)))
        out_data[image_id - 1] = np.copy(image)

    return out_data


def estimate_frozen_segments(images):
    frame_ids = list(sorted(images.keys()))
    assert frame_ids[0] == 0
    assert len(frame_ids) == frame_ids[-1] + 1

    out_segments = []
    last_segment = [0, 1]
    for i in range(1, len(frame_ids)):
        prev_frame_id = frame_ids[i - 1]
        cur_frame_id = frame_ids[i]

        prev_image = images[prev_frame_id]
        cur_image = images[cur_frame_id]

        abs_diff = np.abs(cur_image.astype(np.int32) - prev_image.astype(np.int32))
        if np.max(abs_diff) == 0:
            last_segment[1] += 1
        else:
            out_segments.append(last_segment)
            last_segment = [cur_frame_id, cur_frame_id + 1]

    out_segments.append(last_segment)

    return out_segments


def estimate_motion(segments, images, magnitude_threshold, area_threshold):
    motion = np.zeros([len(images)], dtype=np.uint8)

    first_segment = segments[0]
    prev_grayscale = to_grayscale(images[first_segment[0]])
    for segment_id in range(1, len(segments)):
        cur_segment = segments[segment_id]
        cur_grayscale = to_grayscale(images[cur_segment[0]])

        flow = cv2.calcOpticalFlowFarneback(prev_grayscale, cur_grayscale, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        motion_mask = magnitude > magnitude_threshold
        motion_area = float(np.sum(motion_mask)) / float(magnitude.shape[0] * magnitude.shape[1])
        if motion_area > area_threshold:
            motion[cur_segment[0]:cur_segment[1]] = True

            if first_segment is not None:
                motion[first_segment[0]:first_segment[1]] = True
                first_segment = None

        if segment_id == 1:
            first_segment = None

        prev_grayscale = cur_grayscale

    return motion.tolist()


def process_record(record, data_dir, out_dir, mag_threshold, move_threshold):
    images = load_images(record, data_dir)
    frozen_segments = estimate_frozen_segments(images)
    motions = estimate_motion(frozen_segments, images, mag_threshold, move_threshold)

    out_data_path = join(out_dir, '{}.txt'.format(record.path))
    with open(out_data_path, 'w') as output_stream:
        for frame_id, motion_detected in enumerate(motions):
            output_stream.write('{};{:d}\n'.format(frame_id, motion_detected))


def process_records(records, data_dir, out_dir, mag_threshold, move_threshold):
    for record in tqdm(records):
        process_record(record, data_dir, out_dir, mag_threshold, move_threshold)


def process_records_parallel(num_proc, records, data_dir, out_dir, mag_threshold, move_threshold):
    Parallel(n_jobs=num_proc, verbose=100)(
        delayed(process_record)(record, data_dir, out_dir, mag_threshold, move_threshold)
        for record in records
    )


def main():
    parser = ArgumentParser()
    parser.add_argument('--annotation', '-a', type=str, required=True)
    parser.add_argument('--data_dir', '-d', type=str, required=True)
    parser.add_argument('--out_dir', '-o', type=str, required=True)
    parser.add_argument('--magnitude_threshold', '-mt', type=float, required=False, default=5.0)
    parser.add_argument('--area_threshold', '-at', type=float, required=False, default=5e-3)
    parser.add_argument('--num_proc', '-n', type=int, required=False, default=1)
    args = parser.parse_args()

    assert exists(args.annotation)
    assert exists(args.data_dir)

    records = load_annotation(args.annotation)
    print('Loaded records: {}'.format(len(records)))

    if not exists(args.out_dir):
        makedirs(args.out_dir)

    if args.num_proc == 1:
        process_records(records, args.data_dir, args.out_dir,
                        args.magnitude_threshold, args.area_threshold)
    else:
        process_records_parallel(args.num_proc, records, args.data_dir, args.out_dir,
                                 args.magnitude_threshold, args.area_threshold)


if __name__ == '__main__':
    main()
