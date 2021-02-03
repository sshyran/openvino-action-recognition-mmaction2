import copy
import os.path as osp
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import mmcv
import torch
import numpy as np
from torch.utils.data import Dataset
from terminaltables import AsciiTable

from .pipelines import Compose


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for datasets.

    All datasets to process video should subclass it.
    All subclasses should overwrite:

    - Methods:`load_annotations`, supporting to load information from an
    annotation file.

    - Methods:`prepare_train_frames`, providing train data.

    - Methods:`prepare_test_frames`, providing test data.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Default: False.
        num_classes (int): Number of classes of the dataset, used in
            multi-class datasets. Default: None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 1.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
    """

    allowed_metrics = []

    def __init__(self,
                 source,
                 root_dir,
                 ann_file,
                 data_subdir,
                 pipeline,
                 kpts_subdir=None,
                 load_kpts=False,
                 test_mode=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 logger=None):
        super().__init__()

        assert isinstance(source, str)
        self.dataset_ids_map = {0: source}

        ann_file = osp.join(root_dir, source, ann_file)
        assert osp.exists(ann_file), f'Annotation file does not exist: {ann_file}'

        data_prefix = osp.join(root_dir, source, data_subdir)
        assert osp.exists(data_prefix), f'Data root dir does not exist: {data_prefix}'

        kpts_prefix = None
        if kpts_subdir is not None and load_kpts:
            kpts_prefix = osp.join(root_dir, source, kpts_subdir)
            assert osp.exists(kpts_prefix), f'Kpts root dir does not exist: {kpts_prefix}'

        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        self.logger = logger

        self.pipeline = Compose(pipeline)
        if self.logger is not None:
            self.logger.info(f'Pipeline:\n{str(self.pipeline)}')

        self.video_infos = self._load_annotations(ann_file, data_prefix)
        self.video_infos = self._add_dataset_info(self.video_infos, dataset_id=0, dataset_name=source)
        self.video_infos = self._add_kpts_info(self.video_infos, data_prefix, kpts_prefix, load_kpts)

    @staticmethod
    def _add_dataset_info(records, dataset_id, dataset_name):
        for record in records:
            record['dataset_id'] = dataset_id
            record['dataset_name'] = dataset_name

        return records

    @staticmethod
    def _add_kpts_info(records, data_prefix, kpts_prefix, enable):
        if not enable or kpts_prefix is None:
            return records

        cut_len = len(osp.abspath(data_prefix)) + 1
        for record in records:
            if 'frame_dir' in record:
                rel_path = osp.abspath(record['frame_dir'])[cut_len:]
            elif 'filename' in record:
                rel_path = osp.abspath(record['filename'])[cut_len:].split('.')[0]
            else:
                raise ValueError(f'Record has unknown data format. Keys: {record.keys()}')

            kpts_file = osp.join(kpts_prefix, f'{rel_path}.json')
            record['kpts_file'] = kpts_file

        return records

    @abstractmethod
    def _load_annotations(self, ann_file, data_prefix):
        """Load the annotation according to ann_file into video_infos."""
        pass

    # json annotations already looks like video_infos, so for each dataset,
    # this func should be the same
    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            if self.data_prefix is not None:
                path_value = video_infos[i][path_key]
                path_value = osp.join(self.data_prefix, path_value)
                video_infos[i][path_key] = path_value

            if self.multi_class:
                assert self.num_classes is not None
                onehot = torch.zeros(self.num_classes)
                onehot[video_infos[i]['label']] = 1.
                video_infos[i]['label'] = onehot
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]

        return video_infos

    def evaluate(self, results, metrics, **kwargs):
        """Evaluation for the dataset.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.

        Returns:
            dict: Evaluation results dict.
        """

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        for metric in metrics:
            if metric not in self.allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        return self._evaluate(results, metrics, **kwargs)

    @abstractmethod
    def _evaluate(self, results, metrics, **kwargs):
        """Evaluation for the dataset.

        Args:
            results (list): Output results.
            metrics (sequence[str]): Metrics to be performed.

        Returns:
            dict: Evaluation results dict.
        """

        pass

    def dump_results(self, results, out):
        """Dump data to json/yaml/pickle strings or files."""

        return mmcv.dump(results, out)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""

        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""

        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""

        return len(self.video_infos)

    def __add__(self, other):
        other_dataset_ids_map = other.dataset_ids_map
        assert len(other_dataset_ids_map) == 1

        next_dataset_id = max(self.dataset_ids_map.keys()) + 1
        self.dataset_ids_map[next_dataset_id] = other_dataset_ids_map[0]

        for record in other.video_infos:
            record['dataset_id'] = next_dataset_id
            self.video_infos.append(record)

        return self

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def _parse_data(self):
        all_ids = defaultdict(list)
        for record in self.video_infos:
            dataset_id = record['dataset_id']
            all_ids[dataset_id].append(record['label'])

        out_ids = {}
        for dataset_id, dataset_items in all_ids.items():
            labels, counts = np.unique(dataset_items, return_counts=True)
            out_ids[dataset_id] = {label: size for label, size in zip(labels, counts)}

            actual_size = len(labels)
            expected_size = max(labels) + 1
            if actual_size != expected_size:
                print('Expected {} labels but found {} for the {} dataset.'
                      .format(expected_size, actual_size, self.dataset_ids_map[dataset_id]))

        return out_ids

    def __repr__(self):
        datasets = self._parse_data()

        data_info = []
        total_num_labels, total_num_items = 0, 0
        for dataset_id, labels in datasets.items():
            dataset_name = self.dataset_ids_map[dataset_id]
            dataset_num_labels = len(labels)

            class_sizes = list(labels.values())
            dataset_num_items = sum(class_sizes)
            dataset_imbalance_info = '{:.2f}'.format(max(class_sizes) / min(class_sizes))

            data_info.append([dataset_name, dataset_num_labels, dataset_num_items, dataset_imbalance_info])
            total_num_labels += dataset_num_labels
            total_num_items += dataset_num_items
        data_info.append(['total', total_num_labels, total_num_items, ''])

        header = ['name', '# labels', '# items', 'imbalance']
        table_data = [header] + data_info
        table = AsciiTable(table_data)
        msg = table.table

        return msg

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_frames(idx)
        else:
            return self.prepare_train_frames(idx)

    def num_classes(self):
        datasets = self._parse_data()

        return [max(list(datasets[dataset_id].keys())) + 1 for dataset_id in sorted(datasets.keys())]

    def class_sizes(self):
        datasets = self._parse_data()

        return [datasets[dataset_id] for dataset_id in sorted(datasets.keys())]
