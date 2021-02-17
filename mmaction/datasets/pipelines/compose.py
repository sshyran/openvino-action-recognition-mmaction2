from collections.abc import Sequence

import numpy as np
from mmcv.utils import build_from_cfg

from ..registry import PIPELINES


@PIPELINES.register_module()
class Compose(object):
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be callable or a dict, '
                                f'but got {type(transform)}')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
            dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None

        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'

        return format_string


@PIPELINES.register_module()
class ProbCompose(object):
    def __init__(self, transforms, probs):
        assert isinstance(transforms, Sequence)
        assert isinstance(probs, Sequence)
        assert len(transforms) == len(probs)
        assert all(p >= 0.0 for p in probs)

        sum_probs = float(sum(probs))
        assert sum_probs > 0.0
        norm_probs = [float(p) / sum_probs for p in probs]
        self.limits = np.cumsum([0.0] + norm_probs)

        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be callable or a dict, '
                                f'but got {type(transform)}')

    def __call__(self, data):
        rand_value = np.random.rand()
        transform_id = np.max(np.where(rand_value > self.limits)[0])

        transform = self.transforms[transform_id]
        data = transform(data)

        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'

        return format_string
