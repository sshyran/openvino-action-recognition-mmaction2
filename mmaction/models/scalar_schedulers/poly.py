import numpy as np

from ..registry import SCALAR_SCHEDULERS
from .base import BaseScalarScheduler


@SCALAR_SCHEDULERS.register_module()
class PolyScalarScheduler(BaseScalarScheduler):
    def __init__(self, start_scale, end_scale, num_epochs, power=1.2):
        super(PolyScalarScheduler, self).__init__()

        self._start_s = start_scale
        assert self._start_s > 0.0
        self._end_s = end_scale
        assert self._end_s > 0.0
        self._num_epochs = num_epochs
        assert self._num_epochs > 0
        self._power = power
        assert self._power > 0.0

    def _get_scale(self, step, iters_per_epoch):
        steps_to_end = int(self._num_epochs * iters_per_epoch)

        if step < steps_to_end:
            factor = (self._end_s - self._start_s) / (1.0 - self._power)
            var_a = factor / (steps_to_end ** self._power)
            var_b = -factor * self._power / float(steps_to_end)

            out_value = var_a * np.power(step, self._power) + var_b * step + self._start_s
        else:
            out_value = self._end_s

        return out_value
