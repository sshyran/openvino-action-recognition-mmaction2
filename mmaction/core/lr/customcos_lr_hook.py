import math

from mmcv.runner.hooks import HOOKS

from .base_lr_hook import BaseLrUpdaterHook


@HOOKS.register_module()
class CustomcosLrUpdaterHook(BaseLrUpdaterHook):
    def __init__(self, periods, restart_weights=None, min_lr_ratio=1e-3, **kwargs):
        super(CustomcosLrUpdaterHook, self).__init__(**kwargs)

        if restart_weights is None:
            restart_weights = [1.0] * len(periods)
        assert len(periods) == len(restart_weights)

        self.periods = periods
        self.restart_weights = restart_weights
        self.min_lr_ratio = min_lr_ratio

        self.cumulative_periods = [
            sum(self.periods[0:(i + 1)]) for i in range(len(self.periods))
        ]

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter
        target_lr = base_lr * self.min_lr_ratio

        idx = self._get_position_from_periods(progress, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((progress - nearest_restart) / current_periods, 1)
        out_lr = self._annealing_cos(base_lr, target_lr, alpha, current_weight)

        return out_lr

    @staticmethod
    def _get_position_from_periods(iteration, cumulative_periods):
        for i, period in enumerate(cumulative_periods):
            if iteration < period:
                return i

        raise ValueError(f'Current iteration {iteration} exceeds '
                         f'cumulative_periods {cumulative_periods}')

    @staticmethod
    def _annealing_cos(start, end, factor, weight=1.0):
        cos_out = math.cos(math.pi * factor) + 1.0
        return end + 0.5 * weight * (start - end) * cos_out
