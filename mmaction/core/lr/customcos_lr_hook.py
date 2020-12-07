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

        self.epoch_periods = periods
        self.restart_weights = restart_weights
        self.min_lr_ratio = min_lr_ratio

        self.iter_periods = None
        self.iter_cumulative_periods = None
        self.max_iters = None

    def before_train_epoch(self, runner):
        super(CustomcosLrUpdaterHook, self).before_train_epoch(runner)

        self.iter_periods = [
            period * self.epoch_len for period in self.epoch_periods
        ]
        self.iter_cumulative_periods = [
            sum(self.iter_periods[0:(i + 1)]) for i in range(len(self.iter_periods))
        ]
        self.max_iters = self.iter_cumulative_periods[-1]

    def get_lr(self, runner, base_lr):
        progress = runner.iter
        skip_iters = self.fixed_iters + self.warmup_iters
        if progress <= skip_iters:
            return base_lr
        elif progress > skip_iters + self.max_iters:
            return base_lr * self.min_lr_ratio

        progress -= skip_iters

        idx = self._get_position_from_periods(progress, self.iter_cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.iter_cumulative_periods[idx - 1]
        current_periods = self.iter_periods[idx]

        alpha = min((progress - nearest_restart) / current_periods, 1)
        target_lr = base_lr * self.min_lr_ratio
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
