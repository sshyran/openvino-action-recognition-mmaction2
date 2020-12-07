import math
from abc import ABCMeta, abstractmethod

from mmcv.runner.hooks import Hook


class BaseLrUpdaterHook(Hook, metaclass=ABCMeta):
    schedulers = ['constant', 'linear', 'cos']

    def __init__(self,
                 fixed=None,
                 fixed_epochs=0,
                 fixed_ratio=1.0,
                 warmup=None,
                 warmup_epochs=0,
                 warmup_ratio=0.1):
        if fixed is not None:
            assert fixed in self.schedulers
            assert fixed_epochs > 0

        if warmup is not None:
            assert warmup in self.schedulers
            assert warmup_epochs > 0
            assert 0 < warmup_ratio <= 1.0

        self.warmup_policy = warmup
        self.warmup_epochs = warmup_epochs if warmup else 0
        self.warmup_start_ratio = warmup_ratio

        self.fixed_policy = fixed
        self.fixed_epochs = fixed_epochs if fixed else 0
        self.fixed_start_ratio = fixed_ratio
        self.fixed_end_ratio = self.warmup_start_ratio if warmup is not None else 1.0

        self.base_lr = []
        self.epoch_len = None
        self.fixed_iters = None
        self.warmup_iters = None

    @staticmethod
    def _set_lr(runner, lr_groups):
        if isinstance(runner.optimizer, dict):
            for k, optimizer in runner.optimizer.items():
                for param_group, lr in zip(optimizer.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(runner.optimizer.param_groups,
                                       lr_groups):
                param_group['lr'] = lr

    @staticmethod
    def _get_lr(policy, cur_iters, regular_lr, max_iters, start_scale, end_scale):
        progress = float(cur_iters) / float(max_iters)
        if policy == 'constant':
            k = start_scale
        elif policy == 'linear':
            k = (end_scale - start_scale) * progress + start_scale
        elif policy == 'cos':
            k = end_scale + 0.5 * (start_scale - end_scale) * (math.cos(math.pi * progress) + 1.0)
        else:
            raise ValueError(f'Unknown policy: {policy}')

        return [_lr * k for _lr in regular_lr]

    @abstractmethod
    def get_lr(self, runner, base_lr):
        pass

    def get_regular_lr(self, runner):
        if isinstance(runner.optimizer, dict):
            lr_groups = {}
            for k in runner.optimizer.keys():
                _lr_group = [
                    self.get_lr(runner, _base_lr)
                    for _base_lr in self.base_lr[k]
                ]
                lr_groups.update({k: _lr_group})

            return lr_groups
        else:
            return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    def get_fixed_lr(self, cur_iters, regular_lr):
        return self._get_lr(
            self.fixed_policy,
            cur_iters,
            regular_lr,
            self.fixed_iters,
            self.fixed_start_ratio,
            self.fixed_end_ratio
        )

    def get_warmup_lr(self, cur_iters, regular_lr):
        return self._get_lr(
            self.warmup_policy,
            cur_iters,
            regular_lr,
            self.warmup_iters,
            self.warmup_start_ratio,
            1.0
        )

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        if isinstance(runner.optimizer, dict):
            self.base_lr = {}
            for k, optimizer in runner.optimizer.items():
                for group in optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                _base_lr = [
                    group['initial_lr'] for group in optimizer.param_groups
                ]
                self.base_lr.update({k: _base_lr})
        else:
            for group in runner.optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            self.base_lr = [
                group['initial_lr'] for group in runner.optimizer.param_groups
            ]

    def before_train_epoch(self, runner):
        self.epoch_len = len(runner.data_loader)
        self.fixed_iters = self.fixed_epochs * self.epoch_len
        self.warmup_iters = self.warmup_epochs * self.epoch_len

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        regular_lr = self.get_regular_lr(runner)

        if cur_iter >= self.warmup_iters + self.fixed_iters:
            self._set_lr(runner, regular_lr)
        elif cur_iter >= self.fixed_iters:
            warmup_lr = self.get_warmup_lr(cur_iter - self.fixed_iters, regular_lr)
            self._set_lr(runner, warmup_lr)
        else:
            fixed_lr = self.get_fixed_lr(cur_iter, regular_lr)
            self._set_lr(runner, fixed_lr)
