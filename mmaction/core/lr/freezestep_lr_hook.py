from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.lr_updater import StepLrUpdaterHook


@HOOKS.register_module()
class FreezestepLrUpdaterHook(StepLrUpdaterHook):
    def __init__(self, fixed_iters=0, fixed_ratio=1.0, **kwargs):
        super(FreezestepLrUpdaterHook, self).__init__(**kwargs)

        if self.by_epoch:
            self.fixed_epochs = fixed_iters
            self.fixed_iters = None
        else:
            self.fixed_epochs = None
            self.fixed_iters = fixed_iters

        self.fixed_ratio = fixed_ratio

    def get_fixed_lr(self):
        return [_lr * self.fixed_ratio for _lr in self.regular_lr]

    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return

        epoch_len = len(runner.data_loader)
        self.fixed_iters = self.fixed_epochs * epoch_len

        if self.warmup_by_epoch:
            self.warmup_iters = self.warmup_epochs * epoch_len

        self.regular_lr = self.get_regular_lr(runner)
        self._set_lr(runner, self.regular_lr)

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters + self.fixed_iters:
                self._set_lr(runner, self.regular_lr)
            elif cur_iter >= self.fixed_iters:
                warmup_lr = self.get_warmup_lr(cur_iter - self.fixed_iters)
                self._set_lr(runner, warmup_lr)
            else:
                fixed_lr = self.get_fixed_lr()
                self._set_lr(runner, fixed_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters + self.fixed_iters:
                return
            elif cur_iter == self.warmup_iters + self.fixed_iters:
                self._set_lr(runner, self.regular_lr)
            elif cur_iter >= self.fixed_iters:
                warmup_lr = self.get_warmup_lr(cur_iter - self.fixed_iters)
                self._set_lr(runner, warmup_lr)
            else:
                fixed_lr = self.get_fixed_lr()
                self._set_lr(runner, fixed_lr)
