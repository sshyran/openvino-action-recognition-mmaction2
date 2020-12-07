from mmcv.runner.hooks import HOOKS

from .base_lr_hook import BaseLrUpdaterHook


@HOOKS.register_module()
class CustomstepLrUpdaterHook(BaseLrUpdaterHook):
    def __init__(self, step, gamma=0.1, **kwargs):
        super(CustomstepLrUpdaterHook, self).__init__(**kwargs)

        assert isinstance(step, (list, int))
        if isinstance(step, list):
            for s in step:
                assert isinstance(s, int) and s > 0
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')

        self.step = step
        self.gamma = gamma

    def get_lr(self, runner, base_lr):
        progress = runner.epoch

        if isinstance(self.step, int):
            return base_lr * (self.gamma**(progress // self.step))

        exp = len(self.step)
        for i, s in enumerate(self.step):
            if progress < s:
                exp = i
                break

        return base_lr * self.gamma**exp
