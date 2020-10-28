from abc import ABCMeta, abstractmethod


class BaseScalarScheduler(metaclass=ABCMeta):
    def __init__(self, start_iter=0, iters_per_epoch=1):
        super(BaseScalarScheduler, self).__init__()

        self.iters_per_epoch = iters_per_epoch
        assert self.iters_per_epoch > 0
        self.iter = start_iter
        assert self.iter >= 0

    def get_scale_and_increment_step(self):
        scale = self._get_scale(self.iter, self.iters_per_epoch)
        self.iter += 1

        return scale

    @abstractmethod
    def _get_scale(self, step, iters_per_epoch):
        pass
