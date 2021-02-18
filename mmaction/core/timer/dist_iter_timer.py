import time

import torch
import torch.distributed as dist

from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module(force=True)
class DistIterTimerHook(Hook):
    def before_epoch(self, runner):
        self.t = time.time()

    def before_iter(self, runner):
        runner.log_buffer.update({'data_time': self._sync(time.time() - self.t)})

    def after_iter(self, runner):
        runner.log_buffer.update({'time': self._sync(time.time() - self.t)})
        self.t = time.time()

    @staticmethod
    def _sync(cpu_value):
        if dist.is_available() and dist.is_initialized():
            gpu_value = torch.tensor(cpu_value).cuda()
            dist.all_reduce(gpu_value, dist.ReduceOp.MAX)

            out_value = gpu_value.item()
        else:
            out_value = cpu_value

        return out_value
