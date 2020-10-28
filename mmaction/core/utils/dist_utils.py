from collections import OrderedDict

import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, _take_tensors
from mmcv.runner import OptimizerHook


def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)


def allreduce_tensors(tensors, coalesce=True, bucket_size_mb=-1):
    world_size = dist.get_world_size()
    if coalesce:
        _allreduce_coalesced(tensors, world_size, bucket_size_mb)
    else:
        for tensor in tensors:
            dist.all_reduce(tensor.div_(world_size))


class DistOptimizerHook(OptimizerHook):
    def __init__(self, grad_clip=None, coalesce=True, bucket_size_mb=-1):
        super().__init__(grad_clip)

        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb

    def after_epoch(self, runner):
        tensors = [t for n, t in runner.model.named_buffers() if 'num_batches_tracked' not in n]
        allreduce_tensors(tensors, self.coalesce, self.bucket_size_mb)
