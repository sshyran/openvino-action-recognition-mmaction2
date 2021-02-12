from collections import OrderedDict
from typing import Union, Iterable

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, _take_tensors
from mmcv.runner import OptimizerHook

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


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


def _unit_wise_norm(x):
    shape = x.size()
    sum_dims = tuple(range(1, len(shape)))
    norms = torch.sqrt(torch.sum(x ** 2, dim=sum_dims, keepdim=True))

    return norms


class DistOptimizerHook(OptimizerHook):
    def __init__(self, grad_clip=None, coalesce=True, bucket_size_mb=-1):
        super().__init__(grad_clip)

        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb

    def after_epoch(self, runner):
        tensors = [t for n, t in runner.model.named_buffers() if 'num_batches_tracked' not in n]
        allreduce_tensors(tensors, self.coalesce, self.bucket_size_mb)

    def clip_grads(self, params):
        assert self.grad_clip is not None

        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) == 0:
            return None

        method = self.grad_clip.get('method', 'default')
        grad_clip_cfg = {_key: _value for _key, _value in self.grad_clip.items() if _key != 'method'}

        if method == 'default':
            return clip_grad.clip_grad_norm_(params, **grad_clip_cfg)
        elif method == 'adaptive':
            return self._adaptive_clip_grad_norm(params, **grad_clip_cfg)
        else:
            ValueError(f'Unknown gradient clipping method: {method}')

    @staticmethod
    def _adaptive_clip_grad_norm(parameters: _tensor_or_tensors, clip: float) -> torch.Tensor:
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        all_clip_coef = []
        for p in parameters:
            with torch.no_grad():
                p_norms = _unit_wise_norm(p)
                g_norms = _unit_wise_norm(p.grad)

                max_p_norms = float(clip) * p_norms.clamp_min(1e-3)
                max_g_norms = g_norms.clamp_min(1e-6)

                clip_coef = torch.where(g_norms > max_p_norms,
                                        max_p_norms / max_g_norms,
                                        torch.ones_like(g_norms))
                all_clip_coef.append(torch.mean(clip_coef))

            p.grad.detach().mul_(clip_coef)

        return sum(all_clip_coef) / float(max(1, len(all_clip_coef)))
