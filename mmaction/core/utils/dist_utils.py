from collections import OrderedDict
from typing import Union, Iterable

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, _take_tensors
from mmcv.runner import Hook, HOOKS

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
    if len(shape) <= 1:
        norms = torch.abs(x)
    else:
        sum_dims = tuple(range(1, len(shape)))
        norms = torch.sqrt(torch.sum(x ** 2, dim=sum_dims, keepdim=True))

    return norms


@HOOKS.register_module()
class DistOptimizerHook(Hook):
    def __init__(self, grad_clip=None, coalesce=True, bucket_size_mb=-1):
        super().__init__()

        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb

    def after_epoch(self, runner):
        tensors = [t for n, t in runner.model.named_buffers() if 'num_batches_tracked' not in n]
        allreduce_tensors(tensors, self.coalesce, self.bucket_size_mb)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()

        if self.grad_clip is not None:
            grad_norm_info = self.clip_grads(runner.model.parameters())
            if len(grad_norm_info) > 0:
                runner.log_buffer.update(grad_norm_info, runner.outputs['num_samples'])

        runner.optimizer.step()

    def clip_grads(self, params):
        assert self.grad_clip is not None

        grads_info = dict()

        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) == 0:
            return grads_info

        method = self.grad_clip.get('method', 'default')
        grad_clip_cfg = {_key: _value for _key, _value in self.grad_clip.items() if _key != 'method'}

        if method == 'default':
            grads_info.update(self._default_clip_grad_norm(params, **grad_clip_cfg))
        elif method == 'adaptive':
            grads_info.update(self._adaptive_clip_grad_norm(params, **grad_clip_cfg))
        else:
            ValueError(f'Unknown gradient clipping method: {method}')

        return grads_info

    @staticmethod
    def _default_clip_grad_norm(parameters: _tensor_or_tensors, **kwargs) -> dict:
        out_info = dict()

        grad_norm = clip_grad.clip_grad_norm_(parameters, **kwargs)
        if grad_norm is not None:
            out_info['grad_norm'] = float(grad_norm)

        return out_info

    @staticmethod
    def _adaptive_clip_grad_norm(parameters: _tensor_or_tensors, clip: float) -> dict:
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        all_num_invalids, all_invalid_clip_coef = [], []
        total_num_elements = 0
        for p in parameters:
            with torch.no_grad():
                p_norms = _unit_wise_norm(p)
                g_norms = _unit_wise_norm(p.grad)

                max_p_norms = float(clip) * p_norms.clamp_min(1e-3)
                max_g_norms = g_norms.clamp_min(1e-6)

                scales = max_p_norms / max_g_norms
                invalid_mask = g_norms > max_p_norms
                clip_coef = torch.where(invalid_mask, scales, torch.ones_like(scales))

                num_invalids = torch.sum(invalid_mask).float().item()
                all_invalid_clip_coef.append(torch.sum(scales[invalid_mask]).float().item() / max(1.0, num_invalids))
                all_num_invalids.append(num_invalids)
                total_num_elements += invalid_mask.nelement()

            p.grad.detach().mul_(clip_coef)

        out_info = {
            'invalid_grad_scale': sum(all_invalid_clip_coef) / float(max(1, len(all_invalid_clip_coef))),
            'invalid_grad_ratio': sum(all_num_invalids) / float(max(1, total_num_elements)),
        }

        return out_info
