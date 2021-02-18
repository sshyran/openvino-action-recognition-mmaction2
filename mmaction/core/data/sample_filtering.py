import torch
import torch.distributed as dist

from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class SampleInfoAggregatorHook(Hook):
    def after_iter(self, runner):
        local_meta = runner.model.module.train_meta
        sync_meta = {meta_name: self._sync(meta_data) for meta_name, meta_data in local_meta.items()}

        dataset = runner.data_loader.dataset
        dataset.update_meta_info(**sync_meta)

    @staticmethod
    def _sync(data):
        if dist.is_available() and dist.is_initialized():
            batch_size = data.size(0)
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            shared_shape = [world_size * batch_size] + list(data.shape[1:])
            shared_data = torch.zeros(shared_shape, dtype=data.dtype, device=data.device)

            shared_data[rank*batch_size:(rank + 1)*batch_size] = data
            dist.all_reduce(shared_data, dist.ReduceOp.SUM)

            out_data = shared_data
        else:
            out_data = data

        return out_data.cpu().numpy()
