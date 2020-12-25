import torch.nn as nn
from mmcv.runner import _load_checkpoint
from terminaltables import AsciiTable


def _is_compatible(shape_a, shape_b):
    return shape_a[0] == shape_b[0] \
           and shape_a[1] == shape_b[1] \
           and shape_a[3] == shape_b[3] \
           and shape_a[4] == shape_b[4]


def inflate_weights(model, state_dict_2d, logger=None):
    if isinstance(state_dict_2d, str):
        state_dict_2d = _load_checkpoint(state_dict_2d, map_location='cpu')
        if 'state_dict' in state_dict_2d:
            state_dict_2d = state_dict_2d['state_dict']
    assert isinstance(state_dict_2d, dict)

    missing_keys = []
    shape_mismatch_pairs = []
    shape_inflated_pairs = []
    copied_pairs = []

    for name, module in model.named_modules():
        trg_name = name + '.weight'

        if isinstance(module, nn.Conv3d) and trg_name in state_dict_2d:
            old_weight = state_dict_2d[name + '.weight'].data
            old_weight_shape = old_weight.shape
            assert len(old_weight_shape) in [2, 4]

            if len(old_weight.shape) == 2:
                old_weight = old_weight.unsqueeze(2).unsqueeze(3)
            old_weight = old_weight.unsqueeze(2)

            if not _is_compatible(old_weight.shape, module.weight.data.shape):
                shape_mismatch_pairs.append([name, list(old_weight_shape), list(module.weight.size())])
                continue

            new_weight = old_weight.expand_as(module.weight) / module.weight.data.shape[2]
            module.weight.data.copy_(new_weight)
            shape_inflated_pairs.append([trg_name, list(old_weight_shape), list(module.weight.size())])

            if hasattr(module, 'bias') and module.bias is not None:
                trg_name = name + '.bias'
                new_bias = state_dict_2d[trg_name].data
                module.bias.data.copy_(new_bias)
                copied_pairs.append([trg_name, list(new_bias.size())])
        elif isinstance(module, nn.BatchNorm3d) and trg_name in state_dict_2d:
            for attr_name in ['weight', 'bias', 'running_mean', 'running_var']:
                trg_name = name + '.' + attr_name
                old_attr = state_dict_2d[trg_name].data

                new_attr = getattr(module, attr_name)
                new_attr.data.copy_(old_attr)
                copied_pairs.append([trg_name, list(new_attr.size())])
        elif isinstance(module, (nn.Conv3d, nn.BatchNorm3d)):
            missing_keys.append(name)

    if len(missing_keys) > 0:
        msg = 'Missing keys in source state_dict: {}\n'.format(', '.join(missing_keys))
        if logger is not None:
            logger.warning(msg)

    if shape_mismatch_pairs:
        header = ['key', '2d shape', '3d shape']
        table_data = [header] + shape_mismatch_pairs
        table = AsciiTable(table_data)
        if logger is not None:
            logger.warning('These keys have mismatched shape:\n' + table.table)

    if copied_pairs:
        header = ['key', 'shape']
        table_data = [header] + copied_pairs
        table = AsciiTable(table_data)
        if logger is not None:
            logger.info('These keys have been copied:\n' + table.table)

    if shape_inflated_pairs:
        header = ['key', '2d shape', '3d shape']
        table_data = [header] + shape_inflated_pairs
        table = AsciiTable(table_data)
        if logger is not None:
            logger.info('These keys have been shape inflated:\n' + table.table)
