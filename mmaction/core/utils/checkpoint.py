import torch
from terminaltables import AsciiTable
from mmcv.runner.checkpoint import _load_checkpoint
from mmcv.runner.dist_utils import get_dist_info


def load_state_dict(module, state_dict, strict=False, logger=None, force_matching=False,
                    show_converted=False, ignore_prefixes=None, ignore_suffixes=None):
    rank, _ = get_dist_info()

    unexpected_keys = []
    converted_pairs = []
    shape_mismatch_pairs = []
    shape_casted_pairs = []

    own_state = module.state_dict()
    for name, param in state_dict.items():
        ignored_prefix = ignore_prefixes is not None and name.startswith(ignore_prefixes)
        ignored_suffix = ignore_suffixes is not None and name.endswith(ignore_suffixes)
        if ignored_prefix or ignored_suffix:
            continue

        if name not in own_state:
            unexpected_keys.append(name)
            continue

        if isinstance(param, torch.nn.Parameter):
            param = param.data

        src_shape = param.size()
        trg_shape = own_state[name].size()
        if src_shape != trg_shape:
            is_valid = False
            if force_matching:
                is_valid = len(src_shape) == len(trg_shape)
                for i in range(len(src_shape)):
                    is_valid &= src_shape[i] >= trg_shape[i]

            if is_valid:
                ind = [slice(0, d) for d in list(trg_shape)]
                own_state[name].copy_(param[ind])

                shape_casted_pairs.append([name, list(own_state[name].size()), list(param.size())])
            else:
                shape_mismatch_pairs.append([name, list(own_state[name].size()), list(param.size())])
        else:
            own_state[name].copy_(param)
            if show_converted:
                converted_pairs.append([name, list(own_state[name].size())])

    missing_keys = list(set(own_state.keys()) - set(state_dict.keys()))

    err_msg = []
    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))

    if shape_mismatch_pairs:
        casted_info = 'these keys have mismatched shape:\n'
        header = ['key', 'expected shape', 'loaded shape']
        table_data = [header] + shape_mismatch_pairs
        table = AsciiTable(table_data)
        err_msg.append(casted_info + table.table)

    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)

    ok_message = []
    if converted_pairs:
        converted_info = 'These keys have been matched correctly:\n'
        header = ['key', 'shape']
        table_data = [header] + converted_pairs
        table = AsciiTable(table_data)
        ok_message.append(converted_info + table.table)

    if len(ok_message) > 0 and rank == 0:
        ok_message = '\n'.join(ok_message)
        if logger is not None:
            logger.info(ok_message)

    warning_msg = []
    if shape_casted_pairs:
        casted_info = 'these keys have been shape casted:\n'
        header = ['key', 'expected shape', 'loaded shape']
        table_data = [header] + shape_casted_pairs
        table = AsciiTable(table_data)
        warning_msg.append(casted_info + table.table)

    if len(warning_msg) > 0 and rank == 0:
        warning_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        warning_msg = '\n'.join(warning_msg)
        if logger is not None:
            logger.warning(warning_msg)


def load_checkpoint(model, filename, map_location='cpu', strict=False, logger=None,
                    force_matching=False, show_converted=False,
                    ignore_prefixes=None, ignore_suffixes=None):
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')

    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}

    model = model.module if hasattr(model, 'module') else model
    load_state_dict(model, state_dict,
                    strict, logger, force_matching, show_converted,
                    ignore_prefixes, ignore_suffixes)

    return checkpoint
