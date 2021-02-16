
TARGET_TYPES = 'MapFlippedLabels', 'MixUp', 'CrossNorm'


def propagate_root_dir(cfg, root_dir=None):
    if root_dir is not None:
        cfg.root_dir = root_dir

    assert cfg.root_dir is not None and cfg.root_dir != ''

    cfg.data.root_dir = cfg.root_dir

    for trg_data_type in ['train', 'val', 'test']:
        for stage in cfg.data[trg_data_type].pipeline:
            if 'type' in stage and stage['type'] in TARGET_TYPES:
                stage['root_dir'] = cfg.root_dir

    return cfg
