
TARGET_TYPES = 'MapFlippedLabels', 'MixUp', 'CrossNorm'


def _propagate_data_pipeline(pipeline, root_dir):
    for stage in pipeline:
        if 'type' not in stage:
            continue

        if stage['type'] == 'ProbCompose':
            _propagate_data_pipeline(stage['transforms'], root_dir)
        elif stage['type'] in TARGET_TYPES:
            stage['root_dir'] = root_dir


def propagate_root_dir(cfg, root_dir=None):
    if root_dir is not None:
        cfg.root_dir = root_dir

    assert cfg.root_dir is not None and cfg.root_dir != ''

    cfg.data.root_dir = cfg.root_dir

    for trg_data_type in ['train', 'val', 'test']:
        _propagate_data_pipeline(cfg.data[trg_data_type].pipeline, cfg.root_dir)

    return cfg
