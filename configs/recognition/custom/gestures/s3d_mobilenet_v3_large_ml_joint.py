# global parameters
num_videos_per_gpu = 14
num_workers_per_gpu = 3
train_sources = 'jester', 'iso_gd', 'msasl'
test_sources = 'jester', 'iso_gd', 'msasl'

root_dir = 'data'
work_dir = None
load_from = None
resume_from = None
reset_layer_prefixes = ['cls_head']
reset_layer_suffixes = None

# model settings
input_img_size = 224
clip_len = 16
trg_fps = 15

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='MobileNetV3_S3D',
        num_input_layers=3,
        mode='large',
        pretrained=None,
        pretrained2d=False,
        width_mult=1.0,
        pool1_stride_t=1,
        # block ids:      0  1  2  3  4  5  6  7  8  9  10 11 12 13 14
        # spatial strides 1  2  1  2  1  1  2  1  1  1  1  1  1  2  1
        temporal_strides=(1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1),
        temporal_kernels=(5, 3, 3, 3, 3, 5, 5, 3, 3, 5, 3, 3, 3, 3, 3),
        use_dw_temporal= (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        use_temporal_avg_pool=True,
        out_conv=True,
        dropout_cfg=dict(
            dist='gaussian',
            p=0.1,
            mu=0.1,
            sigma=0.03,
        ),
    ),
    reducer=dict(
        type='AggregatorSpatialTemporalModule',
        modules=[
            dict(type='AverageSpatialTemporalModule',
                 temporal_size=4,
                 spatial_size=7),
        ],
    ),
    cls_head=dict(
        type='ClsHead',
        num_classes=101,
        temporal_size=1,
        spatial_size=1,
        dropout_ratio=None,
        in_channels=960,
        embedding=True,
        embd_size=256,
        num_centers=1,
        st_scale=10.0,
        reg_weight=1.0,
        reg_threshold=0.1,
        enable_sampling=False,
        adaptive_sampling=False,
        sampling_angle_std=3.14 / 2 / 5,
        enable_class_mixing=False,
        class_mixing_alpha=0.2,
        loss_cls=dict(
            type='AMSoftmaxLoss',
            target_loss='ce',
            scale_cfg=dict(
                type='PolyScalarScheduler',
                start_scale=30.0,
                end_scale=5.0,
                power=1.2,
                num_epochs=41.276,
            ),
            pr_product=False,
            margin_type='cos',
            margin=0.35,
            gamma=0.0,
            t=1.0,
            conf_penalty_weight=0.085,
            filter_type='positives',
            top_k=None,
            enable_class_weighting=False,
            enable_adaptive_margins=False,
        ),
        losses_extra=dict(
            loss_lpush=dict(
                type='LocalPushLoss',
                margin=0.1,
                weight=1.0,
                smart_margin=True,
            ),
        ),
    ),
)

# model training and testing settings
train_cfg = dict(
    self_challenging=dict(enable=False, drop_p=0.33),
    clip_mixing=dict(enable=False, mode='logits', weight=0.2)
)
test_cfg = dict(
    average_clips=None
)

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_bgr=False
)
train_pipeline = [
    dict(type='StreamSampleFrames',
         clip_len=clip_len,
         trg_fps=trg_fps,
         num_clips=1,
         temporal_jitter=True,
         min_intersection=1.0),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomRotate', delta=10, prob=0.5),
    dict(type='RatioPreservingCrop',
         input_size=input_img_size,
         scale_limits=(1, 0.875)),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='MapFlippedLabels', map_file=dict(jester='flip_labels_map.txt')),
    dict(type='BlockDropout', scale=0.2, prob=0.1),
    dict(type='PhotometricDistortion',
         brightness_range=(65, 190),
         contrast_range=(0.6, 1.4),
         saturation_range=(0.7, 1.3),
         hue_delta=18),
    dict(type='MixUp',  annot='imagenet_train_list.txt', imgs_root='imagenet/train', alpha=0.5, beta=10.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', targets=['imgs']),
    dict(type='Collect', keys=['imgs', 'label', 'dataset_id'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'dataset_id'])
]
val_pipeline = [
    dict(type='StreamSampleFrames',
         clip_len=clip_len,
         trg_fps=trg_fps,
         num_clips=1,
         test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=input_img_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'dataset_id'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'dataset_id'])
]
data = dict(
    videos_per_gpu=num_videos_per_gpu,
    workers_per_gpu=num_workers_per_gpu,
    train_dataloader=dict(
        drop_last=True
    ),
    shared=dict(
        type='StreamDataset',
        data_subdir='global_crops',
        filename_tmpl=['{:05d}.jpg', '{:05d}.jpg', 'img_{:05d}.jpg']
    ),
    train=dict(
        source=train_sources,
        ann_file='train.txt',
        pipeline=train_pipeline,
    ),
    val=dict(
        source=test_sources,
        ann_file='val.txt',
        pipeline=val_pipeline
    ),
    test=dict(
        source=test_sources,
        ann_file='val.txt',
        pipeline=val_pipeline
    )
)

# optimizer
optimizer = dict(
    type='SGD',
    lr=1e-2,
    momentum=0.9,
    weight_decay=1e-4
)
optimizer_config = dict(
    grad_clip=dict(
        max_norm=40,
        norm_type=2
    )
)

# parameter manager
params_config = dict(
    type='FreezeLayers',
    epochs=5,
    open_layers=['cls_head']
)

# learning policy
lr_config = dict(
    policy='customstep',
    step=[30, 50],
    fixed='constant',
    fixed_epochs=5,
    fixed_ratio=1.0,
    gamma=0.1,
    warmup='linear',
    warmup_epochs=5,
    warmup_ratio=1e-2,
)
total_epochs = 65

# workflow
workflow = [('train', 1)]
checkpoint_config = dict(
    interval=1
)
evaluation = dict(
    interval=1,
    metrics=['top_k_accuracy', 'mean_class_accuracy', 'ranking_mean_average_precision'],
    topk=(1, 5),
)

log_level = 'INFO'
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)

# runtime settings
dist_params = dict(
    backend='nccl'
)
find_unused_parameters = True
