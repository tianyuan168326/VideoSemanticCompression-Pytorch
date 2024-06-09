model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetTSM',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False,
        shift_div=8,
        num_segments=8,
        test_mode='hd',
        test_compress_q=35),
    cls_head=dict(
        type='TSMHead',
        num_classes=51,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True,
        num_segments=8),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
optimizer = dict(
    type='SGD',
    constructor='TSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    lr=0.00015,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 50
checkpoint_config = dict(interval=1)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/data_video/code/lbvu/mmaction/checkpoints/tsm_k400_pretrained_r50_1x1x8_25e_hmdb51_rgb_20210630-10c74ee5.pth'
resume_from = None
workflow = [('train', 1)]
dataset_type = 'RawframeDataset'
data_root = '/data_video/code/lbvu/mmaction/data/hmdb51/rawframes'
data_root_val = '/data_video/code/lbvu/mmaction/data/hmdb51/rawframes'
data_root_compress = '/data_video/code/lbvu/mmaction/data/hmdb51/rawframes'
data_root_val_compress = '/data_video/code/lbvu/mmaction/data/hmdb51/rawframes'
split = 1
ann_file_train = '/data_video/code/lbvu/mmaction/tools/data/hmdb51/hmdb51_train_split_1_rawframes.txt'
ann_file_val = '/data_video/code/lbvu/mmaction/tools/data/hmdb51/hmdb51_val_split_1_rawframes.txt'
ann_file_test = '/data_video/code/lbvu/mmaction/tools/data/hmdb51/hmdb51_val_split_1_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(
        type='Collect',
        keys=['imgs', 'imgs_compress', 'img_norm_cfg', 'label'],
        meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'imgs_compress', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(
        type='Collect',
        keys=['imgs', 'imgs_compress', 'img_norm_cfg', 'label'],
        meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'imgs_compress'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(
        type='Collect',
        keys=['imgs', 'imgs_compress', 'img_norm_cfg', 'label'],
        meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'imgs_compress'])
]
data_name_tmpl = 'img_{:05}.jpg'
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=8),
    train=dict(
        type='RawframeDataset',
        ann_file=
        '/data_video/code/lbvu/mmaction/tools/data/hmdb51/hmdb51_train_split_1_rawframes.txt',
        data_prefix='/data_video/code/lbvu/mmaction/data/hmdb51/rawframes',
        data_prefix_compress=
        '/data_video/code/lbvu/mmaction/data/hmdb51/rawframes',
        filename_tmpl='img_{:05}.jpg',
        pipeline=[
            dict(
                type='SampleFrames', clip_len=1, frame_interval=1,
                num_clips=8),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(
                type='MultiScaleCrop',
                input_size=224,
                scales=(1, 0.875, 0.75, 0.66),
                random_crop=False,
                max_wh_scale_gap=1,
                num_fixed_crops=13),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(
                type='Collect',
                keys=['imgs', 'imgs_compress', 'img_norm_cfg', 'label'],
                meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'imgs_compress', 'label'])
        ]),
    val=dict(
        type='RawframeDataset',
        ann_file=
        '/data_video/code/lbvu/mmaction/tools/data/hmdb51/hmdb51_val_split_1_rawframes.txt',
        data_prefix='/data_video/code/lbvu/mmaction/data/hmdb51/rawframes',
        data_prefix_compress=
        '/data_video/code/lbvu/mmaction/data/hmdb51/rawframes',
        filename_tmpl='img_{:05}.jpg',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=8,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(
                type='Collect',
                keys=['imgs', 'imgs_compress', 'img_norm_cfg', 'label'],
                meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'imgs_compress'])
        ]),
    test=dict(
        type='RawframeDataset',
        ann_file=
        '/data_video/code/lbvu/mmaction/tools/data/hmdb51/hmdb51_val_split_1_rawframes.txt',
        data_prefix='/data_video/code/lbvu/mmaction/data/hmdb51/rawframes',
        data_prefix_compress=
        '/data_video/code/lbvu/mmaction/data/hmdb51/rawframes',
        filename_tmpl='img_{:05}.jpg',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=8,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(
                type='Collect',
                keys=['imgs', 'imgs_compress', 'img_norm_cfg', 'label'],
                meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'imgs_compress'])
        ]))
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
find_unused_parameters = True
work_dir = './work_dirs/tsm_r50_1x1x8_50e_hmdb51_rgb/'
gpu_ids = range(0, 2)
omnisource = False
module_hooks = []
