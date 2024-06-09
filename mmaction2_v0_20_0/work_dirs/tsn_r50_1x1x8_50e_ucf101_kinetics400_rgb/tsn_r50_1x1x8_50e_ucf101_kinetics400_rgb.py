model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=101,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01),
    train_cfg=None,
    test_cfg=dict(average_clips=None))
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 50
checkpoint_config = dict(interval=5)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x8_100e_kinetics400_rgb/tsn_r50_256p_1x1x8_100e_kinetics400_rgb_20200817-883baf16.pth'
resume_from = None
workflow = [('train', 1)]
split = 1
dataset_type = 'RawframeDataset'
data_root = '/data_video/ucf101_jpg'
data_root_val = '/data_video/ucf101_jpg'
data_root_compress = '/data_video/ucf101_jpg'
data_root_val_compress = '/data_video/ucf101_jpg'
ann_file_train = '/data_video/code/lbvu/VideoDatasetList/ucf101/train_tsn_01.txt'
ann_file_val = '/data_video/code/lbvu/VideoDatasetList/ucf101/test_tsn_01.txt'
ann_file_test = '/data_video/code/lbvu/VideoDatasetList/ucf101/test_tsn_01.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
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
    dict(type='CenterCrop', crop_size=256),
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
    dict(type='CenterCrop', crop_size=256),
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
data_name_tmpl = '{:06}.jpg'
data = dict(
    videos_per_gpu=12,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RawframeDataset',
        ann_file=
        '/data_video/code/lbvu/VideoDatasetList/ucf101/train_tsn_01.txt',
        data_prefix='/data_video/ucf101_jpg',
        data_prefix_compress='/data_video/ucf101_jpg',
        filename_tmpl='{:06}.jpg',
        pipeline=[
            dict(
                type='SampleFrames', clip_len=1, frame_interval=1,
                num_clips=8),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='RandomResizedCrop'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
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
        '/data_video/code/lbvu/VideoDatasetList/ucf101/test_tsn_01.txt',
        data_prefix='/data_video/ucf101_jpg',
        data_prefix_compress='/data_video/ucf101_jpg',
        filename_tmpl='{:06}.jpg',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=8,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=256),
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
        '/data_video/code/lbvu/VideoDatasetList/ucf101/test_tsn_01.txt',
        data_prefix='/data_video/ucf101_jpg',
        data_prefix_compress='/data_video/ucf101_jpg',
        filename_tmpl='{:06}.jpg',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=8,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=256),
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
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
work_dir = './work_dirs/tsn_r50_1x1x8_50e_ucf101_kinetics400_rgb/'
gpu_ids = range(0, 10)
find_unused_parameters = True
omnisource = False
module_hooks = []
