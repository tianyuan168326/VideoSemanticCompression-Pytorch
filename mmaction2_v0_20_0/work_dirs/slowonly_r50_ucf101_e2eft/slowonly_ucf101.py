model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained='torchvision://resnet50',
        lateral=False,
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        norm_eval=False,
        test_mode='lbvu_train',
        clip_len=8,
        pretrained_lbvu_path=
        '/data_video/code/lbvu/checkpoints/TSM_50/train_edge01_20f/checkpoint/action_backbone.TSM_k60k_brrand_resnet50_segment8__best_network.pth.tar'
    ),
    cls_head=dict(
        type='I3DHead',
        in_channels=2048,
        num_classes=101,
        spatial_type='avg',
        dropout_ratio=0.5),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
checkpoint_config = dict(interval=1)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_k400_pretrained_r50_8x4x1_40e_ucf101_rgb/slowonly_k400_pretrained_r50_8x4x1_40e_ucf101_rgb_20210630-ee8c850f.pth'
resume_from = None
workflow = [('train', 1)]
dataset_type = 'RawframeDataset'
data_root = '/data_video/ucf101_jpg'
data_root_val = '/data_video/ucf101_jpg'
data_root_compress = '/data_video/ucf101_jpg_crf47/ucf101_jpg/'
data_root_val_compress = '/data_video/ucf101_jpg_crf47/ucf101_jpg/'
split = 1
ann_file_train = '/data_video/code/lbvu/ucf101/train_tsn_01.txt'
ann_file_val = '/data_video/code/lbvu/ucf101/test_tsn_01.txt'
ann_file_test = '/data_video/code/lbvu/ucf101/test_tsn_01_shell.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=4, num_clips=1),
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
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs', 'imgs_compress', 'img_norm_cfg', 'label'],
        meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'imgs_compress', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=4,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs', 'imgs_compress', 'img_norm_cfg', 'label'],
        meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'imgs_compress'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=4,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs', 'imgs_compress', 'img_norm_cfg', 'label'],
        meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'imgs_compress'])
]
data_name_tmpl = '{:06}.jpg'
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=2),
    train=dict(
        filename_tmpl='{:06}.jpg',
        type='RawframeDataset',
        ann_file='/data_video/code/lbvu/ucf101/train_tsn_01.txt',
        data_prefix_compress='/data_video/ucf101_jpg_crf47/ucf101_jpg/',
        data_prefix='/data_video/ucf101_jpg',
        pipeline=[
            dict(
                type='SampleFrames', clip_len=8, frame_interval=4,
                num_clips=1),
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
            dict(type='FormatShape', input_format='NCTHW'),
            dict(
                type='Collect',
                keys=['imgs', 'imgs_compress', 'img_norm_cfg', 'label'],
                meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'imgs_compress', 'label'])
        ]),
    val=dict(
        filename_tmpl='{:06}.jpg',
        type='RawframeDataset',
        ann_file='/data_video/code/lbvu/ucf101/test_tsn_01.txt',
        data_prefix='/data_video/ucf101_jpg',
        data_prefix_compress='/data_video/ucf101_jpg_crf47/ucf101_jpg/',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=4,
                num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(
                type='Collect',
                keys=['imgs', 'imgs_compress', 'img_norm_cfg', 'label'],
                meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'imgs_compress'])
        ]),
    test=dict(
        type='RawframeDataset',
        ann_file='/data_video/code/lbvu/ucf101/test_tsn_01_shell.txt',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=4,
                num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(
                type='Collect',
                keys=['imgs', 'imgs_compress', 'img_norm_cfg', 'label'],
                meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'imgs_compress'])
        ],
        filename_tmpl='{:06}.jpg',
        data_prefix='/data_video/ucf101_jpg',
        data_prefix_compress='/data_video/ucf101_jpg_crf47/ucf101_jpg/'))
evaluation = dict(interval=4, metrics=['top_k_accuracy'])
optimizer = dict(type='SGD', lr=0.0001, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', step=[10, 20])
total_epochs = 25
find_unused_parameters = True
work_dir = './work_dirs/slowonly_r50_ucf101_e2eft'
gpu_ids = range(0, 8)
omnisource = False
module_hooks = []
