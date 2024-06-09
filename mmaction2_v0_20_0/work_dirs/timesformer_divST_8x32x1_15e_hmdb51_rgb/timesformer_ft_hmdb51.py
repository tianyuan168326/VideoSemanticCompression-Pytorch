checkpoint_config = dict(interval=1)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb/timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth'
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='TimeSformer',
        pretrained=
        'https://download.openmmlab.com/mmaction/recognition/timesformer/vit_base_patch16_224.pth',
        num_frames=8,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.0,
        transformer_layers=None,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-06)),
    cls_head=dict(type='TimeSformerHead', num_classes=101, in_channels=768),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
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
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
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
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs', 'imgs_compress', 'img_norm_cfg', 'label'],
        meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'imgs_compress', 'label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs', 'imgs_compress', 'img_norm_cfg', 'label'],
        meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'imgs_compress', 'label'])
]
data_name_tmpl = 'img_{:05}.jpg'
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
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
                type='SampleFrames',
                clip_len=8,
                frame_interval=32,
                num_clips=1),
            dict(type='RawFrameDecode'),
            dict(type='RandomRescale', scale_range=(256, 320)),
            dict(type='RandomCrop', size=224),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
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
                clip_len=8,
                frame_interval=32,
                num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(
                type='Collect',
                keys=['imgs', 'imgs_compress', 'img_norm_cfg', 'label'],
                meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'imgs_compress', 'label'])
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
                clip_len=8,
                frame_interval=32,
                num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 224)),
            dict(type='ThreeCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(
                type='Collect',
                keys=['imgs', 'imgs_compress', 'img_norm_cfg', 'label'],
                meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'imgs_compress', 'label'])
        ]))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])
optimizer = dict(
    type='SGD',
    lr=0.0005,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys=dict({
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        })),
    weight_decay=0.0001,
    nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[5, 10])
total_epochs = 15
work_dir = './work_dirs/timesformer_divST_8x32x1_15e_hmdb51_rgb'
find_unused_parameters = True
gpu_ids = range(0, 8)
omnisource = False
module_hooks = []
