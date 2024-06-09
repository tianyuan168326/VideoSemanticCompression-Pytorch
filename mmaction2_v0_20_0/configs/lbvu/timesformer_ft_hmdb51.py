_base_ = ['../_base_/default_runtime.py']

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='TimeSformer',
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/recognition/timesformer/vit_base_patch16_224.pth',  # noqa: E501
        num_frames=8,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.,
        transformer_layers=None,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(type='TimeSformerHead', num_classes=101, in_channels=768),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/data_video/code/lbvu/mmaction/data/hmdb51/rawframes'
data_root_val = '/data_video/code/lbvu/mmaction/data/hmdb51/rawframes'
data_root_compress = '/data_video/code/lbvu/mmaction/data/hmdb51/rawframes'
data_root_val_compress= '/data_video/code/lbvu/mmaction/data/hmdb51/rawframes'
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'/data_video/code/lbvu/mmaction/tools/data/hmdb51/hmdb51_train_split_1_rawframes.txt'
ann_file_val = f'/data_video/code/lbvu/mmaction/tools/data/hmdb51/hmdb51_val_split_1_rawframes.txt'
ann_file_test = f'/data_video/code/lbvu/mmaction/tools/data/hmdb51/hmdb51_val_split_1_rawframes.txt'


img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', "imgs_compress", 'img_norm_cfg', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', "imgs_compress", 'label'])
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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', "imgs_compress", 'img_norm_cfg','label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs',"imgs_compress", 'label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs',"imgs_compress", 'img_norm_cfg', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs',"imgs_compress", 'label'])
]

data_name_tmpl = 'img_{:05}.jpg'

data = dict(
    videos_per_gpu=4,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=16),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        data_prefix_compress=data_root_compress,
        filename_tmpl=data_name_tmpl,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        data_prefix_compress=data_root_val_compress,
        filename_tmpl=data_name_tmpl,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        data_prefix_compress=data_root_val_compress,
        filename_tmpl=data_name_tmpl,
        pipeline=test_pipeline))

evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.0005,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        }),
    weight_decay=1e-4,
    nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[5, 10])
total_epochs = 15

# runtime settings
checkpoint_config = dict(interval=1)
load_from='https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb/timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth'
work_dir = './work_dirs/timesformer_divST_8x32x1_15e_hmdb51_rgb'
find_unused_parameters = True