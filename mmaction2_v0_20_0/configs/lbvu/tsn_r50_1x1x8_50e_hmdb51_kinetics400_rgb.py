_base_ = [
    '../_base_/models/tsn_r50.py', '../_base_/schedules/sgd_50e.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(cls_head=dict(num_classes=51))

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
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', "imgs_compress", 'img_norm_cfg','label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs',"imgs_compress", 'label'])
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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', "imgs_compress", 'img_norm_cfg','label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', "imgs_compress"])
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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', "imgs_compress", 'img_norm_cfg','label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs',"imgs_compress"])
]
data_name_tmpl = 'img_{:05}.jpg'
data = dict(
    videos_per_gpu=12,
    workers_per_gpu=4,
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
        data_prefix_compress=data_root_compress,
        filename_tmpl=data_name_tmpl,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        data_prefix_compress=data_root_compress,
        filename_tmpl=data_name_tmpl,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))

# optimizer
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0001)

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/tsn_r50_1x1x8_50e_hmdb51_kinetics400_rgb/'
load_from = 'https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x8_100e_kinetics400_rgb/tsn_r50_256p_1x1x8_100e_kinetics400_rgb_20200817-883baf16.pth'  # noqa: E501
gpu_ids = range(0, 1)
