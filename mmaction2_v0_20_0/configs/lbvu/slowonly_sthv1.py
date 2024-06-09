_base_ = [
    '../_base_/models/slowonly_r50.py',
    '../_base_/schedules/sgd_150e_warmup.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(backbone=dict(with_pool1=False,test_mode = 'hd',test_compress_q = 35), 
cls_head=dict(num_classes=174))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1'
data_root_val_compress = '/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1'
data_root_compress = '/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1'
data_root_val = '/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1'
ann_file_train = '/data_video/code/lbvu/somethingv1/train.txt'
ann_file_val = '/data_video/code/lbvu/somethingv1/valid.txt'
ann_file_test = '/data_video/code/lbvu/somethingv1/valid.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=4, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(112, 112), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=4,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs',"imgs_compress", 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=4,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 128)),
    # dict(type='ThreeCrop', crop_size=128),
    dict(type='CenterCrop', crop_size=128),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', "imgs_compress", 'img_norm_cfg','label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs',"imgs_compress"])
]

data = dict(
    videos_per_gpu=8,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=16),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        filename_tmpl='{:05}.jpg',
        data_prefix=data_root,
        data_prefix_compress=data_root_compress,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        filename_tmpl='{:05}.jpg',
        data_prefix=data_root_val,
        data_prefix_compress=data_root_val_compress,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        filename_tmpl='{:05}.jpg',
        data_prefix=data_root_val,
        data_prefix_compress=data_root_val_compress,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(lr=0.1)  # this lr is used for 8 gpus
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=10)
total_epochs = 64

# runtime settings
work_dir = './work_dirs/slowonly_r50_8x4x1_64e_sthv1_rgb'
