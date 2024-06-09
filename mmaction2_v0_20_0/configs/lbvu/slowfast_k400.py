_base_ = [
    '../_base_/models/slowfast_r50.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(cls_head=dict(num_classes=400))

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/data_video/k400_train'
data_root_val = '/data_video/k400_val'
data_root_compress = '/data_video/k400_train'
data_root_val_compress= '/data_video/k400_val'
ann_file_train = '/data_video/code/lbvu/data_process/k400_ds/kinetics400_mmaction_video.txt'
ann_file_val = '/data_video/code/lbvu/data_process/k400_ds/kinetics400_mmaction_video.txt'
ann_file_test = '/data_video/code/lbvu/data_process/k400_ds/kinetics400_mmaction_video.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs',"imgs_compress","img_norm_cfg", 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs',"imgs_compress"])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs',"imgs_compress","img_norm_cfg", 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs',"imgs_compress"])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs',"imgs_compress", "img_norm_cfg",'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs',"imgs_compress"])
]
# data_name_tmpl = '{:06}.jpg'
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=8),
    train=dict(

        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        data_prefix_compress=data_root_compress,
        pipeline=train_pipeline),
    val=dict(

        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        data_prefix_compress=data_root_compress,
        pipeline=val_pipeline),
    test=dict(

        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        data_prefix_compress=data_root_compress,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[15, 30])
total_epochs = 40

# runtime settings
# work_dir = './work_dirs/slowfast_k400_pretrained_r50_4x16x1_40e_ucf101_rgb'
# load_from = 'https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb/slowfast_r50_4x16x1_256e_kinetics400_rgb_20210722-04e43ed4.pth'  # noqa: E501
# find_unused_parameters = True
