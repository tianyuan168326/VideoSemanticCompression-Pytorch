_base_ = [
    '../_base_/models/slowonly_r50.py', '../_base_/default_runtime.py'
]

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
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
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
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs',"imgs_compress", 'img_norm_cfg', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs',"imgs_compress"])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=24),
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
        data_prefix_compress=data_root_val_compress,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        data_prefix_compress=data_root_val_compress,
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[90, 130],
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=10)
total_epochs = 150

# runtime settings
checkpoint_config = dict(interval=4)
work_dir = ('./work_dirs/slowonly_imagenet_pretrained_r50_8x8x1_150e'
            '_kinetics400_rgb')
find_unused_parameters = False


# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=1600 tools/dist_test.sh 
# /data_video/code/lbvu/mmaction/configs/lbvu/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb.py 
# https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb_20200912-3f9ce182.pth
# 4 --eval top_k_accuracy --cfg-options model.backbone.test_mode=hd model.backbone.test_compress_q=39