_base_ = [
    '../_base_/models/slowonly_r50.py', '../_base_/schedules/sgd_50e.py',
    '../_base_/default_runtime.py'
]
##slowonly_k400_pretrained_r50_8x4x1_40e_ucf101_rgb
# model settings
model = dict(cls_head=dict(num_classes=51),backbone=dict(test_mode = 'hd'))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/data_video/hmdb51_extracted'
data_root_val = '/data_video/hmdb51_extracted'
data_root_compress = '/data_video/hmdb51_extracted'
data_root_val_compress= '/data_video/hmdb51_extracted'
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'/data_video/code/lbvu/mmaction/tools/data/hmdb51/hmdb51_train_split_1_rawframes.txt'
ann_file_val = f'/data_video/code/lbvu/mmaction/tools/data/hmdb51/hmdb51_val_split_1_rawframes.txt'
ann_file_test = f'/data_video/code/lbvu/mmaction/tools/data/hmdb51/hmdb51_val_split_1_rawframes.txt'


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=4, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs',"imgs_compress",'img_norm_cfg', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs',"imgs_compress" ,'label'])
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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs',"imgs_compress",'img_norm_cfg', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs',"imgs_compress"])
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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', "imgs_compress", 'img_norm_cfg','label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs',"imgs_compress"])
]
data_name_tmpl = 'img_{:05}.jpg'
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=4),
    train=dict(
        filename_tmpl=data_name_tmpl,
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix_compress=data_root_compress,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        filename_tmpl=data_name_tmpl,
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        data_prefix_compress=data_root_val_compress,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        filename_tmpl=data_name_tmpl,
        data_prefix=data_root_val,
        data_prefix_compress=data_root_val_compress,
        ))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy'])

# optimizer
optimizer = dict(
    lr=0.0001,  # this lr is used for 8 gpus
)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[15, 30])
total_epochs = 40

# runtime settings
work_dir = './work_dirs/slowonly_hmdb_rgb'
load_from = '/data_video/code/lbvu/mmaction/checkpoints/slowonly_k400_pretrained_r50_8x4x1_40e_hmdb51_rgb_20210630-cee5f725.pth'  # noqa: E501
find_unused_parameters = True
