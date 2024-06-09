_base_ = [
    '../_base_/models/tsm_r50.py', '../_base_/schedules/sgd_tsm_50e.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(cls_head=dict(num_classes=51),
backbone=dict(test_mode = 'hd',test_compress_q = -1))

# model = dict(cls_head=dict(num_classes=51),
# backbone=dict())

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '../VideoDatasetList/hmdb51_extracted'
# data_root = '/data/video_datasets/hmdb51_extracted'

data_root_val = data_root
data_root_compress = data_root
data_root_val_compress= data_root
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'tools/data/hmdb51/hmdb51_train_split_1_rawframes.txt'
ann_file_val = f'tools/data/hmdb51/hmdb51_val_split_1_rawframes.txt'
ann_file_test = f'tools/data/hmdb51/hmdb51_val_split_1_rawframes.txt'

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
    dict(type='CenterCrop', crop_size=224),
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
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', "imgs_compress", 'img_norm_cfg','label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs',"imgs_compress"])
]
data_name_tmpl = 'img_{:05}.jpg'
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=64),
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
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        data_prefix_compress=data_root_val_compress,
        filename_tmpl=data_name_tmpl,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy'])

# optimizer
optimizer = dict(weight_decay=0.0001,lr=0.00015) ### 2 card and fine-tune from pretrained
find_unused_parameters = True
# runtime settings
work_dir = './work_dirs/tsm_r50_1x1x8_50e_hmdb51_rgb/'
### tesing
load_from="../pretrain_models/action/tsm_r50_1x1x8_50e_hmdb51_rgb/tsm_hmdb51_best_top1_acc_epoch_10.pth"