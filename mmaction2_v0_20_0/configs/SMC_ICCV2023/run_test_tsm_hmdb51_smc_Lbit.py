import os

for codec_type in ['h266']:
    for run_mode in ['smc_iccv2023']:
        if run_mode == 'raw':
            crfs = [47,43,39,35]
        elif run_mode == 'smc_iccv2023':
            crfs = [23,27,31,35]
        for crf in crfs:
            cmd = '''
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=2024 tools/dist_test.sh \
configs/lbvu/tsm_hmdb.py \
/data_video/code/masked_object_vcs/mmaction/mmaction/work_dirs/tsm_r50_1x1x8_50e_hmdb51_rgb/best_top1_acc_epoch_10.pth 4 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode={run_mode} model.backbone.test_codec={codec_type} model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.pretrained_lbvu_path=/data_video/code/lbvu/Important_models/SMC_ICCV2023/baseline_bpp4_mae0_8f.pth.tar \
model.backbone.test_compress_q={crf} model.backbone.clip_len=8 \
# > configs/SMC_ICCV2023/resutls/tsm_hmdb51_{run_mode}_{codec_type}_crf{crf}.log
            '''.format(crf = crf,run_mode =run_mode, codec_type = codec_type)
            os.system(cmd)