import os

for codec_type in ['h266']:
    for run_mode in ['smc_iccv2023']:
        if run_mode == 'raw':
            crfs = [47,43,39,35]
        elif run_mode == 'smc_iccv2023':
            crfs = [27,31,35]
        for crf in crfs:
            cmd = '''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=2024 tools/dist_test.sh \
configs/lbvu/slowfast_hmdb51.py \
/data_video/code/lbvu/mmaction/work_dirs/slowfast_k400_pretrained_r50_4x16x1_40e_hmdb51_rgb/latest.pth 8 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode={run_mode} model.backbone.test_codec={codec_type} model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.pretrained_lbvu_path=/data_video/code/lbvu/Important_models/SMC_ICCV2023/baseline_bpp4_mae0_8f.pth.tar \
model.backbone.test_compress_q={crf} model.backbone.clip_len=32 data.test_dataloader.videos_per_gpu=4 \
> configs/SMC_ICCV2023/resutls/slowfast_hmdb51_{run_mode}_{codec_type}_crf{crf}.log
            '''.format(crf = crf,run_mode =run_mode, codec_type = codec_type)
            os.system(cmd)