import os
q_list = [51,47,43,39,35]
codec_type = "h266"
run_mode = 'smc_plus'
for q in q_list:
        
        cmd = f'''
CUDA_VISIBLE_DEVICES=0,1 PORT=2022 tools/dist_test.sh \
configs/lbvu/tsm_ucf101.py \
../pretrain_models/action/tsm_ucf101_tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb_20210630-1fae312b.pth 2 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode={run_mode} model.backbone.test_codec={codec_type} model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.pretrained_lbvu_path=../pretrain_models/coding/SMC_plus.pth.tar \
model.backbone.test_compress_q={q} model.backbone.clip_len=8 \
>configs/SMC_Plus/results/tsm_ucf101_{q}.log
        '''
        os.system(cmd)
        #### begin
        cmd = f'''
CUDA_VISIBLE_DEVICES=0,1 PORT=2022 tools/dist_test.sh \
configs/lbvu/tsm_diving48.py \
../pretrain_models/action/tsm_diving48_tsm_r50_video_1x1x8_50e_diving48_rgb_20210426-aba5aa3d.pth 2 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode={run_mode} model.backbone.test_codec={codec_type} model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.pretrained_lbvu_path=../pretrain_models/coding/SMC_plus.pth.tar \
model.backbone.test_compress_q={q} model.backbone.clip_len=8 \
>configs/SMC_Plus/results/tsm_diving48_{q}.log
        '''
        os.system(cmd)
        
        cmd = '''
CUDA_VISIBLE_DEVICES=0,1 PORT=2022 tools/dist_test.sh \
configs/lbvu/timesformer_ft_hmdb51.py \
../pretrain_models/action/timesformer_hmdb51_best_top1_acc_epoch_5.pth 2 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode={run_mode} model.backbone.test_codec={codec_type} model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.pretrained_lbvu_path=../pretrain_models/coding/SMC_plus.pth.tar \
model.backbone.test_compress_q={q} model.backbone.clip_len=8 data.test_dataloader.videos_per_gpu=8 \
>configs/SMC_Plus/results/timesformer_ft_hmdb51_{q}.log
        '''
        os.system(cmd)
        
        cmd = '''
CUDA_VISIBLE_DEVICES=0,1 PORT=2022 tools/dist_test.sh \
configs/lbvu/timesformer_ft_ucf101.py \
../pretrain_models/action/timesformer_ucf101_best_top1_acc_epoch_8.pth 2 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode={run_mode} model.backbone.test_codec={codec_type} model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.pretrained_lbvu_path=../pretrain_models/coding/SMC_plus.pth.tar \
model.backbone.test_compress_q={q} model.backbone.clip_len=8 data.test_dataloader.videos_per_gpu=8 \
>configs/SMC_Plus/results/timesformer_ft_ucf101_{q}.log
        '''
        os.system(cmd)
        
        cmd = f'''
CUDA_VISIBLE_DEVICES=0,1 PORT=2022 tools/dist_test.sh \
configs/lbvu/slowfast_hmdb51_epoch_33.py \
../pretrain_models/action/slowfast_ucf101_epoch_40.pth 2 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode={run_mode} model.backbone.test_codec={codec_type} model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.pretrained_lbvu_path=../pretrain_models/coding/SMC_plus.pth.tar \
model.backbone.test_compress_q={q} model.backbone.clip_len=32 data.test_dataloader.videos_per_gpu=2 \
>configs/SMC_Plus/results/slowfast_hmdb51_{q}.log
        '''
        os.system(cmd)
                        
 
for q in q_list:
        pass
        cmd = f'''
CUDA_VISIBLE_DEVICES=0,1 PORT=2022 tools/dist_test.sh \
configs/lbvu/tsm_k400_tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.py \
../pretrain_models/action/slowfast_ucf101_epoch_40.pth 2 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode={run_mode} model.backbone.test_codec={codec_type} model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.pretrained_lbvu_path=../pretrain_models/coding/SMC_plus.pth.tar \
model.backbone.test_compress_q={q} model.backbone.clip_len=8  \
>configs/SMC_Plus/results/tsm_k400_{q}.log
        '''
        os.system(cmd)
        cmd = '''
CUDA_VISIBLE_DEVICES=0,1 PORT=2022 tools/dist_test.sh \
configs/lbvu/slowfast_k400_slowfast_r50_4x16x1_256e_kinetics400_rgb_20210722-04e43ed4.py \
../pretrain_models/action/slowfast_ucf101_epoch_40.pth 2 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode={run_mode} model.backbone.test_codec={codec_type} model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.pretrained_lbvu_path=../pretrain_models/coding/SMC_plus.pth.tar \
model.backbone.test_compress_q={crf} model.backbone.clip_len=32 data.test_dataloader.videos_per_gpu=2 \
>configs/SMC_Plus/results/slowfast_k400_{q}.log
        '''
        os.system(cmd)
