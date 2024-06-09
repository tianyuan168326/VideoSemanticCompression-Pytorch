import os

    
    
q_list = [51,47,43,39,35]
codec_type = "h266"
run_mode = 'smc_plus'

for q in q_list:
    os.system(f'''
            CUDA_VISIBLE_DEVICES=0,1 PORT=2066 tools/dist_test.sh \
configs/lbvu/tsm_hmdb.py \
../pretrain_models/action/tsm_hmdb51_best_top1_acc_epoch_10.pth 2 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode={run_mode} model.backbone.test_codec={codec_type} model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.pretrained_lbvu_path=../pretrain_models/coding/SMC_plus.pth.tar \
model.backbone.test_compress_q={q} model.backbone.clip_len=8 \
> configs/SMC_Plus/results/tsm_hmdb51_{q}.log
           ''')
    
