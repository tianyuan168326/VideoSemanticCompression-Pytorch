## performance on HD video

import os
q_list = [47,43,39,35]
# q_list = [47]
for q in q_list:
    os.system(f'''
           CUDA_VISIBLE_DEVICES=7 PORT=2013 tools/dist_test.sh \
configs/lbvu/slowfast_ucf101.py \
/data_video/code/lbvu/mmaction/work_dirs/slowfast_k400_pretrained_r50_4x16x1_40e_ucf101_rgb/latest.pth 1 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode=roi_comp model.backbone.test_codec=h266 model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.pretrained_lbvu_path=/data_video/code/lbvu/checkpoints/h265_latent256_temporal/bpp4_lpips05_con0001_local_mixH2645/test_model/action_backbone.TSM_k60k_brrand_resnet50_segment8__network.pth.tar \
model.backbone.test_compress_q={q} model.backbone.clip_len=32 \
    > configs/roi_comp/results/slowfast_ucf101_vvc{q}.log
           ''')
    
for q in q_list:
    os.system(f'''
           CUDA_VISIBLE_DEVICES=7 PORT=2013 tools/dist_test.sh \
configs/lbvu/slowfast_hmdb51.py \
/data_video/code/lbvu/mmaction/work_dirs/slowfast_k400_pretrained_r50_4x16x1_40e_hmdb51_rgb/latest.pth 1 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode=roi_comp model.backbone.test_codec=h266 model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.pretrained_lbvu_path=/data_video/code/lbvu/checkpoints/h265_latent256_temporal/bpp4_lpips05_con0001_local_mixH2645/test_model/action_backbone.TSM_k60k_brrand_resnet50_segment8__network.pth.tar \
model.backbone.test_compress_q={q} model.backbone.clip_len=32 \
    > configs/roi_comp/results/slowfast_hmdb51_vvc{q}.log
           ''')


for q in q_list:
    os.system(f'''
           CUDA_VISIBLE_DEVICES=7 PORT=2013 tools/dist_test.sh \
configs/lbvu/slowfast_k400.py \
https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb/slowfast_r50_4x16x1_256e_kinetics400_rgb_20210722-04e43ed4.pth 1 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode=roi_comp model.backbone.test_codec=h266 model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.pretrained_lbvu_path=/data_video/code/lbvu/checkpoints/h265_latent256_temporal/bpp4_lpips05_con0001_local_mixH2645/test_model/action_backbone.TSM_k60k_brrand_resnet50_segment8__network.pth.tar \
model.backbone.test_compress_q={q} model.backbone.clip_len=32 \
    > configs/roi_comp/results/slowfast_k400_vvc{q}.log
           ''')
    
   