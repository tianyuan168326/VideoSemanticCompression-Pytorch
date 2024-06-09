
import os
# q_list = [51,47,43,39,35]
# for q in q_list:
#     os.system(f'''
#            CUDA_VISIBLE_DEVICES=2,3 PORT=2082 tools/dist_test.sh \
# configs/lbvu/tsm_ucf101.py \
# https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb_20210630-1fae312b.pth 2 \
# --eval top_k_accuracy \
# --cfg-options model.backbone.test_mode=roi_comp model.backbone.test_codec=h266 model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
# model.backbone.pretrained_lbvu_path=/data_video/code/lbvu/checkpoints/h265_latent256_temporal/bpp4_lpips05_con0001_local_mixH2645/test_model/action_backbone.TSM_k60k_brrand_resnet50_segment8__network.pth.tar \
# model.backbone.test_compress_q={q} model.backbone.clip_len=8 \
# > configs/roi_comp/results/tsm_ucf101_vvc{q}.log
#            ''')
    
    
q_list = [47,43,39,35]
for q in q_list:
    os.system(f'''
           CUDA_VISIBLE_DEVICES=3 PORT=2082 tools/dist_test.sh \
configs/lbvu/tsm_hmdb.py \
/data_video/code/lbvu/mmaction/work_dirs/tsm_r50_1x1x8_50e_hmdb51_rgb/best_top1_acc_epoch_10.pth 1 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode=roi_comp model.backbone.test_codec=h266 model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.pretrained_lbvu_path=/data_video/code/lbvu/checkpoints/h265_latent256_temporal/bpp4_lpips05_con0001_local_mixH2645/test_model/action_backbone.TSM_k60k_brrand_resnet50_segment8__network.pth.tar \
model.backbone.test_compress_q={q} model.backbone.clip_len=8 \
> configs/roi_comp/results/tsm_hmdb51_vvc{q}.log
           ''')


q_list = [47,43,39,35]
for q in q_list:
    os.system(f'''
           CUDA_VISIBLE_DEVICES=3 PORT=2082 tools/dist_test.sh \
configs/lbvu/tsm_diving48.py \
https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_video_1x1x8_50e_diving48_rgb/tsm_r50_video_1x1x8_50e_diving48_rgb_20210426-aba5aa3d.pth 1 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode=roi_comp model.backbone.test_codec=h266 model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.pretrained_lbvu_path=/data_video/code/lbvu/checkpoints/h265_latent256_temporal/bpp4_lpips05_con0001_local_mixH2645/test_model/action_backbone.TSM_k60k_brrand_resnet50_segment8__network.pth.tar \
model.backbone.test_compress_q={q} model.backbone.clip_len=8 \
> configs/roi_comp/results/tsm_diving48_vvc{q}.log
           ''')
    
    
q_list = [47,43,39,35]
for q in q_list:
    os.system(f'''
           CUDA_VISIBLE_DEVICES=3 PORT=2082 tools/dist_test.sh \
configs/lbvu/tsm_sthv1.py \
https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_flip_randaugment_1x1x8_50e_sthv1_rgb/tsm_r50_flip_randaugment_1x1x8_50e_sthv1_rgb_20210324-76937692.pth 1 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode=roi_comp model.backbone.test_codec=h266 model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.pretrained_lbvu_path=/data_video/code/lbvu/checkpoints/h265_latent256_temporal/bpp4_lpips05_con0001_local_mixH2645/test_model/action_backbone.TSM_k60k_brrand_resnet50_segment8__network.pth.tar \
model.backbone.test_compress_q={q} model.backbone.clip_len=8 \
> configs/roi_comp/results/tsm_sthv1_vvc{q}.log
           ''')
    
q_list = [47,43,39,35]
for q in q_list:
    os.system(f'''
           CUDA_VISIBLE_DEVICES=3 PORT=2082 tools/dist_test.sh \
configs/lbvu/tsm_k400.py \
https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth 1 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode=roi_comp model.backbone.test_codec=h266 model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.pretrained_lbvu_path=/data_video/code/lbvu/checkpoints/h265_latent256_temporal/bpp4_lpips05_con0001_local_mixH2645/test_model/action_backbone.TSM_k60k_brrand_resnet50_segment8__network.pth.tar \
model.backbone.test_compress_q={q} model.backbone.clip_len=8 \
> configs/roi_comp/results/tsm_k400_vvc{q}.log
           ''')