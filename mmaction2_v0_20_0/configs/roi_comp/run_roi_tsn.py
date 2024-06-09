
import os

    
    
q_list = [47,43,39,35]
for q in q_list:
    os.system(f'''
           CUDA_VISIBLE_DEVICES=5 PORT=2082 tools/dist_test.sh \
configs/lbvu/tsn_r50_1x1x8_50e_ucf101_kinetics400_rgb.py \
/data_video/code/lbvu/mmaction/work_dirs/tsn_r50_1x1x8_50e_ucf101_kinetics400_rgb/epoch_50.pth 1 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode=roi_comp model.backbone.test_codec=h266 model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.test_compress_q={q} model.backbone.clip_len=8 \
> configs/roi_comp/results/tsn_ucf101_vvc{q}.log
           ''')

q_list = [47,43,39,35]
for q in q_list:
    os.system(f'''
           CUDA_VISIBLE_DEVICES=5 PORT=2082 tools/dist_test.sh \
configs/lbvu/tsn_r50_1x1x8_50e_hmdb51_kinetics400_rgb.py \
https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_kinetics400_rgb/tsn_r50_1x1x8_50e_hmdb51_kinetics400_rgb_20201123-7f84701b.pth 1 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode=roi_comp model.backbone.test_codec=h266 model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.test_compress_q={q} model.backbone.clip_len=8 \
> configs/roi_comp/results/tsn_hmdb51_vvc{q}.log
           ''')
    
    