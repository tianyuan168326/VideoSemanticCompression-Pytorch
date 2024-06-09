
import os

    
    
# q_list = [47 ,43 ,39,35]
q_list = [47 ,43 ,39]
for q in q_list:
    os.system(f'''
           CUDA_VISIBLE_DEVICES=5 PORT=2182 tools/dist_test.sh \
configs/lbvu/timesformer_ft_ucf101.py \
/data_video/code/lbvu/mmaction/work_dirs/timesformer_divST_8x32x1_15e_ucf101_rgb/best_top1_acc_epoch_8.pth 1 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode=roi_comp model.backbone.test_codec=h266 model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.test_compress_q={q} model.backbone.clip_len=8 \
> configs/roi_comp/results/timesformer_ucf101_vvc{q}.log
           ''')

for q in q_list:
    os.system(f'''
           CUDA_VISIBLE_DEVICES=5 PORT=2182 tools/dist_test.sh \
configs/lbvu/timesformer_ft_hmdb51.py \
/data_video/code/lbvu/mmaction/work_dirs/timesformer_divST_8x32x1_15e_hmdb51_rgb/best_top1_acc_epoch_5.pth 1 \
--eval top_k_accuracy \
--cfg-options model.backbone.test_mode=roi_comp model.backbone.test_codec=h266 model.backbone.latent_dim=256 model.backbone.keyint=10 model.backbone.temporal_compression=True \
model.backbone.test_compress_q={q} model.backbone.clip_len=8 \
> configs/roi_comp/results/timesformer_hmdb51_vvc{q}.log
           ''')
    
    