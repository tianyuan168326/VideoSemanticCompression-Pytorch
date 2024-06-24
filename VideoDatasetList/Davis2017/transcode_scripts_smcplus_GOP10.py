import os
import shutil
crf_list  = [51,47,43,39,35]
codec_list = ['h266']
for codec in codec_list:
    for q in crf_list:
        src_dir = "XMem/dataset_space/DAVIS_raw_480"
        dest_dir = f"XMem/dataset_space/DAVIS_raw_480_{codec}_smcplus_crf{q}"
        if not os.path.exists(dest_dir):
            print(f"copying {src_dir} to {dest_dir}")
            shutil.copytree(src_dir, dest_dir)
        log_file = f"VideoDatasetList/Davis2017/coding_logs/smc_plus_{codec}_crf_{q}.log"
        print(f"transcoding under setting,codec :{codec}, crf {q}" )
        cmd = f'''
        CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port=11217 \
        --nproc_per_node 2 VideoDatasetList/Davis2017/transcode_davis.py \
        --fp 32 --lr 0.0001 -b 1 -j 3 \
        --root_path ./ --dataset k60k_brrand --dataset_eval k400_video \
        --type action_backbone.TSM --aug_crop_scale 1 --no_video_scale \
        --agg_type agg \
        --input_size 224 --arch resnet50 --num_segments 10 --beta 2 --alpha 4 \
        --checkpoint_dir ./checkpoints/temp \
        --temporal_compression \
        --encoder_net_type smc_plus --encoder_1st vanilla --enc_dim 32 --last_Tlayer_causual --lambda_bpp 4 \
        --test_crf {q} --codec_type h266  --evaluate --test_codec_gop 10 \
        --resume_main pretrain_models/coding/SMC_plus.pth.tar \
        --trancode_src_dir {src_dir} --trancode_dest_dir {dest_dir} \
        > {log_file}
        '''
        
        print(cmd)
        os.system(cmd)
        print(f"Finish transcoding under setting,codec :{codec}, crf {q},please check {log_file} for bitrate, the transcoded directory is {dest_dir} " )
        
        