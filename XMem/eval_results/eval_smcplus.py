import os
from joblib import Parallel,delayed
parallel_pool  = Parallel(n_jobs=16)
gpu_id = 0
codec_type = 'h266'
qps = ['35','39','43','47',"51"]
qps = ["51"]
if not os.path.exists("eval_results/logs_inferseg"):
    os.mkdir("eval_results/logs_inferseg")
if not os.path.exists("eval_results/logs_evalseg"):
    os.mkdir("eval_results/logs_evalseg")
for crf in qps:
    cmd = '''
        CUDA_VISIBLE_DEVICES={gpu_id} python eval.py \
        --d17_path dataset_space/DAVIS_raw_480_{codec_type}_smcplus_crf{crf} \
        --output eval_results/results_{codec_type}_smcplus_crf{crf} \
        > eval_results/logs_inferseg/results_{codec_type}_smcplus_crf{crf}.log
        '''.format(gpu_id = gpu_id,codec_type = codec_type,crf = crf)
    os.system(cmd)
for crf in qps:
    cmd = '''
    CUDA_VISIBLE_DEVICES={gpu_id} python davis2017-evaluation-master/evaluation_method.py \
    --results_path eval_results/results_{codec_type}_smcplus_crf{crf} \
    --task semi-supervised --davis_path /data/code_space/smc_plus_opensource/XMem/dataset_space/DAVIS_raw_480/trainval \
    # > eval_results/logs_evalseg/results_{codec_type}_smcplus_crf{crf}.log
    '''.format(gpu_id = gpu_id,codec_type = codec_type,crf = crf)
    print(cmd)
    os.system(cmd)


