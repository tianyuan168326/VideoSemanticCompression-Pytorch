# SMC++: Masked Learning of Unsupervised Video Semantic Compression

PyTorch Implementation of our paper:

> **SMC++: Masked Learning of Unsupervised Video Semantic Compression**
>
> Yuan Tian, Guo Lu, and Guangtao Zhai
> [[ArXiv](https://arxiv.org/)]

This paper is an extended version of our work originally presented at ICCV 2023 "Non-Semantics Suppressed Mask Learning for Unsupervised Video SemanticCompression".

## Main Contribution
Most video compression methods focus on human visual perception, neglecting semantic preservation. This leads to severe semantic loss during the compression, hampering downstream video analysis tasks.
In this paper, we propose a **Masked Video Modeling (MVM)**-powered compression framework that particularly preserves video semantics, by jointly mining and compressing the semantics in a self-supervised manner.
While MVM is proficient at learning generalizable semantics through the masked patch prediction task, it may also encode non-semantic information like trivial textural details, wasting bitcost and bringing semantic noises. To suppress this, we explicitly regularize the non-semantic entropy of the compressed video in the MVM token space.
The proposed framework is instantiated as a simple Semantic-Mining-then-Compression (SMC) model.
Furthermore, we extend SMC as an advanced SMC++ model from several aspects. First, we equip it with a masked motion prediction objective, leading to better temporal semantic learning ability.
Second, we introduce a Transformer-based compression module, to improve the semantic compression efficacy.
Considering that directly mining the complex redundancy among heterogeneous features in different coding stages is non-trivial, we introduce a compact blueprint semantic representation to align these features into a similar form, fully unleashing the power of the Transformer-based compression module.
Extensive results demonstrate the proposed SMC and SMC++ models show remarkable superiority over previous methods.

<!-- <div align=center><img src="banner_figs/framework.jpg" width="800"></div> -->


## Environment Setup 
1. Create conda env
    ```
    conda env create -f smc_env.yml
    ```
2. Install VVenC 1.5.0

* Download the binary VVenC library (For ubuntu only, if you use other Linux version, please build VVenC from the source code)
    ```
    Link: https://pan.baidu.com/s/1wm92tY27QJW4lm8d3Dxz_A?pwd=tzkf 
    Password: tzkf 
    ```
* Put the VVenC folder under this project

## Pre-trained SMC++ Model

1. Download the pre-trained SMC++ model, and put it under the folder "pretrain_models/coding"
    ```
    Link: https://pan.baidu.com/s/1vk6bRpj5G3ffeg7MwIdpIA?pwd=ekuh 
    Password: ekuh 
    ```

##  Video Object Segmentation (VOS) Task

In this section, we adopt the  [XMEM](https://arxiv.org/abs/2207.07115) approach as the VOS task evaluator.

1. Download the official XMEM models, `XMem.pth' and `XMem-s012.pth', from the link, and save them to the folder "XMem/saves".

    ```
    Link: https://pan.baidu.com/s/1pPUunMOmRn5LoOaW6z641g?pwd=isol 
    Password: isol 
    ```

2. Download our re-organized DAVIS2017 dataset, and put the file "DAVIS_raw_480.tar.gz"  under the folder "XMem/dataset_space"

    ```
    Link: https://pan.baidu.com/s/1O0HD1bldq1EmXVJUPp2n8Q?pwd=r17g 
    Password: r17g  
    ```

    Note: For this datasete, we download the original 1080p frames, and down-sample the frames into 480p.
    Compared to the officially released 480p frames, our re-organized version is with less artifacts, thus more suitable for evaluating different compression methods.

3. De-compress the DAVIS2017 dataset
    ```
    cd XMem/dataset_space
    tar zxvf DAVIS_raw_480.tar.gz
    ```

4. Apply semantic compression to the DAVIS2017 dataset
    ```
    python VideoDatasetList/Davis2017/transcode_scripts_smcplus_GOP10.py
    ```

5. Evaluate the XMEM model on the compressed DAVIS2017 dataset
    Chang the "davis_path " option within eval_results/eval_smcplus.py to your  "...DAVIS_raw_480/trainval" path.

    ```
    cd XMem/
    python eval_results/eval_smcplus.py
    ```
    The evaluation result for VOS task will be saved in "val_results/logs_evalseg".


##  Video Action Recognition Task

We re-use the codes of mmaction2 to inherit their rich pre-trained action models.
Configuring mmaction2 is a bit tedious, please be patient.
<!-- 1. Make sure your G++/GCC version >6.0.0. cuda_toolkit version is 11.6.1. -->

1. Install mmcv and mmaction2
```
cd mmcv_1_6_0
pip install -e . 

cd mmaction2_v0_20_0
python setup.py develop
```

2. Download all pre-trained action models and put them into the folder `/data/code_space/smc_plus_opensource/pretrain_models/action'.
    ```
    Link: https://pan.baidu.com/s/1ixi5AYd0YQyQQQlnHFZtkw?pwd=ri6q 
    Password: ri6q 
    ```

    <div align="center">
    <table>
    <thead>
    <tr>
    <th align="center">Dataset-Action Model Pair</th>
    <th align="center">Model file name</th>
    </tr>
    </thead>

    <tbody>

    <tr>
    <td align="center">timesformer-hmdb51</td>
    <td align="center" >timesformer_hmdb51_best_top1_acc_epoch_5.pth</td>
    </tr>

    <tr>
    <td align="center">timesformer_ucf101</td>
    <td align="center" >timesformer_ucf101_best_top1_acc_epoch_8.pth</td>
    </tr>

    <tr>
    <td align="center">tsm_hmdb51</td>
    <td align="center" >tsm_hmdb51_best_top1_acc_epoch_10.pth</td>
    </tr>

    <tr>
    <td align="center">slowfast_hmdb51</td>
    <td align="center" >slowfast_hmdb51_epoch_33.pth</td>
    </tr>

    <tr>
    <td align="center">slowfast_ucf101</td>
    <td align="center" >slowfast_ucf101_epoch_40.pth</td>
    </tr>

    <tr>
    <td align="center">slowfast_k400</td>
    <td align="center" >slowfast_k400_slowfast_r50_4x16x1_256e_kinetics400_rgb_20210722-04e43ed4.pth</td>
    </tr>

    <tr>
    <td align="center">tsm_ucf101</td>
    <td align="center" >tsm_ucf101_tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb_20210630-1fae312b.pth</td>
    </tr>

    <tr>
    <td align="center">tsm_k400</td>
    <td align="center" >tsm_k400_tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth</td>
    </tr>

    <tr>
    <td align="center">tsm_diving48</td>
    <td align="center" >tsm_diving48_tsm_r50_video_1x1x8_50e_diving48_rgb_20210426-aba5aa3d.pth</td>
    </tr>

    </tbody>

    </table>
    </div>

3. Video action dataset preparation

    We provide our well-organized HMDB51/UCF101 Dataset for quick start-up.
    For other large-scale datasets like Kinetic400 and Diving48, please download them from the official weblink.

    * HMDB51:

    ```
    Link: https://pan.baidu.com/s/1Yvv4YSGlmHdDRqaEH7ijjg?pwd=qo62 
    Password: qo62 
    ```
    Put the downloaded "hmdb51_extracted.tar.gz" under the folder "VideoDatasetList",and decompress it.
    ```
    tar zxvf hmdb51_extracted.tar.gz
    ```

    * UCF101:
    ```
    Link: https://pan.baidu.com/s/1FXXisNGnAu7Uv9opHaIGrw?pwd=964p 
    Password: 964p 
    ```
    Put the downloaded "ucf101_jpg.tar.gz" under the folder "VideoDatasetList",and decompress it.
    ```
    tar zxvf ucf101_jpg.tar.gz
    ```


3. Re-produce action results

    * Slowfast-UCF101:
    

    Set the `data_root/data_root_val/data_root_compress/data_root_val_compress` field in the configuration file `mmaction2_v0_20_0/configs/lbvu/slowfast_ucf101.py' to the directory of your extracted UCF101 dataset;
    ```
    cd mmaction2_v0_20_0
    python configs/SMC_Plus/test_slowfast_ucf101.py
    ```
    If the GPU memory is insufficient, please reduce the value of the `data.test_dataloader.videos_per_gpu' field.


    * TSM- HMDB51:
    Set the `data_root/data_root_val/data_root_compress/data_root_val_compress` field in the configuration file `mmaction2_v0_20_0/configs/lbvu/tsm_hmdb.py' to the directory of your extracted HMDB51 dataset;
    ```
    cd mmaction2_v0_20_0
    python configs/SMC_Plus/test_tsm_hmdb.py
    ```
    If the GPU memory is insufficient, please reduce the value of the `data.test_dataloader.videos_per_gpu' field.

    * Other Action models and datasets:

    Please refer to `mmaction2_v0_20_0/configs/SMC_Plus/test_other_action_models.py'.



## Other Info

### References

This repository is built upon the following projects.

- [TSM](https://github.com/mit-han-lab/temporal-shift-module)
- [MMaction2](https://github.com/open-mmlab/mmaction2)
- [XMem](https://github.com/hkchengrex/XMem)

### Citation

Please **[★star]** this repo and **[cite]** the following papers if you feel our project and codes useful to your research:

```
@inproceedings{tian2023non,
  title={Non-Semantics Suppressed Mask Learning for Unsupervised Video Semantic Compression},
  author={Tian, Yuan and Lu, Guo and Zhai, Guangtao and Gao, Zhiyong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13610--13622},
  year={2023}
}

@article{tian2024smcplus,
  title={SMC++: Masked Learning of Unsupervised Video Semantic Compression},
  author={Tian, Yuan and Lu, Guo and Zhai, Guangtao},
  journal={arXiv},
  year={2024}
}
```


### Contact

For any questions, please feel free to open an issue or contact:

```
Yuan Tian: tianyuan168326@outlook.com
```