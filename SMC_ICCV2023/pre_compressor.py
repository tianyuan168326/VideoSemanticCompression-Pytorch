import sys
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

class VideoPreCompressor(nn.Module):
    def __init__(self,test_mode,latent_dim,clip_len,temporal_compression\
        ,restortion_net,lambda_bpp,last_Tlayer_causual,test_codec_rescale,
        pretrained_lbvu_path,test_codec,test_compress_q,codec_tune,keyint,
        codec_config_file,network_spatial_group = -1,network_temporal_group = 1,base_codec_patch = (1,1),
        vpt_coding_depth = 2,
        vpt_MLP_multi = 1
        ):
        super().__init__()
        self.vpt_coding_depth = vpt_coding_depth
        self.vpt_MLP_multi = vpt_MLP_multi
        self.network_spatial_group= network_spatial_group
        self.network_temporal_group= network_temporal_group
        self.test_mode = test_mode
        ##### old models 
        self.dummy = nn.Linear(1,1)    
        if self.test_mode in ['smc_plus']:
            opt = {
                "nce_type":"none",
                "app_avg_block":0,
                "enc_dim":32,
                "diff_dim":8,
                "diff_group_num":-1,
                "latent_dim":256,
                "smc_multi_divid_strategy":"softmax_mask",
                'use_enc_transformer':False,
                'checkpoint_dir':"./",
                'num_segments':clip_len,
                "enc_net_group_num":8,
                "layers_16x":3,
                "temporal_compression":temporal_compression,
                "K_enc":32,
                "vpt_win_size":8,
                "vpt_mask_dim":128,
                "vpt_coding_depth":3,
                "vpt_MLP_multi":1,
                "vis_mode":False,
            }
            
            from SMC_ICCV2023.SMC_Plus_network import DiffCompensate_SMCPlus
            from munch import DefaultMunch
            opt = DefaultMunch.fromDict(opt)
            self.pre_code_net2 = DiffCompensate_SMCPlus(opt)
            if ".pth" in pretrained_lbvu_path: ## if is a model file
                checkpoint = torch.load(pretrained_lbvu_path,map_location='cpu')
                ckeck = checkpoint['state_dict']
                ckeck_copy = {}
                ks  =ckeck.keys()
                for k in ks:
                    if "object_reason" in k:
                        continue
                    new_k = k
                    if k.startswith("module.pre_code_net."):
                        new_k = k[20:]
                    else:
                        continue
                    ckeck_copy[new_k] = ckeck[k]
                self.pre_code_net2.load_state_dict(ckeck_copy,strict=False)
            self.pre_code_net2.eval()
        
        ###only load pre_code
        self.test_codec_rescale = test_codec_rescale
        from SMC_ICCV2023.Quantization_video_compression import Quantization_H265
        self.test_compress_q  =test_compress_q
        self.quantization_m = Quantization_H265(q = test_compress_q,
            type = test_codec,\
            scale_times_in  = test_codec_rescale,
            scale_times_out  = test_codec_rescale,
            tune = codec_tune,keyint = keyint,codec_config_file = codec_config_file,patch_for_accelerate=base_codec_patch)
        
            
    def d2_to_d3(self,x,t ):
        bt,c,h,w = x.size()
        b = bt//t
        return x.reshape(b,t,c,h,w).transpose(1,2)
    def d3_to_d2(self,x):
        b,c,t,h,w = x.size()
        return x.transpose(1,2).reshape(b*t,c,h,w)
    def forward(self,x,img_norm_cfg):
        
        x_format = "3d"
        if len(x.size()) == 4:
            ### 2D input
            bt,h,w = x.size(0),x.size(-2),x.size(-1)
            b = img_norm_cfg["mean"].size(0)
            clip_length = bt//b
            x_format = "2d"
        elif len(x.size()) == 5:
            b,_,t,h,w = x.size()
            clip_length = t
            x_format = "3d"
        
            
        if self.test_mode in ['hd','feature_comp']:
            code_bpp = torch.zeros((x.size(0),1))
            h265_bitcosts = torch.zeros((x.size(0),1))
        elif self.test_mode == 'raw':
            if img_norm_cfg:
                mean = img_norm_cfg['mean']/255.0
                std = img_norm_cfg['std']/255.0
                std = std[0:1].unsqueeze(-1).unsqueeze(-1)
                mean = mean[0:1].unsqueeze(-1).unsqueeze(-1)
                if x_format == "3d":
                    std = std.unsqueeze(-1)
                    mean = mean.unsqueeze(-1)
                x_pre_norm = x *std + mean
            else:
                x_pre_norm =x
            input_clean = x_pre_norm
            if x_format == "3d":
                codec_input= input_clean
            elif x_format == "2d":
                codec_input = self.d2_to_d3(input_clean,clip_length)
            input_com,h265_bitcosts = self.quantization_m(codec_input)
            input_com  =input_com.to(input_clean.device)
            h265_bitcosts = h265_bitcosts.unsqueeze(-1).to(input_clean.device)
            code_bpp = torch.zeros((h265_bitcosts.size(0),1)).to(input_clean.device)
            if x_format == "3d":
                input_com= input_com
            elif x_format == "2d":
                input_com  =self.d3_to_d2(input_com)
            if img_norm_cfg:
                x = (input_com-mean)/std
            else:
                x = input_com
          
        elif self.test_mode in ["smc_plus"]:
            mean = img_norm_cfg['mean']/255.0
            std = img_norm_cfg['std']/255.0
            std = std[0:1].unsqueeze(-1).unsqueeze(-1)
            mean = mean[0:1].unsqueeze(-1).unsqueeze(-1)
            if x_format == "3d":
                std = std.unsqueeze(-1)
                mean = mean.unsqueeze(-1)
            x_pre_norm = x *std + mean
            if x_format == "2d":
                input_clean =self.d2_to_d3(x_pre_norm,clip_length) 
            elif x_format == "3d":
                input_clean = x_pre_norm
            input_com,h265_bitcosts = self.quantization_m(input_clean)
            input_com  =input_com.cuda(input_clean.device)
            
            if self.network_spatial_group>1 and input_clean.size(-1)>1024:
                wid_per_group = input_clean.size(-1)//self.network_spatial_group
                t_per_group = input_clean.size(-3)//self.network_temporal_group
                recons_clean = []
                code_bpp = 0
                for div_i in range(self.network_spatial_group):
                    recons_clean_t = []
                    for div_i_t in range(self.network_temporal_group):
                        recons_clean_subgroup,code_bpp_subgroup  = self.pre_code_net2([input_clean[:,:,div_i_t*t_per_group: (div_i_t+1)*t_per_group,:,div_i*wid_per_group : (div_i+1)*wid_per_group],\
                            input_com[:,:,div_i_t*t_per_group: (div_i_t+1)*t_per_group,:,div_i*wid_per_group : (div_i+1)*wid_per_group]],'enc_dec')
                        recons_clean_t += [recons_clean_subgroup]
                        code_bpp += code_bpp_subgroup[1]
                    recons_clean += [torch.cat(recons_clean_t,dim=-3)]
                recons_clean =   torch.cat(recons_clean,dim=-1)
                
            else:
                recons_clean,code_bpp  = self.pre_code_net2([input_clean,\
                        input_com],'enc_dec')
                code_bpp=  code_bpp[1]
           
            
            h265_bitcosts = h265_bitcosts.unsqueeze(-1)
            code_bpp = code_bpp.unsqueeze(-1)
            if x_format == "2d":
                recons_clean = self.d3_to_d2(recons_clean)
            elif x_format == "3d":
                recons_clean = recons_clean
            x = (recons_clean-mean)/std
        
        return h265_bitcosts,code_bpp,x