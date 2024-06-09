import os
import sys
import time
import torch.nn as nn
import torch.nn.functional as F
import torch
from SMC_ICCV2023.resnet import resnet18
import torch.distributed as dist
import numpy as np
from SMC_ICCV2023.network_arch import PreNormConv,ResBlockSimple
from SMC_ICCV2023.network_arch import  AlignFt1_toFt2HeterMask,BitEstimator_MVSC
from SMC_ICCV2023.network_arch import upconv2x2,d2_to_d3,d3_to_d2,save_attention_mask
from SMC_ICCV2023.network_arch import pad_3d,window,window_2d,dewindow2d
from SMC_ICCV2023.vit_layers import TransformerLearnedRes
import time
import math


def replace_bn_with_identity_and_relu_with_leakyrelu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # print(name,"....")
            # Replace BatchNorm2d with Identity
            # setattr(module, name, nn.BatchNorm2d(child.num_features, momentum=0.001))
            setattr(module, name, nn.Identity(child.num_features))
            # setattr(module, name, nn.GroupNorm(32,child.num_features))
            # setattr(module, name, LayerNorm2d(child.num_features))
            pass
        elif isinstance(child, nn.ReLU):
            # Replace ReLU with LeakyReLU
            setattr(module, name, nn.LeakyReLU(0.01, inplace=True))
            pass
        else:
            # Recursively apply the replacements
            replace_bn_with_identity_and_relu_with_leakyrelu(child)

def estimate_bits_factorized(bitest,quant_mv,t,img_h,img_w):
    bt,c,h,w = quant_mv.size() ## bt c h w
    
    def iclr18_estrate_bits_mv(mv):
        prob = bitest(mv + 0.5) - bitest(mv - 0.5)
        # est_bits = torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50)
        est_bits = -1.0 * torch.log(prob + 1e-5) / math.log(2.0)
        est_bits_src_dim = est_bits.sum(1,keepdim=True)
        est_bits = est_bits.reshape(bt//t,t,c,h,w).sum(-1).sum(-1).sum(-1).sum(-1)
        return est_bits, est_bits_src_dim,prob
    
    
    
    bits_in_batch, est_bits_src_dim,_ = iclr18_estrate_bits_mv(quant_mv)
    vid_B = bt//t
    bpp =  bits_in_batch.reshape(vid_B)/(t*img_h*img_w)
    return bpp,bits_in_batch





class FNetSMC(nn.Module):
    def __init__(self,opt,norm = "BN"):
        self.opt = opt
        super().__init__()
        K_enc = 32
        clip_len  =self.opt.num_segments
        ResBlock = ResBlockSimple
        self.down_scale4 = nn.Sequential(
            nn.Conv2d(3, K_enc, 7,2,3),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(K_enc, K_enc*2, 5,2,2),
        )
        self.down_scale8 = nn.Sequential(
            nn.Conv2d(K_enc*2, K_enc*2, 5,1,2,groups = K_enc*2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(K_enc*2, K_enc*2, 5,1,2,groups = K_enc*2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(K_enc*2, K_enc*4, 5,2,2),
        )
        self.down_scale16 = nn.Sequential(
            nn.Conv2d(K_enc*4, K_enc*4, 5,1,2,groups = K_enc*4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(K_enc*4, K_enc*4, 5,1,2,groups = K_enc*4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(K_enc*4, K_enc*8, 3,2,1),
        )
        self.up_dim = nn.Sequential(
            upconv2x2(512, K_enc*8, mode = 'transpose'),
        )
        
        
        
        self.dec_net_16x = [
            ResBlock(K_enc*8, K_enc*8,norm  =norm ,video_len=clip_len,is_temporal=True)
            for _ in range(opt.layers_16x)
        ]
        self.dec_net_16x = nn.Sequential(*self.dec_net_16x)
        self.dec_net2= nn.Sequential(
            upconv2x2(K_enc*8, K_enc*4, mode = 'transpose'),
        )
        
        self.dec_net2_tail= nn.Sequential(
            ResBlock(K_enc*4, K_enc*4,norm  =norm ,video_len=clip_len,is_temporal=True),
            ResBlock(K_enc*4, K_enc*4,norm  =norm ,video_len=clip_len,is_temporal=True),
        )
        self.dec_net3= nn.Sequential(
            upconv2x2(K_enc*4, K_enc*2, mode = 'transpose'),
        )
        self.dec_net3_tail = nn.Sequential(
            ResBlock(K_enc*2, K_enc*2,norm  =norm ,video_len=clip_len,is_temporal=True),
            ResBlock(K_enc*2, K_enc*2,norm  =norm ,video_len=clip_len,is_temporal=True)
        )
        self.fuse_16x = nn.Sequential(
            nn.Conv2d(K_enc*16, K_enc*16, 1,1,0),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(K_enc*16, K_enc*16, 3,1,1,groups=16),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(K_enc*16, K_enc*8, 1,1,0),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(K_enc*8, K_enc*8, 3,1,1,groups=8),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(K_enc*8, K_enc*8, 1,1,0),
        )
        self.fuse_8x = nn.Sequential(
            nn.Conv2d(K_enc*8, K_enc*8, 1,1,0),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(K_enc*8, K_enc*8, 3,1,1,groups=8),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(K_enc*8, K_enc*4, 1,1,0),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(K_enc*4, K_enc*4, 3,1,1,groups=4),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(K_enc*4, K_enc*4, 1,1,0),
        )
        
        self.fuse_4x = nn.Sequential(
           nn.Conv2d(K_enc*4, K_enc*4, 1,1,0),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(K_enc*4, K_enc*4, 3,1,1,groups=4),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(K_enc*4, K_enc*2, 1,1,0),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(K_enc*2, K_enc*2, 3,1,1,groups=2),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(K_enc*2, K_enc*2, 1,1,0),
        )
        self.dec_net_tail = [
            upconv2x2(K_enc*2, K_enc, mode = 'transpose'),
            ResBlock(K_enc, K_enc,norm  =norm ,video_len=clip_len,is_temporal=True),
            upconv2x2(K_enc, K_enc, mode = 'transpose'), 
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(K_enc, 3, 3,1,1)
        ]
        self.dec_net_tail = nn.Sequential(*self.dec_net_tail)
    def forward_32x_to_16x(self,received_code):
        dec_root_16x = self.up_dim(received_code)
        return dec_root_16x
    def forward(self,comp_video,comp_ft,received_code,received_code_8x):
        # comp_video = comp_video*0
        # if not type(received_code_8x) == torch.Tensor:
        #     received_code_8x = img_8x *0
        img_4x_compressed =   self.down_scale4(comp_video)
        img_8x_compressed = self.down_scale8(img_4x_compressed)
        img_16x_compressed = self.down_scale16(img_8x_compressed)
        
        dec_root_16x = self.up_dim(received_code)
        
        fused_r1 = self.fuse_16x(torch.cat([img_16x_compressed,dec_root_16x],dim=1))
        img_16x = fused_r1 + dec_root_16x
        img_16x = self.dec_net_16x(img_16x)
        dec_out2 = self.dec_net2(img_16x)
        fused_r2 = self.fuse_8x(torch.cat([img_8x_compressed,dec_out2],dim=1))
        img_8x = fused_r2 + dec_out2 
        
        
        img_8x = self.dec_net2_tail(img_8x)
        dec_out3 = self.dec_net3(img_8x)
        fused_r3 = self.fuse_4x(torch.cat([img_4x_compressed,dec_out3],dim=1))
        img_4x = fused_r3 + dec_out3  
        img_4x = self.dec_net3_tail(img_4x)
        restored_video = self.dec_net_tail(img_4x)
        return restored_video

class DiffCompensate_SMCPlus(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.mxrange = 50
        self.EMB_DIM = 256
        self.opt = opt
        
        

        self.clip_len = self.opt.num_segments
        is_temporal = self.opt.temporal_compression
        
        ###sem_net_{s} for VVC lossy video
        N = 64
        self.sem_net = nn.Sequential(
            nn.Conv2d(3,N,3,2,1),
            nn.LeakyReLU(0.01,True),
            PreNormConv(N,N*2,3,2,1,is_temporal = is_temporal,video_len = self.clip_len),
            
            nn.LeakyReLU(0.01,True),
            nn.Conv2d(N*2,N*2,7,1,3,groups = N),
            nn.LeakyReLU(0.01,True),
            PreNormConv(N*2,N*4,3,2,1,is_temporal = is_temporal,video_len = self.clip_len),
            
            nn.LeakyReLU(0.01,True),
            nn.Conv2d(N*4,N*4,7,1,3,groups = N*2),
            nn.LeakyReLU(0.01,True),
            PreNormConv(N*4,N*6,3,2,1,is_temporal = is_temporal,video_len = self.clip_len),
            
            nn.LeakyReLU(0.01,True),
            nn.Conv2d(N*6,N*6,7,1,3,groups = N*3),
            nn.LeakyReLU(0.01,True),
            PreNormConv(N*6,N*8,3,2,1,is_temporal = is_temporal,video_len = self.clip_len),
        )
        
        C_latent = 512
        self.vpt_anchor_updim = nn.Sequential(
            nn.Conv2d(128, 128, 3,1,1),
            ResBlockSimple(128, 128,video_len=self.clip_len,is_temporal=False,norm="NON"),
            nn.Conv2d(128, 256, 3,1,1),
            ResBlockSimple(256, 256,video_len=self.clip_len,is_temporal=False,norm="NON"),
            nn.Conv2d(256, 512, 3,1,1),
        )
        self.vpt_aligner = AlignFt1_toFt2HeterMask(512,128)
        self.vpt_aligner_adapter1 =self.vpt_aligner_adapter2=self.vpt_aligner_adapter3= nn.Identity( )
        self.ft_late_fusion_pre =  nn.Identity( )
        self.ft_late_fusion = nn.Sequential(
            nn.Conv2d( C_latent*2, C_latent, 3,1,1),
            ResBlockSimple(C_latent*1, C_latent*1,video_len=self.clip_len,is_temporal=True,norm="NON"),
            ResBlockSimple(C_latent*1, C_latent*1,video_len=self.clip_len,is_temporal=True,norm="NON"),
            ResBlockSimple(C_latent*1, C_latent*1,video_len=self.clip_len,is_temporal=True,norm="NON"),
            nn.Conv2d( C_latent, C_latent, 1,1,0),
            )
        self.vpt_anchor_select_net = nn.Sequential(
        nn.Conv2d(512, 256, 3,1,1),
        ResBlockSimple(256, 256,video_len=self.clip_len,is_temporal=True,norm="NON"),
        nn.Conv2d(256, 128, 3,1,1),
        ResBlockSimple(128, 128,video_len=self.clip_len,is_temporal=True,norm="NON"),
        nn.Conv2d(128, 128, 3,1,1),
        )
        self.vpt_anchor_enc = nn.Sequential(
            nn.Conv2d(512+128, 256, 3,1,1),
            ResBlockSimple(256, 256,video_len=self.clip_len,is_temporal=True,norm="NON"),
            nn.Conv2d(256, 128, 3,1,1),
            ResBlockSimple(128, 128,video_len=self.clip_len,is_temporal=True,norm="NON"),
            nn.Conv2d(128, self.opt.vpt_mask_dim, 3,1,1),
        )
        self.vpt_anchor_dec = nn.Sequential(
            nn.Conv2d(512+self.opt.vpt_mask_dim, 256, 3,1,1),
            ResBlockSimple(256, 256,video_len=self.clip_len,is_temporal=True,norm="NON"),
            nn.Conv2d(256, 128, 3,1,1),
            ResBlockSimple(128, 128,video_len=self.clip_len,is_temporal=True,norm="NON"),
            nn.Conv2d(128, 128, 3,1,1),
        )
        
            
        point_token = 5
        atten_type = "abs"
        self.vpt_contextEnc_transformer = TransformerLearnedRes( 512,self.opt.vpt_coding_depth,16,512//16,512*self.opt.vpt_MLP_multi,token_num = point_token,atten_type = atten_type) ## 512 mask
        self.vpt_contextDec_transformer = TransformerLearnedRes( 512,self.opt.vpt_coding_depth,16,512//16,512*self.opt.vpt_MLP_multi,token_num = point_token,atten_type = atten_type) ## 512 mask
        self.vpt_SpatialEnc_transformer = TransformerLearnedRes( 256,self.opt.vpt_coding_depth,16,256//16,512*self.opt.vpt_MLP_multi,token_num = 64,atten_type = atten_type) ## 512 mask
        self.vpt_SpatialDec_transformer = TransformerLearnedRes( 256,self.opt.vpt_coding_depth,16,256//16,512*self.opt.vpt_MLP_multi,token_num = 64,atten_type = atten_type) ## 512 mask
        
        self.vpt_contextEnc_linear =nn.Sequential(
            ResBlockSimple(512, 384,video_len=self.clip_len,is_temporal=False,norm="NON"),
            ResBlockSimple(384, 256,video_len=self.clip_len,is_temporal=False,norm="NON"),
        )
        self.vpt_contextDec_linear = nn.Sequential(
            ResBlockSimple(256, 384,video_len=self.clip_len,is_temporal=False,norm="NON"),
            ResBlockSimple(384, 512,video_len=self.clip_len,is_temporal=False,norm="NON"),
        )
        self.enc_MLP  = nn.Sequential(
            nn.Conv2d(256,256,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,256,3,1,1),
        )
        self.dec_MLP  = nn.Sequential(
            nn.Conv2d(256,256,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,256,3,1,1),
        )
             
        
        self.f_net = FNetSMC(opt,norm = "NON")
        self.encoder = resnet18(weights=False,is_temporal = is_temporal,video_len = self.clip_len,attention_branch  =False)
        replace_bn_with_identity_and_relu_with_leakyrelu(self.encoder)
        
        self.bitEstimator_mv = BitEstimator_MVSC(self.opt.vpt_mask_dim)
        self.bitEstimator_mv1 = BitEstimator_MVSC(self.opt.latent_dim)
            
        
            
    def forward(self,data,mode):
        if mode == "enc":
            _1,_2,_3,_4  = self.enc(data[0],data[1])
            return _1,_2
        elif mode == "dec":
            return self.dec(data[0],data[1])
        elif mode == "dec_vid_only":
            return self.dec_vid(data[0],data[1],data[2])
        elif mode =='enc_dec':
            input_clean,input_com = data[0],data[1]
            code,codebpp,ft_comp,ft_clean = self.enc(input_clean,input_com)
            recon_v,_,_  = self.dec_vid(code,input_com )
            return recon_v,codebpp
    
    
    def Q(self,code):
        quant_mv = torch.round(code)
        return quant_mv
    
   
                
    def enc(self, clean,noise):
        b,c,t,h,w = clean.size()
        self.img_t,self.img_h,self.img_w = t,h,w
        
        clean = clean.permute(0,2,1,3,4).reshape(b*t,c,h,w)
        comp = noise.permute(0,2,1,3,4).reshape(b*t,c,h,w)
        
        ##Sem-Net for high resolution video
        ft_clean,x_32x,x_16x,x_8x,x_4x = self.encoder(clean,0,0,0,0)
        self.ft_clean = ft_clean
        ##Sem-Net_{s} for VVC lossy video
        ft_comp_4x = self.sem_net[0:3](comp)  
        ft_comp_8x = self.sem_net[3:7](ft_comp_4x)  
        ft_comp_16x = self.sem_net[7:11](ft_comp_8x)
        ft_comp = self.sem_net[11:](ft_comp_16x)
            
        
        

        
        win_size = self.opt.vpt_win_size
        ft_b,ft_c,ft_h,ft_w = ft_clean.size()
        
        _b = ft_b//self.clip_len
        _c = ft_c
        
        ft_anchor_raw = self.vpt_anchor_select_net(ft_clean)
        ft_anchor_enc = self.vpt_anchor_enc(torch.cat([ft_anchor_raw,ft_comp],dim=1))
        ft_anchor_enc[:,32:] = 0
        ft_anchor_Qcode = self.Q(ft_anchor_enc)
        ft_anchor = self.vpt_anchor_dec(torch.cat([ft_anchor_Qcode ,ft_comp],dim=1))
        ft_clean_3d = d2_to_d3(ft_clean,self.clip_len)
        ft_comp_3d = d2_to_d3(ft_comp,self.clip_len)
        ft_anchor_3d = d2_to_d3(ft_anchor,self.clip_len)
        h_win_num = ft_h//self.opt.vpt_win_size
        w_win_num = ft_w//self.opt.vpt_win_size
        if ft_h%self.opt.vpt_win_size>0:
            h_win_num = h_win_num+1
        if ft_w%self.opt.vpt_win_size>0:
            w_win_num = w_win_num+1
        _b_win_num = _b*h_win_num * w_win_num
        #
        ft_clean_hat_pre = torch.zeros((_b,ft_c,ft_h,ft_w)).cuda(ft_clean.device) ## ft_clean_hat_window_3d_pre
        spatialwise_Qcodes = []
        contextual_feature_predicted = []
        for t_i in range(self.clip_len):
            if self.opt. only_cal_cost:
                torch.cuda.synchronize()
                start = time.time()*1000
            ft_clean_cur_frame = ft_clean_3d[:,:,t_i]
            ###contexts
            ft_clean_hat_pre
            ft_comp_cur = ft_comp_3d[:,:,t_i]
            if t_i==0:
                ft_comp_pre = ft_comp_cur*0
            else:
                ft_comp_pre = ft_comp_3d[:,:,t_i-1]
            ft_anchor_cur = ft_anchor_3d[:,:,t_i]
            ft_comp_cur_no_aligned = ft_comp_cur
            all_contexts_before_align = torch.cat([self.vpt_aligner_adapter1(ft_clean_hat_pre),
                                                    self.vpt_aligner_adapter2(ft_comp_cur),
                                                    self.vpt_aligner_adapter3(ft_comp_pre)],dim=0)
            ft_anchor_all = torch.cat([ft_anchor_cur,ft_anchor_cur,ft_anchor_cur],dim=0)
            all_contexts_after_align = self.vpt_aligner(all_contexts_before_align,ft_anchor_all )
            ft_clean_hat_pre,ft_comp_cur,ft_comp_pre = torch.chunk(all_contexts_after_align,3,dim=0)
            
                
            ft_anchor_cur = self.vpt_anchor_updim(ft_anchor_cur)
            
            ft_clean_hat_pre_wind = window_2d(ft_clean_hat_pre,win_size).reshape(_b_win_num,_c, win_size*win_size ).transpose(1,2)
            ft_comp_cur_wind = window_2d(ft_comp_cur,win_size).reshape(_b_win_num,_c, win_size*win_size ).transpose(1,2)
            ft_comp_pre_wind = window_2d(ft_comp_pre,win_size).reshape(_b_win_num,_c, win_size*win_size ).transpose(1,2)
            ft_anchor_cur_wind = window_2d(ft_anchor_cur,win_size).reshape(_b_win_num,_c, win_size*win_size ).transpose(1,2)
            ft_clean_cur_frame_wind = window_2d(ft_clean_cur_frame,win_size).reshape(_b_win_num,_c, win_size*win_size ).transpose(1,2)
            
            ft_clean_hat_pre_wind_flat = ft_clean_hat_pre_wind.reshape(_b_win_num*win_size*win_size,1,_c)
            ft_comp_cur_wind_flat = ft_comp_cur_wind.reshape(_b_win_num*win_size*win_size,1,_c)
            ft_comp_pre_wind_flat = ft_comp_pre_wind.reshape(_b_win_num*win_size*win_size,1,_c)
            ft_anchor_cur_wind_flat = ft_anchor_cur_wind.reshape(_b_win_num*win_size*win_size,1,_c)
            ft_clean_cur_frame_wind_flat = ft_clean_cur_frame_wind.reshape(_b_win_num*win_size*win_size,1,_c)
            
            
            pointwise_code = self.vpt_contextEnc_transformer(torch.cat([
            ft_clean_hat_pre_wind_flat,ft_comp_cur_wind_flat,ft_comp_pre_wind_flat,ft_anchor_cur_wind_flat,ft_clean_cur_frame_wind_flat
            ],dim=1))[:,4:]
            _c = 256
            def window1d_to_image(x,_c):
                return dewindow2d(
                x.reshape(_b_win_num,win_size*win_size,_c ).transpose(1,2),
                _b,h_win_num,w_win_num,_c,win_size
                )[:,:,0:ft_h,0:ft_w]
            def image_to_window1d(x,_c):
                return window_2d(x,win_size).reshape(_b_win_num,_c, win_size*win_size ).transpose(1,2)
            pointwise_code = image_to_window1d(self.vpt_contextEnc_linear(window1d_to_image(pointwise_code,512)),256)
            
            spatialwise_code = self.vpt_SpatialEnc_transformer(
                pointwise_code
            )
            spatialwise_Qcode = self.Q(image_to_window1d(self.enc_MLP(window1d_to_image(spatialwise_code,_c)),_c))
            
            spatialwise_Qcodes += [dewindow2d(
                spatialwise_Qcode.reshape(_b_win_num,win_size*win_size,_c ).transpose(1,2),
                _b,h_win_num,w_win_num,_c,win_size
                )[:,:,0:ft_h,0:ft_w]
                ]
            spatialwise_code = image_to_window1d(self.dec_MLP(window1d_to_image(spatialwise_Qcode,_c)),_c)
            
            spatialwise_code = self.vpt_SpatialDec_transformer(
                spatialwise_code
            ).reshape(_b_win_num,win_size*win_size,_c )
            _c = 512
            pointwise_code = image_to_window1d(self.vpt_contextDec_linear(window1d_to_image(spatialwise_code,256)),512).reshape(_b_win_num*win_size*win_size,1,_c)
            
            pointwise_code = self.vpt_contextDec_transformer(torch.cat([
                ft_clean_hat_pre_wind_flat,ft_comp_cur_wind_flat,ft_comp_pre_wind_flat,ft_anchor_cur_wind_flat,pointwise_code
                ],dim=1))[:,4:]
            ft_clean_hat_cur_frame =pointwise_code.reshape(_b_win_num,win_size*win_size,_c ).transpose(1,2)
            ft_clean_hat_cur_frame = dewindow2d(ft_clean_hat_cur_frame,_b,h_win_num,w_win_num,_c,win_size)[:,:,0:ft_h,0:ft_w]
            ft_clean_hat_pre = ft_clean_hat_cur_frame
            
            contextual_feature_predicted += [ft_clean_hat_cur_frame]
            
        ft_residual_Q1 = ft_anchor_Qcode
        ft_residual_Q2 = torch.stack(spatialwise_Qcodes, dim=2)
        ft_residual_Q2= d3_to_d2(ft_residual_Q2)
        contextual_feature_predicted = torch.stack(contextual_feature_predicted,dim=2)
        contextual_feature_predicted = d3_to_d2(contextual_feature_predicted)
        ft_clean_hat =contextual_feature_predicted+self.ft_late_fusion(torch.cat([contextual_feature_predicted,ft_comp],1))
        
      
        
        bpp, bits_in_batch  =ft_clean.mean()*0,0
        bpp_in_batch1, bits_in_batch1  =estimate_bits_factorized(self.bitEstimator_mv,ft_residual_Q1,t,h,w)
        bpp_in_batch2, bits_in_batch2  =estimate_bits_factorized(self.bitEstimator_mv1,ft_residual_Q2,t,h,w)
        
        bpp_in_batch, bits_in_batch  = bpp_in_batch1+bpp_in_batch2,  bits_in_batch1 +bits_in_batch2
                
            
        
        bpp = bpp_in_batch.mean()
            
            
            
        self.bpp_sum = bpp * b*t*h*w
        self.ft_comp = ft_comp
        return ft_clean_hat, [bpp, bits_in_batch],ft_comp,ft_clean
   
    def dec_vid(self,code,I_decode_last ):	
        ft_comp  =self.ft_comp
        b,c,t,h,w = I_decode_last.size()
        x = I_decode_last.permute(0,2,1,3,4).reshape(b*t,c,h,w)
        restored_video_2d = self.f_net(x,ft_comp,code,0)
        
        
        
        return d2_to_d3(restored_video_2d,t),[],[]
    
    
    def dec(self,code,I_decode_last ):
        restored_video,aux_losses,vis_maps = self.dec_vid(code, I_decode_last )
        
        return restored_video,{
            "map_vis":vis_maps
        },[]\
            ,[]
                

