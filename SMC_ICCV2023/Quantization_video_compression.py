import torch
import torch.nn as nn
import skvideo.io
import numpy as np
import os
import random
import time
import threading
import hashlib
import subprocess
import shutil
from common import *
cheap_hash = lambda input: hashlib.md5(input).hexdigest()[:8]
project_root_path = current_file_dir = os.path.dirname(os.path.dirname( os.path.abspath(__file__)))
def convert_1video(input,other_p):
    _c,_t,_h,_w =input.size()
    if _c == 3:
        ##normal image
        return convert_1video_inner(input,other_p)
    if _c== 64:
        input = input.reshape(1,8,8,_t,_h,_w).permute(0,3,1,4,2,5).reshape(1,_t,8*_h,8*_w)
    if _c== 80:
        input = input.reshape(5,4,4,_t,_h,_w).permute(0,3,1,4,2,5).reshape(5,_t,4*_h,4*_w)
    if _c== 128:
        input = input.reshape(2,8,8,_t,_h,_w).permute(0,3,1,4,2,5).reshape(2,_t,8*_h,8*_w)
    if _c== 256:
        ##resnet50 output
        input = input.reshape(1,16,16,_t,_h,_w).permute(0,3,1,4,2,5).reshape(1,_t,16*_h,16*_w)
    if _c== 512:
        ##resnet50 output
        input = input.reshape(2,16,16,_t,_h,_w).permute(0,3,1,4,2,5).reshape(2,_t,16*_h,16*_w)
    if _c== 768:
        ##Transformer output
        # input = input.reshape(3,16,16,_t,_h,_w).permute(0,3,1,4,2,5).reshape(3,_t,16*_h,16*_w)
        input = torch.cat([input,input[0:256]*0],dim=0)
        input = input.reshape(1,32,32,_t,_h,_w).permute(0,3,1,4,2,5).reshape(1,_t,32*_h,32*_w)
    if _c== 1024:
        ##resnet50 output
        input = input.reshape(1,32,32,_t,_h,_w).permute(0,3,1,4,2,5).reshape(1,_t,32*_h,32*_w)
    c,t,h,w =input.size()
    
    if c%3 == 0:
        group = c//3
        last_pad = 0
    else:
        group = c//3 +1
        last_pad = 3 - c%3
    bpp_sum = 0
    ft_sum = []
    for g_i in range(group):
        if g_i == (group-1) and last_pad>0:
            current_ft  = torch.cat([
                input[g_i*3:],
                input[0: last_pad]*0
            ],dim=0)
        else:
            current_ft  = input[g_i*3: (g_i+1)*3]
        ## for video classification task, we adopt this implementatiomn
        if other_p["feature_norm_location"] == 'in_group':
            current_ft = current_ft.float()
            max_v = current_ft.max()
            min_v = current_ft.min()
            space_v = (max_v -min_v )/256
            current_ft = ((current_ft-min_v)/space_v).round()
        
        
        comp_current_ft,comp_bpp =   convert_1video_inner(current_ft,other_p)
        ## for video classification task, we adopt this implementatiomn
        if other_p["feature_norm_location"] == 'in_group':
            comp_current_ft  =  comp_current_ft* space_v + min_v
        
        ft_sum += [comp_current_ft]
        ## for video classification task, we adopt this implementatiomn
        if other_p["feature_norm_location"] == 'in_group':
            bpp_sum  = bpp_sum + comp_bpp + 32*2
        else:
            bpp_sum  = bpp_sum + comp_bpp 
    ft_sum = torch.cat(ft_sum,dim=0)[0:c]
    if _c== 64:
        ft_sum = ft_sum.reshape(1,_t,8,_h,8,_w).permute(0,2,4,1,3,5).reshape(_c,_t,_h,_w)
    if _c== 80:
        ft_sum = ft_sum.reshape(5,_t,4,_h,4,_w).permute(0,2,4,1,3,5).reshape(_c,_t,_h,_w)
    if _c== 128:
        ft_sum = ft_sum.reshape(2,_t,8,_h,8,_w).permute(0,2,4,1,3,5).reshape(_c,_t,_h,_w)
    if _c== 256:
        ft_sum = ft_sum.reshape(1,_t,16,_h,16,_w).permute(0,2,4,1,3,5).reshape(_c,_t,_h,_w)
    if _c== 512:
        ft_sum = ft_sum.reshape(2,_t,16,_h,16,_w).permute(0,2,4,1,3,5).reshape(_c,_t,_h,_w) 
    if _c== 768:
        # ft_sum = ft_sum.reshape(3,_t,16,_h,16,_w).permute(0,2,4,1,3,5).reshape(_c,_t,_h,_w) 
        ft_sum = ft_sum.reshape(1,_t,32,_h,32,_w).permute(0,2,4,1,3,5).reshape(256+_c,_t,_h,_w) 
        ft_sum = ft_sum[0:768]
    if _c== 1024:
        ft_sum = ft_sum.reshape(1,_t,32,_h,32,_w).permute(0,2,4,1,3,5).reshape(_c,_t,_h,_w) 
    # print(F.mse_loss(ft_sum,input))
    return ft_sum, bpp_sum
        

def convert_1video_inner(input,other_p):
    output = input
    c,t,h,w =output.size()
    
        
    
    frames = output.permute(1,2,3,0)## t h w c
    frames = frames.cpu().numpy().astype(np.uint8)
    pid = str(other_p["batch_id"] )
    # pid = cheap_hash(pid.encode("utf-8"))
    # print(pid,"pid")
    file_n = other_p["file_random_name"]+"_"+pid
    
    type = other_p['type']
    tune = other_p['tune']
    keyint = other_p['keyint']
    codec_config_file = other_p['codec_config_file']
    if type in ["h264",'h265']:
        video_name = os.path.join(project_root_path, "coding_buffer_space/outputvideo"+file_n+"cvb.mkv")
        if type == 'h265':
            p = {
                '-c:v': 'libx265',
                '-preset':'veryfast',
                "-s":str(w)+"x"+str(h),
                '-pix_fmt': 'yuv420p',
                "-x265-params":"crf="+str(other_p["q"])+":keyint="+str(keyint)+":no-info=1"
            }
            ##PLVC
            # p["-preset"] = 'medium'
        elif type == 'h264':
            if other_p["q"]<=51:
                ### normal q
                p = {
                    '-c:v': 'libx264',
                    '-preset':'veryfast',
                    "-s":str(w)+"x"+str(h),
                    '-pix_fmt': 'yuv420p',
                    '-crf': str(other_p["q"]),
                }
                if keyint>0:
                    p = {
                    '-c:v': 'libx264',
                    '-preset':'veryfast',
                    "-s":str(w)+"x"+str(h),
                    '-pix_fmt': 'yuv420p',
                    '-crf': str(other_p["q"]),
                    "-x264opts":"keyint={}:min-keyint={}".format(keyint,keyint)
                    }
            else:
                ### const bit mode
                p = {
                    '-c:v': 'libx264',
                    '-preset':'veryfast',
                    "-s":str(w)+"x"+str(h),
                    '-pix_fmt': 'yuv420p',
                    '-maxrate':'5k',
                    '-bufsize':'5k',
                }
        if not tune == 'default':
            p["-tune"] = tune
        writer = skvideo.io.FFmpegWriter(video_name,outputdict = p,verbosity = 0)
        for i in range(t):
            writer.writeFrame(frames[i, :, :, :])
        writer.close()

        file_size = os.path.getsize(video_name)
        bit_cost = file_size*8.0
        reader = skvideo.io.FFmpegReader(video_name,
                        inputdict= {},
                        outputdict={},verbosity = 0)
        # iterate through the frames
        decoded_frames = []
        for frame in reader.nextFrame():
            # do something with the ndarray frame
            # decoded_frames += [torch.from_numpy(frame).cuda(input.device)]
            decoded_frames += [torch.from_numpy(frame)]
        # print('runing time2 %s ms' % ((T3 - T2)*1000))
        decoded_frames = torch.stack(decoded_frames,dim=0)
        decoded_frames = decoded_frames.permute(3,0,1,2)
    elif type == "h266":
        
        video_root =  os.path.join(project_root_path,"coding_buffer_space/outputvideo"+file_n+"_")
        p = {
                    "-s":str(w)+"x"+str(h),
                    '-pix_fmt': 'yuv420p',
                }
        input_vide_path = video_root +"input.yuv"
        writer = skvideo.io.FFmpegWriter(input_vide_path,outputdict = p,verbosity = 0)
        for i in range(t):
            writer.writeFrame(frames[i, :, :, :])
        writer.close()

        vtm_sequence_cfg = video_root+"vtm_seq.cfg"
        template_IO_cfg = os.path.join(project_root_path, "SMC_ICCV2023/VVCenc_VTM_config/vtm_sequence.cfg")
        if len(codec_config_file) == 0:
            if keyint == "10" or keyint == 10:
                codec_cfg = os.path.join(project_root_path, "SMC_ICCV2023/VVCenc_VTM_config/vtm_lowdeplayP.cfg")
            elif keyint == "10_128CTUMid":
                codec_cfg = os.path.join(project_root_path, "SMC_ICCV2023/VVCenc_VTM_config/vtm_lowdeplayP_medium.cfg")
            elif keyint == "32" or keyint == 32  :
                codec_cfg = os.path.join(project_root_path, "SMC_ICCV2023/VVCenc_VTM_config/lowdelay_medium.cfg")
            else:
                raise Exception("keyint must be 10 or 32")
        else:
            codec_cfg = codec_config_file
        filename_266 = video_root +".bin"
        filename_266_dec = video_root+"dec.yuv"
        with open(template_IO_cfg,'r') as vtm_sequence_F:
            vtm_sequence_str = vtm_sequence_F.read()
            vtm_sequence_str = vtm_sequence_str.replace("1920NUMBER",str(w))
            vtm_sequence_str = vtm_sequence_str.replace("1080NUMBER",str(h))
            vtm_sequence_str = vtm_sequence_str.replace("600NUMBER",str(t))
            vtm_sequence_str = vtm_sequence_str.replace("32QPVALUE", str(other_p["q"]))
            vtm_sequence_str = vtm_sequence_str.replace("dummy.yuvSTRING",input_vide_path)
            vtm_sequence_str = vtm_sequence_str.replace("str.binSTRING",filename_266)
            vtm_sequence_str = vtm_sequence_str.replace("rec.yuvSTRING",filename_266_dec)
            
            
        with open(vtm_sequence_cfg,'w') as vtm_sequence_F:   
            vtm_sequence_F.write(vtm_sequence_str)
            vtm_sequence_F.flush()
        p =subprocess.Popen(
            [os.path.join(project_root_path,"vvenc_150/vvenc/bin/release-static/vvencFFapp"), "-c",\
                 codec_cfg, "-c", vtm_sequence_cfg],
                 stdout  = subprocess.DEVNULL
            )

        p.wait()
        if p.stdin:
            p.stdin.close()
        if p.stdout:
            p.stdout.close()
        if p.stderr:
            p.stderr.close()
        try:
            p.kill()
        except:
            pass
        file_size = os.path.getsize(filename_266)
        bit_cost = file_size*8.0
        # for i in range(10):
        reader = skvideo.io.FFmpegReader(filename_266_dec,
                        inputdict= {
                            "-s":str(w)+"x"+str(h),
                            '-pix_fmt': 'yuv420p',
                        },
                        outputdict={},verbosity = 0)
        decoded_frames = []
        for frame in reader.nextFrame():
            f = torch.from_numpy(np.copy(frame))
            if torch.numel(f) == 0:
                print(filename_266_dec,"filename_266_dec")
                exit()
            decoded_frames += [f]
        reader.close()
            # if len(decoded_frames)==t:
            #     break
        # for i in range(t):
        #    decoded_frames += [torch.from_numpy(frames[i, :, :, :])]
        try:
            decoded_frames = torch.stack(decoded_frames,dim=0)
        except:
            ##
            print(filename_266_dec,"error filename_266_dec")
            decoded_frames = []
            for i in range(t):
                decoded_frames += [torch.from_numpy(frames[i, :, :, :])]
            decoded_frames = torch.stack(decoded_frames,dim=0)
        
        decoded_frames = decoded_frames.permute(3,0,1,2)

    return decoded_frames,bit_cost
from joblib import Parallel, delayed
import torch.nn.functional as F

# class Quant(torch.autograd.Function):
    
#     @staticmethod
#     def forward(ctx, input):
#         if Quantization_H265.scale_times_in>1:
#             input = rescale_3d_video(input, 1/Quantization_H265.scale_times_in)
#         b,c,t,h,w = input.size()
#         output = torch.zeros_like(input,device = input.device)
#         video_list = []
#         bpp_list = []
#         other_p = {
#             "file_random_name":Quantization_H265.file_random_name,
#             "q":Quantization_H265.q,
#             'type': Quantization_H265.type,
#             'keyint': Quantization_H265.keyint,
#             'tune': Quantization_H265.tune,
#             'codec_config_file': Quantization_H265.codec_config_file,
#             # "scale_times": Quantization_H265.scale_times
#         }
#         if Quantization_H265.value_range == 1:
#             input_on_cpu = (input*255.0)
#         # input_on_cpu_min,input_on_cpu_max = input_on_cpu.min(), input_on_cpu.max()
#         # input_on_cpu_space = (input_on_cpu_max  - input_on_cpu_min)/256
#         # input_on_cpu = ((input_on_cpu - input_on_cpu_min)/input_on_cpu_space).floor()
#         input_on_cpu  =input_on_cpu.cpu()
#         # print(input_on_cpu.size(),"input_on_cpu")
        
#         v_crt,bpp_list = zip(*Quant.fuck (delayed(convert_1video)(input_on_cpu[b_i,:,:],other_p) for b_i in range(b)) )
#         output = torch.stack(v_crt,dim=0)
#         output = output.cuda(input.device)
#         # output = output *input_on_cpu_space + input_on_cpu_min
#         if Quantization_H265.value_range == 1:
#             output =output/255.0
#         output = output.to(input.device)
#         if Quantization_H265.scale_times_out>1:
#             output = rescale_3d_video(output, Quantization_H265.scale_times_out)
#         bpp_tensor = torch.FloatTensor(bpp_list).to(input.device)
#         return output,bpp_tensor

class ParallelShell():
    def __init__(self) -> None:
        pass
ParallelShell.processpool  = Parallel(n_jobs=16)
import random
class Quantization_H265(nn.Module):
    def __init__(self,q = 17,keyint = -1,\
        scale_times_in = 1,
        scale_times_out = 1,
        type = 'h265',tune='zerolatency',codec_config_file = "",feature_norm_location='in_group',patch_for_accelerate = (1,1)):
        super(Quantization_H265, self).__init__()
        self.q = q
        Quantization_H265.keyint = keyint
        self.scale_times_in = scale_times_in
        self.scale_times_out = scale_times_out
        self.patch_for_accelerate = patch_for_accelerate
        Quantization_H265.file_random_name = None
        Quantization_H265.type = type
        Quantization_H265.tune = tune
        Quantization_H265.codec_config_file = codec_config_file
        self.feature_norm_location = feature_norm_location
    def forward(self, input):
        if not Quantization_H265.file_random_name:
            Quantization_H265.file_random_name = cheap_hash(str(time.time()).encode("utf-8"))+cheap_hash(str(os.getpid()).encode("utf-8"))
            print("initilize Quantization_H265.file_random_name",Quantization_H265.file_random_name)
        if self.scale_times_in>1:
            input = rescale_3d_video(input, 1/self.scale_times_in)
        b,c,t,h,w = input.size()
        # output = torch.zeros_like(input,device = input.device)
        video_list = []
        bpp_list = []
        other_p = {
            "file_random_name":Quantization_H265.file_random_name,
            "q":self.q,
            'type': Quantization_H265.type,
            'keyint': Quantization_H265.keyint,
            'tune': Quantization_H265.tune,
            'codec_config_file': Quantization_H265.codec_config_file,
            "feature_norm_location": self.feature_norm_location
        }
        if self.feature_norm_location  == 'in_batch':
            b,c,t,h,w = input.size()
            max_values_b, _ = torch.max(input.reshape(b,-1), dim=1)
            max_values_b = max_values_b.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            min_values_b, _ = torch.min(input.reshape(b,-1), dim=1)
            min_values_b = min_values_b.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            val_space_b = (max_values_b  - min_values_b)/256
            input = ((input - min_values_b)/val_space_b).floor()
        if input.max()<2:
            input_on_cpu = (input*255.0).cpu()
        else:
            input_on_cpu = input.cpu()
        is_patchfy = self.patch_for_accelerate  == (1,1) and h > 720
        if not is_patchfy:
            h_p = self.patch_for_accelerate[0]
            w_p = self.patch_for_accelerate[1]
            input_on_cpu_patched = input_on_cpu.reshape(b, c,t,h_p,h//h_p, w_p,w//w_p)\
                .permute(0,3,5,1,2,4,6)\
                    .reshape(b*h_p*w_p, c,t,h//h_p,w//w_p )
            patch_b = b*h_p*w_p
        else:
            input_on_cpu_patched = input_on_cpu
            patch_b = b
        v_crt,bpp_list = zip(*ParallelShell.processpool (delayed(convert_1video)(input_on_cpu_patched[b_i,:,:],{**other_p, "batch_id":b_i}) for b_i in range(patch_b)) )
        output = torch.stack(v_crt,dim=0)
        if not is_patchfy:
            output = output.reshape(b,h_p,w_p, c,t,h//h_p,w//w_p )\
                .permute(0,3,4,1,5,2,6)\
            .reshape(b, c,t,h_p*h//h_p, w_p*w//w_p)
            bpp_tensor = torch.FloatTensor(bpp_list).to(input.device)
            bpp_tensor = bpp_tensor.reshape(b,h_p,w_p).sum(-1).sum(-1)
            # print(bpp_tensor)
            # exit()
        else:
            output = output
            bpp_tensor = torch.FloatTensor(bpp_list).to(input.device)
        output = output.to(input.device)
        if output.max()>2:
            output =output/255.0
        if self.scale_times_out>1:
            output = rescale_3d_video(output, self.scale_times_out)
        
        return output,bpp_tensor

import torchvision
class Quantization_Codec_ROI(nn.Module):
    def __init__(self,q = 17,keyint = -1,\
        scale_times_in = 1,
        scale_times_out = 1,
        type = 'h266_roi',tune='zerolatency',codec_config_file = ""):
        super(Quantization_Codec_ROI, self).__init__()
        if type == 'h266_roi':
            type = 'h266'
        self.inner_codec = Quantization_H265(q,keyint,scale_times_in,scale_times_out\
            ,type,tune,codec_config_file)
        self.saliency_net = torchvision.models.detection.retinanet_resnet50_fpn(weights= torchvision.models.detection.retinanet. RetinaNet_ResNet50_FPN_Weights.DEFAULT)
        self.saliency_net.eval()
        self.dilation_kernel = torch.ones((15, 15), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    def d3_to_d2(self,x):
        b,c,t,h,w = x.size()
        return x.transpose(1,2).reshape(b*t,c,h,w)
    def d2_to_d3(self,x,t ):
        bt,c,h,w = x.size()
        b = bt//t
        return x.reshape(b,t,c,h,w).transpose(1,2)
    def forward(self, input):
        b,c,clip_length,h,w = input.size()
        input_clean =  self.d3_to_d2(input)
        b_2d = input_clean.size(0)
        roi_maps = []
        bin_size = 32
        all_results = []
        Thresh = 0.5
        for b_2d_i in range(b_2d//bin_size):
            imgs = input_clean[b_2d_i*bin_size:(b_2d_i+1)*bin_size]
            preds = self.saliency_net(imgs)
            for pred in preds:
                boxes = pred['boxes']
                scores = pred['scores']
                # print(boxes)
                # exit()
                filtered_boxes = boxes[scores > Thresh]
                image_tensor = torch.zeros((1,h, w)).cuda(input.device)
                for box in filtered_boxes:
                    x1, y1, x2, y2 = box
                    image_tensor[0,int(y1):int(y2), int(x1):int(x2)] = 1
                roi_maps += [image_tensor]
        b_2d_i = b_2d//bin_size-1
        if b_2d%bin_size>0:
            imgs = input_clean[(b_2d_i+1)*bin_size:]
            preds = self.saliency_net(imgs)
            for pred in preds:
                boxes = pred['boxes']
                scores = pred['scores']
                filtered_boxes = boxes[scores > Thresh]
                image_tensor = torch.zeros((1,h, w)).cuda(input.device)
                for box in filtered_boxes:
                    x1, y1, x2, y2 = box
                    image_tensor[0,int(y1):int(y2), int(x1):int(x2)] = 1
                print(image_tensor.size(),"image_tensor")
                roi_maps += [image_tensor]
        roi_maps = torch.stack(roi_maps,dim=0)
        roi_maps[roi_maps>0] = 1.0
        roi_maps[roi_maps<=0] = 0.0
        roi_maps = F.conv2d(roi_maps, self.dilation_kernel.cuda(input.device),padding  = 7)
        roi_maps = roi_maps.clamp(0,1)
        bt,c,h,w = roi_maps.size()
        b,t = bt//clip_length,clip_length
        b,c,h,w = input_clean.size()
        input_clean_flat = input_clean.reshape(b*c,1,h,w)
        # input_clean_downinfo = F.conv2d(input_clean_flat,self.gaussian_kernel.cuda(input_clean.device),padding = self.gaussian_kernel_size//2)
        s = 2
        input_clean_downinfo = F.upsample_bilinear(F.upsample_bilinear(input_clean_flat,scale_factor=1/s)
                                                    ,scale_factor=s)
        input_clean_downinfo = input_clean_downinfo.reshape(b,c,h,w)
        # input_clean_downinfo = 0
        input_clean = input_clean * roi_maps + input_clean_downinfo *(1-roi_maps)
        input_clean = input_clean.clamp(0,1)
        # imgs = torchvision.utils.make_grid(torch.cat([roi_maps.repeat(1,3,1,1),input_clean],dim=0),nrow=10,normalize =False)
        # torchvision.utils.save_image(imgs,"2.jpg")
        input_com,h265_bitcosts = self.inner_codec(self.d2_to_d3(input_clean,clip_length))
        return input_com,h265_bitcosts
    


