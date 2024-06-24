import argparse
from PIL import Image, ImageOps
import os
current_file_dir = os.path.dirname( os.path.abspath(__file__))
import sys
sys.path.append(os.path.dirname(os.path.dirname(current_file_dir)))
import torchvision
import time
import shutil
import torch
import torch.nn as nn
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import numpy as np

# from dataset import VideoDataSet
from dataset_online_seg import VideoDataSetSegOnline
from opts import parser
import datasets_video
import torch.nn.functional as F
# from piq import MultiScaleSSIMLoss, multi_scale_ssim
from piq import FID
from SMC_ICCV2023.pre_compressor import VideoPreCompressor
from transforms import Stack,ToTorchFormatTensor

best_prec1 = 0
# STAGE1 = False
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

def main():
    

    global args, best_prec1
    args = parser.parse_args()

    categories, train_list, val_list, root_path, prefix = datasets_video.return_dataset(args.dataset,args.root_path)
    num_class = len(categories)


    global store_name 
    store_name = '_'.join([args.type, args.dataset, args.arch, 'segment%d'% args.num_segments, args.store_name])
    print(('storing name: ' + store_name))

    
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    spatial_group = 1
    if args.test_codec_gop in  [32,'32']:
        ## to avoid OOM error
        spatial_group = 2
    student_model  = VideoPreCompressor( args.encoder_net_type,args.latent_dim,args.num_segments,args.temporal_compression\
    ,"rnet",1,args.last_Tlayer_causual,1,
    args.resume_main,args.codec_type,args.test_crf,"zerolatency",args.test_codec_gop,
    '',network_spatial_group=spatial_group)
    if torch.cuda.is_available():
        student_model = torch.nn.parallel.DistributedDataParallel(student_model.cuda(), device_ids=[torch.cuda.current_device()],\
            broadcast_buffers=False , find_unused_parameters=True)
    student_model.eval()
    ## begin configure val ds
    categories, train_list, val_list, root_path, prefix = datasets_video.return_dataset(args.dataset_eval,args.root_path)
    val_ds = VideoDataSetSegOnline(
        "XMem/dataset_space/DAVIS_raw_480/trainval/JPEGImages/480p",
     "VideoDatasetList/Davis2017/valid.txt",
             num_segments=10,only_clean  =True,
                   image_tmpl='{:05d}.png',
                   random_shift=False,
                   dup_num = 1,
                   begin_index = 0,
                   transform=torchvision.transforms.Compose([
                       Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                   ]))
    
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,drop_last=False,sampler=DistributedSampler(val_ds))

    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    log_training = None
    
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = None
    if dist.get_rank() <= 0:
        tb_logger = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir,'board'),flush_secs=10)
    

    

    teacher_model = 0
    if args.evaluate:
        prec1 = validate(val_loader,student_model, teacher_model, criterion,\
             0,0,log_training,tb_logger,\
                 args)
        return
    
from util import *
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()
    return rt
def dict2str(opt, indent_l=1):
    if dict2str == None:
        return ""
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            if isinstance(v, torch.Tensor):
                v = v.item()
                v = '%.4f'%v
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) 
    
    return msg

def restore_to_original_size(input_clean, dest_h,dest_w):
    b,c,t,h,w = input_clean.size()
    i = input_clean.permute(0,2,1,3,4).reshape(b*t,c,h,w)
    i = F.upsample(i, size = (dest_h,dest_w),mode ='bilinear')
    i = i.reshape(b,t,c,dest_h,dest_w).permute(0,2,1,3,4)
    input_clean = i
    return input_clean
def run_model(_,input_clean,target,student_model,criterion,args):
    clip_len =input_clean.size(1)//3
    input_clean = input_clean.view((-1, clip_len,3) + input_clean.size()[-2:])
    input_clean = input_clean.transpose(1,2).contiguous()### b c t h w
    video_scale = clip_len *input_clean.size(-2) *input_clean.size(-1) 
    
    
    input_clean_copy = input_clean.clone()
    src_h,src_w = input_clean.size(-2), input_clean.size(-1)
    pad_h,pad_w = (src_h //128), (src_w //128)
    if src_h%128>0:
        pad_h = pad_h+1
    if src_w*128>0:
        pad_w = pad_w+1
    pad_h  =pad_h *128
    pad_w  =pad_w *128
    input_clean_padded = torch.zeros((input_clean.size(0),input_clean.size(1),clip_len,pad_h,pad_w))
    input_clean_padded[:,:,:,0:src_h,0:src_w] = input_clean
    input_clean = input_clean_padded
    video_scale = clip_len * src_h * src_w

    h,w = input_clean.size(-2),input_clean.size(-1)
    P_h = 1
    P_w = 1

    half_h,half_w = h//P_h,w//P_w
    b,c,t,h,w = input_clean.size()

    input_clean_p = input_clean.reshape(b,c,t,P_h,half_h,P_w,half_w).permute(0,3,5,1,2,4,6).reshape(b*P_h*P_w,c,t,half_h,half_w)
    code_meta_sum = 0
    code_h265_sum = 0
    recons = []
    if True:
        for i in range(b*P_h*P_w):
            h265_bitcosts,code_bpp,recons_clean = student_model(input_clean_p[i:i+1],
                                                                {
                                                                    "mean":torch.zeros([1,3],device = input_clean_p.device),
                                                                    "std":torch.ones([1,3],device = input_clean_p.device).fill_(255),
                                                                })
            recons += [recons_clean]
            code_h265_sum += h265_bitcosts
            code_meta_sum += code_bpp
        if not type(recons[0]) == type(None):
            recons_clean = torch.cat(recons,dim=1)
            recons_clean = recons_clean.reshape(b,P_h,P_w,c,t,half_h,half_w).permute(0,3,4,1,5,2,6).reshape(b,c,t,h,w)
            recons_clean_output  = recons_clean[:,:,:, 0:src_h,0:src_w]
        else:
            recons_clean = None
    elif args.encoder_net_type == "null":
        recons_clean = None


    input_clean_output = input_clean_copy

    

    
    h265_bitcosts = code_h265_sum/video_scale
    code_meta_sum = code_meta_sum/video_scale
    pixel_recon_loss  = 0
    
    loss_action = 0
    
    
    loss_sum = loss_action*args.lambda_action + (pixel_recon_loss*args.lambda_pixel )
    vis_dict = {}
    
    acc1 = 0
    acc5 = 0
    
    recons_clean_output = recons_clean_output.clamp(0,1)
    input_clean_output = input_clean_output.clamp(0,1)
    return vis_dict, loss_sum,acc1,acc5,0,recons_clean_output,input_clean_output,h265_bitcosts,code_meta_sum,{
        "input_com":0
    }



    
def cal_fid(x,y):
    fid_metric = FID()
    first_feats = fid_metric.compute_feats(x)
    second_feats = fid_metric.compute_feats(y)
    fid = fid_metric(first_feats, second_feats)
    return fid

def validate(val_loader, student_model, teacher_model, criterion,epoch_index, iter, log ,tb_logger,opt):

    # switch to evaluate mode
    end = time.time()
    i = 0
    psnr_recon_list = []
    ssim_recon_list = []
    if dist.get_rank() == 0:
        H265_br_list = []
        meta_br_list = []
        sum_br_list = []
    with torch.no_grad():
        for i, (_,input_clean, target,img_path_l) in enumerate(val_loader):
            input_clean = input_clean.cuda(non_blocking = True)
            batch_index = i
            # if i ==2:
            # 	break
            img_path_l = map(lambda x : x[0],img_path_l)
            img_path_l = list(img_path_l)
            ###  aussuming one clip on one gpu
            vis_dict, loss_sum,acc1,acc5,input_com, recons_clean,input_clean,h265_bitcost,code_meta_sum,aux_v =\
                 run_model(_,input_clean,target,student_model,criterion,args)
            input_com = aux_v["input_com"]
            src_path = "/DAVIS_raw_480/"
            
            recons_clean = recons_clean[0]
            for i in range(len(img_path_l)):
                ## recons_clean 1 c t h w
                p_now = img_path_l[i]
                
                if args.encoder_net_type == "smc_plus":
                    if opt.test_codec_gop in [10,"10"]:
                        dest_path = "/DAVIS_raw_480_{}_smcplus_gop10_crf{}/".format(opt.codec_type,opt.test_crf)
                    elif opt.test_codec_gop in [32,"32"]:
                        dest_path = "/DAVIS_raw_480_{}_smcplus_gop32_crf{}/".format(opt.codec_type,opt.test_crf)
                    path_now_recon =  p_now.replace(args.trancode_src_dir,args.trancode_dest_dir)
                    if not os.path.exists(os.path.dirname(path_now_recon)):
                        os.makedirs(os.path.dirname(path_now_recon),exist_ok=True)
                    print("saving",recons_clean.device, path_now_recon, "index",batch_index)
                    im_recon = Image.fromarray (
                        (recons_clean[:,i].permute(1,2,0)*255).cpu().numpy().astype(np.uint8)
                    )
                    im_recon.save(path_now_recon,quality =100)
                else :
                    print("unkow encoder_net_type" ,args.encoder_net_type)
                    exit()
                

            h265_bitcost = h265_bitcost.to(code_meta_sum.device)
            sum_br = h265_bitcost+ code_meta_sum
            
            dist.all_reduce(h265_bitcost, op=dist.ReduceOp.SUM)
            h265_bitcost = h265_bitcost/dist.get_world_size()
            dist.all_reduce(code_meta_sum, op=dist.ReduceOp.SUM)
            code_meta_sum = code_meta_sum/dist.get_world_size()
            dist.all_reduce(sum_br, op=dist.ReduceOp.SUM)
            sum_br = sum_br/dist.get_world_size()
            if dist.get_rank() == 0:
                print("bitcost  h265 meta all",h265_bitcost.item(),code_meta_sum.item(),sum_br.item() )
                H265_br_list += [h265_bitcost.item()]
                meta_br_list += [code_meta_sum.item()]
                sum_br_list += [sum_br.item()]
    if dist.get_rank() == 0:
        print("average  H265_br {}, meta_br {}, sum_br {}".format(
            sum(H265_br_list)/len(H265_br_list),
            sum(meta_br_list)/len(meta_br_list),
            sum(sum_br_list)/len(sum_br_list),
        ))







if __name__ == '__main__':
    main()
