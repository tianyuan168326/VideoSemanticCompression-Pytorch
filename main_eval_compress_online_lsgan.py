import argparse
import os
import sys


import time
import shutil
import torch
import torch.nn as nn
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from torch.nn.utils import clip_grad_value_ as clip_grad_value 
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from precode_agg import PrecodeTemporalModel
import copy
from common import set_requires_grad,d3_to_d2,d2_to_d3
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
from dataset_rand_root import RandCrfVideoDataSet
from transforms import *
from video_gan import *
from opts import parser
import datasets_video
from dataset_rand_root_video import TSNDataSet
import torch.nn.functional as F
best_prec1 = 0
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
def ema1(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())## life long
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        if ("mom" in k):
            par1[k].data.mul_(decay).add_(par2[("module."+k).replace("mom_","")].data.to(par1[k].device), alpha=1 - decay)
        # else:
        # 	par1[k].data.mul_(0).add_(par2["module."+k].data.to(par1[k].device), alpha=1 )

def ema2(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())## life long
    par2 = dict(model2.named_parameters())

    for k in par2.keys():
        if ("mom" in k):
            par1["module."+k].data.mul_(0).add_(par2[k].data.to(par1["module."+k].device), alpha=1 )
def main():
    cudnn.benchmark = True
    global args, best_prec1
    args = parser.parse_args()
    check_rootfolders()

    categories, train_list, val_list, root_path, prefix = datasets_video.return_dataset(args.dataset,args.root_path)
    num_class = len(categories)
    num_class  =101
        
     
 


    global store_name 
    store_name = '_'.join([args.type, args.dataset, args.arch, 'segment%d'% args.num_segments, args.store_name])
    print(('storing name: ' + store_name))

    ft_action = args.ft_action
    from precode_agg import PrecodeTemporalModel
    student_model = PrecodeTemporalModel(ft_action,args.input_size,num_class, args.num_segments, model = args.type, backbone=args.arch, 
                        alpha = args.alpha, beta = args.beta, crf = 43, 
                        dropout = args.dropout,opt = args)
    crop_size = student_model.crop_size
 

    
    
    
    
    
    train_augmentation = student_model.get_augmentation(args.aug_crop_scale,args.no_video_scale)
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    if args.gan_d == "default":
        dis_model = NLayerDiscriminator(3,args.dis_c,n_layers = args.dis_depth)
    elif args.gan_d == 'patch_ms':
        dis_model = NLayerDiscriminatorPatchMS(3,args.dis_c,n_layers = args.dis_depth,norm_type =args.gan_norm_type )
    elif args.gan_d == 'large':
        dis_model = LargeDiscriminator(3,args.dis_c,n_layers = args.dis_depth,norm_type =args.gan_norm_type )
    elif args.gan_d == "default_sn":
        dis_model = NLayerDiscriminatorSN(3,args.dis_c,3)
    if args.gan_d == "default_vid":
        dis_model = NLayerVideoDiscriminator(3,args.dis_c,4)
    if args.gan_d == "ms":
        dis_model = MultiscaleDiscriminator(3,args.dis_c,3)
   
        
  
  
    if os.path.isfile(args.resume_main):
        print(("=> loading checkpoint '{}'".format(args.resume_main)))
        checkpoint = torch.load(args.resume_main,map_location='cpu')
        ckeck = checkpoint['state_dict']
        ckeck_copy = {}
        ks  =ckeck.keys()
        for k in ks:
            new_k = k
            if k.startswith("module"):
                new_k = k[7:]
            if 'lpips_perceptual_loss' in k:
                continue
            if 'vgg_perceptual_loss' in k:
                continue
            if ".down_dim." in k:
                if "0" in k or "3" in k or "6" in k or "9" in k or "12" in k :
                    continue
            if ".up_dim." in k:
                if "0" in k or "3" in k or "6" in k or "9" in k or "12" in k :
                    continue
            if "mean_scale_pred" in k:
                continue
            if "pre_code_net.sem_net.0" in k:
                continue
            if "pre_code_net.encoder.conv.weight" in k:
                continue
            if "reason_backbone.pos_embed" in k :
                continue
            if "reason_backbone.decoder_pos_embed" in k :
                continue
            # if "pre_code_net.sem_net.0" in k:
            #     continue
            # if ".encoder." in k or ".sem_net." in k:
            #     continue
            # if "reason_backbone.mask_token_pos_embed" in k:
            #     continue
            ckeck_copy[new_k] = ckeck[k]
        student_model.load_state_dict(ckeck_copy,strict=False)
        
        
            
        if args.moco:
            model_ema = copy.deepcopy(student_model).requires_grad_(False)
        load_epoch = -1
        if "epoch" in checkpoint:
            load_epoch = checkpoint['epoch']
        print(("=> loaded main model checkpoint '{}' (epoch {})"
                .format(args.evaluate,load_epoch )))
    
    if os.path.isfile(args.resume_dis):
        print(("=> loading checkpoint '{}'".format(args.resume_dis)))
        checkpoint = torch.load(args.resume_dis,map_location='cpu')
        ckeck = checkpoint['state_dict']
        ckeck_copy = {}
        ks  =ckeck.keys()
        ### remove module
        for k in ks:
            if k.startswith("module"):
                ckeck_copy[k[7:]] = ckeck[k]
        dis_model.load_state_dict(ckeck_copy,strict=False)
        print(("=> loaded dis_model  checkpoint '{}' (epoch {})"
                .format(args.evaluate, checkpoint['epoch'])))
    
    
    # params = student_model.parameters()
    # for param in params:
    #     param.data = torch.clamp(param.data, -1, 1)
    if torch.cuda.is_available():
        student_model = torch.nn.parallel.DistributedDataParallel(student_model.cuda(), device_ids=[torch.cuda.current_device()],\
             broadcast_buffers=False , find_unused_parameters=True)
        dis_model = torch.nn.parallel.DistributedDataParallel(dis_model.cuda(), device_ids=[torch.cuda.current_device()],\
             broadcast_buffers=False , find_unused_parameters=False)
    # model_ema = EMA(student_model, decay=0.999)
    # model_ema.register()
    
    dis_model.apply(ganD_weights_init)
    teacher_model  =None
    # Data loading code
    # normalize = GroupNormalize(input_mean, input_std)
    if "rand" in args.dataset:
        if args.dataset in ['k60k_brrand','k60k_brrand_a6000','k60k_brrand_cloud']:
            video_list_compressed =[
                "k400_random50k320_crf39_h265","k400_random50k320_crf47_h265",
                'k400_random50k320_crf51_h265',
                'k400_random50k320_crf31_h265',
                ]
            train_ds = TSNDataSet(
            "kinetics",
            "",
            train_list,
            num_segments=args.num_segments,
            modality="RGB",
            image_tmpl=prefix,
            transform=torchvision.transforms.Compose([
                train_augmentation,
                Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                ]),
            dense_sample=False,
            video_list_compressed = video_list_compressed,
            sample_strategy = args.dataset_sample_strategy
            )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.workers, pin_memory=True,sampler=DistributedSampler(train_ds,shuffle=True))
    
    
   
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    policy_precode,policy_action = \
        get_optim_policies(student_model,[1,1,1,0],ft_action_on_small = args.ft_action_on_small,opt=args)
    optimizer = torch.optim.AdamW(policy_precode,
                            args.lr,betas=(args.adam_beta1, args.adam_beta2),
                            weight_decay=args.weight_decay)

    
     
    optimizer_D = torch.optim.AdamW(dis_model.parameters(),
                                    args.lr*(2 if args.gan_turr else 1),betas=(args.adam_beta1, args.adam_beta2))
    log_training = None
    if dist.get_rank() == 0:
        log_training = open(os.path.join(args.checkpoint_dir,'log', '%s.csv' % store_name), 'w')
        log_training.write(args.__str__()+"\n")
        log_training.write(str(student_model.__str__()))
        str_g_summary = count_parameters(student_model,["_loss","base_model"],args.print_model_keyword)
        str_d_summary = count_parameters(dis_model,["_loss","base_model"])
        log_training.write("student_model summary: ")
        log_training.write(str_g_summary)
        log_training.write(str(dis_model.__str__()))
        log_training.write("dis_model summary :")
        log_training.write(str_d_summary)
        log_training.write("================================\n")
        print(str_g_summary)
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = None
    if dist.get_rank() <= 0:
        tb_logger = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir,'board'),flush_secs=10)
    

    
    if args.only_cal_cost:
        from thop import profile
        from thop import clever_format
        torch.set_default_tensor_type('torch.FloatTensor')
        input1 = torch.randn(1, 3,8, args.input_size, args.input_size).cuda(0)
        macs, params = profile(student_model, inputs=(input1,input1,input1,True))
        macs, params = clever_format([macs, params], "%.3f")
        print("computation cost",macs, params)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch, args.lr_steps)
        

        # train for one epoch
        train(train_loader, student_model, dis_model,0,criterion, optimizer, optimizer_D,epoch, log_training,tb_logger,ft_action,args)
        save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': student_model.state_dict(),
                    'best_prec1': 0,
                }, True,"network",args.save_model_freq)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': dis_model.state_dict(),
            'best_prec1': 0,
        }, True,"dis",args.save_model_freq)
        
        
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
import inspect

def get_variable_file(variable_name):
    frame = inspect.currentframe()
    while frame:
        if variable_name in frame.f_locals:
            file_path = frame.f_code.co_filename
            return file_path
        frame = frame.f_back
    return None
def run_model(input_com,input_clean,target,student_model,criterion,args):
    input_com = input_com.cuda(non_blocking = True)
    input_clean = input_clean.cuda(non_blocking = True)
    
    
    target = target.cuda(non_blocking=True)
    clip_len = args.num_segments
    input_com = input_com.view((-1, clip_len,3) + input_com.size()[-2:])
    input_com = input_com.transpose(1,2).contiguous()
    input_clean = input_clean.view((-1, clip_len,3) + input_clean.size()[-2:])
    input_clean = input_clean.transpose(1,2).contiguous()### b c t h w
    # input_clean_now = input_clean
    
    if args.vis_feature:
        torchvision.utils.save_image(input_clean[0,:,0,:,:],"1.png")
        torchvision.utils.save_image(input_com[0,:,0,:,:],"2.png")

    
    if student_model.training:
        code_bpp_tuple,pixel_loss,total_nce_losses,recons_clean,aux_v = \
        student_model(input_clean,input_com,input_clean,False)
    else:
        # print(type(student_model))
        code_bpp_tuple,pixel_loss,total_nce_losses,recons_clean,aux_v = \
        student_model(input_clean,input_com,input_clean,False)
        
    code_bpp_train = code_bpp_tuple[0].mean()
    code_meta_sum = code_bpp_tuple[1]
    # print("code_meta_sum",code_meta_sum,code_bpp_train)
    # print("code_meta_sum2",code_meta_sum2,code_bpp_train2)
    if args.test_crf<0:
        h265_bitcosts = code_meta_sum ## just palce holder  for h265_bitcosts during training
    nce_loss  = 0
    nce_vis = {
        "nce_loss":0,
    }
    if student_model.training:
        if total_nce_losses:
            for i in range(len(total_nce_losses)):
                nce_loss+= total_nce_losses[i].mean()
                nce_vis["nce_loss"+str(i)] = total_nce_losses[i]
            nce_vis["nce_loss"]  =nce_loss
            # nce_vis = {
            # "":,
            # "total_nce_losses0":[0],
            # "total_nce_losses1":total_nce_losses[1],
            # "total_nce_losses2":total_nce_losses[2],
            # }
       
    
    
    loss_action = 0
    acc1 = 0
    acc5 = 0
    
    loss_sum = loss_action*args.lambda_action + (pixel_loss.mean()*args.lambda_pixel )
    vis_dict = {}
    if student_model.training:
        vis_dict = {
            "code_bpp_train":code_bpp_train,
            'pixel_loss':pixel_loss.mean(),
            "loss_action": loss_action,
            "prec1":acc1,
            "prec5":acc5,
            "lr":-1,
            **nce_vis
        }
    return vis_dict, loss_sum,acc1,acc5,recons_clean,input_clean,h265_bitcosts,code_meta_sum,aux_v
def cal_loss_GAN_mse(dis_model,fake,real=None):
    # print(fake)
    func = F.relu
    # func = lambda x:x
    if type(real) == type(None):
        pred_fake = dis_model( d3_to_d2(fake))
        # ff = lambda x: -x
        pred_fake = [F.mse_loss(x,torch.ones_like(x)) for x in pred_fake]
        loss_G = sum(pred_fake)/len(pred_fake)
        return loss_G
    pred_fakes = dis_model(d3_to_d2(fake.detach()) )
    # ff = lambda x: func(1. + x)
    pred_fake = [F.mse_loss(x,torch.zeros_like(x)) for x in pred_fakes]
    
    loss_D_fake = sum(pred_fake)/len(pred_fake)
    loss_D_fake = loss_D_fake
    # Real
    pred_real = dis_model( d3_to_d2(real.detach()) )
    # ff = lambda x: func(1. - x)
    pred_real = [F.mse_loss(x,torch.ones_like(x)) for x in pred_real]
    pred_real = sum(pred_real)/(len(pred_real))
    loss_D_real = pred_real
    # combine loss and calculate gradients
    loss_D = (loss_D_fake + loss_D_real) * 0.5
    return loss_D
def cal_loss_GAN(dis_model,fake,real=None):
    # print(fake)
    if type(real) == type(None):
        pred_fake = dis_model( d3_to_d2(fake))
        ff = lambda x: -x
        pred_fake = [(ff(x)).mean() for x in pred_fake]
        loss_G = sum(pred_fake)
        return loss_G
    pred_fakes = dis_model(d3_to_d2(fake.detach()) )
    ff = lambda x: F.relu(1. + x)
    # ff = lambda x: (1. + x)
    pred_fake = [(ff(x)).mean() for x in pred_fakes]
    loss_D_fake = sum(pred_fake)
    loss_D_fake = loss_D_fake
    # Real
    pred_real = dis_model( d3_to_d2(real.detach()) )
    ff = lambda x: F.relu(1. - x)
    # ff = lambda x: (1. - x)
    pred_real = [(ff(x)).mean() for x in pred_real]
    pred_real = sum(pred_real)
    loss_D_real = pred_real
    # combine loss and calculate gradients
    loss_D = (loss_D_fake + loss_D_real) * 0.5
    return loss_D

def calculate_adaptive_weight( nll_loss, g_loss, last_layer=None):
    if last_layer is not None:
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    d_weight = d_weight 
    return d_weight


train_iter = 0

def train(train_loader, student_model,dis_model,model_ema,criterion, optimizer,optimizer_D, epoch, log,tb_logger,ft_action,args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    reg_losses = AverageMeter()
    rep_losses1 = AverageMeter()
    rep_losses2 = AverageMeter()
    rep_losses3 = AverageMeter()
    rep_losses4 = AverageMeter()
    reg_loss_fuses = AverageMeter()
    prob_pens = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode

    student_model.train()

    end = time.time()
    train_loader.sampler.set_epoch(epoch)
    for i, (input_com,input_clean, target,_) in enumerate(train_loader):
        if i>3000 and i%5000  == 0:
            best_prec1 = 0
            is_best = False
            save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': student_model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best,"network",args.save_model_freq)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': dis_model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best,"dis",args.save_model_freq)
        # measure data loading time
        data_time.update(time.time() - end)
        vis_dict, loss_sum,top1,top5,recons_clean,input_clean,h265_bitcost,code_meta_sum,aux_v = run_model(input_com,input_clean,target,student_model,criterion,args)
        
        # exit()
        vis_dict["lr"] = optimizer.param_groups[-1]['lr']

        global train_iter
        train_iter  = train_iter+1
        nce_loss = vis_dict['nce_loss']
        gan_func = cal_loss_GAN
        if args.gan_loss == "mse":
            gan_func = cal_loss_GAN_mse
        code_bpp_train = vis_dict["code_bpp_train"]
        if args.lambda_gan > 0:
            # img_4x_aux,img_8x_aux,input_clean4,input_clean8 = aux_v
            set_requires_grad(dis_model, True)
            optimizer_D.zero_grad()
            # recons_clean_sample,input_clean_sample = rand_crop_videos([recons_clean,input_clean],224)
            recons_clean_sample,input_clean_sample =recons_clean, input_clean
            t = recons_clean.size(2)
            loss_D1x = gan_func(dis_model,recons_clean_sample,input_clean_sample)
            loss_D = (loss_D1x)* args.lambda_gan
            loss_D.backward()
            if args.clip_gd_type == 'norm':
                total_norm = clip_grad_norm(dis_model.parameters(), args.clip_gradient)
                if total_norm > args.clip_gradient:
                    print("too large norm clippping",total_norm)
            if args.clip_gd_type == 'value':
                clip_grad_value(dis_model.parameters(), args.clip_gradient)
            total_norm = clip_grad_norm(dis_model.parameters(), args.clip_gradient)
            optimizer_D.step()
            set_requires_grad(dis_model, False)
            optimizer.zero_grad()
            loss_G1x = gan_func(dis_model,recons_clean_sample)
            loss_G = (loss_G1x)
            # nce_loss1 =vis_dict['nce_loss0']
            # nce_loss2 =vis_dict['nce_loss1'] ## real ncer 
            loss_sum  =  (loss_sum+ args.lambda_gan* loss_G )
            if dist.get_rank() <= 0:
                vis_dict["loss_D"] = loss_D
                vis_dict["loss_G"] = loss_G
        # if args.lambda_nce>0:
        loss_sum  =loss_sum+nce_loss * args.lambda_nce 
        loss_sum = loss_sum + (code_bpp_train) * args.lambda_bpp
        # loss_sum = loss_sum + (code_bpp_train+code_bpp_train2) * args.lambda_bpp
        if dist.get_rank() <= 0:
            vis_dict['loss_sum'] = loss_sum
        loss_sum.backward()
        if args.moco:
            ema1(model_ema, student_model,0.999)
            ema2(student_model, model_ema,0)
        if dist.get_rank() <= 0 and train_iter%50 == 0:
            for k, v in vis_dict.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                # tensorboard logger
                
                tb_logger.add_scalar(k, v, train_iter)
        if args.clip_gd_type == 'norm':
            total_norm = clip_grad_norm(student_model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                pass
        if args.clip_gd_type == 'value':
            clip_grad_value(student_model.parameters(), args.clip_gradient)
        optimizer.step()
        
        # model_ema.update()
        # model_ema.apply()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0 and dist.get_rank() == 0:
            # for i in optimizer.param_groups:
            # 	print(i['lr'])
            # exit()
            output = ('Epoch: [{0}][{1}/{2}],'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time))
            output += dict2str(vis_dict)
            print(output)
            log.write(output + '\n')
            log.flush()
        # break
        


def validate(val_loader, student_model, teacher_model, criterion,epoch_index, iter, log ,tb_logger,quantization_m):
    batch_time = AverageMeter()
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    sum_bit = AverageMeter()
    h265_bit = AverageMeter()
    latent_bit = AverageMeter()

    # switch to evaluate mode
    student_model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input_com,input_clean, target,video_lens) in enumerate(val_loader):
            
            vis_dict, loss_sum,acc1,acc5,recons_clean,input_clean,h265_bitcost,code_meta_sum,aux_v =\
                 run_model(input_com,input_clean,target,student_model,criterion,args,quantization_m)
            
            video_lens = video_lens.cuda(h265_bitcost.device)
            # print(h265_bitcost.size(),code_meta_sum.size(),video_lens.size())
            # exit()
            vid_time = video_lens/25
            H265_br = h265_bitcost/(vid_time)
            meta_br = code_meta_sum/(vid_time)
            H265_br = H265_br.mean()
            meta_br = meta_br.mean()
            # reduced_acc1 = (acc1)
            # reduced_acc5 = (acc5)
            # reduced_latent_bit = (vis_dict["code_bpp"])

            reduced_H265_br = reduce_tensor(H265_br).item()/1000
            reduced_meta_br = reduce_tensor(meta_br).item()/1000
            sum_br = reduced_H265_br+ reduced_meta_br
            # reduced_latent_bit = reduce_tensor(vis_dict["code_bpp"])
            top1.update(acc1.item(), input_com.size(0))
            top5.update(acc5.item(), input_com.size(0))
            sum_bit.update(sum_br, input_com.size(0))
            latent_bit.update(reduced_meta_br , input_com.size(0))
            h265_bit.update(reduced_H265_br, input_com.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and dist.get_rank() <= 0:
                output = ('Test: [{0}/{1}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'sum_bit@1 {sum_bit.val:.6f} ({sum_bit.avg:.6f})\t'
                    'latent_bit@1 {latent_bit.val:.6f} ({latent_bit.avg:.6f})\t'
                    'h265_bit@1 {h265_bit.val:.6f} ({h265_bit.avg:.6f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time,
                    sum_bit = sum_bit,
                    latent_bit = latent_bit,
                    h265_bit = h265_bit,
                       top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()
            # break
        
    output = ('Testing Results: bit {latent_bit.avg:.6f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} '
          .format(latent_bit = latent_bit,top1=top1, top5=top5))
    # tensorboard logger
    if dist.get_rank() <= 0:
        tb_logger.add_scalar("val_top1", top1.avg, epoch_index)
        tb_logger.add_scalar("val_top5", top5.avg, epoch_index)
        tb_logger.add_scalar("val_latent_bpp", latent_bit.avg, epoch_index)
    print(output)
    output_best = '\nBest Prec@1: %.3f'%(best_prec1)
    print(output_best)
    if log is not None:
        log.write(output + ' ' + output_best + '\n')
        log.flush()

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint',save_model_freq = "10"):
    if dist.get_rank() == 0:
        torch.save(state, '%s/%s_%s.pth.tar' % (os.path.join(args.checkpoint_dir,'checkpoint'), store_name,filename))
        if is_best:
            shutil.copyfile('%s/%s_%s.pth.tar' % (os.path.join(args.checkpoint_dir,'checkpoint'), store_name,filename),
            '%s/%s_best_%s.pth.tar' % (os.path.join(args.checkpoint_dir,'checkpoint'), store_name,filename))
        if state['epoch'] %save_model_freq == 0:
            shutil.copyfile(
                '%s/%s_%s.pth.tar' % (os.path.join(args.checkpoint_dir,'checkpoint'), store_name,filename),
            '%s/%s_%s_%d.pth.tar' % (os.path.join(args.checkpoint_dir,'checkpoint'), store_name,filename,state['epoch'])
            )

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 """
    decay = 0.5 ** (sum(epoch >= np.array(lr_steps)))
    
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        if "decay_mult" in param_group:
            param_group['weight_decay'] = decay * param_group['decay_mult']
            if lr <1e-6:
                param_group['weight_decay'] = 0


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0,keepdim = True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.checkpoint_dir,os.path.join(args.checkpoint_dir,'log'), os.path.join(args.checkpoint_dir,'checkpoint')]
    for folder in folders_util:
        if not os.path.exists(folder):
            print(('creating folder ' + folder))
            os.mkdir(folder)
    
def judge_is_most_sub_module(m):
    if len(list(m.children())) == 0:
        return True
    else:
        return False
def get_optim_policies(model,lr_multies,ft_action_on_small,opt):
    # return model.parameters()
    pre_code_param_list_encoder = []
    pre_code_param_list_decoder = []
    bit_param_list = []
    mae_param_list = []
    param_fuck = []
    # vq_param_list = []
    bit_param_list_multi = lr_multies[0]
    pre_code_param_list_decoder_multi = lr_multies[0]
    pre_code_param_list_encoder_multi = lr_multies[0]
    
    
    
    for n,m in model.named_parameters():
        if "pos_emb" in n or "pos_embedding" in n :
            pre_code_param_list_decoder.append(m)
        
    for n,m in model.named_modules():
        if "vgg_perceptuall_loss" in n:
            continue
        if "lpips_perceptual_loss" in n:
            continue
        if "hed_edge_loss" in n:
            continue
        if ".videomae_model." in n:
            continue
        if ".dinov2_model." in n:
            continue
        if ".swin_model." in n:
            continue
        if ("pre_code_net" in n) or 'fvc_model' in n:
            if judge_is_most_sub_module(m):
                if ".object_reason." in n:
                    mae_param_list += list(m.parameters())
                elif "bitEstimator_" in n:
                    bit_param_list += list(m.parameters())
                elif "pre_code_net.encoder." in n:
                    pre_code_param_list_encoder += list(m.parameters())
                else:
                    print(n)
                    pre_code_param_list_decoder += list(m.parameters())
            else:
                pass
            
    return [
                {
                    'params': pre_code_param_list_encoder, 'lr_mult':1,
                     'weight_decay': opt.pre_code_wd,
                'name': "pre_code_param_list_encoder",
                },
                {
                    'params': pre_code_param_list_decoder, 'lr_mult':1,
                     'weight_decay': opt.pre_code_wd,
                'name': "pre_code_param_list_decoder",
                },
                 {
                    'params': param_fuck, 'lr_mult':1,
                     'weight_decay': opt.pre_code_wd,
                    'name': "param_fuck",
                },
                {
                    'params': bit_param_list, 'lr_mult': 1,
                     'weight_decay': opt.pre_code_wd*0,
                '   name': "bit_param_list",
                },
                {
                    'params': mae_param_list, 'lr_mult': 2,'weight_decay': 0.001,
                'name': "mae_param_list",
                }
            ],None
if __name__ == '__main__':
    main()
