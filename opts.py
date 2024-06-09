import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of action recognition models")
parser.add_argument('--dataset', type=str,
                   default = 'somethingv1')
parser.add_argument('--train_compressor_type', type=str,
                   default = 'h264')
parser.add_argument('--dataset_eval', type=str,
                   default = 'somethingv1')
parser.add_argument('--root_path', type = str, default = '../',
                    help = 'root path to video dataset folders')
parser.add_argument('--store_name', type=str, default="")
parser.add_argument('--local_rank', type=int, default="0")
parser.add_argument('--local-rank', type=int, default="0")
parser.add_argument('--augname', type=str, default="")


# ========================= Model Configs ==========================
parser.add_argument('--type', type=str, default="GST",
# choices=['GST','R3D','S3D','STCeption','STCeption_Reg'],
                    help = 'type of temporal models, currently support GST,Res3D and S3D')
parser.add_argument('--arch', type=str, default="resnet50",
                    help = 'backbone networks, currently only support resnet')
parser.add_argument('--agg_type', type=str, default="TSN",
                    help = 'video classifier,default TSN')
parser.add_argument('--train_stage', type=str, default="entropy_model",
                    help = 'entropy_model')        
parser.add_argument('--num_segments', type=int, default=8)
parser.add_argument('--fp', type=int, default=32, help = 'numerical precision')
parser.add_argument('--alpha', type=int, default=4, help = 'spatial temporal split for output channels')
parser.add_argument('--beta', type=int, default=2, choices=[1,2], help = 'channel splits for input channels, 1 for GST-Large and 2 for GST')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=24, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument( '--sr_scale', default=1, type=int,
                    metavar='N', help='sr_scale')

                    
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_decay_rate', '--lr_decay_rate', default=0.1, type=float,
                     help='lr_decay_rate')
parser.add_argument('--lr_steps', default=[50, 60], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--dropout', '--dp', default=0.3, type=float,
                    metavar='dp', help='dropout ratio')
parser.add_argument('--lambda_mae', default=0, type=float, help='lambda_mae')
parser.add_argument('--mae_loss_type', default="l2", type=str, metavar='M',
                    help='mae_loss_type')
parser.add_argument('--mae_group_num', default=-1, type=int, help='mae_group_num')

parser.add_argument('--lambda_mae_nss', default=0, type=float, help='lambda_mae_nss')
parser.add_argument('--lambda_vit_mse', default=0, type=float, help='lambda_vit_mse')

parser.add_argument('--warmup', default=False, action="store_true", help='if using warm up')
parser.add_argument('--warmup_epoch', default=0, type=int,
                    metavar='N', help='warmup_epoch (default: 0)')
parser.add_argument('--nesterov', default=False, action="store_true", help='if using nesterov')
parser.add_argument('--ft_action_on_small', default=False, action="store_true", help='ft_action_on_small')
parser.add_argument('--tail_unet_enhance', default=False, action="store_true", help='tail_enhance')
parser.add_argument('--ada_Q', default=False, action="store_true", help='tail_enhance')
parser.add_argument('--psnr_adaptor_layer', default=False, action="store_true", help='mae_motion')
parser.add_argument('--SMC_multi', default=False, action="store_true", help='SMC_multi')
parser.add_argument('--mae_motion', type=str, default="None",
                    help = 'None')
parser.add_argument('--entropy_quant_style', type=str, default="None",
                    help = 'None')


parser.add_argument('--mae_ratio', default=0.9, type=float, help='lambda_mae')



#========================= Optimizer Configs ==========================
parser.add_argument('--optim',type=str,  required=False,
                    help = 'sgd or adam')

parser.add_argument('--adam_beta1', default=0.9, type=float, metavar='M',
                    help='adam_beta1')
parser.add_argument('--adam_beta2', default=0.999, type=float, metavar='M',
                    help='adam_beta2')
parser.add_argument('--optim_stage',default="all",type=str,  required=False,
                    help = 'entropy_and_decoder or all')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--pre_code_wd', default=0, type=float, metavar='M',
                    help='pre_code_wd')

parser.add_argument('--weight_decay', '--wd', default=3e-4, type=float,   
                    metavar='W', help='weight decay (default: 3e-4)')


parser.add_argument('--clip-gradient', '--gd', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: 20)')
parser.add_argument('--clip-gradient-dis', '--gddis', default=-1, type=float,
                    metavar='W', help='gradient norm clipping (default: 20)')

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--save-model-freq', '-smf', default=10, type=int,
                    metavar='N', help='save model frequency (default: 10)')

parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 1)')

# ========================= Input Data Configs ==========================
parser.add_argument('--input_size', '-i', default=112, type=int,
                    metavar='N', help='input size default 112')

parser.add_argument('--aug_crop_scale', '-acs', nargs='*')                

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--resume_main', default='', type=str, metavar='PATH',
                    help='path to latest resume_main checkpoint (default: none)')
parser.add_argument('--resume_dis', default='', type=str, metavar='PATH',
                    help='path to latest resume_dis checkpoint (default: none)')
parser.add_argument('--print_model_keyword', default='', type=str,required=False,
                    help='print_model_keyword')
parser.add_argument('--encoder_net_type', default='sem_enc', type=str,required=False,
                    help='encoder_net_type')
parser.add_argument('--dataset_clean_rootdir', default=' ', type=str,required=False,
                    help='dataset_clean_rootdir')
                    
parser.add_argument('--only_cal_cost', default=False, action="store_true", help='use dense sample for video dataset')
parser.add_argument('--vis_feature', default=False, action="store_true", help='use dense sample for video dataset')
parser.add_argument('--no_align_coding', default=False, action="store_true", help='use dense sample for video dataset')
                    
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--checkpoint_dir',type=str,  required=False,
                    help = 'folder to restore checkpoint and training log')
parser.add_argument('--fvc_model',type=str,  required=False,
                    help = 'folder to restore checkpoint and training log')
parser.add_argument('--restortion_net', default='rnet',type=str,  required=False,
                    help = 'restore_net')
parser.add_argument('--dfgm_type', default='none',type=str,  required=False,
                    help = 'dfgm_type')

#==============================For multi crop test================
parser.add_argument('-ts', '--test_segments', default=-1, type=int, metavar='N')
parser.add_argument('-tc', '--test_crops', default=1, type=int, metavar='N')
parser.add_argument('--test_video_first_resize', default=-1, type=int, metavar='N')
parser.add_argument('--test_codec_gop', default=10, type=int, metavar='N')
parser.add_argument('-nc', '--num_clips', default=1, type=int, metavar='N')
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')
parser.add_argument('--multi_clip_test', default=False, action="store_true", help='multi clip test')
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--shift', default=False, action="store_true", help='use shift for models')
parser.add_argument('--ft_action', default=False, action="store_true", help='ft_action')


####-------------------------###for compressed video recognition
parser. add_argument("--crf_list", nargs="+", default=[])
parser. add_argument("--logfile", default=None, type=str)
parser.add_argument('--lambda_action', default=1, type=float, metavar='M',
                    help='lambda_action')
parser.add_argument('--lambda_bpp', default=1, type=float, metavar='M',
                    help='lambda_bpp')
parser.add_argument('--distortion_metric', default=None, type=str)
parser.add_argument('--lambda_pixel', default=1, type=float, metavar='M',
                    help='lambda_pixel')
parser.add_argument('--lambda_segrecon', default=1, type=float, metavar='M',
                    help='lambda_segrecon')
parser.add_argument('--lambda_nce', default=1, type=float, metavar='M',
                    help='lambda_nce')
parser.add_argument('--nce_lpips', default=1, type=float, metavar='M',
                    help='nce_lpips')
parser.add_argument('--nce_edge', default=0, type=float, metavar='M',
                    help='nce_edge')
parser.add_argument('--nce_edge_local', default=0, type=float, metavar='M',
                    help='nce_edge_local')
parser.add_argument('--nce_edge_global', default=0, type=float, metavar='M',
                    help='nce_edge_global')
parser.add_argument('--nce_edge_post',type=str,  default= 'median', required=False,
                    help = 'nce_edge_post')
parser.add_argument('--dataset_sample_strategy',type=str,  default= 'random', required=False,
                    help = 'dataset_sample_strategy')
parser.add_argument('--train_bitrate_strategy',type=str,  default= 'legacy', required=False,
                    help = 'train_bitrate_strategy')


parser.add_argument('--quant_func',type=str,  default= 'default', required=False,
                    help = 'quant_func')
parser.add_argument('--quant_range', default=0, type=float, metavar='M',
                    help='quant_range')
parser.add_argument('--gan_loss',type=str,  default= 'hinge', required=False,
                    help = 'gan_loss')
parser.add_argument('--gan_dis_recon', default=False, action="store_true", help='gan_dis_recon')

parser.add_argument('--decode_ft_name',type=str,  default= 'quant_mv', required=False,
                    help = 'decode_ft_name')

parser.add_argument('--lambda_gan', default=0, type=float, metavar='M',
                    help='lambda_gan')
parser.add_argument('--lambda_distill_global', default=1, type=float, metavar='M',
                    help='lambda_distill_global')
parser.add_argument('--lambda_distill_16x', default=1, type=float, metavar='M',
                    help='lambda_distill_16x')
# parser.add_argument('--qp_max', default=320, type=float, metavar='M',
#                     help='lambda_distill_16x')
# parser.add_argument('--qp_min', default=8, type=float, metavar='M',
#                     help='lambda_distill_16x')
parser.add_argument('--qp_range',type=str,  default= '256_128_64_32_16', required=False,
                    help = 'qp_range')
parser.add_argument('--promt_num', type=int, default=1)
parser.add_argument('--distill_32x_globalize', default=False, action="store_true", help='multi clip test')
parser.add_argument('--remove_dycontext', default=False, action="store_true", help='multi clip test')
parser.add_argument('--I_hyper', default=False, action="store_true", help='multi clip test')
parser.add_argument('--I_bits_scale', type=int, default=1)
parser.add_argument('--semantic_source',type=str,  default= 'videomae_dinov2_swin', required=False,
                    help = 'semantic_source')
parser.add_argument('--bridge_mode',type=str,  default= 'net_shared_prompt_unshared', required=False,
                    help = 'bridge_mode')
parser.add_argument('--sem_loss',type=str,  default= 'mse', required=False,
                    help = 'sem_loss')
parser.add_argument('--variable_qp_transform_dec',type=str,  default= 'null', required=False,
                    help = 'variable_qp_transform_dec')

parser.add_argument('--hidden_state_fuser',type=str,  default= '2d', required=False,
                    help = 'hidden_state_fuser')




parser.add_argument('--app_avg_block', default=1, type=int, help='app_avg_block')
parser.add_argument('--test_crf', default=-1, type=int, help='test_crf')
parser.add_argument('--test_save_image', default=False, action="store_true", help='test_save_image')
parser.add_argument('--train_crf', default=-1, type=int, help='train_crf')
parser.add_argument('--code_scale', default=1, type=float, help='code_scale')
parser.add_argument('--codec_type',type=str,  required=False,
                    help = 'codec_type')
parser.add_argument('--encoder_1st',type=str,  required=False,
                    help = 'encoder_1st')
parser.add_argument('--sem_net',type=str, default="default", required=False,
                    help = 'sem_net')

parser.add_argument('--decoder_resblock',type=str,  required=False,default= 'simple',
                    help = '')
parser.add_argument('--distill_layer',type=str, default= 'transformer', required=False,
                    help = 'distill_layer')
parser.add_argument('--distill_backbone_loc',type=str, default= '32_16_8x', required=False,
                    help = 'distill_backbone_loc')
parser.add_argument('--distill_target',type=str, default= 'vanilla', required=False,
                    help = 'distill_target')
parser.add_argument('--videomae_model',type=str, default= "/home/ubuntu/research/A6000_lbvu/pretrain_models/vit_b_hybrid_pt_800e.pth", required=False,
                    help = 'videomae_model')
parser.add_argument('--mocoswin_model',type=str, default= "/data_video/code/lbvu/pretrain_models/swin_base_patch4_window7_224_22k.pth", required=False,
                    help = 'mocoswin_model')
parser.add_argument('--vfm_norm_type',type=str, default= "whole", required=False,
                    help = 'vfm_norm_type')

parser.add_argument('--dec_net3_layer',type=str, default= 'conv', required=False,
                    help = 'dec_net3_layer')
parser.add_argument('--base_encoder_network',type=str, default= '==', required=False,
                    help = 'base_encoder_network')
parser.add_argument('--variable_qp_transform',type=str, default= '===', required=False,
                    help = 'variable_qp_transform')
parser.add_argument('--detach_location',type=str, default= '8x', required=False,
                    help = 'detach_location')
parser.add_argument('--entropy_st_adaptive', default=False, action="store_true", help='entropy_st_adaptive')
parser.add_argument('--variable_qp_latentQ', default=False, action="store_true", help='variable_qp_latentQ')

parser.add_argument('--distill_st_adaptive', default=False, action="store_true", help='distill_st_adaptive')

parser.add_argument('--hyper_entropy',type=str, default= 'laplace', required=False,
                    help = 'encoder_1st')
# parser.add_argument('--entropy_gumbel',type=str, default= 'softmax', required=False,
#                     help = 'encoder_1st')
parser.add_argument('--base_entropy',type=str, default= 'full_factor', required=False,
                    help = 'encoder_1st')
parser.add_argument('--disable_ftres_enc', default=False, action="store_true", help='disable_ftres_enc')
parser.add_argument('--use_layernorm', default=False, action="store_true", help='use_layernorm')
parser.add_argument('--use_CA', default=False, action="store_true", help='use_CA')
parser.add_argument('--ft_res_lossless', default=False, action="store_true", help='ft_res_lossless')


parser.add_argument('--enc_dim', default=16, type=int, help='enc_dim')
parser.add_argument('--quant_qp', default=1, type=int, help='quant_qp')
parser.add_argument('--train_bpp_interval', default=1, type=int, help='enc_dim')
parser.add_argument('--diff_dim', default=8, type=int, help='diff_dim')
parser.add_argument('--diff_group_num', default=-1, type=int, help='enc_dim')
parser.add_argument('--nce_type',type=str,  default= '',required=False,
                    help = 'nce_type')
parser.add_argument('--pixel_distor_type',type=str,  default= 'mse', required=False,
                    help = 'pixel_distor_type')
# parser.add_argument('--lpips_layer_range',type=str,  required=False,
#                     help = 'lpips_layer_range  split by -')
parser.add_argument('--gan_d',type=str,  required=False,
                    help = 'gan_d')
parser.add_argument('--gan_lr_decay',default=False, action="store_true",
                    help = 'gan_lr_decay')
parser.add_argument('--dis_depth',default=6,type=int,  required=False,
                    help = 'dis_depth')
parser.add_argument('--gan_turr',default=1,type=int,  required=False,
                    help = 'gan_turr')
parser.add_argument('--gan_norm_type',type=str,default= 'bn',  required=False,
                    help = 'gan_norm_type')
parser.add_argument('--dis_c',default=64,type=int,  required=False,
                    help = 'dis_c')
parser.add_argument('--dis_r1_reg', default=False, action="store_true", help='dis_r1_reg')


parser.add_argument('--latent_dim',default=256,type=int,  required=False,
                    help = 'latent_dim')
parser.add_argument('--latent_scale',default="32x",type=str,  required=False,
                    help = 'latent_scale')

parser.add_argument('--moco', default=False, action="store_true", help='moco')
parser.add_argument('--mask_comp', default=False, action="store_true", help='mask_comp')
parser.add_argument('--mask_ratio', default=1, type=float, help='mask_ratio')
parser.add_argument('--disable_bnmap', default=False, action="store_true", help='disable_bnmap')


parser.add_argument('--use_base_hyperprior', default=False, action="store_true", help='use_base_hyperprior')
parser.add_argument('--use_normal_prior', default=False, action="store_true", help='use_normal_prior')
parser.add_argument('--no_use_dynamic', default=False, action="store_true", help='no_use_dynamic')
parser.add_argument('--last_Tlayer_causual', default=False, action="store_true", help='last_Tlayer_causual')
parser.add_argument('--dec_tail_fuse', default=False, action="store_true", help='dec_tail_fuse')
parser.add_argument('--dec_post_enhance', default=False, action="store_true", help='dec_post_enhance')
parser.add_argument('--dec_att_no_sigmoid', default=False, action="store_true", help='dec_tail_fuse')
parser.add_argument('--entropy_static', default=False, action="store_true", help='entropy_static')
parser.add_argument('--MEMC', default=False, action="store_true", help='MEMC')


parser.add_argument('--dec_tail_fuse_method',type=str,default= 'add',  required=False,
                    help = 'dec_tail_fuse_method')
parser.add_argument('--fnet_version',type=str,default= 'default',  required=False,
                    help = 'dec_tail_fuse_method')
parser.add_argument('--smc_plus_compensate', default=False,
                    action="store_true", help='smc_plus_compensate')

parser.add_argument('--decoder_net',type=str,default= 'default',  required=False,
                    help = 'decoder_net')
parser.add_argument('--decoder_spatial_r', default=0, type=float, help='decoder_spatial_r')


parser.add_argument('--clip_gd_type',type=str,default= 'norm',  required=False,
                    help = 'clip_gd_type')

parser.add_argument('--run_downstream', default=False, action="store_true", help='run_downstream')
parser.add_argument('--temprature', default=0.07, type=float, help='temprature')
parser.add_argument('--temporal_compression', default=False, action="store_true", help='temporal_compression')
parser.add_argument('--vis_mode', default=False, action="store_true", help='vis_mode')



parser.add_argument('--no_video_scale', default=False, action="store_true", help='no_video_scale')
parser.add_argument('--use_enc_transformer', default=False, action="store_true", help='use_enc_transformer')
parser.add_argument('--nce_on_patch', default=False, action="store_true", help='nce_on_patch')
parser.add_argument('--nce_queue_size', default=32768,type=int,  help='nce_queue_size')


parser.add_argument('--gan_t_sample', default=-1,type=int,  help='gan_t_sample')
parser.add_argument('--gan_s_sample', default=224,type=int,  help='gan_s_sample')
parser.add_argument('--layers_16x', default=3,type=int,  help='layers_16x')
parser.add_argument('--enc_Tenhance', default=False, action="store_true", help='enc_Tenhance')
parser.add_argument('--is_depthwise',type=str, default="depthwise_1dconv",required=False, help='is_depthwise')
parser.add_argument('--test_mode', default='smc', type=str,required=False,
                    help='test_mode')
parser.add_argument('--K_enc', default=32,type=int,  help='K_enc')

parser.add_argument('--enc_net_group_num', default=8,type=int,  help='enc_net_group_num')
parser.add_argument('--enhance_model',type=str,  required=False,
                    help = 'enhance_model')
parser.add_argument('--trancode_src_dir',type=str,  required=False,
                    help = 'trancode_src_dir')
parser.add_argument('--trancode_dest_dir',type=str,  required=False,
                    help = 'trancode_src_dir')