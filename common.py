import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

class GaussianBlur(nn.Module):
    def __init__(self, radius, sigma, channels=3):
        super(GaussianBlur, self).__init__()
        self.radius = radius
        self.sigma = sigma
        self.channels = channels
        self.kernel_size = 2 * radius + 1
        self.padding = radius

        self.blur_weights = self.create_blur_weights()

    def create_blur_weights(self):
        kernel = torch.exp(-(torch.arange(self.kernel_size) - self.radius) ** 2 / (2 * self.sigma ** 2))
        kernel = kernel / kernel.sum()
        weights = torch.outer(kernel, kernel).reshape(1, 1, self.kernel_size, self.kernel_size)
        weights = weights.repeat(self.channels, 1, 1, 1)
        return weights

    def forward(self, input):
        blurred_output = F.conv2d(input, self.blur_weights.to(input.device), padding=self.padding, groups=self.channels)
        return blurred_output

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def sub_mean(x):
    mean = x.mean(2, keepdim=True).mean(3, keepdim=True)
    x -= mean
    return x, mean

def InOutPaddings(x):
    w, h = x.size(3), x.size(2)
    padding_width, padding_height = 0, 0
    if w != ((w >> 7) << 7):
        padding_width = (((w >> 7) + 1) << 7) - w
    if h != ((h >> 7) << 7):
        padding_height = (((h >> 7) + 1) << 7) - h
    paddingInput = nn.ReflectionPad2d(padding=[padding_width // 2, padding_width - padding_width // 2,
                                               padding_height // 2, padding_height - padding_height // 2])
    paddingOutput = nn.ReflectionPad2d(padding=[0 - padding_width // 2, padding_width // 2 - padding_width,
                                                0 - padding_height // 2, padding_height // 2 - padding_height])
    return paddingInput, paddingOutput





class UpConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, mode='transpose', norm=False):
        super(UpConvNorm, self).__init__()

        if mode == 'transpose':
            self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        elif mode == 'shuffle':
            self.upconv = nn.Sequential(
                ConvNorm(in_channels, 4*out_channels, kernel_size=3, stride=1, norm=norm),
                PixelShuffle(2))
        else:
            # out_channels is always going to be the same as in_channels
            self.upconv = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                ConvNorm(in_channels, out_channels, kernel_size=1, stride=1, norm=norm))
    
    def forward(self, x):
        out = self.upconv(x)
        return out


def rescale_3d_video(input,s):
    b,c,t,h,w = input.size()
    vid_2d = input.transpose(1,2).reshape(b*t,c,h,w)
    vid_2d = F.upsample(vid_2d,scale_factor= s,mode= 'bicubic')
    h = int(h* s)
    w = int(w*s)
    return vid_2d.reshape(b,t,c,h,w).transpose(1,2)
class meanShift(nn.Module):
    def __init__(self, rgbRange, rgbMean, sign, nChannel=3):
        super(meanShift, self).__init__()
        if nChannel == 1:
            l = rgbMean[0] * rgbRange * float(sign)

            self.shifter = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
            self.shifter.weight.data = torch.eye(1).view(1, 1, 1, 1)
            self.shifter.bias.data = torch.Tensor([l])
        elif nChannel == 3:  
            r = rgbMean[0] * rgbRange * float(sign)
            g = rgbMean[1] * rgbRange * float(sign)
            b = rgbMean[2] * rgbRange * float(sign)

            self.shifter = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
            self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
            self.shifter.bias.data = torch.Tensor([r, g, b])
        else:
            r = rgbMean[0] * rgbRange * float(sign)
            g = rgbMean[1] * rgbRange * float(sign)
            b = rgbMean[2] * rgbRange * float(sign)
            self.shifter = nn.Conv2d(6, 6, kernel_size=1, stride=1, padding=0)
            self.shifter.weight.data = torch.eye(6).view(6, 6, 1, 1)
            self.shifter.bias.data = torch.Tensor([r, g, b, r, g, b])

        # Freeze the meanShift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)

        return x

    



## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, reduction, bias=True,
            norm=False, act=nn.ReLU(True), downscale=False, return_ca=False):
        super(RCAB, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size, stride=2 if downscale else 1, norm=norm),
            act,
            ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
            CALayer(out_feat, reduction)
        )
        self.downscale = downscale
        if downscale:
            self.downConv = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1)
        self.return_ca = return_ca

    def forward(self, x):
        res = x
        out, ca = self.body(x)
        if self.downscale:
            res = self.downConv(res)
        out += res

        if self.return_ca:
            return out, ca
        else:
            return out


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, Block, n_resblocks, n_feat, kernel_size, reduction, act, norm=False):
        super(ResidualGroup, self).__init__()

        modules_body = [Block(n_feat, n_feat, kernel_size, reduction, bias=True, norm=norm, act=act)
            for _ in range(n_resblocks)]
        modules_body.append(ConvNorm(n_feat, n_feat, kernel_size, stride=1, norm=norm))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

def pixel_shuffle(input, scale_factor):
    batch_size, channels, in_height, in_width = input.size()

    out_channels = int(int(channels / scale_factor) / scale_factor)
    out_height = int(in_height * scale_factor)
    out_width = int(in_width * scale_factor)

    if scale_factor >= 1:
        input_view = input.contiguous().view(batch_size, out_channels, scale_factor, scale_factor, in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    else:
        block_size = int(1 / scale_factor)
        input_view = input.contiguous().view(batch_size, channels, out_height, block_size, out_width, block_size)
        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

    return shuffle_out.view(batch_size, out_channels, out_height, out_width)


class PixelShuffle(nn.Module):
    def __init__(self, scale_factor):
        super(PixelShuffle, self).__init__()
        self.scale_factor = scale_factor
    
    def forward(self, x):
        return pixel_shuffle(x, self.scale_factor)
    def extra_repr(self):
        return 'scale_factor={}'.format(self.scale_factor)
def d2_to_d3(x,t):
    # assert t==20,"fuck wrong vid len"
    bt,c,h,w = x.size()
    b = bt//t
    return x.reshape(b,t,c,h,w).transpose(1,2)

def d3_to_d2(x):
    b,c,t,h,w = x.size()
    return x.transpose(1,2).reshape(b*t,c,h,w)
def conv(in_channels, out_channels, kernel_size, 
         stride=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=kernel_size//2,
        stride=1,
        bias=bias,
        groups=groups)
def compute_classification_accuracy(output, label):
    _, predicted = torch.max(output, dim=1)  # 获取预测类别
    correct = (predicted == label).sum().item()  # 统计预测正确的样本数量
    total = label.size(0)  # 总样本数
    accuracy = correct / total  # 计算准确率

    return accuracy
def normalize_image_by_imagenetstatics(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).cuda(image_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).cuda(image_tensor.device)
    normalized_tensor = (image_tensor - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
    return normalized_tensor
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def conv1x1(in_channels, out_channels, stride=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=bias,
        groups=groups)

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def conv5x5(in_channels, out_channels, stride=1, 
            padding=2, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=5,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def conv7x7(in_channels, out_channels, stride=1, 
            padding=3, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=7,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose',k=3):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1)
    elif mode == 'shuffle':
        if k==3:
            return nn.Sequential(
                conv3x3(in_channels, 4*out_channels),
                PixelShuffle(2))
        elif k==5:
            return nn.Sequential(
                conv5x5(in_channels, 4*out_channels,groups=8),
                PixelShuffle(2))
    else:
        # out_channels is always going to be the same as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
            conv1x1(in_channels, out_channels))



class Interpolation(nn.Module):
    def __init__(self, n_resgroups, n_resblocks, n_feats, 
                 reduction=16, act=nn.LeakyReLU(0.2, True), norm=False):
        super(Interpolation, self).__init__()

        # define modules: head, body, tail
        self.headConv = conv3x3(n_feats, n_feats)

        modules_body = [
            ResidualGroup(
                RCAB,
                n_resblocks=n_resblocks,
                n_feat=n_feats,
                kernel_size=3,
                reduction=reduction, 
                act=act, 
                norm=norm)
            for _ in range(n_resgroups)]
        self.body = nn.Sequential(*modules_body)

        self.tailConv = conv3x3(n_feats, n_feats)

    def forward(self, x0):
        # Build input tensor
        # x = torch.cat([x0, x1], dim=1)
        x = x0
        x = self.headConv(x)

        res = self.body(x)
        res += x

        out = self.tailConv(res)
        return out
import random
def rand_crop_videos(vids,size):
    cropped_vs = []
    for v in vids:
        b,c,t,h,w = v.size()
        if h<= size or w<= size:
            cropped_vs += [v]
            continue
        h_beg = random.randint(0, h - size-1)
        w_beg = random.randint(0, w - size-1)
        v = v [:,:,:,h_beg:h_beg+size, w_beg:w_beg+size]
        cropped_vs += [v]
    return cropped_vs


class Interpolation_res(nn.Module):
    def __init__(self, n_resgroups, n_resblocks, n_feats,
                 act=nn.LeakyReLU(0.2, True), norm=False):
        super(Interpolation_res, self).__init__()

        # define modules: head, body, tail (reduces concatenated inputs to n_feat)
        self.headConv = conv3x3(n_feats * 2, n_feats)

        modules_body = [ResidualGroup(ResBlock, n_resblocks=n_resblocks, n_feat=n_feats, kernel_size=3,
                            reduction=0, act=act, norm=norm)
                        for _ in range(n_resgroups)]
        self.body = nn.Sequential(*modules_body)

        self.tailConv = conv3x3(n_feats, n_feats)

    def forward(self, x0, x1):
        # Build input tensor
        x = torch.cat([x0, x1], dim=1)
        x = self.headConv(x)

        res = x
        for m in self.body:
            res = m(res)
        res += x

        x = self.tailConv(res)

        return x
from torchvision.ops.deform_conv import DeformConv2d
def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
   pad = (kernel_size - 1) * dilation
   return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)
class VideoTemporalCausal(nn.Module):
    def __init__(self,i,o,k,is_depthwise=True,vid_len=8,d=1,input_type ='3d'):
        super().__init__()
        k = 3
        self.input_type = input_type
        self.is_depthwise = is_depthwise
        if is_depthwise == 'depthwise_1dconv':
            self.conv1 = CausalConv1d(i, o, kernel_size=k,dilation = d, groups=i)
        elif is_depthwise == '1dconv':
            self.conv1 = CausalConv1d(i, o, kernel_size=k,dilation = d)
        elif is_depthwise == 'shift_1dconv':
            self.conv1 = CausalConv1d(i, o, kernel_size=k,dilation = d)
            self.shift_spatial_conv = nn. Conv3d(i//2, o//2, kernel_size=(1,7,7),\
                padding =(0,3,3),groups = i//2)
        elif is_depthwise == 'DCN':
            self.conv1 = CausalConv1d(i, o, kernel_size=k,dilation = d)
            self.shift_spatial_conv = nn. Conv3d(i//2, o//2, kernel_size=(1,7,7),\
                padding =(0,3,3),groups = i//2)
            self.off_set_c = 8*(2*3*3)
            self.offset_pre_net = nn.Sequential(
                nn.Conv2d(i * 2, i*2, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.1,inplace=True),
                # nn.Conv2d(i * 2, i*2, 3, 1, 1, bias=True), 
                # nn.LeakyReLU(0.1,inplace=True),
                nn.Conv2d(i * 2, i, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.1,inplace=True),
                nn.Conv2d(i, self.off_set_c, 3, 1, 1, bias=True), 
            )
            self.L1_dcnpack = DeformConv2d(i, i, 3, stride=1, padding=1, dilation=1,groups=8)
            self.enhance_net =  nn.Sequential(
                nn.Conv2d(i * 3, i*2, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.1,inplace=True),
                nn.Conv2d(i * 2, i, 3, 1, 1, bias=True), 
            )
        elif is_depthwise == '3dconv':
            self.before_conv1 = nn. Conv3d(i, i//2, 1,1,0)
            self.conv1 = nn. Conv3d(i//2, o//2, kernel_size=k, padding =(k - 1,k//2,k//2))
            self.after_conv1 = nn. Conv3d(o//2, o, 1,1,0)
        elif is_depthwise == '3dconv_4scale':
            self.before_conv1 = nn. Conv3d(i, i//4, 1,1,0)
            self.conv1 = nn. Conv3d(i//4, o//4, kernel_size=k, padding =(k - 1,k//2,k//2))
            self.after_conv1 = nn. Conv3d(o//4, o, 1,1,0)
        else:
            print("not in type VideoTemporalCausal") 
            import traceback
            traceback.print_stack()
            exit()
        self.vid_len = vid_len
    def forward(self, x):
        ### b c t h w
        res = x
        vid_len = self.vid_len
        if self.input_type == '2d':
            x = d2_to_d3(x,t = vid_len)
        b,c,t,h,w = x.size()
        if self.is_depthwise in ['depthwise_1dconv','1dconv']:
            x_t = x.permute(0,3,4,1,2).reshape(b*h*w,c,t)
            x_t = self.conv1(x_t)
            x_t = x_t[:, :, :-self.conv1.padding[0]]  # remove trailing padding
            x = x_t.reshape(b,h,w,c,t).permute(0,3,4,1,2)
        if self.is_depthwise in ['shift_1dconv']:
            x_up,x_down = torch.chunk(x,2,dim=1)
            x = torch.cat([self.shift_spatial_conv(x_up),x_down],dim=1)
            x_t = x.permute(0,3,4,1,2).reshape(b*h*w,c,t)
            x_t = self.conv1(x_t)
            x_t = x_t[:, :, :-self.conv1.padding[0]]  # remove trailing padding
            x = x_t.reshape(b,h,w,c,t).permute(0,3,4,1,2)
        if self.is_depthwise in ['DCN']:
           
            ## deformable
            pre_frame = torch.cat([x[:,:,0:1],x[:,:,0:t-1] ],dim=2).transpose(1,2).reshape(b*t,c,h,w)
            pre_frame2 = torch.cat([x[:,:,0:2],x[:,:,0:t-2] ],dim=2).transpose(1,2).reshape(b*t,c,h,w)
            cur_frame = x.transpose(1,2).reshape(b*t,c,h,w)
            offset =self.offset_pre_net(torch.cat([pre_frame,cur_frame],dim=1))
            offseted_pre_frame = self.L1_dcnpack(pre_frame,offset)
            
            offset2 =self.offset_pre_net(torch.cat([pre_frame2,cur_frame],dim=1))
            offseted_pre_frame2 = self.L1_dcnpack(pre_frame2,offset2)
            
            cur_frame = cur_frame + self.enhance_net(
                torch.cat([offseted_pre_frame,offseted_pre_frame2,cur_frame],dim=1)
            )
            x = cur_frame.reshape(b,t,c,h,w).transpose(1,2)
            ## 1D
            x_up,x_down = torch.chunk(x,2,dim=1)
            x_t = torch.cat([self.shift_spatial_conv(x_up),x_down],dim=1)
            x_t = x_t.permute(0,3,4,1,2).reshape(b*h*w,c,t)
            x_t = self.conv1(x_t)
            x_t = x_t[:, :, :-self.conv1.padding[0]]  # remove trailing padding
            x = x_t.reshape(b,h,w,c,t).permute(0,3,4,1,2)+x##b c t h w
        elif self.is_depthwise in ['3dconv']:
            x = self.before_conv1(x)
            x = self.conv1(x)
            x = self.after_conv1(x)
            x = x[:,:, :-self.conv1.padding[0],:,:]
            x = F.leaky_relu(x,0.2)
        if self.input_type == '2d':
            x = d3_to_d2(x)
        return x + res

class MultiscaleCausalConv(nn.Module):
    def __init__(self,c_final,vid_len,input_type ='3d',is_depthwise = "",use_residual=False,\
        use_layernorm=False,use_CA = False):
        super().__init__()
        self.input_type = input_type
        self.conv_fuse = nn.Conv3d(c_final, c_final, 1, 1, 0, bias=False)
        c_final_2 = c_final
        
        self.conv_causual_up = VideoTemporalCausal(c_final_2, c_final_2, 3,is_depthwise = is_depthwise,vid_len = vid_len)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.vid_len  =vid_len
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual
        if use_layernorm:
            self.layernorm = LayerNorm2d(c_final)
        self.CA = None
        # print(use_CA)
        
        if use_CA:
            self.CA = CALayer(c_final,2)

    def forward(self, x):
        res = x
        vid_len = self.vid_len
        if self.input_type == '2d':
            x = d2_to_d3(x,t = vid_len)
        if self.use_layernorm:
            x = d3_to_d2(x)
            x = self.layernorm(x)
            x = d2_to_d3(x,t = vid_len)
        x5 = self.lrelu(self.conv_fuse(x) )
        x5 = self.conv_causual_up( x5)
        if self.CA:
            x5 = d3_to_d2(x5)
            x5 = self.CA(x5)
            x5 = d2_to_d3(x5,t = vid_len)
        if self.input_type == '2d':
            x5 = d3_to_d2(x5)
        if self.use_residual:
            x5 = x5 + res
        return x5

class RealMultiscaleCausalConv(nn.Module):
    def __init__(self,c_in,c_out,vid_len=-1,input_type ='3d',is_depthwise = ""):
        super().__init__()
        self.input_type = input_type
        self.conv_fuse = nn.Conv3d(c_out, c_out, 1, 1, 0, bias=False)
        c_in_2 = c_in//2
        c_out_2 = c_out//2
        self.conv_causual_up = VideoTemporalCausal(c_in_2, c_out_2, 3,is_depthwise = is_depthwise,vid_len = vid_len)
        self.conv_causual_down = VideoTemporalCausal(c_in_2, c_out_2, 5,is_depthwise = is_depthwise,vid_len = vid_len)
        self.vid_len  =vid_len

    def forward(self, x):
        # return x
        vid_len = self.vid_len
        if self.input_type == '2d':
            x = d2_to_d3(x,t = vid_len)
        x5_up,x5_down = torch.chunk(x,2,dim=1)
        x5_up = self.conv_causual_up( x5_up)
        x5_down = self.conv_causual_down(x5_down)
        x5 = torch.cat([x5_up,x5_down],dim=1)
        x5 = self.conv_fuse(x5)
        if self.input_type == '2d':
            x5 = d3_to_d2(x5)
        return x5

class VideoTemporalCausalOldBug(nn.Module):
    def __init__(self,i,o,k,vid_len=-1,d=1,input_type ='3d'):
        super().__init__()
        self.input_type = input_type
        self.conv1 = CausalConv1d(i, o, kernel_size=k,dilation = d, groups=i)
        self.vid_len = vid_len
    def forward(self, x):
        ### b c t h w
        res = x
        vid_len = self.vid_len
        if self.input_type == '2d':
            if not self.training:
                vid_len = x.size(0)
            x = d2_to_d3(x,t = vid_len)
        b,c,t,h,w = x.size()
        x_t = x.permute(0,3,4,1,2).reshape(b*h*w,c,t)
        x_t = self.conv1(x_t)
        x_t = x_t[:, :, :-self.conv1.padding[0]]  # remove trailing padding
        x = x_t.reshape(b,h,w,c,t).permute(0,3,4,1,2)
        if self.input_type == '2d':
            x = d3_to_d2(x)
        return x + res
class MultiscaleCausalConvOldBug(nn.Module):
    def __init__(self,c_final,vid_len=-1,input_type ='3d'):
        super().__init__()
        self.input_type = input_type
        self.conv_fuse = nn.Conv3d(c_final, c_final, 1, 1, 0, bias=False)
        c_final_2 = c_final//2
        # self.conv_causual1 = nn.Identity()
        self.conv_causual_up = VideoTemporalCausalOldBug(c_final_2, c_final_2, 3)
        self.conv_causual_down = VideoTemporalCausalOldBug(c_final_2, c_final_2, 5,2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.vid_len  =vid_len

    def forward(self, x):
        # return x
        vid_len = self.vid_len
        if self.input_type == '2d':
            x = d2_to_d3(x,t = vid_len)
        x5 = self.lrelu(self.conv_fuse(x) )
        c_2 = x5.size(1)//2
        x5 = torch.cat(
            [
                self.conv_causual_up(x5[:,0:c_2]),
                self.conv_causual_down(x5[:,c_2:c_2*2]),
            ],dim=1
        )

        if self.input_type == '2d':
            x5 = d3_to_d2(x5)
        return x5
    
def cal_imagetensor_batchbits_byLaplace(res_quant_mean, res_quant_scale,res_ft_encoded_Q):
    gauss_G0 = torch.distributions.laplace.Laplace(res_quant_mean, res_quant_scale)
    prob1 = gauss_G0.cdf(res_ft_encoded_Q + 0.5) - gauss_G0.cdf(res_ft_encoded_Q - 0.5)
    total_bits = -1.0 * torch.log(prob1 + 1e-5) / math.log(2.0)
    bit_curframe_res = total_bits.clamp_min(0).sum(-1).sum(-1).sum(-1)
    return bit_curframe_res

