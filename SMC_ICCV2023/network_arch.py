import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import functools
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from collections import OrderedDict
import torchvision
import time


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        # print(self.weight, self.bias)
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    
def save_batch_img(imgs,p,t= 8):
    ### t c h w
    imgs = torchvision.utils.make_grid(imgs,nrow=t,normalize =True)
    torchvision.utils.save_image(imgs,p)
def save_attention_mask(src_input,vis_probs,path,time_inter = 10000):
    ## C T H W ## N T H W
    c,t,h,w = src_input.size()
    if  ("0" in str(src_input.device) or "cpu" in str(src_input.device)) and int(time.time())%time_inter == 0:
        src_input_1st = src_input[:,:,:,:].transpose(0,1)
        vis = [src_input_1st]
        if vis_probs:
            for vis_prob_1st in vis_probs:
                # print("vis_prob_1st",vis_prob_1st.size())
                # vis_prob_1st = F.upsample(vis_prob_1st.unsqueeze(0),size=(t,h,w),mode="trilinear")
                # vis_prob_1st = vis_prob_1st.unsqueeze(2).repeat(1,1,3,1,1).reshape(-1,3,h,w)
                #t 3 h w
                if vis_prob_1st.size(0)  ==1:
                    vis_prob_1st = vis_prob_1st.repeat(3,1,1,1)
                vis_prob_1st = vis_prob_1st.transpose(0,1)

                vis_prob_1st  =F.upsample_bilinear(vis_prob_1st,size = (h,w))
                vis += [vis_prob_1st]
        vis = torch.cat(vis,dim=0)
        save_batch_img(vis,path,t)
def Conv2D(
        in_channels: int, out_channels: int,
        kernel_size: int, stride: int, padding: int,
        is_seperable: bool = False, has_relu: bool = False,
        has_bn: bool = False
):
    modules = OrderedDict()

    if is_seperable:
        modules['depthwise'] = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False,
        )
        modules['pointwise'] = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=True,
        )
    else:
        modules['conv'] = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            bias=True,
        )
    if has_bn:
        modules['bn'] = nn.BatchNorm2d(out_channels)
    if has_relu:
        modules['relu'] = nn.LeakyReLU(0.2,inplace=True)

    return nn.Sequential(modules)

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

def upconv2x2(in_channels, out_channels, mode='shuffle'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1)
    elif mode == 'shuffle':
        return nn.Sequential(
            conv3x3(in_channels, 4*out_channels),
            PixelShuffle(2))
    else:
        # out_channels is always going to be the same as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
            conv1x1(in_channels, out_channels))
def d2_to_d3(x,t):
    bt,c,h,w = x.size()
    b = bt//t
    return x.reshape(b,t,c,h,w).transpose(1,2)

def d3_to_d2(x):
    b,c,t,h,w = x.size()
    return x.transpose(1,2).reshape(b*t,c,h,w)
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        # self.norm = nn.LayerNorm(dim)
        # self.norm = nn.BatchNorm1d(dim)
        self.norm = nn.Identity()
        self.fn = fn
    def forward(self, x, **kwargs):
        b,t,c = x.size()
        x_reshape = x.reshape(b*t,c)
        x_reshape = self.norm(x_reshape)
        x_normed = x_reshape.reshape(b,t,c)
        return self.fn(x_normed, **kwargs)

class ResMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc1 = nn.Linear(c,c)
        self.fc2= nn.Linear(c,c)
    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(x),True))
    
class ResMLP3(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.resmlp1 = ResMLP(c)
        self.resmlp2 = ResMLP(c)
        self.resmlp3 = ResMLP(c)
        self.final_l = nn.Linear(c,c)
    def forward(self, x):
        return x + self.final_l(self.resmlp3(self.resmlp2(self.resmlp1(x))))

class PreNormConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True,is_temporal = False,video_len = 0):
        """
        Custom Convolutional layer with Batch Normalization before convolution.

        Parameters:
        - in_channels: Number of input channels
        - out_channels: Number of output channels
        - kernel_size: Size of the convolutional kernel
        - stride: Stride of the convolution
        - padding: Padding of the convolution
        - groups: Number of groups in the convolution
        - bias: Whether to include bias in the convolution
        """
        super(PreNormConv, self).__init__()

        # Batch Normalization layer
        # self.bn = nn.BatchNorm2d(in_channels,momentum=0.001)
        # self.bn = nn.GroupNorm(32,in_channels)
        # self.bn = LayerNorm2d(in_channels)
        self.bn = nn.Identity()
        self.temporal_conv = 0
        if is_temporal:
            self.temporal_conv = VideoTemporalCausal_MVSC(in_channels,in_channels,3,vid_len=video_len,input_type="2d")
        # Convolutional layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )

    def forward(self, x):
        # Apply Batch Normalization before convolution
        x = self.bn(x)
        if self.temporal_conv:
            x = self.temporal_conv(x)
        # Apply convolution
        x = self.conv(x)

        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Identity(),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Identity(),
            nn.ReLU(True),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        # print("x",x.size())
        b,n,c = x.size()
        x_reshaped = x.reshape(b*n,c)
        x_reshaped =  self.net(x_reshaped)
        x_reshaped = x_reshaped.reshape(b,n,c)
        return x_reshaped

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x
class ConvNorm(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride=1, norm="BN"):
        super(ConvNorm, self).__init__()

        reflection_padding = kernel_size // 2
        # self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_feat, out_feat, stride=stride, kernel_size=kernel_size,padding=reflection_padding, bias=True)

        self.norm = norm
        if norm == 'IN':
            self.norm = nn.InstanceNorm2d(out_feat, track_running_stats=True)
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(out_feat,momentum=0.05)
        elif norm == 'BN001':
            self.norm = nn.BatchNorm2d(out_feat,momentum=0.01)
        elif norm == 'GN':
            self.norm = nn.GroupNorm(32,out_feat)
        elif norm == 'LN':
            self.norm = LayerNorm2d(out_feat)
        elif norm == 'NON':
            self.norm = nn.Identity(out_feat)

    def forward(self, x):
        # out = self.reflection_pad(x)
        
        out = self.conv(x)
        # if self.norm:
        #     out = self.norm(out)
        return out

class Bitparm(nn.Module):
    '''
    save params
    '''
    def __init__(self, channel,type, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        if type == '2d':
            self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
            self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
            if not final:
                self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
            else:
                self.a = None
        elif type == '1d':
            self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, 1,-1), 0, 0.01))
            self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, 1,-1), 0, 0.01))
            if not final:
                self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, 1,-1), 0, 0.01))
            else:
                self.a = None

    def forward(self, x):
        if self.final:
            return F.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + F.tanh(x) * F.tanh(self.a)

class BitEstimator_MVSC(nn.Module):
    '''
    Estimate bit
    '''
    def __init__(self, channel,type='2d'):
        super(BitEstimator_MVSC, self).__init__()
        self.f1 = Bitparm(channel,type)
        self.f2 = Bitparm(channel,type)
        self.f3 = Bitparm(channel,type)
        self.f4 = Bitparm(channel,type, True)
        
    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)
class GaussianEstimator_MVSC(nn.Module):
    '''
    Estimate bit
    '''
    def __init__(self, channel,type='2d'):
        super(GaussianEstimator_MVSC, self).__init__()
        self.std_l = nn.Parameter(torch.nn.init.constant_(torch.empty(channel).view(1,channel ,1,1), 1))
        self.mean_l = nn.Parameter(torch.nn.init.constant_(torch.empty(channel).view(1,channel ,1,1), 0))
    def forward(self, inputs):
        eps = 1e-6
        self.m_normal_dist =torch.distributions. normal.Normal (self.mean_l, (self.std_l**2).clamp(1e-5,1e5))
        # now_v = (inputs - 0 ) / (self.stds_pos_emb)
        now_v_cdf = self.m_normal_dist.cdf(inputs)
        return now_v_cdf
    
    
def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
   pad = (kernel_size - 1) * dilation
   return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)
class VideoTemporalCausal_MVSC(nn.Module):
    def __init__(self,i,o,k,vid_len=8,d=1,input_type ='2d'):
        super().__init__()
        self.input_type = input_type
        self.conv1 = CausalConv1d(i, o, kernel_size=k,dilation = d, groups=i)
        self.vid_len = vid_len
        assert(not vid_len == 0,"VideoTemporalCausal_MVSC  vid_len can not 0")

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
        x = F.leaky_relu(x,0.2,inplace=True)
        if self.input_type == '2d':
            x = d3_to_d2(x)
        return x + res

class MultiscaleCausalConv_MVSC(nn.Module):
    def __init__(self,c_final,clip_len,input_type ='3d'):
        super().__init__()
        self.clip_len = clip_len
        self.input_type = input_type
        self.conv_fuse = nn.Conv3d(c_final, c_final, 1, 1, 0, bias=False)
        c_final_2 = c_final//2
        # self.conv_causual1 = nn.Identity()
        self.conv_causual_up = VideoTemporalCausal_MVSC(c_final_2, c_final_2, 3,vid_len=clip_len, input_type= input_type )
        self.conv_causual_down = VideoTemporalCausal_MVSC(c_final_2, c_final_2, 5,vid_len= clip_len ,input_type= input_type)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # return x
        vid_len = self.clip_len
        if self.input_type == '2d':
            if not self.training:
                vid_len = x.size(0)
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

class ResBlockSimple(nn.Module):
    def __init__(self, in_feat, out_feat,kernel_size=3, act=nn.LeakyReLU(0.2,True),norm = "BN",\
        downscale=False,is_temporal = False, video_len = 0):
        super(ResBlockSimple, self).__init__()
  
        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size=kernel_size, stride=2 if downscale else 1,norm = norm),
            act,
            ConvNorm(out_feat, out_feat, kernel_size=kernel_size, stride=1,norm = norm)
        )
        self.downscale = None
        if downscale:
            self.downscale = nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=2)
        self.res_transform = None
        if not in_feat == out_feat:
            self.res_transform = nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=1)
        self.temporal_conv = None
        if is_temporal:
            self.temporal_conv = VideoTemporalCausal_MVSC(in_feat,in_feat,3,vid_len=video_len,input_type="2d")
    def forward(self, x):
        res = x
        if self.temporal_conv:
            x  =self.temporal_conv(x)
        out = self.body(x)
        if self.downscale is not None:
            res = self.downscale(res)
        elif self.res_transform:
            res = self.res_transform(res)
        out += res

        return out 



class ResBlockSimpleDepth(nn.Module):
    def __init__(self, in_feat, out_feat,kernel_size=3, act=nn.LeakyReLU(0.2,True), downscale=False,is_temporal = False, video_len = 0,norm=None):
        super(ResBlockSimpleDepth, self).__init__()
        if norm:
            self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size=kernel_size, stride=2 if downscale else 1,norm = norm),
            act,
            nn.Conv2d(out_feat, out_feat, stride=1, kernel_size=kernel_size,padding=kernel_size//2, bias=True,groups=out_feat),
            act,
            nn.Conv2d(out_feat, out_feat, stride=1, kernel_size=1,padding=0, bias=True),
            )
        else:
            self.body = nn.Sequential(
                ConvNorm(in_feat, out_feat, kernel_size=kernel_size, stride=2 if downscale else 1,norm=norm),
                act,
                nn.Conv2d(out_feat, out_feat, stride=1, kernel_size=kernel_size,padding=kernel_size//2, bias=True,groups=out_feat),
                act,
                nn.Conv2d(out_feat, out_feat, stride=1, kernel_size=1,padding=0, bias=True),
            )
        self.downscale = None
        if downscale:
            self.downscale = nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=2)
        self.res_transform = None
        if not in_feat == out_feat:
            self.res_transform = nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=1)
        self.temporal_conv = None
        if is_temporal:
            self.temporal_conv = VideoTemporalCausal_MVSC(in_feat,in_feat,3,vid_len=video_len,input_type="2d")
    def forward(self, x):
        res = x
        if self.temporal_conv:
            x  =self.temporal_conv(x)
        out = self.body(x)
        if self.downscale is not None:
            res = self.downscale(res)
        elif self.res_transform:
            res = self.res_transform(res)
        out += res

        return out 
    
class ResBlockSimpleFullDepth(nn.Module):
    def __init__(self, in_feat, out_feat,group_num = 4,kernel_size=3, act=nn.LeakyReLU(0.2,True), downscale=False,is_temporal = False, video_len = 0,norm=None):
        super(ResBlockSimpleFullDepth, self).__init__()
        self.body = nn.Sequential(
        nn.Conv2d(in_feat, in_feat, stride=1, kernel_size=1,padding=0, bias=True,groups=1),
        act,
        nn.Conv2d(in_feat, out_feat, stride=1, kernel_size=kernel_size,padding=kernel_size//2, bias=True,groups=group_num),
        act,
        nn.Conv2d(out_feat, out_feat, stride=1, kernel_size=1,padding=0, bias=True,groups=1),
        # act,
        # nn.Conv2d(out_feat, out_feat, stride=1, kernel_size=kernel_size,padding=kernel_size//2, bias=True,groups=group_num),
        # act,
        # nn.Conv2d(out_feat, out_feat, stride=1, kernel_size=1,padding=0, bias=True),
        )
        
        self.downscale = None
        if downscale:
            self.downscale = nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=2)
        self.res_transform = None
        if not in_feat == out_feat:
            self.res_transform = nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=1)
        self.temporal_conv = None
        if is_temporal:
            self.temporal_conv = VideoTemporalCausal_MVSC(in_feat,in_feat,3,vid_len=video_len,input_type="2d")
            self.temporal_conv1 = VideoTemporalCausal_MVSC(out_feat,out_feat,3,vid_len=video_len,input_type="2d")
    def forward(self, x):
        res = x
        if self.temporal_conv:
            x  =self.temporal_conv(x)
        out = self.body(x)
        if self.temporal_conv:
            out  =self.temporal_conv(out)
        if self.downscale is not None:
            res = self.downscale(res)
        elif self.res_transform:
            res = self.res_transform(res)
        out += res

        return out 

class ResBlockReLU(nn.Module):
    def __init__(self, in_feat, out_feat,kernel_size=3, act=nn.ReLU(True), downscale=False,is_temporal = False, video_len = 0):
        super(ResBlockReLU, self).__init__()
  
        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size=kernel_size, stride=2 if downscale else 1),
            nn.LeakyReLU(0.01,True),
            ConvNorm(out_feat, out_feat, kernel_size=kernel_size, stride=1),
            nn.LeakyReLU(0.01,True),
            ConvNorm(out_feat, out_feat, 1),
            
        )
        self.downscale = None
        if downscale:
            self.downscale = nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=2)
        self.res_transform = None
        if not in_feat == out_feat:
            self.res_transform = nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=1)
        self.temporal_conv = None
        if is_temporal:
            self.temporal_conv = VideoTemporalCausal_MVSC(in_feat,in_feat,3,vid_len=video_len,input_type="2d")
    def forward(self, x):
        res = x
        if self.temporal_conv:
            x  =self.temporal_conv(x)
        out = self.body(x)
        if self.downscale is not None:
            res = self.downscale(res)
        elif self.res_transform:
            res = self.res_transform(res)
        out += res
        out = F.leaky_relu(out,0.01,True)
        return out 


""" CONV - (BN) - RELU - CONV - (BN) """
class ResBlock_MVSC(nn.Module):
    def __init__(self, in_feat, out_feat, vid_len=8,kernel_size=3, reduction=False, bias=True, # 'reduction' is just for placeholder
                 norm=None, act=nn.ReLU(True), downscale=False,use_temporal=False):
        super(ResBlock_MVSC, self).__init__()
        self.use_temporal = use_temporal
        if self.use_temporal:
            self.temporal = VideoTemporalCausal_MVSC(in_feat,in_feat,3,vid_len=vid_len,input_type="2d")

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size=kernel_size, stride=2 if downscale else 1),
            act,
            ConvNorm(out_feat, out_feat, kernel_size=kernel_size, stride=1)
        )
        
        self.downscale = None
        if downscale:
            self.downscale = nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=2)
        self.res_transform = None
        if not in_feat == out_feat:
            self.res_transform = nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=1)

    def forward(self, x):
        res = x
        if self.use_temporal:
            x = self.temporal(x)
        out = self.body(x)
        if self.downscale is not None:
            res = self.downscale(res)
        elif self.res_transform:
            res = self.res_transform(res)
        out += res

        return out 



class TemporalAggregation_MVSC(nn.Module):
    def __init__(self,dim,clip_len,opt,spatial_group = 8,input_type='2d'):
        super().__init__()
        self.input_type = input_type
        self.clip_len = clip_len
        def Layer_Gen(dim):
            return MultiscaleCausalConv_MVSC(dim,clip_len)
        self.group_enc1 = nn.Sequential(
            Layer_Gen(dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, (1,3,3),1,(0,1,1),groups=spatial_group),
            Layer_Gen(dim),
        )
        self.group_enc2 = nn.Sequential(
            Layer_Gen(dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, (1,3,3),1,(0,1,1),groups=spatial_group),
            Layer_Gen(dim),
        )
        self.group_enc3 = nn.Sequential(
            Layer_Gen(dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, (1,3,3),1,(0,1,1),groups=spatial_group),
            Layer_Gen(dim),
        )
        self.head_conv = nn.Sequential(
            nn.Conv3d(dim, dim, 1,1,0),
        )
        self.tail_conv = nn.Sequential(
            nn.Conv3d(dim, dim, 1,1,0),
        )

    def forward(self,x):
        res = x
        if self.input_type == '2d':
            res =d2_to_d3(x,self.clip_len)
        res =  self.head_conv(res)
        x = res
        x = x + self.group_enc1(x)
        x = x + self.group_enc2(x)
        x =  res + self.group_enc3(x)
        x =  self.tail_conv(x)
        if self.input_type == '2d':
            x = d3_to_d2(x)
        return x


class CTHyperEncoder(nn.Module):
    def __init__(self,dim,clip_len):
        super().__init__()
        self.clip_len = clip_len
        self.group_enc1 = nn.Sequential(
            nn.Conv3d(dim, dim, (3,1,1),1,(1,0,0),groups=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, (1,3,3),1,(0,1,1),groups=8),
            nn.Conv3d(dim, dim, (3,1,1),1,(1,0,0),groups=1),
        )
        self.group_enc2 = nn.Sequential(
            nn.Conv3d(dim, dim, (3,1,1),1,(1,0,0),groups=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, (1,3,3),1,(0,1,1),groups=8),
            nn.Conv3d(dim, dim, (3,1,1),1,(1,0,0),groups=1),
        )
        self.group_enc3 = nn.Sequential(
            nn.Conv3d(dim, dim, (3,1,1),1,(1,0,0),groups=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, (1,3,3),1,(0,1,1),groups=8),
            nn.Conv3d(dim, dim, (3,1,1),1,(1,0,0),groups=1),
        )
        self.conv4 = nn.Conv3d(dim, dim, (1,1,1),1,(0,0,0),groups=1)
        # nn.init.constant_(self.group_enc3[-1].weight,0)
        # nn.init.constant_(self.group_enc3[-1].bias,0)

    def forward(self,x):
        res =d2_to_d3(x,self.clip_len)
        x = res
        x = F.relu(x + self.group_enc1(x),True)
        x = F.relu(x + self.group_enc2(x),True)
        x =  self.conv4(res + self.group_enc3(x))
        x = x.mean(-1).mean(-1)
        x = self.tconv(x).unsqueeze(-1).unsqueeze(-1)
        x = d3_to_d2(x) 
        return x


class CTHyperDecoder(nn.Module):
    def __init__(self,dim,clip_len):
        super().__init__()
        self.clip_len = clip_len
        self.group_enc1 = nn.Sequential(
            nn.Conv3d(dim, dim, (3,1,1),1,(1,0,0),groups=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, (1,3,3),1,(0,1,1),groups=8),
            nn.Conv3d(dim, dim, (3,1,1),1,(1,0,0),groups=1),
        )
        self.group_enc2 = nn.Sequential(
            nn.Conv3d(dim, dim, (3,1,1),1,(1,0,0),groups=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, (1,3,3),1,(0,1,1),groups=8),
            nn.Conv3d(dim, dim, (3,1,1),1,(1,0,0),groups=1),
        )
        self.group_enc3 = nn.Sequential(
            nn.Conv3d(dim, dim, (3,1,1),1,(1,0,0),groups=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, (1,3,3),1,(0,1,1),groups=8),
            nn.Conv3d(dim, dim, (3,1,1),1,(1,0,0),groups=1),
        )
        self.conv4 = nn.Conv3d(dim, dim, (1,1,1),1,(0,0,0),groups=1)
        self.tconv = nn.Sequential(
            nn.Conv1d(dim,dim,5,1,1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim,dim,5,1,1),
        )
        # nn.init.constant_(self.group_enc3[-1].weight,0)
        # nn.init.constant_(self.group_enc3[-1].bias,0)

    def forward(self,x):
        x = d2_to_d3(x)


        res =d2_to_d3(x,self.clip_len)
        x = res
        x = F.relu(x + self.group_enc1(x),True)
        x = F.relu(x + self.group_enc2(x),True)
        x =  self.conv4(res + self.group_enc3(x))
        x = x.mean(-1).mean(-1)
        x = self.tconv(x)

        x = d3_to_d2(x)
        return x

class DiffEncoder_MVSC(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.opt = opt
        self.use_global_transformer = opt.use_enc_transformer

        diff_dim = opt.diff_dim
        clean_dim = opt.enc_dim

        self.dynconv1 = DiffGuidedConv_MVSC(12,clean_dim,12,diff_dim,opt,dyn_m="identity")
        dyn_m = 'dynamic_conv'
        if self.opt.no_use_dynamic:
            dyn_m = 'plain_conv'
        self.dynconv2 = DiffGuidedConv_MVSC(clean_dim,clean_dim*2,diff_dim,diff_dim*2,opt,heavy_visual_enc=1,stride=2,dyn_m=dyn_m,kernel_map_bs=8)
        self.dynconv3 = DiffGuidedConv_MVSC(clean_dim*2,clean_dim*4,diff_dim*2,diff_dim*4,opt,heavy_visual_enc=1,stride=2,dyn_m=dyn_m,kernel_map_bs=4)
        self.dynconv4 = DiffGuidedConv_MVSC(clean_dim*4,clean_dim*8,diff_dim*4,diff_dim*8,opt,heavy_visual_enc=1,stride=2,dyn_m=dyn_m,kernel_map_bs=2)
        self.dynconv5 = DiffGuidedConv_MVSC(clean_dim*8,clean_dim*16,diff_dim*8,diff_dim*16,opt,heavy_visual_enc=1,stride=2,dyn_m=dyn_m,kernel_map_bs=1)
        
        if self.use_global_transformer:
            self.transformer_compress = nn.Conv2d(clean_dim*16,clean_dim*4,1,1,0)
            self.global_trans = Transformer(clean_dim*4,1,4,clean_dim//1,clean_dim*4,0)
            self.transformer_expand = nn.Conv2d(clean_dim*4,clean_dim*16,1,1,0)
            self.register_parameter("enc_pos_embedding",torch.nn.Parameter(torch.FloatTensor(clean_dim*4,7*7)))
            torch.nn.init.normal(self.enc_pos_embedding)
        if self.opt.temporal_compression:
            self.temporal_encoder = TemporalAggregation_MVSC(clean_dim*16,self.opt.num_segments,self.opt)

    def forward(self,clean_d2,diff_d2):
        cleanft, diffft = self.dynconv1(clean_d2,diff_d2)
        cleanft, diffft = self.dynconv2(cleanft, diffft)
        cleanft, diffft = self.dynconv3(cleanft, diffft)
        cleanft, diffft = self.dynconv4(cleanft, diffft)
        # import matplotlib.pyplot as plt
        # plt.axis("off")
        # vis = cleanft[2].abs().sum(0)
        # vis = torch.pow(1.2,vis)
        # plt.imshow(vis.data.cpu().numpy(),cmap='jet')
        # plt.savefig(('feature_maps4.png'), bbox_inches='tight')
        cleanft, diffft = self.dynconv5(cleanft, diffft)
        vid_len = self.opt.num_segments
        code = cleanft
        # code = 
        # # code = code + self.group_enc1(code)
        # code = d3_to_d2(code)
        if self.use_global_transformer:
            code_res = code
            code = self.transformer_compress(code)
            b,c,h,w = code.size()
            code = code.reshape(b,c,h*w)
            ax_loc_emb = self.enc_pos_embedding.unsqueeze(0).repeat(b,1,1)
            code = code+ax_loc_emb
            code = code.transpose(1,2)
            code = self.global_trans(code)
            code = code.transpose(1,2).reshape(b,c,h,w)
            code = code_res + self.transformer_expand(code)
        if self.opt.temporal_compression:
            code = self.temporal_encoder(code)

        return code
from torchvision.ops.deform_conv import DeformConv2d

class AlignFt1_toFt2(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.off_set_c = 8*(2*3*3)
        self.offset_pre_net1 = nn.Sequential(
            nn.Conv2d(c * 2, c*2, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(c*2, c, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(c, self.off_set_c, 1, 1, 0, bias=True), 
        )
        self.L1_dcnpack = DeformConv2d(c, c, 3, stride=1, padding=1, dilation=1,groups=8)
        self.fusion_net1 =  nn.Sequential(
            nn.Conv2d(c * 3, c, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(c , c, 3, 1, 1, bias=True),
        )

    def forward(self,f1,ft2):
        offset =self.offset_pre_net1(torch.cat([f1,ft2],dim=1))
        offseted_pre_frame = self.L1_dcnpack(f1,offset)
        return offseted_pre_frame + self.fusion_net1(torch.cat([f1,ft2,offseted_pre_frame],dim=1))

class ResidualSeperable2DConv(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(c,c,3,1,1,groups=c),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(c,c,1,1,0,groups=1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(c,c,3,1,1,groups=c),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(c,c,1,1,0,groups=1),
        )
        

    def forward(self,x): 
        return x + self.layers(x)

class AlignFt1_toFt2Heter(nn.Module):
    def __init__(self,c1,c2,groups=8):
        super().__init__()
        self.off_set_c = 8*(2*3*3)
        mid_c = min(256,c1+c2 )
        self.offset_pre_net1 = nn.Sequential(
            nn.Conv2d(c1+c2, mid_c, 3, 1, 1, bias=True),
            ResidualSeperable2DConv(mid_c),
            ResidualSeperable2DConv(mid_c),
            ResidualSeperable2DConv(mid_c),
            nn.Conv2d(mid_c, self.off_set_c, 3, 1, 1, bias=True), 
            
        )
        self.L1_dcnpack = DeformConv2d(c1, c1, 3, stride=1, padding=1, dilation=1,groups=8)
        self.fusion_net1 =  nn.Sequential(
            nn.Conv2d(c1+c1, c1, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(c1, c1, 3, 1, 1, bias=True),
        )

    def forward(self,f1,ft2):
        offset =self.offset_pre_net1(torch.cat([f1,ft2],dim=1))
        offseted_pre_frame = self.L1_dcnpack(f1,offset)
        return offseted_pre_frame + self.fusion_net1(torch.cat([f1,offseted_pre_frame],dim=1))
    
class AlignFt1_toFt2HeterDense(nn.Module):
    def __init__(self,c1,c2,groups=8):
        super().__init__()
        self.off_set_c = 8*(2*3*3)
        mid_c = min(256,c1+c2 )
        # self.mask_generator = nn.Sequential(
        #     nn.Conv2d(c1+c2, 256, 3, 1, 1, bias=True),
        #     nn.LeakyReLU(0.1,True),
        #     nn.Conv2d(256, c1*3, 3, 1, 1, bias=True),
        # )
        self.offset_pre_net1 = nn.Sequential(
            nn.Conv2d(c1+c2, mid_c, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(mid_c, mid_c, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(mid_c, self.off_set_c, 3, 1, 1, bias=True), 
        )
        self.L1_dcnpack = DeformConv2d(c1, c1, 3, stride=1, padding=1, dilation=1,groups=8)
        self.fusion_net1 =  nn.Sequential(
            nn.Conv2d(c1+c2, c1, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(c1, c1, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(c1, c1, 3, 1, 1, bias=True),
        )

    def forward(self,f1,ft2):
        offset =self.offset_pre_net1(torch.cat([f1,ft2],dim=1))
        offseted_pre_frame = self.L1_dcnpack(f1,offset)
        return offseted_pre_frame + self.fusion_net1(torch.cat([f1,ft2],dim=1))

class AlignFt1_toFt2HeterMask(nn.Module):
    def __init__(self,c1,c2,groups=8):
        super().__init__()
        self.off_set_c = 8*(2*3*3)
        mid_c = min(256,c1+c2 )
        self.mask_generator = nn.Sequential(
            nn.Conv2d(c1+c2,c1+c2, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(c1+c2, c1*3, 1, 1, 0, bias=True),
            nn.Tanh()
        )
        self.offset_pre_net1 = nn.Sequential(
            nn.Conv2d(c1+c2, mid_c, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(mid_c, mid_c, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(mid_c, self.off_set_c, 3, 1, 1, bias=True), 
        )
        self.L1_dcnpack = DeformConv2d(c1, c1, 3, stride=1, padding=1, dilation=1,groups=8)
        self.fusion_net1 =  nn.Sequential(
            nn.Conv2d(c1+c2, c1, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(c1, c1, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(c1, c1, 3, 1, 1, bias=True),
        )

    def forward(self,f1,ft2):
        # m1,m2,m3 = torch.chunk(self.mask_generator(torch.cat([f1,ft2],dim=1)),3,1)
        offset =self.offset_pre_net1(torch.cat([f1,ft2],dim=1))
        offseted_pre_frame = self.L1_dcnpack(f1,offset)
        return offseted_pre_frame+ self.fusion_net1(torch.cat([f1,ft2],dim=1))
class AlignFt1_toFt2HeterDepth(nn.Module):
    def __init__(self,c1,c2,groups=8):
        super().__init__()
        self.off_set_c = 8*(2*3*3)
        mid_c = min(256,c1+c2 )
        # self.mask_generator = nn.Sequential(
        #     nn.Conv2d(c1+c2, 256, 3, 1, 1, bias=True),
        #     nn.LeakyReLU(0.1,True),
        #     nn.Conv2d(256, c1*3, 3, 1, 1, bias=True),
        # )
        self.offset_pre_net1 = nn.Sequential(
            nn.Conv2d(c1+c2, mid_c, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1,True),
            ResBlockSimpleFullDepth(mid_c, mid_c,video_len=-1,is_temporal=False,norm="NON"),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(mid_c, self.off_set_c, 3, 1, 1, bias=True), 
        )
        self.L1_dcnpack = DeformConv2d(c1, c1, 3, stride=1, padding=1, dilation=1,groups=8)
        self.fusion_net1 =  nn.Sequential(
            nn.Conv2d(c1+c2, c1, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1,True),
            ResBlockSimpleFullDepth(c1, c1,video_len=-1,is_temporal=False,norm="NON"),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(c1, c1, 3, 1, 1, bias=True),
        )
        self.mask_net = nn.Sequential(
            ResBlockSimpleFullDepth(c1+c2, c1,video_len=-1,is_temporal=False,norm="NON"),
            nn.Conv2d(c1, c1*2, 1, 1, 0, bias=True),
        )

    def forward(self,f1,ft2):
        offset =self.offset_pre_net1(torch.cat([f1,ft2],dim=1))
        offseted_pre_frame = self.L1_dcnpack(f1,offset)
        mask = self.mask_net(torch.cat([f1,ft2],dim=1))
        mask = F.tanh(mask)
        m1,m2 = torch.chunk(mask,2,dim=1)
        return offseted_pre_frame*m1 + self.fusion_net1(torch.cat([f1,ft2],dim=1))*m2
    
class AlignFt1_byFt2toFt3(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.off_set_c = 8*(2*3*3)
        self.offset_pre_net1 = nn.Sequential(
            nn.Conv2d(c * 2, c*2, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(c*2, c, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(c, self.off_set_c, 1, 1, 0, bias=True), 
        )
        self.L1_dcnpack = DeformConv2d(c, c, 3, stride=1, padding=1, dilation=1,groups=8)
        self.fusion_net1 =  nn.Sequential(
            nn.Conv2d(c * 3, c, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(c , c, 3, 1, 1, bias=True),
        )

    def forward(self,f1,ft2,ft3):
        offset =self.offset_pre_net1(torch.cat([ft2,ft3],dim=1))
        offseted_pre_frame = self.L1_dcnpack(f1,offset)
        return offseted_pre_frame + self.fusion_net1(torch.cat([f1,ft3,offseted_pre_frame],dim=1))
    
    
def pad_3d(x_3d,win_size):
    b,c,t,h,w = x_3d.size()
    old_h,old_w = h,w
    new_h,new_w = h,w
    if not h%win_size ==0:
        new_h = (h//win_size+1)*win_size
    if not w%win_size ==0:
        new_w = (w//win_size+1)*win_size
    new_tensor = torch.zeros((b,c,t,new_h,new_w),device = x_3d.device)
    new_tensor[:,:,:,0:h,0:w] = x_3d
    return new_tensor,old_h,old_w
def window(x_3d,win_size):
    b,c,t,h,w = x_3d.size()
    ft_h,ft_w = x_3d.size(-2),x_3d.size(-1)
    win_num = (ft_h//win_size) *(ft_w//win_size)
    x_3d_win = x_3d.reshape(b,c,t, ft_h//win_size,win_size,ft_w //win_size,win_size)\
        .permute(0,3,5,1,2,4,6).reshape(b*win_num,c,t,win_size,win_size)
    return x_3d_win
def window_2d(x_2d,win_size):
    b,c,h,w = x_2d.size()
    ft_h,ft_w = x_2d.size(-2),x_2d.size(-1)
    new_ft_h = ft_h
    new_ft_w = ft_w
    if not ft_h %win_size  ==0:
        new_ft_h = (ft_h//win_size+1)*win_size
    if not ft_w  %win_size  ==0:
        new_ft_w = (ft_w//win_size+1)*win_size
    x_2d_padded = torch.zeros((b,c,new_ft_h,new_ft_w),device = x_2d.device)
    x_2d_padded[:,:,0:ft_h,0:ft_w] = x_2d
        
    win_num = (new_ft_h//win_size) *(new_ft_w//win_size)
    x_2d_win = x_2d_padded.reshape(b,c, new_ft_h//win_size,win_size,new_ft_w //win_size,win_size)\
        .permute(0,2,4,1,3,5).reshape(b*win_num,c,win_size,win_size)
    return x_2d_win
def dewindow2d(x,_b,h_win_num,w_win_num,_c,win_size):
    ##_b_win_num,_c,win_size,win_size
    x = x.reshape(_b,h_win_num,w_win_num,_c,win_size,win_size  )\
        .permute(0,3,1,4,2,5)\
        .reshape(_b,_c, h_win_num*win_size,w_win_num*win_size )
    return x