from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.hidden_features = hidden_features
        if self.hidden_features>0:
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.hidden_features>0:
            x = self.fc1(x)
            x = self.act(x)
            # x = self.drop(x)
            # commit this for the orignal BERT implement 
            x = self.fc2(x)
            x = self.drop(x)
            return x
        else:
            return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn_map = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_map)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x,attn_map


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        ft,attn_map  =self.attn(self.norm1(x))
        if self.gamma_1 is None:
            x = x + self.drop_path(ft)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * ft)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x, attn_map


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # B 768 t//1 h/16 w/16
        x = self.proj(x).flatten(2).transpose(1, 2)
        
        return x
    
# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False, 
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_mean_pooling=True,
                 left_padding_token_num = 0
                ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        self.left_padding_token_num = left_padding_token_num
        num_patches = self.patch_embed.num_patches + left_padding_token_num
        if self.left_padding_token_num>0:
            self.nerf_token_embed = nn.Parameter(torch.zeros(1, self.left_padding_token_num, embed_dim))
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        # print(x.size())
        # exit()
        B, _, _ = x.size()##b nc
        # print(self.nerf_token_embed.size(),x.size()) 
        if self.left_padding_token_num>0:
            x = torch.cat(
            [
                self.nerf_token_embed.repeat(B,1,1),
                x
            ],
            dim=1
            )
        
        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        return x
        x = self.norm(x)
        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        else:
            return x[:, 0]

    def forward(self, x):
        x_feature = self.forward_features(x)
        return x_feature
        x = self.head(x)
        return x

class FinerTransformerST(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
            input_num, left_padding_token_num,embed_dim,
            depth,mlp_ratio = 4,num_heads=12,psize = 16,only_ft_extractor = False,video_len = -1
                 ):
        super().__init__()
        self.video_len = video_len
        self.psize = psize
        self.only_ft_extractor = only_ft_extractor
        self.left_padding_token_num = left_padding_token_num
        num_patches =input_num + left_padding_token_num
        embed_dim = embed_dim
        if self.left_padding_token_num >0:
            self.nerf_token_embed = nn.Parameter(torch.zeros(1, left_padding_token_num, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.video_len, num_patches, embed_dim))
        
        # self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
        drop_path_rate  = 0
        self.pos_drop = nn.Dropout(p=0)
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        init_values  =  0
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                drop=0, attn_drop=0, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.blocks_temporal = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=1, qkv_bias=True, qk_scale=None,
                drop=0, attn_drop=0, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        if not self.only_ft_extractor :
            decoder_embed_dim = embed_dim//2
            self.mask_token_pos_embed = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.video_len,num_patches, decoder_embed_dim))
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.decoder_blocks = nn.ModuleList([
                Block(
                    dim=decoder_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                    drop=0, attn_drop=0, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
                for i in range(1)])
            self.decoder_blocks_temporal = nn.ModuleList([
                Block(
                    dim=decoder_embed_dim, num_heads=num_heads, mlp_ratio=1, qkv_bias=True, qk_scale=None,
                    drop=0, attn_drop=0, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
                for i in range(1)])
            self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
            self.decoder_pred = nn.Linear(decoder_embed_dim, (self.psize)**2 *3, bias=True) # decoder to patch

            self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x,imgs_clean):
        # x = self.patch_embed(x)
        # print(x.size())
        # exit()
        B, N, C = x.size()##b nc
        t = self.video_len
        b,t = B//t,t
        if self.pos_embed is not None:
            x_st_expand = x.reshape(b,t,N,C)
            x_st_expand = x_st_expand + self.pos_embed.repeat(b, 1, 1,1)
            x = x_st_expand.reshape(B,N,C)
        x = self.pos_drop(x)
        x_unmask_features, masks, ids_restores = self.random_masking(x, 0.9)
        Group_N = len(masks)
        mae_losses = []
        x_unmask_feature_stacked = []
        for group_i in range(Group_N):
            x_unmask_feature= x_unmask_features[group_i] ## B N C
            x_unmask_feature_stacked += [x_unmask_feature]
        x_unmask_feature_stacked = torch.cat(x_unmask_feature_stacked, dim=0)
        
        b_big = x_unmask_feature_stacked.size(0)//t
        blk_i = 0
        N = x_unmask_features[0].size(1)
        for blk in self.blocks:
            x_unmask_feature_stacked = x_unmask_feature_stacked.reshape(b_big,t,N,C)\
                .transpose(1,2).reshape(b_big*N,t,C)
            x_unmask_feature_stacked,_ = self.blocks_temporal[blk_i](x_unmask_feature_stacked)
            x_unmask_feature_stacked = x_unmask_feature_stacked.reshape(b_big,N,t,C).transpose(1,2).reshape(b_big*t,N,C)
            x_unmask_feature_stacked,_ = blk(x_unmask_feature_stacked)
            blk_i =  blk_i +1
        x_unmask_feature_unstacked = x_unmask_feature_stacked.reshape(Group_N, b*t,N,C)
        for group_i in range(Group_N):
            mask= masks[group_i]
            ids_restore= ids_restores[group_i]
            x_unmask_feature = x_unmask_feature_unstacked[group_i]
            pred = self.forward_decoder(x_unmask_feature, ids_restore)  # [N, L, p*p*3]
            mae_loss = self.forward_loss(imgs_clean, pred, mask)
            mae_losses += [mae_loss]
        return x_unmask_feature, pred,sum(mae_losses)/len(mae_losses)
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        target_target = target
        self.norm_pix_loss =False
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        # print(target,target.max(), target.min(),target_target.max(), target_target.min())
        # exit()
        ## 
        #imgs: [BT, 3, H, W]
        b,t = imgs.size(0)//self.video_len,self.video_len
        ## target  bt (H//p)*(W//p) pp3
        pnum = 256//self.psize
        # target = target.reshape(b,t//2,pnum*pnum,self.psize*self.psize*3*2)\
        #     .reshape(b,t*pnum*pnum//2, self.psize*self.psize*3*2)
        target = target.reshape(b*t,pnum*pnum,self.psize*self.psize*3)
            # .reshape(bt,pnum*pnum, self.psize*self.psize*3*2)
        loss = (pred - target)
        loss = loss.abs()
        # loss = loss**2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    def forward_decoder(self, x, ids_restore):
        # embed tokens
        B,N,C = x.size()
        t = self.video_len
        b,t = B//t,t
        # B = B2//2
        # x = x.reshape(B,2,N,256).permute(0,2,1,3).reshape(B,N,512)
        x = self.decoder_embed(x)
        C = x.size(-1)
        # append mask tokens to sequence
        # print(ids_restore.size(),x.size())
        # mask_tokens = self.mask_token_pos_embed.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        mask_tokens = self.mask_token_pos_embed.repeat(x.shape[0], ids_restore.shape[1]  - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        # print(x_.size())
        # exit()
        
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        x = x_
        N = x.size(1)
        # add pos embed
        x_st_expand = x.reshape(b,t,N,C)
        x_st_expand = x_st_expand + self.decoder_pos_embed
        x = x_st_expand.reshape(B,N,C)
        # apply Transformer blocks
        block_i = 0
        for blk in self.decoder_blocks:
            x_t_expand  =x.reshape(b,t,N,C).transpose(1,2).reshape(b*N,t,C)
            x_t_expand,_ = self.decoder_blocks_temporal[block_i](x_t_expand)
            x  =x_t_expand.reshape(b,N,t,C).transpose(1,2).reshape(B,N,C)
            x,_ = blk(x)
            block_i = block_i +1
            
            
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        # remove cls token
        # x = x[:, 1:, :]

        return x
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        # x  = x.reshape(N//8,8,L,D)
        
        
        noise = torch.rand(N//self.video_len, L, device=x.device)  # noise in [0, 1]
        noise = noise.unsqueeze(1).repeat(1,self.video_len,1).reshape(N,L)
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        len_keep = int(L * (1 - mask_ratio))
        group_num = int(1/(1-mask_ratio))
        group_num = 1
        x_maskeds = []
        masks = []
        ids_restores = []
        for group_i in range(group_num):
            ids_keep = ids_shuffle[:, group_i*len_keep:(group_i+1)*len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=x.device)
            mask[:, group_i*len_keep:(group_i+1)*len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
            x_maskeds += [x_masked]
            masks += [mask]
            ids_restores += [ids_restore]

        return x_maskeds, masks, ids_restores
    def forward(self, x):
        x,imgs_clean = x
        x_feature = self.forward_features(x,imgs_clean)
        return x_feature
    def forward_layer_feature(self, x):
        B, N, C = x.size()##b nc
        t = self.video_len
        b,t = B//t,t
        if self.pos_embed is not None:
            x_st_expand = x.reshape(b,t,N,C)
            x_st_expand = x_st_expand + self.pos_embed.repeat(b, 1, 1,1)
            x = x_st_expand.reshape(B,N,C)
        x = self.pos_drop(x)
        blk_i = 0
        att_maps = []
        for blk in self.blocks:
            x = x.reshape(b,t,N,C)\
                .transpose(1,2).reshape(b*N,t,C)
            x,_ = self.blocks_temporal[blk_i](x)
            x = x.reshape(b,N,t,C).transpose(1,2).reshape(b*t,N,C)
            x,att_map = blk(x)
            blk_i =  blk_i +1
            if blk_i in [2,4,6,8]:
                att_maps += [x]
            
        return att_maps

@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_base_patch16_224(pretrained=False, depth=10,left_padding_token_num = 0,\
    patch_size=16,tubelet_size = 2,img_size = 224, vid_len = 8, embed_dim=768,mlp_ratio = 4):
    # model = VisionTransformer(
    #     patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    # model.default_cfg = _cfg()
    # return model
    # model = VisionTransformer(
    #     patch_size=16, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4, qkv_bias=True,
    #     all_frames = 8,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = VisionTransformer(
        img_size = img_size,
        patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=12, mlp_ratio=mlp_ratio, qkv_bias=True,
        all_frames = vid_len,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),left_padding_token_num = left_padding_token_num,tubelet_size=tubelet_size)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_512(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model