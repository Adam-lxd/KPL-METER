""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Hacked together by / Copyright 2020 Ross Wightman
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from pytorch_lightning.utilities.distributed import rank_zero_info


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()

        self.down_proj = nn.Linear(c_in, c_in // reduction, bias=True)
        self.non_linear_func = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(c_in // reduction, c_in, bias=True)
        self.drop = nn.Dropout(0.1)
        # init weights
        self.down_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.up_proj.weight.data.normal_(mean=0.0, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True):
        x = self.down_proj(x)
        x = self.non_linear_func(x)
        x = self.drop(x)
        x = self.up_proj(x)
        return x


class AdapterMultiModal(nn.Module):
    def __init__(self,
                 d_model=None,
                 middle_dim=None,
                 dropout=0.1,
                 adapter_layernorm_option="in",
                 ):
        super().__init__()
        middle_features = middle_dim
        self.n_embd = d_model
        self.activation = nn.ReLU()
        # text-down
        self.text_down_proj = nn.Linear(self.n_embd, middle_features)
        # text-up
        self.text_up_proj = nn.Linear(middle_features, 384)

        # image-down
        self.image_down_proj = nn.Linear(self.n_embd, middle_features)
        # image-up
        self.image_up_proj = nn.Linear(middle_features, 384)

        # share-down
        self.share_down_proj = nn.Linear(self.n_embd, middle_features)
        # share-up
        self.share_up_proj = nn.Linear(middle_features, 384)
        
        self.multi_norm = nn.LayerNorm(self.n_embd)
        self.x_norm = nn.LayerNorm(self.n_embd)

        self.dropout = nn.Dropout(p=dropout)

        # ==== Init weights
        self.text_down_proj.weight.data.normal_(mean=0.0, std=0.02)
        nn.init.zeros_(self.text_down_proj.bias)
        self.text_up_proj.weight.data.normal_(mean=0.0, std=0.02)
        nn.init.zeros_(self.text_up_proj.bias)

        self.image_down_proj.weight.data.normal_(mean=0.0, std=0.02)
        nn.init.zeros_(self.image_down_proj.bias)
        self.image_up_proj.weight.data.normal_(mean=0.0, std=0.02)
        nn.init.zeros_(self.image_up_proj.bias)

        self.share_down_proj.weight.data.normal_(mean=0.0, std=0.02)
        nn.init.zeros_(self.share_down_proj.bias)
        self.share_up_proj.weight.data.normal_(mean=0.0, std=0.02)
        nn.init.zeros_(self.share_up_proj.bias)

        

    def forward(self, x, add_residual=True, mode="text"):
        x = self.x_norm(x)
        # shared features
        shared_x = self.share_down_proj(x)
        shared_x = self.activation(shared_x)
        shared_x = self.dropout(shared_x)
        shared_x = self.share_up_proj(shared_x)
        if mode == "text":
            unimodal_x = self.text_down_proj(x)
            unimodal_x = self.activation(unimodal_x)
            unimodal_x = self.dropout(unimodal_x)
            unimodal_x = self.text_up_proj(unimodal_x)

        else:
            unimodal_x = self.image_down_proj(x)
            unimodal_x = self.activation(unimodal_x)
            unimodal_x = self.dropout(unimodal_x)
            unimodal_x = self.image_up_proj(unimodal_x)
        x = torch.cat([unimodal_x, shared_x], dim=2)
        x = self.activation(x)
        return x

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, mask=None, relative_position_bias=None):
        B, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))
        
        if relative_position_bias is not None:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        with_vlffn=False,
        layer_scale_init_values=0.1,
        max_text_len=40,
        use_adapter=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2_text = norm_layer(dim)
        self.norm2_imag = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.use_adapter = use_adapter
        self.mlp_text = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp_imag = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp_vl = None
        if use_adapter:
                self.image_adapter = Adapter(c_in=dim, reduction=3)
                self.text_adapter = Adapter(c_in=dim, reduction=3)

        if with_vlffn:
            self.mlp_vl = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
            self.norm2_vl = norm_layer(dim)
            # whether use adapter
            if use_adapter:
                self.multi_adaper = AdapterMultiModal(d_model=dim, middle_dim=128)
            
        self.gamma_1 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0
        self.gamma_2 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0

        self.max_text_len = max_text_len

    def forward(self, x, prompt_embeds=None, prompt_masks=None, mask=None, modality_type=None, relative_position_bias=None):
        # if use adapter
        if self.use_adapter:
            if prompt_embeds is not None:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), mask=mask, relative_position_bias=relative_position_bias))

                # replace image parts
                ## cat prompt and image parts
                prompt_embeds = torch.cat([prompt_embeds, x[:, self.max_text_len :]], dim=1)
                ## perform self-attention
                prompt_embeds = prompt_embeds + self.drop_path(self.gamma_1 * self.attn(self.norm1(prompt_embeds), mask=prompt_masks, relative_position_bias=relative_position_bias))
                ## replace image parts
                x[:, self.max_text_len :] = prompt_embeds[:, self.max_text_len :]
                ## split prompt parts
                prompt_embeds = prompt_embeds[:, : self.max_text_len]
            else:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), mask=mask, relative_position_bias=relative_position_bias))

            if modality_type == "image":
                # apply adapter
                x_nrom = self.norm2_imag(x)
                adapter_imag = self.image_adapter(x_nrom, add_residual=False) 
                x =  x + self.drop_path(self.gamma_2 * (self.mlp_imag(x_nrom) + adapter_imag))
            elif modality_type == "text":
                # apply adapter
                x_nrom = self.norm2_text(x)
                adapter_text = self.text_adapter(x_nrom, add_residual=False) 
                x =  x + self.drop_path(self.gamma_2 * (self.mlp_text(x_nrom) + adapter_text))
            else:
                if self.mlp_vl is None:
                    x_text = x[:, : self.max_text_len]
                    x_imag = x[:, self.max_text_len :]

                    x_text_norm = self.norm2_text(x_text)
                    adapter_text = self.text_adapter(x_text_norm, add_residual=False) 
                    x_text =  x_text + self.drop_path(self.gamma_2 * (self.mlp_text(x_text_norm) + adapter_text))

                    x_imag_norm = self.norm2_imag(x_imag)
                    adapter_imag = self.image_adapter(x_imag_norm, add_residual=False)
                    x_imag = x_imag + self.drop_path(self.gamma_2 * (self.mlp_imag(x_imag_norm) + adapter_imag))
                    
                    x = torch.cat([x_text, x_imag], dim=1)

                else:
                    
                    x_nrom = self.norm2_vl(x)
                    x_text = x_nrom[:, : self.max_text_len]
                    x_imag = x_nrom[:, self.max_text_len :]
                    adapter_text = self.multi_adaper(x_text, add_residual=False, mode="text")
                    adapter_imag = self.multi_adaper(x_imag, add_residual=False, mode="image")
                    adapter_multi = torch.cat([adapter_text, adapter_imag], dim=1)
                    x = x + self.drop_path(self.gamma_2 * (self.mlp_vl(x_nrom) + adapter_multi))
                    

            return x, prompt_embeds
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), mask=mask, relative_position_bias=relative_position_bias))

            if modality_type == "image":
                x = x + self.drop_path(self.gamma_2 * self.mlp_imag(self.norm2_imag(x)))
            elif modality_type == "text":
                x = x + self.drop_path(self.gamma_2 * self.mlp_text(self.norm2_text(x)))
            else:
                if self.mlp_vl is None:
                    x_text = x[:, : self.max_text_len]
                    x_imag = x[:, self.max_text_len :]
                    x_text = x_text + self.drop_path(self.gamma_2 * self.mlp_text(self.norm2_text(x_text)))
                    x_imag = x_imag + self.drop_path(self.gamma_2 * self.mlp_imag(self.norm2_imag(x_imag)))
                    x = torch.cat([x_text, x_imag], dim=1)
                else:
                    x = x + self.drop_path(self.gamma_2 * self.mlp_vl(self.norm2_vl(x)))

            return x, prompt_embeds


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        no_patch_embed_bias=False,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False if no_patch_embed_bias else True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # FIXME look at relaxing size constraints
        x = self.proj(x)
        return x


class MultiWayTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        need_relative_position_embed=True,
        use_abs_pos_emb=False,
        layer_scale_init_values=0.1,
        vlffn_start_layer_index=10,
        config=None,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            need_relative_position_embed (bool): enable relative position bias on self-attention
            use_abs_pos_emb (bool): enable abs pos emb
            layer_scale_init_values (float or None): layer scale init values, set None to disable
            vlffn_start_layer_index (int): vl-ffn start index
            config: (dict): other hyper from pytorch-lighting
        """
        super().__init__()
        drop_path_rate = drop_path_rate if config is None else config["drop_path_rate"]
        rank_zero_info("drop path rate: {}".format(drop_path_rate))
        self.use_abs_pos_emb = use_abs_pos_emb
        self.need_relative_position_embed = need_relative_position_embed

        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.vlffn_start_layer_index = vlffn_start_layer_index
        if config["loss_names"]["textmlm"] > 0:
            self.vlffn_start_layer_index = depth
            rank_zero_info("Set vlffn_start_layer_index={} for text-only pretraining".format(self.vlffn_start_layer_index))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) if self.use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        # whether use adapter
        self.use_adapter = config["use_adapter"]
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    with_vlffn=(i >= self.vlffn_start_layer_index),
                    layer_scale_init_values=layer_scale_init_values,
                    max_text_len=config["max_text_len"],
                    use_adapter=self.use_adapter,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def visual_embed(self, _x):
        x = self.patch_embed(_x)
        x = x.flatten(2).transpose(1, 2)
        B, L, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        x_mask = torch.ones(x.shape[0], x.shape[1])

        return x, x_mask


# VLMo base/p16
@register_model
def vlmo_base_patch16(pretrained=False, **kwargs):
    img_size = kwargs.pop("img_size", 224)
    model = MultiWayTransformer(
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
        mlp_ratio=4, qkv_bias=True, vlffn_start_layer_index=10, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# VLMo large/p16
@register_model
def vlmo_large_patch16(pretrained=False, **kwargs):
    img_size = kwargs.pop("img_size", 224)
    model = MultiWayTransformer(
        img_size=img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, 
        mlp_ratio=4, qkv_bias=True, vlffn_start_layer_index=21, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# VLMo base+/p16
@register_model
def vlmo_base_plus_patch16(pretrained=False, **kwargs):
    img_size = kwargs.pop("img_size", 224)
    model = MultiWayTransformer(
        img_size=img_size, patch_size=16, embed_dim=544, depth=24, num_heads=16, 
        mlp_ratio=4, qkv_bias=True, vlffn_start_layer_index=21,
        use_abs_pos_emb=True, need_relative_position_embed=False, 
        layer_scale_init_values=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
