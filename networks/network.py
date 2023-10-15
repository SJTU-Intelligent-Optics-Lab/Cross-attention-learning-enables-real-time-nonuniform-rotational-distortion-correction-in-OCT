from functools import partial
import torch
import torch.nn as nn
from models.vision_transformer import Block
import models

def cross_attention_network(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=4, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

class VisionTransformer(models.vision_transformer.VisionTransformer):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """


    def __init__(
        self,
        line_dim = 512,
        line_number = 1024,
        dropout_rate: float = 0.0,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """
        super(VisionTransformer, self).__init__(embed_dim=embed_dim,**kwargs)

        #define vit
        self.patch_embed_ref = nn.Linear(line_dim, embed_dim)
        self.norm_ref = norm_layer(embed_dim)

        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, line_number, embed_dim))# 位置编码可训练

        # self.feature1 = nn.Conv2d(1, 1024, (1,512), stride=(1,512))
        # self.batch1 = nn.BatchNorm2d(1024)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.feature2 = nn.Conv2d(1, 1024, (1, 512), stride=(1, 512))
        # self.batch2 = nn.BatchNorm2d(1024)
        # self.relu2 = nn.ReLU(inplace=True)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, attn_drop=dropout_rate)
            for i in range(depth)])

        self.blocks_fusion = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, attn_drop=dropout_rate)
            for i in range(1)])

        # self.norm1 = norm_layer(embed_dim)
        # self.proj1 = nn.Linear(embed_dim,256)
        # self.proj2 = nn.Linear(256, 1)

        #self.regress = nn.Linear(embed_dim, line_number)
        self.regress = nn.Linear(line_number, line_number)
        self.regress1 = nn.Linear(line_number, line_number)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights["state_dict"]:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.position_embeddings_3d"]
            )
            self.vit.patch_embedding.cls_token.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.cls_token"]
            )
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.weight"]
            )
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.bias"]
            )

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights["state_dict"]["module.transformer.norm.weight"])
            self.vit.norm.bias.copy_(weights["state_dict"]["module.transformer.norm.bias"])

    def forward(self, ref, obj):
        # ref image as K and V, obj image as Q
        ref, obj = ref.squeeze(1), obj.squeeze(1)
        B,N,Line = ref.shape

        #patch embedding
        ref = self.patch_embed_ref(ref)
        ref = self.norm_ref(ref)
        obj = self.patch_embed_ref(obj)
        obj = self.norm_ref(obj)

        # add pos embed
        ref = ref + self.pos_embed[:, :, :]
        obj = obj + self.pos_embed[:, :, :]

        # apply Transformer blocks
        for blk in self.blocks:
            obj_tmp = blk(ref,obj)
            ref_tmp = blk(obj,ref)
            obj = obj_tmp
            ref = ref_tmp

        ref1 = ref
        obj1 = obj
        for blk in self.blocks_fusion:
            obj1 = blk(ref1,obj1)

        ref2 = ref
        obj2 = obj
        for blk in self.blocks_fusion:
            ref2 = blk(obj2,ref2)

        x1 = obj1.mean(axis=2)
        x1 = self.regress(x1)

        x2 = ref2.mean(axis=2)
        x2 = self.regress(x2)
        return x1,x2

        # #最后一版5.29
        # x1 = obj.mean(axis=2)
        # x1 = self.regress(x1)
        #
        # x2 = ref.mean(axis=2)
        # x2 = self.regress(x2)
        # return x1, x2
