import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Image to patch embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)  # similar to: nn.Linear(in_chans*patch_size*patch_size, embed_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, H*W, C]
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)  # Linear layer similar matrix multiplication with an opition bias
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, dim_per_head]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)  # [B, num_heads, N, N]
        attn = self.attn_drop(attn)
        
        x = (attn @ v).tranpose(1, 2).reshape(B, N, C)  # [B, num_heads, N, dim_per_head] --> [B, N, num_heads, dim_per_head] --> [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


        
        
        
        
        
    
class Block(nn.Module):
    """One transformer layer — the unit that is stacked `depth` times to form the ViT.

    After patch embedding, tokens are independent: each one encodes its own
    16x16 image region and knows nothing about the others. A Block is what
    lets them interact and refine. It does this in two stages:

      1. Self-attention — every token looks at every other token and pulls in
         whatever it finds relevant. This is the only place in the network
         where information moves *between* spatial positions, and it is why
         a ViT has a global receptive field from the very first layer,
         unlike a CNN which grows its receptive field gradually.
      2. MLP — each token is then transformed on its own, giving the model
         nonlinear capacity to process what attention just gathered.

    Both stages are residual (x + sublayer(x)) and pre-normalized
    (LayerNorm applied *before* each sublayer, not after the add). The
    residual path keeps gradients flowing through a deep stack; pre-norm
    keeps activations bounded at depth, which is what makes a 12-layer
    transformer trainable without the warmup tricks post-norm needs.

    Shape is preserved end to end — (B, N, dim) in, (B, N, dim) out — which
    is precisely what allows Blocks to be stacked to arbitrary depth. Each
    successive Block re-mixes the same N tokens at the same width, building
    progressively more abstract features while the CLS token accumulates a
    summary of the whole image.

    Args:
        dim: token dimension (384 for ViT-S, 768 for ViT-B).
        num_heads: attention heads; each attends in a dim/num_heads subspace,
            letting different heads specialize on different relationships.
        mlp_ratio: MLP hidden width as a multiple of dim (4.0 by default).
        drop_path: stochastic depth rate for this block. Ramps from 0 at the
            first block to drop_path_rate at the last.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop=drop)


class VisionTransformer(nn.Module):
    """ViT backbone adapted for DINO.

    Differs from a standard classification ViT: no classification head
    (forward returns the CLS token for the projection head), positional
    embeddings are interpolated at every forward pass so the same backbone
    can take 224x224 global crops and 96x96 local crops, and no batch
    normalization anywhere.

    Module names follow the timm/DeiT layout so official DINO checkpoints
    load with strict=True.

    Args:
        img_size: base resolution used to initialise the position embeddings.
        patch_size: patch side length (16 or 8).
        embed_dim: token dimension (384 for ViT-S, 768 for ViT-B).
        depth: number of transformer blocks.
        num_heads: attention heads per block.
    """
    
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        
        super().__init__()
        self.num_features = embed_dim
        self.embed_dim = embed_dim
        
        self.patch_embed = PatchEmbed(img_size[0], patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1,1, embed_dim))  # nn.Parameter is a Tensor subclass that, when assigned as a Module attribute, is auto-registered as a learnable parameter.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)  # randomly zeroes each element of its input independently with probability p, during training only
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # builds a per-block list of drop-path probabilities that ramp linearly from 0 to drop_path_rate across the depth of the network.
        
        
        