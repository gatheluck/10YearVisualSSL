"""Masked Autoencoder (MAE) -- a self-contained re-implementation.

Ported from the lab's own implementation (`methods/25_mae/models/mae_vit.py` in
the capture), which is an independent implementation of He et al. (2021),
"Masked Autoencoders Are Scalable Vision Learners" (arXiv:2111.06377) -- not a
copy of the CC-BY-NC facebookresearch/mae code, and it uses no pretrained
weights. So MAE ports self-contained here, the same treatment the other
re-implemented methods got: there is no `third_party/` submodule.

  Encoder : ViT (patch embed -> mask visible tokens -> transformer blocks)
  Decoder : lightweight transformer (encoder outputs + mask tokens -> pixels)
  Masking : random token masking, 75% masked
  Loss    : MSE on (optionally normalised) pixel values, masked patches only

`MAEEncoder` (via `MaskedAutoencoder.get_encoder`) exposes the encoder alone and
returns CLS or global-average-pooled patch features -- the representation the
linear probe reads. Only torch is used; the architecture is parameterisable so a
hermetic smoke can build a tiny one.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Positional embeddings
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int,
                            cls_token: bool = False) -> torch.Tensor:
    """2-D sinusoidal positional embedding, shape (grid_size**2 [+1], embed_dim)."""
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack([grid_h.reshape(-1), grid_w.reshape(-1)], dim=0)

    half = embed_dim // 2
    omega = 1.0 / (10000 ** (torch.arange(half // 2, dtype=torch.float32)
                             / (half // 2)))
    emb_h = torch.einsum("n,d->nd", grid[0], omega)
    emb_w = torch.einsum("n,d->nd", grid[1], omega)
    emb_h = torch.cat([emb_h.sin(), emb_h.cos()], dim=-1)
    emb_w = torch.cat([emb_w.sin(), emb_w.cos()], dim=-1)
    emb = torch.cat([emb_h, emb_w], dim=-1)

    if cls_token:
        emb = torch.cat([torch.zeros(1, embed_dim), emb], dim=0)
    return emb


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Split an image into non-overlapping patches and linearly project each."""

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: "int | None" = None,
                 out_dim: "int | None" = None, act_layer=nn.GELU):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, hidden_dim=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Masked Autoencoder
# ---------------------------------------------------------------------------

class MaskedAutoencoder(nn.Module):
    """MAE following He et al. (2021). Step 1 (this port) is ViT-L/16 by recipe;
    the architecture is parameterisable so a smoke can build a tiny one."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        enc_embed_dim: int = 1024,
        enc_depth: int = 24,
        enc_num_heads: int = 16,
        dec_embed_dim: int = 512,
        dec_depth: int = 8,
        dec_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.enc_embed_dim = enc_embed_dim

        # Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans,
                                      enc_embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, enc_embed_dim))
        self.register_buffer(
            "enc_pos_embed",
            torch.zeros(1, self.num_patches + 1, enc_embed_dim),
            persistent=False)
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, norm_layer=norm_layer)
            for _ in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)

        # Decoder
        self.dec_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        self.register_buffer(
            "dec_pos_embed",
            torch.zeros(1, self.num_patches + 1, dec_embed_dim),
            persistent=False)
        self.dec_blocks = nn.ModuleList([
            Block(dec_embed_dim, dec_num_heads, mlp_ratio, norm_layer=norm_layer)
            for _ in range(dec_depth)])
        self.dec_norm = norm_layer(dec_embed_dim)
        self.dec_pred = nn.Linear(dec_embed_dim,
                                  patch_size * patch_size * in_chans, bias=True)

        self._init_weights()

    def _init_weights(self):
        enc_pe = get_2d_sincos_pos_embed(
            self.enc_embed_dim, int(self.num_patches ** 0.5), cls_token=True)
        self.enc_pos_embed.data.copy_(enc_pe.unsqueeze(0))
        dec_pe = get_2d_sincos_pos_embed(
            self.dec_embed.out_features, int(self.num_patches ** 0.5),
            cls_token=True)
        self.dec_pos_embed.data.copy_(dec_pe.unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view(w.size(0), -1))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(imgs.shape[0], h * w, p * p * 3)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        h = w = int(patches.shape[1] ** 0.5)
        x = patches.reshape(patches.shape[0], h, w, p, p, 3)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(patches.shape[0], 3, h * p, w * p)

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        B, N, D = x.shape
        n_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :n_keep]
        x_visible = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask = mask.gather(1, ids_restore)
        return x_visible, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float):
        x = self.patch_embed(x)
        x = x + self.enc_pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls = self.cls_token + self.enc_pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.enc_blocks:
            x = blk(x)
        x = self.enc_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor):
        x = self.dec_embed(x)
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_no_cls = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_no_cls = x_no_cls.gather(
            1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_no_cls], dim=1)
        x = x + self.dec_pos_embed
        for blk in self.dec_blocks:
            x = blk(x)
        x = self.dec_norm(x)
        x = self.dec_pred(x)
        return x[:, 1:, :]

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs: torch.Tensor, mask_ratio: "float | None" = None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def get_encoder(self, pool: str = "cls") -> "MAEEncoder":
        return MAEEncoder(self, pool=pool)


class MAEEncoder(nn.Module):
    """The encoder alone, returning CLS or global-average-pooled patch features
    (He et al. Section 4). This is what the linear probe reads."""

    def __init__(self, mae: MaskedAutoencoder, pool: str = "cls"):
        super().__init__()
        if pool not in {"cls", "avg"}:
            raise ValueError(f"unknown MAE encoder pool: {pool}")
        self.pool = pool
        self.patch_embed = mae.patch_embed
        self.cls_token = mae.cls_token
        self.register_buffer("enc_pos_embed", mae.enc_pos_embed.clone())
        self.enc_blocks = mae.enc_blocks
        self.enc_norm = mae.enc_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.enc_pos_embed[:, 1:, :]
        cls = self.cls_token + self.enc_pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.enc_blocks:
            x = blk(x)
        x = self.enc_norm(x)
        if self.pool == "cls":
            return x[:, 0]
        return x[:, 1:, :].mean(dim=1)


def mae_vit_base_patch16(**kwargs) -> MaskedAutoencoder:
    """ViT-B/16 MAE."""
    defaults = dict(img_size=224, patch_size=16,
                    enc_embed_dim=768, enc_depth=12, enc_num_heads=12,
                    dec_embed_dim=512, dec_depth=8, dec_num_heads=16,
                    mlp_ratio=4.0, norm_pix_loss=True, mask_ratio=0.75)
    defaults.update(kwargs)
    return MaskedAutoencoder(**defaults)


def mae_vit_large_patch16(**kwargs) -> MaskedAutoencoder:
    """ViT-L/16 MAE -- the Step 1 recipe."""
    defaults = dict(img_size=224, patch_size=16,
                    enc_embed_dim=1024, enc_depth=24, enc_num_heads=16,
                    dec_embed_dim=512, dec_depth=8, dec_num_heads=16,
                    mlp_ratio=4.0, norm_pix_loss=True, mask_ratio=0.75)
    defaults.update(kwargs)
    return MaskedAutoencoder(**defaults)


def mae_vit_huge_patch14(**kwargs) -> MaskedAutoencoder:
    """ViT-H/14 MAE -- reference only (very large)."""
    defaults = dict(img_size=224, patch_size=14,
                    enc_embed_dim=1280, enc_depth=32, enc_num_heads=16,
                    dec_embed_dim=512, dec_depth=8, dec_num_heads=16,
                    mlp_ratio=4.0, norm_pix_loss=True, mask_ratio=0.75)
    defaults.update(kwargs)
    return MaskedAutoencoder(**defaults)
